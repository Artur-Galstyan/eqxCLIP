from typing import Optional, Tuple, Union
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


bottleneck_expansion = 4


class Bottleneck(eqx.nn.StatefulLayer):
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    relu1: eqx.nn.Lambda

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    relu2: eqx.nn.Lambda

    avgpool: eqx.nn.AvgPool2d | eqx.nn.Identity

    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    relu3: eqx.nn.Lambda

    downsample: eqx.nn.Sequential | None

    def __init__(self, inplanes, planes, key: PRNGKeyArray, stride=1):
        super().__init__()

        key, *subkeys = jax.random.split(key, 12)

        self.conv1 = eqx.nn.Conv2d(
            inplanes, planes, 1, use_bias=False, key=subkeys[0]
        )

        self.bn1 = eqx.nn.BatchNorm(planes, axis_name="batch")
        self.relu1 = eqx.nn.Lambda(jax.nn.relu)

        self.conv2 = eqx.nn.Conv2d(
            planes, planes, 3, padding=1, use_bias=False, key=subkeys[1]
        )
        self.bn2 = eqx.nn.BatchNorm(planes, axis_name="batch")
        self.relu2 = eqx.nn.Lambda(jax.nn.relu)

        self.avgpool = (
            eqx.nn.AvgPool2d(kernel_size=stride, stride=stride)
            if stride > 1
            else eqx.nn.Identity()
        )

        self.conv3 = eqx.nn.Conv2d(
            planes,
            planes * bottleneck_expansion,
            1,
            use_bias=False,
            key=subkeys[2],
        )
        bn3 = eqx.nn.BatchNorm(planes * bottleneck_expansion, axis_name="batch")
        # Replace bn3.weight with zeros as per the "initialize_parameters" function
        # defined here: https://github.com/openai/CLIP/blob/main/clip/model.py#L312
        new_bn3_weight = jnp.zeros_like(bn3.weight)
        bn3_where = lambda x: x.weight
        self.bn3 = eqx.tree_at(bn3_where, bn3, new_bn3_weight)

        self.relu3 = eqx.nn.Lambda(jax.nn.relu)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * bottleneck_expansion:
            self.downsample = eqx.nn.Sequential(
                [
                    eqx.nn.AvgPool2d(stride, stride),
                    eqx.nn.Conv2d(
                        inplanes,
                        planes * bottleneck_expansion,
                        1,
                        stride=1,
                        use_bias=False,
                        key=subkeys[3],
                    ),
                    eqx.nn.BatchNorm(
                        planes * bottleneck_expansion, axis_name="batch"
                    ),
                ]
            )

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        identity = x
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = self.relu1(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = self.relu2(out)

        out = self.avgpool(out)
        out = self.conv3(out)
        out, state = self.bn3(out, state)

        if self.downsample is not None:
            identity, state = self.downsample(x, state=state)

        out += identity
        out = self.relu3(out)
        return out, state


class DirectPositionalEmbedding(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    spacial_dim: int = eqx.field(static=True)
    weight: Array

    def __init__(self, embed_dim: int, spacial_dim: int, key: PRNGKeyArray):
        super().__init__()
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim
        self.weight = (
            jax.random.normal(key, (spacial_dim**2 + 1, embed_dim))
            / embed_dim**0.5
        )

    def __call__(self, x: Float[Array, "spacial_dim**2+1 embed_dim"]) -> Array:
        return x + self.weight


class AttentionPool2d(eqx.Module):
    spacial_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    output_dim: int | None = eqx.field(static=True)

    positional_embedding: DirectPositionalEmbedding
    mha: eqx.nn.MultiheadAttention

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        key, *subkeys = jax.random.split(key, 5)
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.positional_embedding = DirectPositionalEmbedding(
            embed_dim=embed_dim, spacial_dim=spacial_dim, key=subkeys[0]
        )
        self.mha = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            output_size=output_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=subkeys[1],
        )
        # TODO: Initialize the weights of the mha layer as per the
        # "initialize_parameters" function defined here:
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L312

    def __call__(self, x: Float[Array, "embed_dim h w"]) -> Array:
        x = x.reshape((self.spacial_dim**2, self.embed_dim))
        x = jnp.concatenate([jnp.mean(x, axis=0, keepdims=True), x])
        x = self.positional_embedding(x)
        x = self.mha(query=x[:1], key_=x, value=x)
        x = x.reshape(-1)
        return x


class ModifiedResNet(eqx.Module):
    output_dim: int = eqx.field(static=True)
    input_resolution: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    _inplanes: int = eqx.field(static=False)

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    relu1: eqx.nn.Lambda

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    relu2: eqx.nn.Lambda

    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    relu3: eqx.nn.Lambda

    avgpool: eqx.nn.AvgPool2d

    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential

    attnpool: AttentionPool2d

    def __init__(
        self,
        layers,
        output_dim: int,
        heads: int,
        input_resolution: int = 224,
        width: int = 64,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.width = width

        key, *subkeys = jax.random.split(key, 20)

        # the 3-layer stem
        self.conv1 = eqx.nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
            key=subkeys[0],
        )

        self.bn1 = eqx.nn.BatchNorm(width // 2, axis_name="batch")
        self.relu1 = eqx.nn.Lambda(jax.nn.relu)

        self.conv2 = eqx.nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )

        self.bn2 = eqx.nn.BatchNorm(width // 2, axis_name="batch")
        self.relu2 = eqx.nn.Lambda(jax.nn.relu)
        self.conv3 = eqx.nn.Conv2d(
            width // 2,
            width,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=subkeys[2],
        )
        self.bn3 = eqx.nn.BatchNorm(width, axis_name="batch")
        self.relu3 = eqx.nn.Lambda(jax.nn.relu)

        self.avgpool = eqx.nn.AvgPool2d(2, stride=2)

        # # residual layers
        self._inplanes = (
            width  # this is a *mutable* variable used during construction
        )

        self.layer1 = self._make_layer(width, layers[0], key=subkeys[-1])
        self.layer2 = self._make_layer(
            width * 2, layers[1], stride=2, key=subkeys[-2]
        )
        self.layer3 = self._make_layer(
            width * 4, layers[2], stride=2, key=subkeys[-3]
        )
        self.layer4 = self._make_layer(
            width * 8, layers[3], stride=2, key=subkeys[-4]
        )
        #
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32,
            embed_dim,
            heads,
            output_dim,
            key=subkeys[-5],
        )

    def _make_layer(
        self, planes, blocks, key: PRNGKeyArray, stride=1
    ) -> eqx.nn.Sequential:
        key, *subkeys = jax.random.split(key, 20)
        layers = [Bottleneck(self._inplanes, planes, subkeys[0], stride)]

        self._inplanes = planes * bottleneck_expansion

        for i in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, subkeys[i + 1]))

        return eqx.nn.Sequential(layers)

    def __call__(
        self, x: Array, state: eqx.nn.State
    ) -> Tuple[Array, eqx.nn.State]:
        def stem(x_inner, state_inner):
            conv1 = self.conv1(x_inner)
            x_inner, state_inner = self.bn1(conv1, state=state_inner)
            x_inner = self.relu1(x_inner)

            conv2 = self.conv2(x_inner)
            x_inner, state_inner = self.bn2(conv2, state=state_inner)
            x_inner = self.relu2(x_inner)

            conv3 = self.conv3(x_inner)
            x_inner, state_inner = self.bn3(conv3, state=state_inner)
            x_inner = self.relu3(x_inner)
            x_inner = self.avgpool(x_inner)
            return x_inner, state_inner

        x, state = stem(x, state)
        x, state = self.layer1(x, state=state)
        x, state = self.layer2(x, state=state)
        x, state = self.layer3(x, state=state)
        x, state = self.layer4(x, state=state)
        x = self.attnpool(x)

        return x, state


class ResidualAttentionBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    ln_1: eqx.nn.LayerNorm
    mlp: eqx.nn.Sequential
    ln_2: eqx.nn.LayerNorm
    attn_mask: Array | None

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Optional[Array] = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        key, *subkeys = jax.random.split(key, 12)
        self.attn = eqx.nn.MultiheadAttention(n_head, d_model, key=subkeys[0])
        self.ln_1 = eqx.nn.LayerNorm(shape=(d_model))
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(d_model, d_model * 4, key=subkeys[1]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(d_model * 4, d_model, key=subkeys[2]),
            ]
        )
        self.ln_2 = eqx.nn.LayerNorm(shape=(d_model))

        self.attn_mask = attn_mask

    def __call__(self, x: Array, *, key: PRNGKeyArray):
        seq_len, d_model = x.shape
        ln_1 = jax.vmap(self.ln_1)(x)
        attn = self.attn(ln_1, ln_1, ln_1, mask=self.attn_mask)
        x = x + attn
        ln_2 = jax.vmap(self.ln_2)(x)
        mlp = jax.vmap(self.mlp)(ln_2)
        x = x + mlp
        return x


class Transformer(eqx.Module):
    width: int = eqx.field(static=True)
    layers: int = eqx.field(static=True)

    resblocks: eqx.nn.Sequential

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Optional[Array] = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        key, *subkeys = jax.random.split(key, layers + 1)
        self.resblocks = eqx.nn.Sequential(
            [
                ResidualAttentionBlock(width, heads, attn_mask, key=subkeys[i])
                for i in range(layers)
            ]
        )

    def __call__(self, x: Float[Array, "seq_len heads d_model"]):
        return self.resblocks(x)


class ClassEmbedding(eqx.Module):
    width: int = eqx.field(static=True)
    weight: Array

    def __init__(self, width: int, key: PRNGKeyArray):
        super().__init__()
        self.width = width
        scale = width**-0.5
        self.weight = scale * jax.random.normal(key, shape=(width,))

    def __call__(self, x: Array) -> Array:
        x = jnp.concatenate(
            [
                self.weight.reshape(1, -1) + jnp.zeros(shape=(1, x.shape[-1])),
                x,
            ],
        )
        return x


class PositionalEmbeddingVIT(eqx.Module):
    width: int = eqx.field(static=True)
    input_resolution: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    weight: Array

    def __init__(
        self,
        width: int,
        input_resolution: int,
        patch_size: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size

        scale = width**-0.5

        self.weight = scale * jax.random.normal(
            key=key, shape=((input_resolution // patch_size) ** 2 + 1, width)
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.weight
        return x


class Projection(eqx.Module):
    width: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    weight: Array

    def __init__(
        self, width: int, output_dim: int, *, key: PRNGKeyArray
    ) -> None:
        self.width = width
        self.output_dim = output_dim
        scale = width**-0.5

        self.weight = scale * jax.random.normal(
            key=key, shape=(width, output_dim)
        )

    def __call__(self, x: Array) -> Array:
        x = x @ self.weight
        return x


class VisionTransformer(eqx.Module):
    input_resolution: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d

    class_embedding: ClassEmbedding
    positional_embedding: PositionalEmbeddingVIT

    ln_pre: eqx.nn.LayerNorm
    transformer: Transformer
    ln_post: eqx.nn.LayerNorm

    proj: Projection | None

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 9)
        self.input_resolution = input_resolution
        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=False,
            key=subkeys[0],
        )

        self.class_embedding = ClassEmbedding(width, key=subkeys[1])
        self.positional_embedding = PositionalEmbeddingVIT(
            width, input_resolution, patch_size, key=subkeys[2]
        )

        self.ln_pre = eqx.nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, key=subkeys[3])

        self.ln_post = eqx.nn.LayerNorm(width)
        self.proj = Projection(width, output_dim, key=subkeys[4])

    def __call__(self, x: Array, *, state: eqx.nn.State) -> Array:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], -1)  # shape = [*, width, grid ** 2]
        x = jnp.transpose(x)
        x = self.class_embedding(x)
        x = self.positional_embedding(x)
        x = jax.vmap(self.ln_pre)(x)
        x = self.transformer(x)
        x = self.ln_post(x[0, :])
        if self.proj is not None:
            x = self.proj(x)
        return x


class PositionalEmbeddingCLIP(eqx.Module):
    weight: Array

    def __init__(self, context_length: int, width: int, key: PRNGKeyArray):
        sigma = 0.01
        self.weight = sigma * jax.random.normal(
            key, shape=(context_length, width)
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.weight
        return x


class TextProjection(eqx.Module):
    weight: Array

    def __init__(self, width: int, embed_dim: int, *, key: PRNGKeyArray):
        std = width**-0.5
        self.weight = std * jax.random.normal(key=key, shape=(width, embed_dim))

    def __call__(self, x: Array, text: Array) -> Array:
        x = x[jnp.argmax(text, axis=-1)] @ self.weight
        return x


class CLIP(eqx.Module):
    context_length: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    logit_scale: Array

    visual: VisionTransformer | ModifiedResNet
    transformer: Transformer

    token_embedding: eqx.nn.Embedding
    positional_embedding: PositionalEmbeddingCLIP
    ln_final: eqx.nn.LayerNorm
    text_projection: TextProjection

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: Optional[int],
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        *,
        key: PRNGKeyArray,
    ):
        self.context_length = context_length
        key, *subkeys = jax.random.split(key, 10)
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                key=subkeys[0],
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                key=subkeys[0],
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            key=subkeys[1],
        )

        self.vocab_size = vocab_size
        token_embedding_std = 0.02
        self.token_embedding = eqx.nn.Embedding(
            weight=token_embedding_std
            * jax.random.normal(
                subkeys[2], shape=(vocab_size, transformer_width)
            )
        )
        self.positional_embedding = PositionalEmbeddingCLIP(
            context_length, transformer_width, key=subkeys[3]
        )
        self.ln_final = eqx.nn.LayerNorm(transformer_width)
        self.text_projection = TextProjection(
            width=transformer_width, embed_dim=embed_dim, key=subkeys[4]
        )
        self.logit_scale = jnp.ones([]) * jnp.log(1 / 0.07)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        mask = jnp.full(
            shape=(self.context_length, self.context_length),
            fill_value=float("-inf"),
        )
        mask = jnp.triu(mask, k=1)
        return mask

    def encode_image(
        self, image: Array, state: Optional[eqx.nn.State] = None
    ) -> Array:
        return self.visual(image, state=state)

    def encode_text(self, text):
        x = jax.vmap(self.token_embedding)(text)  # [batch_size, n_ctx, d_model]
        x = self.positional_embedding(x)

        x = self.transformer(x)
        x = jax.vmap(self.ln_final)(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_projection(x, text)
        return x

    def __call__(self, image, text, state: Optional[eqx.nn.State] = None):
        if state is not None:
            image_features, state = self.encode_image(image, state=state)
        else:
            image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / jnp.linalg.norm(
            x=image_features, keepdims=True
        )
        text_features = text_features / jnp.linalg.norm(
            x=text_features, keepdims=True
        )

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # shape = [global_batch_size, global_batch_size]
        if state is not None:
            return logits_per_image, logits_per_text, state
        else:
            return logits_per_image, logits_per_text

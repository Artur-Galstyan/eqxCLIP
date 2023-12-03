import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Bottleneck(eqx.Module):
    expansion: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    relu1: eqx.nn.Lambda

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    relu2: eqx.nn.Lambda

    avgpool: eqx.nn.AvgPool2d

    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    relu3: eqx.nn.Lambda

    downsample: eqx.nn.Sequential | None

    def __init__(self, inplanes, planes, key: PRNGKeyArray, stride=1):
        super().__init__()
        self.expansion = 4

        key, *subkeys = jax.random.split(key, 12)

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = eqx.nn.Conv2d(inplanes, planes, 1, use_bias=False, key=subkeys[0])

        self.bn1 = eqx.nn.BatchNorm(planes, axis_name=0)
        self.relu1 = eqx.nn.Lambda(jax.nn.relu)

        self.conv2 = eqx.nn.Conv2d(
            planes, planes, 3, padding=1, use_bias=False, key=subkeys[1]
        )
        self.bn2 = eqx.nn.BatchNorm(planes, axis_name=0)
        self.relu2 = eqx.nn.Lambda(jax.nn.relu)

        self.avgpool = eqx.nn.AvgPool2d(stride) if stride > 1 else eqx.nn.Identity()

        self.conv3 = eqx.nn.Conv2d(
            planes, planes * self.expansion, 1, use_bias=False, key=subkeys[2]
        )
        self.bn3 = eqx.nn.BatchNorm(planes * self.expansion, axis_name=0)
        self.relu3 = eqx.nn.Lambda(jax.nn.relu)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * self.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = eqx.nn.Sequential(
                [
                    eqx.nn.AvgPool2d(stride),
                    eqx.nn.Conv2d(
                        inplanes,
                        planes * self.expansion,
                        1,
                        stride=1,
                        use_bias=False,
                        key=subkeys[3],
                    ),
                    eqx.nn.BatchNorm(planes * self.expansion, axis_name=0),
                ]
            )

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
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
            identity = self.downsample(x)

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
                jax.random.normal(key, (spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5
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
            output_dim: int = None,
            *,
            key: PRNGKeyArray
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

    def __call__(self, x: Float[Array, "embed_dim h w"]) -> Array:
        x = x.reshape((self.spacial_dim ** 2, self.embed_dim))
        x = jnp.concatenate([jnp.mean(x, axis=0, keepdims=True), x])
        x = self.positional_embedding(x)
        x = self.mha(query=x[:1], key_=x, value=x)
        x = x.reshape(-1)
        return x

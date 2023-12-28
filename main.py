import jax.random
import jax.numpy as jnp
import equinox as eqx
from jaxclip.model import (
    CLIP,
    ResidualAttentionBlock,
    Transformer,
    VisionTransformer,
)


def main():
    key = jax.random.PRNGKey(42)
    seq_len = 77
    spacial_dim = 7
    embed_dim = 2048
    num_heads = 32
    output_dim = 1024

    layers = (3, 4, 6, 3)
    output_dim = 1024
    heads = 32
    input_resolution = 224
    width = 64
    x = jax.random.normal(key, (8, 3, 224, 224))

    # modified_resnet, state = eqx.nn.make_with_state(ModifiedResnet)(
    #     layers=layers,
    #     output_dim=output_dim,
    #     heads=heads,
    #     input_resolution=input_resolution,
    #     width=width,
    #     key=key,
    # )
    # state = eqx.filter_vmap(lambda: state, axis_size=8)()
    # y = eqx.filter_vmap(modified_resnet, axis_name="batch")(x, state)

    d_model = 512
    n_heads = 8
    mask_shape = (77, 77)
    mask = jnp.tril(jnp.zeros(mask_shape), k=0) - jnp.triu(
        jnp.full(mask_shape, float("inf")), k=1
    )

    x_shape = (seq_len, n_heads, d_model)
    x = jax.random.normal(key, x_shape)
    width = 512
    layers = 12

    # t = Transformer(
    #     width=width,
    #     layers=layers,
    #     heads=n_heads,
    #     attn_mask=mask,
    #     key=key,
    # )
    #
    # y = t(x)
    batch_size = 16
    n_images = 3
    input_resolution = 224
    patch_size = 32
    width = 768
    layers = 12
    heads = 12
    output_dim = 512
    x = jax.random.normal(
        key, (batch_size, n_images, input_resolution, input_resolution)
    )
    key, subkey = jax.random.split(key)

    # res = ResidualAttentionBlock(
    #     d_model, n_head=heads, key=subkey, attn_mask=mask
    # )
    #
    # test_x = jax.random.normal(key, (seq_len, d_model))
    # res(test_x, key=subkey)
    # vit = VisionTransformer(
    #     input_resolution=input_resolution,
    #     patch_size=patch_size,
    #     width=width,
    #     layers=layers,
    #     heads=heads,
    #     output_dim=output_dim,
    #     key=key,
    # )
    #
    # y = eqx.filter_vmap(vit)(x)
    embed_dim = 512
    image_resolution = 224
    vision_layers = 12
    vision_width = 768
    vision_patch_size = 32
    context_length = 77
    vocab_size = 49408
    transformer_width = 512
    transformer_heads = 8
    transformer_layers = 12
    text_x = jnp.ones(shape=(77,), dtype=jnp.int32)
    clip = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        key=key,
    )

    x = jax.random.normal(key, (n_images, input_resolution, input_resolution))
    key, subkey = jax.random.split(key)
    clip(x, text_x)


if __name__ == "__main__":
    main()

import jax.random
import jax.numpy as jnp
import equinox as eqx
from jaxclip.model import ModifiedResnet, ResidualAttentionBlock


def main():
    key = jax.random.PRNGKey(42)
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

    x_shape = (77, 8, 512)
    x = jax.random.normal(key, x_shape)
    res_attn_block = ResidualAttentionBlock(
        d_model=d_model, n_head=n_heads, attn_mask=mask, key=key
    )
    y = res_attn_block(x)


if __name__ == "__main__":
    main()

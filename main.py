import equinox as eqx
import jax.random

from eqxclip.model import Bottleneck, AttentionPool2d
import icecream

icecream.install()


def main():
    key = jax.random.PRNGKey(42)
    spacial_dim = 7
    embed_dim = 2048
    num_heads = 32
    output_dim = 1024

    x = jax.random.normal(key, (8, 2048, 7, 7))

    attn = AttentionPool2d(
        spacial_dim=spacial_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        output_dim=output_dim,
        key=key,
    )

    x = eqx.filter_vmap(attn)(x)
    ic(x.shape)


if __name__ == "__main__":
    main()

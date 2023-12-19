import jax.random

from eqxclip.model import ModifiedResnet


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
    x = jax.random.normal(key, (8, 2048, 7, 7))

    modified_resnet = ModifiedResnet(
        layers=layers,
        output_dim=output_dim,
        heads=heads,
        input_resolution=input_resolution,
        width=width,
        key=key,
    )


if __name__ == "__main__":
    main()

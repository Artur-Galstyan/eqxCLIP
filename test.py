# %%
from jaxclip import clip
import jax
import matplotlib.pyplot as plt
import os
import skimage
from PIL import Image
import equinox as eqx
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(0)
model, preprocess = clip.load("ViT-B/32", key=key)
# model, preprocess = clip.load("RN50")

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


image = preprocess(Image.open("CLIP.png"))
print(image.shape)
image = jnp.transpose(image, axes=(2, 1, 0))
text = clip.tokenize(["a diagram", "a dog", "a cat"])
print(f"{image.shape=}, {image.dtype=}")
print(f"{text.shape=}, {text.dtype=}")


image_features = model.encode_image(image)
text_features = eqx.filter_vmap(model.encode_text)(text)
logits_per_image, logits_per_text = eqx.filter_vmap(model, in_axes=(None, 0))(
    image, text
)
probs = jax.nn.softmax(logits_per_image)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

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
model, preprocess = clip.load("ViT-B/16", key=key)
# model, preprocess = clip.load("RN50")

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# images in skimage to use and their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer",
}
original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))

for filename in [
    filename
    for filename in os.listdir(skimage.data_dir)
    if filename.endswith(".png") or filename.endswith(".jpg")
]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

    original_images.append(image)
    preprocessed_image = jnp.transpose(preprocess(image), axes=(2, 1, 0))
    images.append(preprocessed_image)
    texts.append(descriptions[name])


image_input = jax.lax.stop_gradient(jnp.stack(images))
text_tokens = clip.tokenize(["This is " + desc for desc in texts])
print(f"{image_input.shape=}, {type(image_input)=}")
assert isinstance(image_input, jnp.ndarray)
image_features = eqx.filter_vmap(model.encode_image)(image_input)
print(text_tokens.shape)
text_features = eqx.filter_vmap(model.encode_text)(text_tokens)

from jaxclip import CLIP
import torch
import numpy as np
import icecream
import matplotlib.pyplot as plt
import os
import skimage
from PIL import Image
import sys

icecream.install()

ic(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
# model, preprocess = clip.load("RN50")
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
)
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
    images.append(preprocess(image))
    texts.append(descriptions[name])

image_input = torch.tensor(np.stack(images))
text_tokens = clip.tokenize(["This is " + desc for desc in texts])

with torch.no_grad():
    # make images from shape 8, 3, 224, 224 to 16, 3, 224, 224 by doubling
    # the images and add 1 more image to make it 17, 3, 224, 224
    image_input = torch.cat([image_input, image_input, image_input[:1]])
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

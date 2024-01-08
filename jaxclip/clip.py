import hashlib
import operator
import os
import pickle
import urllib.request
import warnings
from functools import reduce, partial
from typing import List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, PRNGKeyArray, PyTree
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import cli, tqdm

from jaxclip.model import CLIP
from jaxclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from jaxclip.utils.pytorch_to_eqx_loading_utils import (
    load_model_from_state_dict,
)
from skimage.transform import resize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC  # noqa: F821
__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()


_MODELS = {
    "RN50": "https://storage.googleapis.com/ml_model_weights/CLIP/1d9c67e05c7a39d55e98769e64eb4ec056fd173df65aab69a18f793dd3da1960/RN50.pt",
    "RN101": "https://storage.googleapis.com/ml_model_weights/CLIP/4157a8f0cbb9ec0e6b5428555d0ad2ef65bed630d7f7a515272a40c01e451747/RN101.pt",
    "RN50x4": "https://storage.googleapis.com/ml_model_weights/CLIP/3b3b040f3b21424c55fe3953b910dd1fa945e9ea351bd6bdb447ebac27826e62/RN50x4.pt",
    "RN50x16": "https://storage.googleapis.com/ml_model_weights/CLIP/7c3669feac5b6cc9ae624648637c5fe79ca008257117782fe0a9d89da2ace60a/RN50x16.pt",
    "RN50x64": "https://storage.googleapis.com/ml_model_weights/CLIP/937ca86e302c770395a4d178c06c1d2067c4cd38956538dc7e848aa4fc31c413/RN50x64.pt",
    "ViT-B/32": "https://storage.googleapis.com/ml_model_weights/CLIP/56cb1d183b5460243b1f8cac644ad96309177d4128618c59b213be530628aa26/ViT-B-32.pt",
    "ViT-B/16": "https://storage.googleapis.com/ml_model_weights/CLIP/97fe4078ceebef511c0e08dc21471f0ca3b6cb6daa6193abc7cd2aedf28fb58b/ViT-B-16.pt",
    "ViT-L/14": "https://storage.googleapis.com/ml_model_weights/CLIP/264c8ee78b6705e24b1c01768d60e93320d844442202ccbe9f0272a74bac2097/ViT-L-14.pt",
    "ViT-L/14@336px": "https://storage.googleapis.com/ml_model_weights/CLIP/883325c9b33bad86fc315dbdb89294f2e6f3a4c043c83bc64d2114b5ea5c5afc/ViT-L-14%40336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file"
        )

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(
        download_target, "wb"
    ) as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def load(
    name: str,
    download_root: str = None,
    *,
    key: PRNGKeyArray = None,
) -> Array:
    if name in _MODELS:
        model_path = _download(
            _MODELS[name],
            download_root or os.path.expanduser("~/.cache/jaxclip"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )
    try:
        state_dict: dict = pickle.load(open(model_path, "rb"))
    except Exception:
        raise RuntimeError(
            f"Failed to load model {name} from {model_path}; please check the file format"
        )

    assert (
        state_dict is not None
    ), "Failed to load model, because state_dict is None"

    if key is None:
        key = jax.random.PRNGKey(0)

    embed_dim = state_dict["embed_dim"]
    image_resolution = state_dict["image_resolution"]
    vision_layers = state_dict["vision_layers"]
    vision_width = state_dict["vision_width"]
    vision_patch_size = (
        state_dict["vision_patch_size"]
        if "vision_patch_size" in state_dict
        else None
    )
    context_length = state_dict["context_length"]
    transformer_width = state_dict["transformer_width"]
    transformer_heads = state_dict["transformer_heads"]
    transformer_layers = state_dict["transformer_layers"]
    vocab_size = state_dict["vocab_size"]

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
    # if isinstance(vision_layers, (tuple, list)):
    #     clip = load_model_from_state_dict(state_dict, clip, visual="resnet")
    # else:
    #     clip = load_model_from_state_dict(state_dict, clip, visual="vit")

    return clip, _transform(image_resolution)


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


# Copied from original CLIP repository, with minor modifications
def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> Union[Array, Array]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [
        [sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts
    ]

    result = jnp.zeros(shape=(len(all_tokens), context_length), dtype=jnp.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result = result.at[i, : len(tokens)].set(
            jnp.array(tokens, dtype=jnp.int32)
        )
    return result


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def jax_preprocess(
    n_px: int,
    image: Image.Image,
) -> Array:
    image = _convert_image_to_rgb(image)
    image = np.array(image)

    image = resize(image, (n_px, n_px), anti_aliasing=True)
    image = jnp.array(image)

    mean = jnp.array((0.48145466, 0.4578275, 0.40821073))
    std = jnp.array((0.26862954, 0.26130258, 0.27577711))
    image = (image - mean) / std

    return image


def _transform(n_px):
    return partial(jax_preprocess, n_px)

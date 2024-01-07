from jaxtyping import PyTree
import numpy as np
import jax.numpy as jnp
import equinox as eqx


def get_attrs_to_delete(visual="resnet"):
    attrs_to_delete = []
    if visual == "resnet":
        attrs_to_delete.append("visual.bn1.running_mean")
        attrs_to_delete.append("visual.bn1.running_var")
        attrs_to_delete.append("visual.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.bn2.running_mean")
        attrs_to_delete.append("visual.bn2.running_var")
        attrs_to_delete.append("visual.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.bn3.running_mean")
        attrs_to_delete.append("visual.bn3.running_var")
        attrs_to_delete.append("visual.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.0.bn1.running_mean")
        attrs_to_delete.append("visual.layer1.0.bn1.running_var")
        attrs_to_delete.append("visual.layer1.0.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.0.bn2.running_mean")
        attrs_to_delete.append("visual.layer1.0.bn2.running_var")
        attrs_to_delete.append("visual.layer1.0.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.0.bn3.running_mean")
        attrs_to_delete.append("visual.layer1.0.bn3.running_var")
        attrs_to_delete.append("visual.layer1.0.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.0.downsample.1.running_mean")
        attrs_to_delete.append("visual.layer1.0.downsample.1.running_var")
        attrs_to_delete.append(
            "visual.layer1.0.downsample.1.num_batches_tracked"
        )
        attrs_to_delete.append("visual.layer1.1.bn1.running_mean")
        attrs_to_delete.append("visual.layer1.1.bn1.running_var")
        attrs_to_delete.append("visual.layer1.1.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.1.bn2.running_mean")
        attrs_to_delete.append("visual.layer1.1.bn2.running_var")
        attrs_to_delete.append("visual.layer1.1.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.1.bn3.running_mean")
        attrs_to_delete.append("visual.layer1.1.bn3.running_var")
        attrs_to_delete.append("visual.layer1.1.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.2.bn1.running_mean")
        attrs_to_delete.append("visual.layer1.2.bn1.running_var")
        attrs_to_delete.append("visual.layer1.2.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.2.bn2.running_mean")
        attrs_to_delete.append("visual.layer1.2.bn2.running_var")
        attrs_to_delete.append("visual.layer1.2.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer1.2.bn3.running_mean")
        attrs_to_delete.append("visual.layer1.2.bn3.running_var")
        attrs_to_delete.append("visual.layer1.2.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.0.bn1.running_mean")
        attrs_to_delete.append("visual.layer2.0.bn1.running_var")
        attrs_to_delete.append("visual.layer2.0.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.0.bn2.running_mean")
        attrs_to_delete.append("visual.layer2.0.bn2.running_var")
        attrs_to_delete.append("visual.layer2.0.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.0.bn3.running_mean")
        attrs_to_delete.append("visual.layer2.0.bn3.running_var")
        attrs_to_delete.append("visual.layer2.0.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.0.downsample.1.running_mean")
        attrs_to_delete.append("visual.layer2.0.downsample.1.running_var")
        attrs_to_delete.append(
            "visual.layer2.0.downsample.1.num_batches_tracked"
        )
        attrs_to_delete.append("visual.layer2.1.bn1.running_mean")
        attrs_to_delete.append("visual.layer2.1.bn1.running_var")
        attrs_to_delete.append("visual.layer2.1.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.1.bn2.running_mean")
        attrs_to_delete.append("visual.layer2.1.bn2.running_var")
        attrs_to_delete.append("visual.layer2.1.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.1.bn3.running_mean")
        attrs_to_delete.append("visual.layer2.1.bn3.running_var")
        attrs_to_delete.append("visual.layer2.1.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.2.bn1.running_mean")
        attrs_to_delete.append("visual.layer2.2.bn1.running_var")
        attrs_to_delete.append("visual.layer2.2.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.2.bn2.running_mean")
        attrs_to_delete.append("visual.layer2.2.bn2.running_var")
        attrs_to_delete.append("visual.layer2.2.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.2.bn3.running_mean")
        attrs_to_delete.append("visual.layer2.2.bn3.running_var")
        attrs_to_delete.append("visual.layer2.2.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.3.bn1.running_mean")
        attrs_to_delete.append("visual.layer2.3.bn1.running_var")
        attrs_to_delete.append("visual.layer2.3.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.3.bn2.running_mean")
        attrs_to_delete.append("visual.layer2.3.bn2.running_var")
        attrs_to_delete.append("visual.layer2.3.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer2.3.bn3.running_mean")
        attrs_to_delete.append("visual.layer2.3.bn3.running_var")
        attrs_to_delete.append("visual.layer2.3.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.0.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.0.bn1.running_var")
        attrs_to_delete.append("visual.layer3.0.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.0.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.0.bn2.running_var")
        attrs_to_delete.append("visual.layer3.0.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.0.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.0.bn3.running_var")
        attrs_to_delete.append("visual.layer3.0.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.0.downsample.1.running_mean")
        attrs_to_delete.append("visual.layer3.0.downsample.1.running_var")
        attrs_to_delete.append(
            "visual.layer3.0.downsample.1.num_batches_tracked"
        )
        attrs_to_delete.append("visual.layer3.1.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.1.bn1.running_var")
        attrs_to_delete.append("visual.layer3.1.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.1.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.1.bn2.running_var")
        attrs_to_delete.append("visual.layer3.1.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.1.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.1.bn3.running_var")
        attrs_to_delete.append("visual.layer3.1.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.2.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.2.bn1.running_var")
        attrs_to_delete.append("visual.layer3.2.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.2.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.2.bn2.running_var")
        attrs_to_delete.append("visual.layer3.2.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.2.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.2.bn3.running_var")
        attrs_to_delete.append("visual.layer3.2.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.3.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.3.bn1.running_var")
        attrs_to_delete.append("visual.layer3.3.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.3.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.3.bn2.running_var")
        attrs_to_delete.append("visual.layer3.3.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.3.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.3.bn3.running_var")
        attrs_to_delete.append("visual.layer3.3.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.4.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.4.bn1.running_var")
        attrs_to_delete.append("visual.layer3.4.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.4.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.4.bn2.running_var")
        attrs_to_delete.append("visual.layer3.4.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.4.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.4.bn3.running_var")
        attrs_to_delete.append("visual.layer3.4.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.5.bn1.running_mean")
        attrs_to_delete.append("visual.layer3.5.bn1.running_var")
        attrs_to_delete.append("visual.layer3.5.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.5.bn2.running_mean")
        attrs_to_delete.append("visual.layer3.5.bn2.running_var")
        attrs_to_delete.append("visual.layer3.5.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer3.5.bn3.running_mean")
        attrs_to_delete.append("visual.layer3.5.bn3.running_var")
        attrs_to_delete.append("visual.layer3.5.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.0.bn1.running_mean")
        attrs_to_delete.append("visual.layer4.0.bn1.running_var")
        attrs_to_delete.append("visual.layer4.0.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.0.bn2.running_mean")
        attrs_to_delete.append("visual.layer4.0.bn2.running_var")
        attrs_to_delete.append("visual.layer4.0.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.0.bn3.running_mean")
        attrs_to_delete.append("visual.layer4.0.bn3.running_var")
        attrs_to_delete.append("visual.layer4.0.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.0.downsample.1.running_mean")
        attrs_to_delete.append("visual.layer4.0.downsample.1.running_var")
        attrs_to_delete.append(
            "visual.layer4.0.downsample.1.num_batches_tracked"
        )
        attrs_to_delete.append("visual.layer4.1.bn1.running_mean")
        attrs_to_delete.append("visual.layer4.1.bn1.running_var")
        attrs_to_delete.append("visual.layer4.1.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.1.bn2.running_mean")
        attrs_to_delete.append("visual.layer4.1.bn2.running_var")
        attrs_to_delete.append("visual.layer4.1.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.1.bn3.running_mean")
        attrs_to_delete.append("visual.layer4.1.bn3.running_var")
        attrs_to_delete.append("visual.layer4.1.bn3.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.2.bn1.running_mean")
        attrs_to_delete.append("visual.layer4.2.bn1.running_var")
        attrs_to_delete.append("visual.layer4.2.bn1.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.2.bn2.running_mean")
        attrs_to_delete.append("visual.layer4.2.bn2.running_var")
        attrs_to_delete.append("visual.layer4.2.bn2.num_batches_tracked")
        attrs_to_delete.append("visual.layer4.2.bn3.running_mean")
        attrs_to_delete.append("visual.layer4.2.bn3.running_var")
        attrs_to_delete.append("visual.layer4.2.bn3.num_batches_tracked")
        #    attrs_to_delete.append("visual.attnpool.k_proj.weight")
        attrs_to_delete.append("visual.attnpool.k_proj.bias")
        #    attrs_to_delete.append("visual.attnpool.q_proj.weight")
        attrs_to_delete.append("visual.attnpool.q_proj.bias")
        #    attrs_to_delete.append("visual.attnpool.v_proj.weight")
        attrs_to_delete.append("visual.attnpool.v_proj.bias")
        #    attrs_to_delete.append("visual.attnpool.c_proj.weight")
        attrs_to_delete.append("visual.attnpool.c_proj.bias")
    if visual == "vit":
        attrs_to_delete.append(
            "visual.transformer.resblocks.0.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.0.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.1.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.1.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.2.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.2.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.3.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.3.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.4.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.4.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.5.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.5.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.6.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.6.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.7.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.7.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.8.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.8.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.9.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.9.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.10.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.10.attn.out_proj.bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.11.attn.in_proj_bias"
        )
        attrs_to_delete.append(
            "visual.transformer.resblocks.11.attn.out_proj.bias"
        )
    attrs_to_delete.append("transformer.resblocks.0.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.0.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.1.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.1.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.2.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.2.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.3.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.3.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.4.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.4.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.5.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.5.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.6.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.6.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.7.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.7.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.8.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.8.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.9.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.9.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.10.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.10.attn.out_proj.bias")
    attrs_to_delete.append("transformer.resblocks.11.attn.in_proj_bias")
    attrs_to_delete.append("transformer.resblocks.11.attn.out_proj.bias")
    attrs_to_delete.append("embed_dim")
    attrs_to_delete.append("image_resolution")
    attrs_to_delete.append("vision_layers")
    attrs_to_delete.append("vision_width")
    attrs_to_delete.append("transformer_width")
    attrs_to_delete.append("transformer_heads")
    attrs_to_delete.append("transformer_layers")
    return attrs_to_delete


def rename_mapping_key(mapping, from_, to):
    cp = dict(mapping)
    cp[to] = cp[from_]
    del cp[from_]
    return cp


def get_nested_attr(pytree, parts):
    if len(parts) == 0:
        return pytree
    part = parts[0]
    if part.isdigit():
        return get_nested_attr(pytree[int(part)], parts[1:])
    elif part != "":
        return get_nested_attr(getattr(pytree, part), parts[1:])


def rename_transformer_resblocks_mlp(mapping, i, visual=False):
    prefix = "visual." if visual else ""
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.mlp.c_fc.weight",
        f"{prefix}transformer.resblocks.{i}.mlp.0.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.mlp.c_fc.bias",
        f"{prefix}transformer.resblocks.{i}.mlp.0.bias",
    )
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.mlp.c_proj.weight",
        f"{prefix}transformer.resblocks.{i}.mlp.2.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.mlp.c_proj.bias",
        f"{prefix}transformer.resblocks.{i}.mlp.2.bias",
    )
    return mapping


def rename_transformer_resblocks_attn_proj(mapping, i, visual=False):
    prefix = "visual." if visual else ""
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.attn.in_proj_weight",
        f"{prefix}transformer.resblocks.{i}.attn.query_proj.weight",
    )
    mapping[
        f"{prefix}transformer.resblocks.{i}.attn.key_proj.weight"
    ] = f"{prefix}transformer.resblocks.{i}.attn.key_proj.weight"
    mapping[
        f"{prefix}transformer.resblocks.{i}.attn.value_proj.weight"
    ] = f"{prefix}transformer.resblocks.{i}.attn.value_proj.weight"
    mapping = rename_mapping_key(
        mapping,
        f"{prefix}transformer.resblocks.{i}.attn.out_proj.weight",
        f"{prefix}transformer.resblocks.{i}.attn.output_proj.weight",
    )
    return mapping


def rename_mapping_keys_in_layers(mapping, n_layers, rename_fn):
    if n_layers == 0:
        return mapping
    else:
        mapping = rename_fn(mapping, n_layers - 1)
        return rename_mapping_keys_in_layers(mapping, n_layers - 1, rename_fn)


def rename_visual_attn_pool_mha(mapping):
    mapping = rename_mapping_key(
        mapping,
        "visual.attnpool.k_proj.weight",
        "visual.attnpool.mha.key_proj.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        "visual.attnpool.q_proj.weight",
        "visual.attnpool.mha.query_proj.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        "visual.attnpool.v_proj.weight",
        "visual.attnpool.mha.value_proj.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        "visual.attnpool.c_proj.weight",
        "visual.attnpool.mha.output_proj.weight",
    )
    return mapping


def rename_visual_downsample_weights(mapping, i):
    mapping = rename_mapping_key(
        mapping,
        f"visual.layer{i + 1}.0.downsample.0.weight",
        f"visual.layer{i + 1}.0.downsample.1.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        f"visual.layer{i + 1}.0.downsample.1.weight",
        f"visual.layer{i + 1}.0.downsample.2.weight",
    )
    mapping = rename_mapping_key(
        mapping,
        f"visual.layer{i + 1}.0.downsample.1.bias",
        f"visual.layer{i + 1}.0.downsample.2.bias",
    )
    return mapping


def visual_rename_transformer_resblocks_attn_proj(mapping, i):
    return rename_transformer_resblocks_attn_proj(mapping, i, visual=True)


def visual_rename_transformer_resblocks_mlp(mapping, i):
    return rename_transformer_resblocks_mlp(mapping, i, visual=True)


def load_model_from_state_dict(
    state_dict: dict, clip: PyTree, visual: str
) -> PyTree:
    attrs_to_delete = get_attrs_to_delete(visual)
    for attr in attrs_to_delete:
        del state_dict[attr]
    n_layers = 12

    proj_weights = []
    for i in range(n_layers):
        attn_in_proj_weight = state_dict[
            f"transformer.resblocks.{i}.attn.in_proj_weight"
        ].numpy()
        query_weight, key_weight, value_weight = np.split(
            attn_in_proj_weight, 3, axis=0
        )
        proj_weights.append((query_weight, key_weight, value_weight))

    mapping = {k: k for k in state_dict}
    mapping = rename_mapping_keys_in_layers(
        mapping, n_layers, rename_transformer_resblocks_mlp
    )
    mapping = rename_mapping_keys_in_layers(
        mapping, n_layers, rename_transformer_resblocks_attn_proj
    )
    if visual == "resnet":
        mapping = rename_mapping_keys_in_layers(
            mapping, 4, rename_visual_downsample_weights
        )
        mapping = rename_visual_attn_pool_mha(mapping)
    if visual == "vit":
        mapping = rename_mapping_keys_in_layers(
            mapping, n_layers, visual_rename_transformer_resblocks_attn_proj
        )
        mapping = rename_mapping_keys_in_layers(
            mapping, n_layers, visual_rename_transformer_resblocks_mlp
        )

    for i in range(n_layers):
        q, k, v = proj_weights[i]
        state_dict[f"transformer.resblocks.{i}.attn.query_proj.weight"] = q
        state_dict[f"transformer.resblocks.{i}.attn.key_proj.weight"] = k
        state_dict[f"transformer.resblocks.{i}.attn.value_proj.weight"] = v

    if visual == "vit":
        for i in range(n_layers):
            attn_in_proj_weight = state_dict[
                f"visual.transformer.resblocks.{i}.attn.in_proj_weight"
            ].numpy()
            query_weight, key_weight, value_weight = np.split(
                attn_in_proj_weight, 3, axis=0
            )
            proj_weights.append((query_weight, key_weight, value_weight))
            for i in range(n_layers):
                q, k, v = proj_weights[i]
                state_dict[
                    f"visual.transformer.resblocks.{i}.attn.query_proj.weight"
                ] = q
                state_dict[
                    f"visual.transformer.resblocks.{i}.attn.key_proj.weight"
                ] = k
                state_dict[
                    f"visual.transformer.resblocks.{i}.attn.value_proj.weight"
                ] = v

    for k in mapping:
        if len(k.split(".")) == 1:
            # skip scalars
            continue
        obj = get_nested_attr(clip, k.split("."))
        if obj is None:
            print(k)
        try:
            clip = eqx.tree_at(
                where=lambda x: get_nested_attr(x, k.split(".")),
                pytree=clip,
                replace=state_dict[mapping[k]],
            )
        except:
            print("ERROR: ", k, state_dict[mapping[k]])

    return clip

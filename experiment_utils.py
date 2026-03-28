VALID_ATTENTION_MODES = ("none", "pre_hidden", "cnn_fusion")
VALID_ATTENTION_SCALES = ("1/2", "1/4", "1/8", "1/16")


def parse_attention_scales(mode, scales_arg):
    if mode not in VALID_ATTENTION_MODES:
        raise ValueError(f"Unsupported attention mode: {mode}")

    if mode == "none":
        return tuple()

    if scales_arg:
        raw_scales = [item.strip() for item in scales_arg.split(",") if item.strip()]
    elif mode == "pre_hidden":
        raw_scales = ["1/16"]
    else:
        raw_scales = ["1/8", "1/4", "1/2"]

    seen = set()
    normalized = []
    for scale in raw_scales:
        if scale not in VALID_ATTENTION_SCALES:
            valid = ", ".join(VALID_ATTENTION_SCALES)
            raise ValueError(f"Unsupported attention scale '{scale}'. Valid values: {valid}")
        if scale not in seen:
            normalized.append(scale)
            seen.add(scale)

    if not normalized:
        raise ValueError("At least one attention scale is required when attention is enabled.")

    return tuple(normalized)


def apply_attention_config(config, mode, scales, reduction):
    config.attention_mode = mode
    config.attention_scales = tuple(scales)
    config.attention_reduction = reduction
    return config


def build_attention_suffix(mode, scales, reduction=None):
    if mode == "none":
        return ""
    scale_tag = "-".join(scale.replace("/", "_") for scale in scales)
    suffix = f"_attn-{mode}-{scale_tag}"
    if reduction is not None:
        suffix += f"-r{reduction}"
    return suffix

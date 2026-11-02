"""Dependency-free validation shared by the thin worker and ML runner."""

CATEGORIES = {
    "full_sleeves_shirt": "top",
    "half_sleeves_shirt": "top",
    "sweater": "top",
    "full_pant": "bottom",
    "half_pant": "bottom",
    "short_skirt": "bottom",
    "long_skirt": "bottom",
    "long_gown": "dress",
    "half_gown": "dress",
    "jacket": "outerwear",
    **{name: name for name in ("auto", "top", "bottom", "dress", "outerwear")},
}


def validate_params(params):
    mode = params.get("tryon_mode", "clothing")
    if mode not in ("clothing", "accessory"):
        return "invalid tryon_mode"
    for field, limit in (
        ("accessory_category", 100),
        ("contact_sentence", 500),
        ("garment_prompt", 500),
    ):
        value = params.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            return f"invalid {field}"
    if mode == "accessory":
        if not (params.get("accessory_category") or "").strip():
            return "accessory_category is required"
        if (
            params.get("garment_type") is not None
            or params.get("garment_prompt") is not None
        ):
            return "garment fields are only valid for clothing mode"
    else:
        if (
            params.get("accessory_category") is not None
            or params.get("contact_sentence") is not None
        ):
            return "accessory fields are only valid for accessory mode"
        if params.get("garment_type") is not None and (
            not isinstance(params["garment_type"], str)
            or params["garment_type"] not in CATEGORIES
        ):
            return "invalid garment_type"
        if params.get("garment_type") and params.get("garment_prompt"):
            return "provide either garment_type or garment_prompt, not both"
    return None

from app.providers.vision import VisionProvider
from app.schemas import VisualMaterial


def extract_visual_materials(
    visual_inputs: list,
    vision_provider: VisionProvider | None = None,
) -> list[VisualMaterial]:
    """
    Normalize VisualMaterial inputs and run VisionProvider on raw file specs.

    Each item in `visual_inputs` can be a pre-built VisualMaterial, a dict with
    `source_path` (and optional `material_type`, `timestamp`), or a path string.
    """
    if not visual_inputs:
        return []

    results: list[VisualMaterial] = []
    for item in visual_inputs:
        if isinstance(item, VisualMaterial):
            if (item.extracted_text or item.summary) or vision_provider is None:
                results.append(item)
            else:
                results.append(
                    vision_provider.analyze_material(
                        item.source_path,
                        material_type=item.material_type,
                        timestamp=item.timestamp,
                    )
                )
        elif isinstance(item, dict):
            if vision_provider is None:
                raise ValueError(
                    "vision_provider is required to analyze raw visual inputs."
                )
            results.append(
                vision_provider.analyze_material(
                    str(item["source_path"]),
                    material_type=item.get("material_type", "slide"),
                    timestamp=item.get("timestamp"),
                )
            )
        else:
            if vision_provider is None:
                raise ValueError(
                    "vision_provider is required to analyze raw visual inputs."
                )
            results.append(vision_provider.analyze_material(str(item)))

    return results


def format_visual_context(materials: list[VisualMaterial]) -> str:
    if not materials:
        return ""

    blocks = []
    for idx, mat in enumerate(materials, start=1):
        ts = f"{mat.timestamp:.2f}s" if mat.timestamp is not None else "unknown"
        parts = [
            f"[Visual {idx} | type={mat.material_type} | t={ts} | source={mat.source_path}]"
        ]
        if mat.summary:
            parts.append(f"Summary: {mat.summary}")
        if mat.extracted_text:
            parts.append(f"Text: {mat.extracted_text}")
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)

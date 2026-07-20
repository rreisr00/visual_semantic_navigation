"""Stable presentation labels and plotting style for experiments."""
from __future__ import annotations

METHOD_LABELS = {
    "random_baseline": "Baseline aleatorio",
    "nearest_node_baseline": "Baseline por proximidad",
    "room_label_baseline": "Baseline por etiqueta de habitación",
    "single_view_siglip": "SigLIP con una vista",
    "multiview_siglip": "SigLIP multivista",
    "siglip_with_objects": "SigLIP con objetos detectados",
    "siglip_with_objects_and_relations": "SigLIP con objetos y relaciones",
    "hybrid_semantic_retrieval": "Recuperación semántica híbrida",
}

PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
METHOD_COLORS = {
    "random_baseline": "#9a9a92",
    "nearest_node_baseline": "#77776f",
    "room_label_baseline": "#b0b0aa",
    "single_view_siglip": PALETTE[0],
    "multiview_siglip": PALETTE[1],
    "siglip_with_objects": PALETTE[2],
    "siglip_with_objects_and_relations": PALETTE[3],
    "hybrid_semantic_retrieval": PALETTE[5],
}


def apply_plot_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.autolayout": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def save_figure(figure, directory, name: str) -> str:
    from pathlib import Path
    target = Path(directory) / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    return str(target)

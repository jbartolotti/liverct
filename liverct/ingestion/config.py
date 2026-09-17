"""YAML configuration for the CT archive ingestion workflow."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "archive": {"session_from": "study_date"},
    "tiering": {
        "min_z_extent_mm": 200.0,
        "min_num_slices": 50,
        "abdomen_terms": ["ABD", "ABDOMEN", "ABD/PEL", "A/P", "PELV", "PELVIS", "TORSO", "CAP"],
        "reject_terms": ["SCOUT", "LOCALIZER", "DOSE", "SAGITTAL", "CORONAL", "REFORMAT", "PROTOCOL"],
        "strong_abdomen_terms": ["ABDOMEN", "ABD", "ABD/PEL", "ABDOMEN/PELVIS", "A/P", "CAP", "TORSO", "TRUNK"],
        "supporting_abdomen_terms": ["PELVIS", "PELV", "LIVER", "HEPATIC", "PORTAL", "RENAL", "KIDNEY", "PANCREAS"],
        "anatomy_exclude_terms": ["HEAD", "BRAIN", "NECK", "C-SPINE", "CERVICAL", "SINUS", "FACIAL", "MAXILLOFACIAL", "CHEST ONLY", "LUNG ONLY", "UPPER EXTREMITY", "LOWER EXTREMITY", "LEG", "FEMUR", "KNEE", "ANKLE", "FOOT", "ARM", "ELBOW", "WRIST", "HAND", "SHOULDER", "CALF", "TIBIA", "TIB/FIB"],
        "derived_exclude_terms": ["DERIVED", "SECONDARY", "LOCALIZER", "SCOUT", "MIP", "MINIP", "VOLUME", "3D", "SCREENSHOT", "CORONAL", "SAGITTAL", "REFORMAT", "RECON", "MPR", "CURVED", "OBLIQUE"],
        "auto_min_score": 110,
        "auto_min_margin": 10,
        "required_modality": "CT",
    },
    "review": {"thumbnail_count": 6, "window_min": -200, "window_max": 300, "max_montage_width": 4096, "detailed_review": False},
    "staging": {"mode": "copy"},
}


@dataclass
class IngestionConfig:
    """Normalized ingestion settings loaded from YAML."""

    values: Dict[str, Any] = field(default_factory=lambda: _copy_defaults())
    path: Optional[Path] = None

    @property
    def archive(self) -> Dict[str, Any]:
        return self.values["archive"]

    @property
    def tiering(self) -> Dict[str, Any]:
        return self.values["tiering"]

    @property
    def review(self) -> Dict[str, Any]:
        return self.values["review"]

    @property
    def staging(self) -> Dict[str, Any]:
        return self.values["staging"]


def _copy_defaults() -> Dict[str, Any]:
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: Optional[Path] = None) -> IngestionConfig:
    """Load a YAML configuration, applying defaults to omitted sections."""
    values = _copy_defaults()
    if path is None:
        logger.info("Using default ingestion configuration (version=%s)", values.get("config_version", "1"))
        return IngestionConfig(values=values)

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML ingestion configuration requires PyYAML") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Ingestion configuration must contain a YAML mapping")

    for section, section_values in loaded.items():
        if section not in values:
            values[section] = section_values
        elif isinstance(section_values, dict):
            values[section].update(section_values)
        else:
            values[section] = section_values
    values["config_version"] = loaded.get("config_version", "1")
    logger.info("Loaded ingestion configuration: path=%s version=%s", config_path, values["config_version"])
    return IngestionConfig(values=values, path=config_path)

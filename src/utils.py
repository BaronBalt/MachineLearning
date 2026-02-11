import re
from pathlib import Path

_VERSION_PATTERN = re.compile(r"v(\d+)\.")

def get_versioned_paths(
        name: str,
        path: str,
        suffix: str,
) -> list[tuple[int, Path]]:
    """
    Returns [(version, path), ...] sorted by version ascending.
    """
    base_dir = Path(path) / name
    if not base_dir.exists():
        return []

    versions = []

    for p in base_dir.glob(f"v*{suffix}"):
        match = _VERSION_PATTERN.match(p.name)
        if match:
            versions.append((int(match.group(1)), p))

    return sorted(versions, key=lambda x: x[0])

def get_latest_model_path(
        model_name: str,
        models_dir: str,
        suffix: str = ".pkl",
) -> Path | None:
    versions = get_versioned_paths(model_name, models_dir, suffix)
    return versions[-1][1] if versions else None

def get_next_model_path(
        model_name: str,
        models_dir: str,
        suffix: str = ".pkl",
) -> Path:
    model_dir = Path(models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    versions = get_versioned_paths(model_name, models_dir, suffix)
    next_version = versions[-1][0] + 1 if versions else 1

    return model_dir / f"v{next_version}{suffix}"

def load_processed(root=".", path="data/processed/features.csv"):
    import os, pandas as pd
    full_path = os.path.join(root, path)
    return pd.read_csv(full_path)
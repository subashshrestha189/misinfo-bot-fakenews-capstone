from pathlib import Path

def ensure_dir(p: str | Path) -> Path:
    """
    Make sure a directory exists and return it as Path.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

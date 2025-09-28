import math


def hash_bucket(x: str, buckets: int = 1_000_003) -> int:
    """Hash a string to a stable bucket id within [0, buckets)."""
    return (hash(x) % buckets)


def norm_dense(v: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Normalize a scalar dense feature value."""
    return (v - mean) / (std if std > 0 else 1.0)

# Note: For large-scale runs, estimate stats via sampled files or online methods.


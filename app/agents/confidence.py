import hashlib
import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

def bucket_null_pct(null_pct: float) -> str:
    """Bucket null percentage into ranges for stable hashing."""
    if null_pct == 0.0:
        return "0%"
    elif null_pct <= 0.05:
        return "0-5%"
    elif null_pct <= 0.15:
        return "5-15%"
    elif null_pct <= 0.30:
        return "15-30%"
    else:
        return "30%+"

def column_signature(column_name: str, dtype: str, null_pct: float) -> str:
    """
    Generate a stable hash for a column based on its name, data type, and missingness bucket.
    This allows the confidence engine to reuse rules across datasets when encountering
    similar column structures.
    """
    null_bucket = bucket_null_pct(null_pct)
    raw = f"{column_name.lower()}|{str(dtype).lower()}|{null_bucket}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

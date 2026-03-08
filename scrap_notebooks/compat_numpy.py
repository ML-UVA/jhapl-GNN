import numpy as _np

# Reintroduce deprecated numpy types if missing
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64  # or _np.float64
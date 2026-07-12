"""Shared encoder + loss, re-exported from the parent DL_model package.

This pivot reuses the validated V4 encoder and SupCon loss WITHOUT forking them.
Any module under summer_work/ should do:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summer_work/
    from shared import RFEncoder, SupervisedContrastiveLoss, get_temperature

or, when the script already has summer_work/ on sys.path:

    from shared import RFEncoder, SupervisedContrastiveLoss, get_temperature
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../DL_model/summer_work
_PARENT = os.path.dirname(_HERE)                            # .../DL_model
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imported from the PARENT DL_model/model.py and DL_model/losses.py — shared, not copied.
from model import RFEncoder                                  # noqa: E402
from losses import SupervisedContrastiveLoss, get_temperature  # noqa: E402

__all__ = ["RFEncoder", "SupervisedContrastiveLoss", "get_temperature"]

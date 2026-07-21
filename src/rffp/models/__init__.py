"""Model zoo: the shared dual-branch encoder, the SupCon loss, and the generalized encoder.

Re-exports the encoder + loss so callers can simply::

    from rffp.models import RFEncoder, SupervisedContrastiveLoss, get_temperature

(This replaces the legacy ``summer_work/shared.py`` re-export shim.)
"""
from rffp.models.encoder import RFEncoder
from rffp.models.losses import SupervisedContrastiveLoss, get_temperature

__all__ = ["RFEncoder", "SupervisedContrastiveLoss", "get_temperature"]

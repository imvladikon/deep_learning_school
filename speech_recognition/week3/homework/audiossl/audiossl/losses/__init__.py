from audiossl.losses.infonce import InfoNCELoss, ContrastiveLoss
from audiossl.losses.ncl_losses import SimSiamLoss, BarlowTwinsLoss, VICRegLoss
from audiossl.losses.classification import LabelSmoothingCrossEntropy

__all__ = [
    "InfoNCELoss",
    "ContrastiveLoss",
    "SimSiamLoss",
    "BarlowTwinsLoss",
    "VICRegLoss",
    "LabelSmoothingCrossEntropy",
]

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
from .FocalLoss import FocalLoss
from .Loss import Loss
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, PoissonNLLLoss, NLLLoss2d, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss
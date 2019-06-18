# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
from .Embedding import Embedding, EmbeddingConf
from .BiLSTM import BiLSTM, BiLSTMConf
from .BiLSTMLast import BiLSTMLast, BiLSTMLastConf
from .BiGRU import BiGRU, BiGRUConf
from .BiGRULast import BiGRULast, BiGRULastConf
from .Linear import Linear, LinearConf
from .BaseLayer import BaseLayer, BaseConf
from .BiLSTMAtt import BiLSTMAtt, BiLSTMAttConf
from .BiQRNN import BiQRNN, BiQRNNConf
from .Conv import Conv, ConvConf
from .Pooling import Pooling, PoolingConf
from .ConvPooling import ConvPooling, ConvPoolingConf

from .Dropout import Dropout, DropoutConf

from .Conv2D import Conv2D, Conv2DConf
from .Pooling2D import Pooling2D, Pooling2DConf

from .embedding import CNNCharEmbedding, CNNCharEmbeddingConf

from .CRF import CRFConf, CRF

from .attentions import FullAttention, FullAttentionConf
from .attentions import Seq2SeqAttention, Seq2SeqAttentionConf
from .attentions import LinearAttention, LinearAttentionConf       # The output rank of this layer can be either unchanged or reduced
from .attentions import BiAttFlow, BiAttFlowConf
from .attentions import MatchAttention, MatchAttentionConf
from .attentions import Attention, AttentionConf
from .attentions import BilinearAttention, BilinearAttentionConf
from .attentions import Interaction, InteractionConf

# Operators
from .op import *

# Math operations
from .math import Add2D, Add2DConf
from .math import Add3D, Add3DConf
from .math import Minus2D, Minus2DConf
from .math import Minus3D, Minus3DConf
from .math import ElementWisedMultiply2D, ElementWisedMultiply2DConf
from .math import ElementWisedMultiply3D, ElementWisedMultiply3DConf
from .math import MatrixMultiply, MatrixMultiplyConf

# Transformer layer
from .Transformer import Transformer, TransformerConf

# Encoder Decoder classes
from .EncoderDecoder import EncoderDecoder, EncoderDecoderConf

from .normalizations import LayerNorm, LayerNormConf

from .HighwayLinear import HighwayLinear, HighwayLinearConf
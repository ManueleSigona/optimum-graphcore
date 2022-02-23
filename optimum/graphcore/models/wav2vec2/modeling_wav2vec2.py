# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

import poptorch
from optimum.utils import logging
from transformers import (
    Wav2Vec2ForPreTraining,
)
from .ipu_layer_drop import IPUWav2Vec2Encoder, IPUWav2Vec2EncoderStableLayerNorm, IPUWav2Vec2Adapter
from .ipu_gumbel_vector_quantizer import IPUWav2Vec2GumbelVectorQuantizer

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


@register(Wav2Vec2ForPreTraining)
class PipelinedWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining, PipelineMixin):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Inject IPU Layer Drop
        if config.do_stable_layer_norm:
            self.wav2vec2.encoder = IPUWav2Vec2EncoderStableLayerNorm(config)
        else:
            self.wav2vec2.encoder = IPUWav2Vec2Encoder(config)

        self.wav2vec2.adapter = IPUWav2Vec2Adapter(config) if config.add_adapter else None
        # Inject IPU Gumbel Vector Quantizer
        self.quantizer = IPUWav2Vec2GumbelVectorQuantizer(config)


    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        sampled_negative_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        ):
        return super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False)




        



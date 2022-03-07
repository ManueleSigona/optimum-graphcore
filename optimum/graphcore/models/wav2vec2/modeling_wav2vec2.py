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
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput
from .ipu_layer_drop import IPUWav2Vec2Encoder, IPUWav2Vec2EncoderStableLayerNorm, IPUWav2Vec2Adapter
from .ipu_gumbel_vector_quantizer import IPUWav2Vec2GumbelVectorQuantizer
from .ipu_wav2vec2_model import IPUWav2Vec2Model

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


@register(Wav2Vec2ForPreTraining)
class PipelinedWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining, PipelineMixin):
    def __init__(self, config) -> None:

        # Override return_dict in config
        
        super().__init__(config)

        self.wav2vec2 = IPUWav2Vec2Model(config)

        # Inject IPU Layer Drop
        # We also pad the sequence length for self-attention
        #     This makes the memory use across tiles more balanced
        if config.do_stable_layer_norm:
            self.wav2vec2.encoder = IPUWav2Vec2EncoderStableLayerNorm(
                config,
                sequence_length_padding_divisor=4
            )
        else:
            self.wav2vec2.encoder = IPUWav2Vec2Encoder(
                config,
                sequence_length_padding_divisor=4
            )

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
        return_dict=None,
    ):

        # Override the return_dict argument
        return_dict = False

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked

            # Change 1: temporary workaround while aten::all unsupported
            #neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            neg_is_pos = (quantized_features != negative_quantized_features).to(torch.int32).sum(-1).bool().logical_not()

            # Change 2: this if-statement is untraceable
            #if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )


    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        #non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        #non_padded_lengths = attention_mask.cumsum(dim=-1)[:, 249999]
        non_padded_lengths = attention_mask.sum(dim=-1)

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
    
    
    def _add_begin_block(self, module, name, ipu_id):

        module = poptorch.BeginBlock(module, name, ipu_id)


    def parallelize(self):

        super().parallelize()

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[0],
            name="Conv[0,2)", ipu_id=0
        )

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[2],
            name="Conv[2,3)", ipu_id=1
        )

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[3],
            name="Conv[3,7)", ipu_id=2
        )

        self._add_begin_block(
            self.wav2vec2.encoder.pos_conv_embed,
            name="PCE+EL[00,02)", ipu_id=3
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[2],
            name="EL[02,06)", ipu_id=4
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[6],
            name="EL[06,10)", ipu_id=5
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[10],
            name="EL[10,12)+ELQ", ipu_id=6
        )

        self._add_begin_block(
            self.quantizer,
            name="Quantizer+Losses", ipu_id=7
        )

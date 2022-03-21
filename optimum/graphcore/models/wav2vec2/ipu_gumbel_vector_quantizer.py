import poptorch
import torch
import warnings
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2GumbelVectorQuantizer,
)


def _ipu_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -(torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_() + 1e-4).log()
    )  # ~Gumbel(0,1)

    # Old workaround for missing `torch.exponential_`
    #exponential_distribution = torch.distributions.exponential.Exponential(1.0)
    #gumbels = -exponential_distribution.sample(logits.size()).log()

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]

        # Same `scatter_` bug as in IPUWav2Vec2GumbelVectorQuantizer inference
        # Needs to be fixed here too
        update_values = torch.ones_like(index, dtype=logits.dtype)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, update_values)
        
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    
    return ret
    
    
class IPUWav2Vec2GumbelVectorQuantizer(Wav2Vec2GumbelVectorQuantizer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        del self.temperature
        self.register_buffer("temperature", torch.Tensor([2.0]))

    def _get_temperature(self):

        decay = 0.999995
        end = 0.5

        decayed_temp = self.temperature * decay
        new_temp = torch.where(
            decayed_temp >= torch.Tensor([end]),
            decayed_temp, torch.Tensor([end])
        )

        self.temperature.copy_(new_temp)

        return poptorch.ipu_print_tensor(self.temperature, "gsvq_temp")[0]


    def forward(self, hidden_states, mask_time_indices=None):
        
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:

            temperature = self._get_temperature()

            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = _ipu_gumbel_softmax(
                hidden_states.float(), tau=temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)

            update_values = torch.ones_like(codevector_idx.view(-1, 1), dtype=hidden_states.dtype)
            
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), update_values
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
        codebook = self.codevectors[0, :, :]
        codebook = codebook.view(self.num_groups, self.num_vars, -1)
        codevectors = torch.einsum('lgv,gvd->lgd', codevector_probs, codebook)

        codevectors = codevectors.reshape(batch_size, sequence_length, -1)

        return codevectors, perplexity

import torch
from torch.nn import functional as F
import warnings
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2GumbelVectorQuantizer,
)

SERIALISATION_FACTOR = 4

def _ipu_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
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
    def forward(self, hidden_states, mask_time_indices=None):

        serialisation_factor = SERIALISATION_FACTOR
        
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = _ipu_gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)
            # Get the indices corresponding to entries with value 1
            index = codevector_probs.argmax(dim=-1)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            index = codevector_idx

            update_values = torch.ones_like(codevector_idx.view(-1, 1), dtype=hidden_states.dtype)
            
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), update_values
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors

        # First, pad codevector_probs with zeros on bs*seqlen axis
        # Want e.g. (999, 640) -> (1000, 640)

        bs_seqlen_product = batch_size * sequence_length
        remainder = bs_seqlen_product % serialisation_factor

        if not remainder == 0:
            padding_for_serialisation = serialisation_factor - remainder

        else:
            padding_for_serialisation = 0

        codevector_probs = F.pad(
            codevector_probs,
            # Pad with zeros on dim=-2, on the right
            (0, 0, 0, padding_for_serialisation),
            mode='constant',
            value=0.0
        )

        # Split codevector_probs into chunks

        items_per_chunk = (bs_seqlen_product + padding_for_serialisation) \
                          // serialisation_factor

        codevector_prob_chunks = torch.split(
            codevector_probs,
            # Use list here as sanity check
            split_size_or_sections=[
                 items_per_chunk for _ in range(serialisation_factor)
            ],
            dim=-2
        )

        codevectors_chunks = []

        # Do same mutiplication as before on each chunk

        for chunk in codevector_prob_chunks:

            codevectors_per_group_chunk = chunk.unsqueeze(-1) * self.codevectors
            codevectors_chunk = (
                codevectors_per_group_chunk.view(
                    items_per_chunk, self.num_groups, self.num_vars, -1
                )
                .sum(-2)
                # Move this view change to the very end until padding removed
                #.view(batch_size, sequence_length, -1)
            )

            codevectors_chunks.append(codevectors_chunk)

        # Each codevectors_chunk has shape (BS * SL / SF, num_groups, cvdim)
        #     where BS = (micro) batch size, SL = sequence length,
        #     SF = serialisation factor

        # so we concatenate on dim=-3

        codevectors = torch.cat(codevectors_chunks, dim=-3)

        # Remove padding
        # Need padding_for_serialisation > 0 condition since otherwise
        #     the slice is [:0] which isn't what we want!
        if padding_for_serialisation > 0:
            codevectors = codevectors[..., :-(padding_for_serialisation), :, :]

        # Perform delayed view change
        codevectors = codevectors.view(batch_size, sequence_length, -1)

        return codevectors, perplexity

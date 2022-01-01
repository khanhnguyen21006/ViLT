import math

import torch
import torch.nn as nn
import torch.onnx.operators

from collections import defaultdict
from typing import Dict


class SinusoidalPositionalEmbedding(nn.Module):
    """Construct sinusoidal positional embeddings of any length.
    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase. This allows attention to learn to use
    absolute and relative positions.
    Timing signals should be added to some precursors of both the query and
    the memory inputs to attention. The use of relative position is possible
    because sin(x+y) and cos(x+y) can be expressed in terms of y, sin(x) and
    cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we generate the
    two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in the
    channels dimension.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    We can think of max_ts as the max length in a text. In the default
    implementation, wavelengths form a geometric progression from 2π to
    10000⋅2π.
    """

    def __init__(self, vocab, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        init_size = init_size + 1  # for padding index
        weights = self.get_embedding(init_size, embedding_dim, padding_idx)
        self.onnx_trace = False
        self.register_buffer('weights', weights)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(n_embeds, embed_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        max_ts = 10000
        min_ts = 1
        n_timescales = embed_dim // 2
        increment = math.log(max_ts / min_ts) / (n_timescales - 1)
        # Example increment: 9 / 384 = 0.024

        timescales = torch.arange(n_timescales, dtype=torch.float)

        # inv_timescales ranges from 1 to 1/10000 with log spacing
        inv_timescales = min_ts * torch.exp(timescales * -increment)
        # inv_timescales.shape == [embed_size // 2]

        positions = torch.arange(n_embeds, dtype=torch.float).unsqueeze(1)
        # positions.shape ==  [n_embeds, 1]

        inv_timescales = inv_timescales.unsqueeze(0)
        # inv_timescales.shape == [1, embed_size // 2]

        scaled_time = positions * inv_timescales
        # scaled_time.shape == [n_embeds, embed_size // 2]

        sin_signal = torch.sin(scaled_time)
        cos_signal = torch.cos(scaled_time)
        signal = torch.cat([sin_signal, cos_signal], dim=1)
        # signal.shape == [n_embeds, embed_dim]

        # Ensure that embed_dim is even
        if embed_dim % 2 == 1:
            signal = torch.cat([signal, torch.zeros(n_embeds, 1)], dim=1)

        if padding_idx is not None:
            signal[padding_idx, :] = 0

        return signal

    def forward(self, X, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        batch_size, seq_len = X.shape
        if incremental_state is not None:
            start_pos = self._get_last_position(incremental_state)
            max_pos = start_pos + seq_len
            self._save_last_position(incremental_state, max_pos)
        else:
            start_pos = 0
            max_pos = seq_len

        # bsz, seq_len = torch.onnx.operators.shape_as_tensor(X)
        # Expand embeddings if needed
        max_pos = max_pos + 1
        if max_pos > self.weights.shape[0]:
            weights = self.get_embedding(max_pos, self.embedding_dim,
                                         self.padding_idx)
            # We need to manually move weights to GPU if needed
            weights = self.weights.new_tensor(weights)

            self.register_buffer('weights', weights)

        # if incremental_state is not None:
        #     # positions is the same for every token when decoding a single step
        #     pos = (timestep.int() + 1).long() if timestep is not None else seq_len
        #     if self.onnx_trace:
        #         return self.weights[self.padding_idx + pos, :].unsqueeze(1).repeat(bsz, 1, 1)
        #     return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            X, self.padding_idx, self.left_pad, self.onnx_trace)
        pos_mask = positions != self.padding_idx
        positions[pos_mask] = positions[pos_mask] + start_pos
        # if self.onnx_trace:
        #     flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
        #     embedding_shape = torch.cat(
        #         (bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
        #     embeddings = torch.onnx.operators.reshape_from_tensor_shape(
        #         flat_embeddings, embedding_shape)
        #     return embeddings

        embeds = self.weights.index_select(0, positions.view(-1))

        embeds = embeds.view(batch_size, seq_len, -1)
        return embeds.detach()

    def _get_last_position(self, incremental_state):
        last_pos = get_incremental_state(self, incremental_state, 'position')
        if last_pos is None:
            last_pos = 0
        return last_pos

    def _save_last_position(self, incremental_state, position):
        set_incremental_state(self, incremental_state, 'position', position)


def make_positions(X, padding_idx, left_pad, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_seq_len = X.shape[1]
    # torch._dim_arange is a temporary hack to allow tracing of arange like
    # constructs with dynamic bounds on arange.  Normal arange is not traceable
    # because it does not take any tensor inputs; if the range you need is
    # based on another tensor, calling this function directly will preserve
    # tracing.  Get rid of this when arange can directly take tensors for
    # bounds (so that it can be traced directly).
    if onnx_trace:
        range_buf = torch._dim_arange(like=X, dim=1) + padding_idx + 1
        mask = X.ne(padding_idx)
        positions = range_buf.expand_as(X)
        if left_pad:
            offsets = max_seq_len - mask.long().sum(dim=1).unsqueeze(1)
            positions = positions - offsets
        return positions * mask.long() + padding_idx * (1 - mask.long())

    max_pos = padding_idx + 1 + X.size(1)

    # Function attributes are used for caching
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = X.new()
    make_positions.range_buf = make_positions.range_buf.type_as(X)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = X.ne(padding_idx)
    positions = make_positions.range_buf[:X.size(1)].expand_as(X)
    if left_pad:
        offsets = max_seq_len - mask.long().sum(dim=1).unsqueeze(1)
        positions = positions - offsets
    return X.clone().masked_scatter_(mask, positions[mask])


# We assign a unique ID to the instance. Given the same class all instances
# will have different ID stored as `_fairseq_instance_id`. Examples of key
# names: LinearizedConvolution.0.input_buffer
INCREMENTAL_STATE_INSTANCE_ID: Dict[str, int] = defaultdict(lambda: 0)


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_wincremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)
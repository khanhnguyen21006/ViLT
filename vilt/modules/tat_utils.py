import torch.nn as nn
import torch.onnx.operators
import torch.nn.functional as F
import string
from collections import defaultdict
from typing import Dict

import pytorch_lightning as pl

import hashlib
import math
import os
import textstat
import torch
from nltk.tokenize import word_tokenize
from spacy.tokens import Doc


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


def get_incremental_state(module, incremental_state, key):
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


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_epoch-{epoch}_global_step-{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_readability_scores(text):
    scores = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'text_standard': textstat.text_standard(text, float_output=True),
        'difficult_words': textstat.difficult_words(text) / len(text.split()),
    }
    return scores


def spacize(text, cache, nlp):
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in cache:
        cache[key] = nlp(text).to_bytes()

    return Doc(nlp.vocab).from_bytes(cache[key])


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities


def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns


def get_narrative_productivity(text):
    doc = word_tokenize(text)
    doc = list(filter(is_word, doc))
    n_words = len(doc)
    n_terms = len(set(doc))

    scores = {
        'basic_ttr': basic_ttr(n_terms, n_words),
        'root_ttr': root_ttr(n_terms, n_words),
        'corrected_ttr': corrected_ttr(n_terms, n_words),
        'herdan': herdan(n_terms, n_words),
        'summer': summer(n_terms, n_words),
        'maas': maas(n_terms, n_words),
    }

    return scores


def basic_ttr(n_terms, n_words):
    """ Type-token ratio (TTR) computed as t/w, where t is the number of unique
    terms/vocab, and w is the total number of words.
    (Chotlos 1944, Templin 1957)
    """
    if n_words == 0:
        return 0
    return n_terms / n_words


def root_ttr(n_terms, n_words):
    """ Root TTR (RTTR) computed as t/sqrt(w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        Also known as Guiraud's R and Guiraud's index.
        (Guiraud 1954, 1960)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(n_words)


def corrected_ttr(n_terms, n_words):
    """ Corrected TTR (CTTR) computed as t/sqrt(2 * w), where t is the number of unique terms/vocab,
        and w is the total number of words.
        (Carrol 1964)
    """
    if n_words == 0:
        return 0
    return n_terms / math.sqrt(2 * n_words)


def herdan(n_terms, n_words):
    """ Computed as log(t)/log(w), where t is the number of unique terms/vocab, and w is the
        total number of words.
        Also known as Herdan's C.
        (Herdan 1960, 1964)
    """
    if n_words <= 1:
        return 0
    return math.log(n_terms) / math.log(n_words)


def summer(n_terms, n_words):
    """ Computed as log(log(t)) / log(log(w)), where t is the number of unique terms/vocab, and
        w is the total number of words.
        (Summer 1966)
    """
    try:
        math.log(math.log(n_terms)) / math.log(math.log(n_words))
    except ValueError:
        return 0


def maas(n_terms, n_words):
    """ Maas's TTR, computed as (log(w) - log(t)) / (log(w) * log(w)), where t is the number of
        unique terms/vocab, and w is the total number of words. Unlike the other measures, lower
        maas measure indicates higher lexical richness.
        (Maas 1972)
    """
    # We cap this score at 0.2
    if n_words <= 1:
        return 0.2
    score = (math.log(n_words) - math.log(n_terms)) / \
        (math.log(n_words) ** 2)
    return min(score, 0.2)


def is_word(tok):
    return tok not in string.punctuation
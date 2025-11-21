#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  

        self.e = lexicon.size(1)  # dimensionality of word embeddings
        self.rnn_dim = rnn_dim

        super().__init__(tagset, vocab, unigram)
        # keep lexicon as a fixed feature matrix on the right device/dtype
        self.register_buffer("E", lexicon.to(self.device, self.dtype))

    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]

        ## debug 

        self.A_base = nn.Parameter(torch.empty(self.k, self.k, device=self.device, dtype=self.dtype))
        nn.init.uniform_(self.A_base, 0, 0.01)
        self.A_pos = nn.Parameter(torch.zeros(self.k, self.k, device=self.device, dtype=self.dtype))
        self.A_word = nn.Parameter(torch.empty(self.k, self.k, self.e, device=self.device, dtype=self.dtype))
        nn.init.xavier_uniform_(self.A_word.view(self.k * self.k, self.e))

        self.B_base = nn.Parameter(torch.empty(self.k, self.V, device=self.device, dtype=self.dtype))
        nn.init.uniform_(self.B_base, 0, 0.01)
        self.B_context = nn.Parameter(torch.empty(self.k, self.e, device=self.device, dtype=self.dtype))
        nn.init.xavier_uniform_(self.B_context)

    @override
    def updateAB(self) -> None:
        # Your non-stationary A_at() and B_at() might not make any use of the
        # stationary A and B matrices computed by the parent.  So we override
        # the parent so that we won't waste time computing self.A, self.B.
        #
        # But if you decide that you want A_at() and B() at to refer to self.A
        # and self.B (for example, multiplying stationary and non-stationary
        # potentials), then you'll still need to compute them; in that case,
        # don't override the parent in this way.
        pass   

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        # You need to override this function to compute your non-stationary features.

        pos_feat = torch.tensor(float(position), device=self.device, dtype=self.dtype)
        w_j, _ = sentence[position]
        word_vec = self.E[w_j]
        # print(f"ConditionalRandomFieldTest.A_at: word_vec={word_vec}")

        lexical_term = torch.einsum("abg,g->ab", self.A_word, word_vec)
        scores = self.A_base + pos_feat * self.A_pos + lexical_term
        scores[self.eos_t, :] = -inf
        scores[:, self.bos_t] = -inf
        return torch.exp(scores)
        
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        # print(f"ConditionalRandomFieldTest.B_at: position={position}")

        w_j, _ = sentence[position]
        word_vec = self.E[w_j]
        scores = self.B_base.clone()
        if w_j != self.bos_w and w_j != self.eos_w:
            scores[:, w_j] = scores[:, w_j] + self.B_context @ word_vec
        scores[self.eos_t, :] = -inf
        scores[self.bos_t, :] = -inf
        return torch.exp(scores)

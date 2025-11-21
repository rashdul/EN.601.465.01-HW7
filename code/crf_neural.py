#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.E = lexicon
        self.e = self.E.size(1) # dimensionality of word's embeddings
        self.h_forward = []  # will hold forward RNN hidden states for current sentence
        self.h_backward = [] # will hold backward RNN hidden states for current sentence

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        # Initialize parameters for biRNN: M, M', U_a, U_b, theta_a, theta_b

        # print(f"all paraneters are: {self.k}, {self.V}, {self.e}, {self.rnn_dim}")

        embed_dim = self.E.shape[1]
        hidden_dim = 1 + 2*self.rnn_dim + 2*self.k
        if self.rnn_dim > 0:
            self.M = nn.Parameter(torch.empty(self.rnn_dim, self.rnn_dim + embed_dim + 1, device=self.device, dtype=self.dtype))
            self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, self.rnn_dim + embed_dim + 1, device=self.device, dtype=self.dtype))
            self.U_a = nn.Parameter(torch.empty(self.rnn_dim + 1, hidden_dim, device=self.device, dtype=self.dtype))
            # emission features: [1; tag_one_hot; word_vec; h_prefix; h_suffix]
            b_input_dim = 1 + self.k + embed_dim + 2 * self.rnn_dim
            self.U_b = nn.Parameter(torch.empty(self.rnn_dim + 1, b_input_dim, device=self.device, dtype=self.dtype))
            nn.init.xavier_uniform_(self.M)
            nn.init.xavier_uniform_(self.M_prime)
            nn.init.xavier_uniform_(self.U_a)
            nn.init.xavier_uniform_(self.U_b)
        else:
            self.M = nn.Parameter(torch.zeros(0, 0, device=self.device, dtype=self.dtype))
            self.M_prime = nn.Parameter(torch.zeros(0, 0, device=self.device, dtype=self.dtype))
            b_input_dim = 1 + self.k + embed_dim  # no h_prefix/h_suffix when rnn_dim=0
            self.U_a = nn.Parameter(torch.zeros(1, hidden_dim, device=self.device, dtype=self.dtype))
            self.U_b = nn.Parameter(torch.zeros(1, b_input_dim, device=self.device, dtype=self.dtype))

        # Final linear weights that map hidden features to a scalar score.
        hidden_out_dim = self.rnn_dim + 1
        self.theta_a = nn.Parameter(torch.empty(hidden_out_dim, device=self.device, dtype=self.dtype))
        self.theta_b = nn.Parameter(torch.empty(hidden_out_dim, device=self.device, dtype=self.dtype))
        nn.init.normal_(self.theta_a, mean=0.0, std=0.01)
        nn.init.normal_(self.theta_b, mean=0.0, std=0.01)
        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""
        n = len(isent)

        if self.rnn_dim == 0:
            zero = torch.zeros(0, device=self.device, dtype=self.dtype)
            self.h_forward = [zero for _ in range(n)]
            self.h_backward = [zero for _ in range(n)]
            return

        h_fwd = [torch.zeros(self.rnn_dim, device=self.device, dtype=self.dtype) for _ in range(n)]
        h_bwd = [torch.zeros(self.rnn_dim, device=self.device, dtype=self.dtype) for _ in range(n)]

        for j in range(1, n):
            w_j, _ = isent[j]
            x = self.E[w_j]
            # [1;h_{j-1};w_j]
            # print(f"shape of h_fwd[{j-1}]: {h_fwd[j-1].shape}, shape of x: {x.shape}")
            bracket = torch.cat((torch.ones(1, device=self.device, dtype=self.dtype), h_fwd[j-1], x))
            h_fwd[j] = torch.sigmoid(self.M @ bracket)

        for j in range(n-1, 0, -1):
            w_j, _ = isent[j]
            x = self.E[w_j]
            # [1;w_j; h'_{j}]
            bracket = torch.cat((torch.ones(1, device=self.device, dtype=self.dtype), x, h_bwd[j]))
            h_bwd[j-1] = torch.sigmoid(self.M_prime @ bracket)

        self.h_forward = h_fwd
        self.h_backward = h_bwd

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        ## avoid for loops

        # f_A = sigmoid( U_a [h_i-2, s, t, h'_i])
        tag_prev = self.eye.unsqueeze(1).expand(self.k, self.k, self.k)
        tag_curr = self.eye.unsqueeze(0).expand(self.k, self.k, self.k)
        h_prefix = self.h_forward[position - 2].view(1, 1, -1).expand(self.k, self.k, -1)
        h_suffix = self.h_backward[position].view(1, 1, -1).expand(self.k, self.k, -1)
        # print(f"h_prefix shape: {h_prefix.shape}, tag_prev shape: {tag_prev.shape}, tag_curr shape: {tag_curr.shape}, h_suffix shape: {h_suffix.shape}")
        ones = torch.ones(self.k, self.k, 1, device=self.device, dtype=self.dtype)
        features_A = torch.cat((ones, h_prefix, tag_prev, tag_curr, h_suffix), dim=2)  # [h_prefix; s_tag; t_tag; h_suffix]
        hidden_A = torch.sigmoid(features_A @ self.U_a.T)  # (k, k, rnn_dim+1)
        scores = torch.tensordot(hidden_A, self.theta_a, dims=([2], [0]))  # (k, k)
        scores[self.eos_t, :] = -inf
        scores[:, self.bos_t] = -inf
        return torch.exp(scores)
            
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        w_j, _ = sentence[position]
        word_vec = self.E[w_j].view(1, -1).expand(self.k, -1)                    # (k, e)
        tag_feat = self.eye                                                     # (k, k)
        h_prefix = self.h_forward[position - 1].view(1, -1).expand(self.k, -1)  # (k, rnn_dim)
        if position == len(sentence):
            h_suffix = torch.zeros(self.k, self.rnn_dim, device=self.device, dtype=self.dtype)
        else:
            h_suffix = self.h_backward[position].view(1, -1).expand(self.k, -1) # (k, rnn_dim)
        ones = torch.ones(self.k, 1, device=self.device, dtype=self.dtype)      # (k, 1)

        features_B = torch.cat((ones, tag_feat, word_vec, h_prefix, h_suffix), dim=1)  # (k, 1+k+e+2r)
        hidden_B = torch.sigmoid(features_B @ self.U_b.T)                                # (k, rnn_dim+1)
        scores = hidden_B @ self.theta_b                                                # (k,)

        B = torch.zeros(self.k, self.V, device=self.device, dtype=self.dtype)
        if w_j != self.eos_w and w_j != self.bos_w:
            B[:, w_j] = torch.exp(scores)
        B[self.eos_t, :] = 0.0
        B[self.bos_t, :] = 0.0
        return B

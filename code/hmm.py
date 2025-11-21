#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked


import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

torch.set_default_dtype(torch.float64)

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        self.bos_w: Optional[int] = vocab.index(BOS_WORD)
        self.eos_w: Optional[int] = vocab.index(EOS_WORD)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        
        self.eye: Tensor = torch.eye(self.k, device=self.device, dtype=self.dtype)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V, device=self.device, dtype=self.dtype)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k, device=self.device, dtype=self.dtype)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        # print(f"HiddenMarkovModel.A_at: position={position}")
        return self.A

    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        # print(f"HiddenMarkovModel.B_at: position={position}")
        return self.B
    
    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        # if self.B_counts[self.eos_t:self.bos_t, :].any() == 0:
        # print(f"emission counts: {self.B_counts}")
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A in the
        # unigram case.

        maskA = torch.ones_like(self.A_counts)
        maskA[:, self.bos_t] = 0.0   # no transitions to BOS
        maskA[self.eos_t, :] = 0.0   # no transitions from EOS

        masked_counts = self.A_counts * maskA + λ * maskA

        if self.unigram:
            # In unigram mode, p(t | s) = p(t) for all s.
            col_sums = masked_counts.sum(dim=0)  # total expected transitions into each tag
            denom = col_sums.sum()
            if denom.item() == 0.0:
                denom = torch.tensor(1.0, device=col_sums.device)
            distr = col_sums / denom
            self.A = distr.unsqueeze(0).repeat(self.k, 1)
            self.A[:, self.bos_t] = 0.0
            self.A[self.eos_t, :] = 0.0
        else:
            row_sums_A = masked_counts.sum(dim=1, keepdim=True)
            row_sums_A[row_sums_A == 0.0] = 1.0
            self.A = masked_counts / row_sums_A
            self.A[:, self.bos_t] = 0   # no tag→BOS
            self.A[self.eos_t, :] = 0   # no EOS→tag

        print("M-step updated A values:")
        for i in range(self.A.size(0)):
            print(f"{i}: {self.A[i, :]}")

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False, device=self.device, dtype=self.dtype)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False, device=self.device, dtype=self.dtype)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      

        self._save_time = time.time()      # mark start of training     
        dev_loss = loss(self)              # evaluate the model at the start of training
        old_dev_loss: float = dev_loss     # loss from the last epoch
        steps: int = 0   # total number of sentences the model has been trained on so far      
        while steps < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            if save_path: self.save(save_path, checkpoint=steps)  # save incompletely trained model in case we crash
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # Save the trained model.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(isent, mult=mult)

        # Check that forward and backward passes found the same total
        # probability of all paths (up to floating-point error).
        assert torch.isclose(log_Z_forward, log_Z_backward), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"


    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute any quantities needed for forward/backward/Viterbi algorithms.
        This method may be overridden in subclasses."""
        pass

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        alpha = [torch.zeros(self.k, device=self.device, dtype=self.dtype) for _ in isent]    
        alpha[0] = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.bos_t]  # vector that is one-hot at BOS_TAG

            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.
        # print("I am in forward pass, the first alpha values are:", alpha[1])
        n = len(isent)

        self.setup_sentence(isent)

        alpha[0][self.bos_t] = 1.0
        # print(f"deintegrized sentence  {len(isent)}")
        # scalling factor
        k = 0
        scale_factors = [torch.tensor(1.0, dtype=self.dtype, device=self.device) for _ in isent]  
        # print(f"self.A = {self.A},\n self.B = {self.B}")
        for j in range(1, n):
            w_j, t_obs = isent[j]
            A = self.A_at(j, isent)  
            B = self.B_at(j, isent)
            alpha_j = torch.zeros(self.k, device=self.device, dtype=self.dtype)
            if t_obs is not None:
                e_j = torch.zeros(self.k, device=self.device, dtype=self.dtype)
                if w_j != self.eos_w and w_j != self.bos_w:
                    e_j[t_obs] = B[t_obs, w_j]
            elif w_j != self.eos_w and w_j != self.bos_w:      
                e_j = B[:, w_j]
            elif w_j == self.eos_w:                           
                e_j = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.eos_t]
            else:                                             
                e_j = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.bos_t]


            alpha_j = (alpha[j - 1] @ A) * e_j


            # print(f"Type of alpha_j before scaling: {type(alpha_j)}, type of e_j: {type(e_j)}, type of alpha[j-1]: {type(alpha[j-1])}, type of A: {type(A)}")
            # scale = alpha_j.sum()
            # scale_factors.append(scale)
            # alpha_j = alpha_j / scale
            scale = alpha_j.sum().clamp_min(torch.tensor(1e-12, dtype=self.dtype, device=self.device))
            scale_factors[j] = scale
            if scale.item() == 0:
                scale = torch.tensor(1e-12, device=alpha_j.device)
            k = k + torch.log(scale)
            alpha[j] = alpha_j / scale

            # print(f"type of alpha_j after scaling: {type(alpha_j)}, type of alpha[j]: {type(alpha[j])}, tupe of scale: {type(scale)},")

            # k += torch.log(torch.sum(alpha_j))
            # alpha[j][t] += prob * alpha[j - 1][t_prev]
                # break
            # break
        # print(f"The final scaling factors are: {scale_factors}, and their logs sum to: {k}")
        self.scale_factors = scale_factors
        self.log_Z = k  
        self.alpha = alpha
        # Z without log
        # print(f"log alphas: {log_Z}")
        # print("The final alpha values are:", alpha)

        return k 

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the backward
        probability). 
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        beta = [torch.zeros(self.k, device=self.device, dtype=self.dtype) for _ in isent]
        beta[-1] = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.eos_t]  # vector that is one-hot at EOS_TAG

        n = len(isent)
        self.setup_sentence(isent)
        k = 0
        if not torch.isfinite(self.log_Z):
            return torch.tensor(float("-inf"), dtype=self.dtype, device=self.device)
        invZ = torch.exp((-self.log_Z).clamp(min=-50.0, max=50.0))

        
        maskA = torch.ones_like(self.A)
        maskA[:, self.bos_t] = 0.0
        maskA[self.eos_t, :] = 0.0

        for j in range(n - 1, 0, -1):
            w_j, t_obs = isent[j]

            A = self.A_at(j, isent)  
            B = self.B_at(j, isent)

            if t_obs is not None:
                # Use the actual emission potential for the observed tag.
                e_j = torch.zeros(self.k, device=self.device, dtype=self.dtype)
                if w_j != self.eos_w and w_j != self.bos_w:
                    e_j[t_obs] = B[t_obs, w_j]
            elif w_j != self.eos_w and w_j != self.bos_w:
                e_j = B[:, w_j]
            elif w_j == self.eos_w:
                e_j = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.eos_t]
            else:
                e_j = torch.eye(self.k, device=self.device, dtype=self.dtype)[self.bos_t]

            tmp = e_j * beta[j]
            beta_j_1 = A @ tmp

            if w_j != self.eos_w and w_j != self.bos_w:
                gamma_j = (self.alpha[j] * beta[j]) * invZ
                gamma_j = gamma_j.clone()
                gamma_j[self.bos_t] = 0.0
                gamma_j[self.eos_t] = 0.0
                if t_obs is not None:
                    self.B_counts[t_obs, w_j] += mult * gamma_j[t_obs]
                else:
                    self.B_counts[:, w_j] += mult * gamma_j

            xi_j = (self.alpha[j - 1].unsqueeze(1) * A) * (tmp.unsqueeze(0)) * invZ
            xi_j = xi_j * maskA
            if t_obs is not None:
                mask_to_t = torch.zeros_like(xi_j)
                mask_to_t[:, t_obs] = 1.0
                xi_j = xi_j * mask_to_t
            self.A_counts += mult * xi_j

            scale = self.scale_factors[j]
            beta[j - 1] = beta_j_1 / scale
            k = k + torch.log(scale)
        self.A_counts[:, self.bos_t] = 0.0
        self.A_counts[self.eos_t, :] = 0.0
        self.B_counts[self.eos_t, :] = 0.0
        self.B_counts[self.bos_t, :] = 0.0
        self.beta = beta
        return k
   
   
    # def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
    #     """Run the backwards algorithm from the handout on a tagged, untagged, 
    #     or partially tagged sentence.  Return log Z (the log of the backward
    #     probability). 
        
    #     As a side effect, add the expected transition and emission counts (times
    #     mult) into self.A_counts and self.B_counts.  These depend on the alpha
    #     values and log Z, which were stored for us (in self) by the forward
    #     pass."""

    #     # Pre-allocate beta just as we pre-allocated alpha.
    #     beta = [torch.zeros(self.k) for _ in isent]
    #     beta[-1] = self.eye[self.eos_t]  # vector that is one-hot at EOS_TAG

    #     # print()
    #     # raise NotImplementedError   # you fill this in!
    #     n = len(isent)
    #     # print(f"length of isent: {n}")
    #     # beta[n-1][self.eos_t] = 1.0
    #     # print(f"possible tags are: {[self.tagset[t] for t in range(self.k)]}")
    #     k = 0

    #     # print(f"self.A_counts before backward pass: {self.A_counts}")

    #     for j in range(n-2, -1, -1): 
    #         w_j1, _ = isent[j+1]
    #         beta_j = torch.zeros(self.k)
    #         for t in range(self.k):
    #             if j == 0 and t != self.bos_t:
    #                 continue  # at position 0, only BOS_TAG is possible
    #             if t == self.eos_t:
    #                 continue  #  already assigned
    #             if j != 0 and t == self.bos_t:
    #                 continue  # at position 0, only BOS_TAG is possible

    #             if w_j1 != self.bos_w and w_j1 != self.eos_w:  # normal word (not EOS_WORD or BOS_WORD):
    #                 self.B_counts[t, w_j1] += beta[j+1][t] * self.alpha[j+1][t] / torch.exp(self.log_Z) 
    #                 # if torch.isnan(self.B_counts[t, w_j1]):
    #                 #     self.B_counts[t, w_j1] = 0.0
                
    #             for t_next in range(self.k):
    #                 if w_j1 != self.bos_w and w_j1 != self.eos_w:  # normal word (not BOS_WORD or EOS_WORD)
    #                     emission_prob = self.B[t_next, w_j1]
    #                 elif w_j1 == self.eos_w and t_next != self.eos_t:  # EOS_WORD and not EOS_TAG
    #                     emission_prob = 0.0
    #                 elif w_j1 == self.bos_w and t_next != self.bos_t:  # BOS_WORD and not BOS_TAG
    #                     emission_prob = 0.0
    #                 else:
    #                     emission_prob = 1.0
                    
    #                 prob = self.A[t, t_next] * emission_prob
    #                 # self.A_counts[t, t_next] += (self.alpha[j][t] * prob * beta[j+1][t_next] / torch.exp(self.log_Z))
    #                 self.A_counts[t, t_next] += (mult * self.alpha[j][t] * prob * beta[j + 1][t_next] / torch.exp(self.log_Z))
    #                 beta_j[t] += prob * beta[j+1][t_next]
    #         scale = torch.sum(beta_j)
    #         if scale.item() == 0:
    #             scale = torch.tensor(1e-12, device=beta_j.device)
    #         beta[j] = beta_j / scale
    #         k += torch.log(scale)
    #                 # print(f"We are at tag {t} and next tag {t_next} at position {j}, p(t_next|t) is {self.A[t, t_next]}, p(w_j1|t_next) is {emission_prob}, prob is {prob}, beta[j] is {beta[j]}")
    #                 # print(f"At position {j}, tag {self.tagset[t]}, next tag {self.tagset[t_next]}, p({self.tagset[t_next]}|{self.tagset[t]}) is {self.A[t, t_next]}, p({w_j1}|{self.tagset[t_next]}) is {emission_prob}, prob is {prob}, beta[{j}][{self.tagset}] is {beta[j]}")

        
    #     self.A_counts[:, self.bos_t] = 0
    #     self.A_counts[self.eos_t, :] = 0
    #     log_Z_backward = k
    #     self.beta = beta

    #     # print(f"betas are: {beta}")
    #     # print(f"self.A_counts after backward pass: {self.A_counts}")
    #     # print(f"self.B_counts after backward pass: {self.B_counts}")

    #     return log_Z_backward 


    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus) # integerized sentence, 2 constituents per position: (word, tag)

        # Precompute any sentence-specific structures needed by A_at/B_at (e.g., RNN contexts).
        self.setup_sentence(isent)

        # See comments in log_forward on preallocation of alpha.
        alpha        = [torch.zeros(self.k)                  for _ in isent]  ## initilize alpha to be a list of garbage values of size of the sentence, each entry is a tensor of all possiblie tags (H, C, _EOS_TAG_, _BOS_TAG_)
        # backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        backpointers = [torch.full((self.k,), -1, dtype=torch.int) for _ in isent]
        tags: List[int]    # you'll put your integerized tagging here
        # print("The sentence to tag is:", isent)
        # print("Viterbi tagging, the first alpha values are:", alpha[1])

        ## The goal of Viterbi is to find the best path (tags) by maximizing the score and finding best previous tag for each current tag at each position

        
        ## isent of 0 and n+1 already has EOSW and EOS
        n = len(isent)


        # all ˆα values are initially 0 (or −∞), and all backpointers are initially None
        # print(f"isent is {isent}")
        alpha[0] = torch.zeros(self.k)
        alpha[0][self.bos_t] = 1.0
        for j in range(1, n):
            w_j, t_j = isent[j]
            w_prev, t_prev = isent[j-1]
            B = self.B_at(j, isent)  
            A = self.A_at(j, isent)

            for t in range(self.k):  # for each possible current tag
                for t_prev in range(self.k): # for each possible previous tag
                    # p ← pA(tj | tj−1) · pB(wj | tj)
                    # probability of transition from t_prev to t is self.A[t_prev, t]
                    # probability of emission of w_j from t is self.B[t, w_j]
                    if w_j < self.V:  # normal word (not EOS_WORD or BOS_WORD)
                        emission_prob = B[t, w_j]
                    elif w_j == self.V:  # EOS_WORD
                        emission_prob = 1.0 if t == self.eos_t else 0.0
                    else:  # BOS_WORD (shouldn't happen in positions j >= 1)
                        emission_prob = 1.0 if t == self.bos_t else 0.0
                    prob = A[t_prev, t] * emission_prob
                    temp_alpha = prob * alpha[j - 1][t_prev]
                    if alpha[j][t] < temp_alpha:
                        alpha[j][t] = temp_alpha
                        backpointers[j][t] = t_prev
        tags = [0] * n
        tags[n-1] = self.eos_t

        # tags[n-1] = self.eos_t
        for j in range(n-1, 0, -1):  
            tags[j-1] = backpointers[j][tags[j]]

        # print(f"updated tags are: {[self.tagset[tags[j]] for j in tags]}")
        

        # Make a new tagged sentence with the old words and the chosen tags
        # (using self.tagset to deintegerize the chosen tags).
        # raise NotImplemented
        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])
    




    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its 
        string representation into the filename and save to a temporary checkpoint file (but only 
        do this save if it's been at least checkpoint_interval seconds since the last save).  If 
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed

        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None   # this is a real save, not a checkpoint
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return   # we already saved too recently to save another temp version
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  # use temp filename
            self._checkpoint_path = path

        # update the elapsed training time since we started training or last saved (if that happened)
        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        del self._save_time
        
        # Save the model with the fields set as above, so that we'll 
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass  # don't complain if the user already removed it manually

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model

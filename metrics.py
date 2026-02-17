import numpy as np
from math import log
from collections import defaultdict
import math
import os
import typing

import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from transformers import logging
logging.set_verbosity_error()

# TODO: for tc

LOG2 = math.log(2)

# For take the total correlation between x0, x1, and y


class TC():
    def __init__(self):
        self.x0 = []
        self.x1 = torch.tensor([])

    def update(self, x0, x1):
        # note that x0, x1, y are all torch tensors
        assert len(x0) == len(x1), "All must have same length"

        # it will be a [8, 1024]
        if self.x1.device != x1.device:
            self.x1 = self.x1.to(x1.device)
        self.x0 += x0
        self.x1 = torch.cat((self.x1, x1), dim=0)

    def entropy_from_counts(self, counts, base=2):
        """
        Compute the empirical entropy H(X) from a dictionary of counts,
        using either base-2 (bits) or base-e (nats).
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                # Use math.log(p, base) for bits (base=2) or nats (base=np.e)
                entropy -= p * log(p, base)
        return entropy

    def compute_total_correlation_x1(self, x1_group):
        """
        Given an array x1_group of shape (N, 16, 16), where each row is
        one sample of x1, estimate T(X1) = sum_j H(X1_j) - H(X1)
        by naive frequency counting.

        We treat each of the 256 pixels in the 16x16 image as a separate
        discrete variable. We compute:
           H(X1)   via joint frequencies over all 256 pixels
           H(X1_j) via marginal frequencies of each pixel j
        Then T(X1) = sum_j H(X1_j) - H(X1).
        """
        N = x1_group.shape[0]
        if N < 2:
            # With <2 samples, you cannot really estimate correlation reliably.
            # We return 0.0 by convention or skip it entirely.
            return 0.0

        # Flatten each image into 256-dimensional vector:
        # shape becomes (N, 1024)
        flattened = x1_group.reshape(N, -1)
        d = flattened.shape[1]  # should be 1024

        # ---- Joint distribution (all 256 dims) ----#
        # We'll store each flattened row as a tuple, then count frequencies.
        joint_counts = defaultdict(int)
        for row in flattened:
            key = tuple(row)   # row is length-1024
            joint_counts[key] += 1
        H_joint = self.entropy_from_counts(joint_counts, base=2)

        # ---- Marginal distributions (one pixel at a time) ----#
        # We'll compute an entropy for each pixel dimension j.
        marginal_entropies = []
        for j in range(d):
            counts_j = defaultdict(int)
            for row in flattened:
                val_j = row[j].long().item()
                counts_j[val_j] += 1
            H_j = self.entropy_from_counts(counts_j, base=2)
            marginal_entropies.append(H_j)

        sum_marginals = sum(marginal_entropies)

        # ---- Total correlation ----#
        T_val = sum_marginals - H_joint
        return T_val, sum_marginals, H_joint

    def compute_conditional_total_correlation_x1_given_x0y(self, x0, x1):
        """
        1) Group all samples by unique x0[i] pair.
           - Here x0[i] is a [1024] tensor
        2) Collect the corresponding x1[i] arrays for each group.
        3) For each group, estimate T(X1) by naive frequency counting
           (thus approximating T(X1 | x0=x0_val, y=y_val)).

        Returns a dict:  (x0_bytes) -> estimated total correlation in bits.
        """

        # Safety checks:
        assert len(x0) == len(x1), "All must have same length"
        N = len(x0)

        # Group x1 by unique x0
        groups = defaultdict(list)
        for i in range(N):
            # We need a hashable key for x0[i], which is shape (1028)
            # Convert to bytes, or a tuple if you prefer
            x0_key = x0[i]
            key = x0_key
            groups[key].append(x1[i])

        # Compute total correlation for each group
        results = {}
        marginals = {}
        joints = {}
        for key, x0_x1_tup in groups.items():
            # x1_list is a tensor, shape each (1024)
            # print(len(x1_list))
            # shape = (num_samples_for_this_group, 1024)
            group = torch.stack(x0_x1_tup, dim=0)
            T_val, sum_marginals, H_joint = self.compute_total_correlation_x1(
                group)
            results[key] = T_val
            marginals[key] = sum_marginals
            joints[key] = H_joint

        return results, joints, marginals

    def compute(self):
        """
        Compute the total correlation of x1 given x0 and y.
        """
        tc_results, joints, marginals = self.compute_conditional_total_correlation_x1_given_x0y(
            self.x0, self.x1)

        avg_tc = np.mean(list(tc_results.values()))
        avg_joints = np.mean(list(joints.values()))
        avg_marginals = np.mean(list(marginals.values()))

        return avg_tc, avg_joints, avg_marginals


class NLL(torchmetrics.aggregation.MeanMetric):
    def update(self,
               value: typing.Union[float, torch.Tensor],
               weight: typing.Union[float, torch.Tensor] = 1.0) -> None:
        """Update state with data.

        Args:
          value: Either a float or tensor containing data.
            Additional tensor dimensions will be flattened
          weight: Either a float or tensor containing weights
            for calculating the average. Shape of weight should
            be able to broadcast with the shape of `value`.
            Default to `1.0` corresponding to simple harmonic
            average.
        """
        # broadcast weight to value shape
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype,
                                    device=self.device)
        if (weight is not None
                and not isinstance(weight, torch.Tensor)):
            weight = torch.as_tensor(weight,
                                     dtype=self.dtype,
                                     device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value,
                                                       weight)

        if value.numel() == 0:
            return
        self.mean_value += value.sum()
        self.weight += weight.sum()


class BPD(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    def __init__(self, gen_ppl_eval_model_name_or_path=None,
                 eval_ppl_batch_size=None) -> None:
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
        metrics.set_dtype(torch.float64)
        self.train_nlls = metrics.clone(prefix='train/')
        self.train_aux = BPD()
        self.valid_nlls = metrics.clone(prefix='val/')
        self.valid_aux = BPD()
        self.gen_ppl = Perplexity()
        self.sample_entropy = torchmetrics.aggregation.MeanMetric()
        self.unique_token_count = 0
        self.tc = TC()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.\
            from_pretrained(gen_ppl_eval_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def to(self, *args, **kwargs):
        self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
        self.sample_entropy = self.sample_entropy.to(*args, **kwargs)
        self.train_nlls = self.train_nlls.to(*args, **kwargs)
        self.train_aux = self.train_aux.to(*args, **kwargs)
        self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
        self.valid_aux = self.valid_aux.to(*args, **kwargs)

    def reset(self):
        self.gen_ppl.reset()
        self.sample_entropy.reset()
        self.train_nlls.reset()
        self.train_aux.reset()
        self.valid_nlls.reset()
        self.valid_aux.reset()

    def update_train(self, nll, aux_loss, num_tokens):
        self.train_nlls.update(nll, num_tokens)
        self.train_aux.update(aux_loss, num_tokens)

    def update_valid(self, nll, aux_loss, num_tokens):
        self.valid_nlls.update(nll, num_tokens)
        self.valid_aux.update(aux_loss, num_tokens)

    @torch.no_grad()
    def _eval_retokenize(self, text_samples, max_length,
                         device):
        """Retokenizes samples for the eval model.

        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        if 'llama2' in self.gen_ppl_eval_model_name_or_path.lower():
            tokenizer_kwargs = {
                'text_samples': text_samples,
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 4096
        elif 'llama3' in self.gen_ppl_eval_model_name_or_path.lower():
            tokenizer_kwargs = {
                'text': text_samples,
                'return_tensors': 'pt',
                'padding': True,
            }
            eval_context_size = 8192
        else:
            tokenizer_kwargs = {
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 1024
        samples = self.tokenizer(text_samples,
                                 **tokenizer_kwargs)
        attn_mask = samples['attention_mask']
        samples = samples['input_ids']
        if 'llama' not in self.gen_ppl_eval_model_name_or_path.lower():
            attn_mask = attn_mask.to(device)
            samples = samples.to(device)
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def record_entropy(self, tokens):
        for sample in tokens:
            _, counts = torch.unique(
                sample, return_counts=True, sorted=False)
            entropy = torch.special.entr(
                counts.float() / counts.sum()).sum().item()
            self.sample_entropy.update(entropy)

    @torch.no_grad()
    def record_unique_tokens(self, samples):
        """Record the count of unique tokens across all samples.
        
        Args:
            samples: torch.Tensor of shape (batch_size, seq_length)
        
        Returns:
            int: Number of unique tokens in the samples
        """
        unique_tokens = torch.unique(samples.flatten())
        self.unique_token_count = len(unique_tokens)
        return self.unique_token_count

    def reset_unique_tokens(self):
        """Reset unique token count."""
        self.unique_token_count = 0

    @torch.no_grad()
    def record_generative_perplexity(
            self,
            text_samples: typing.List[str],
            max_length: int,
            retokenize: bool = True,
            device='cuda') -> None:

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if 'llama' not in self.gen_ppl_eval_model_name_or_path:
            eval_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.gen_ppl_eval_model_name_or_path).eval()
            eval_model = eval_model.to(device)
            # Re-tokenize using eval model's tokenizer
            if retokenize:
                (samples, attn_mask,
                 eval_context_size) = self._eval_retokenize(
                    text_samples, max_length=max_length, device=device)
            else:
                samples = text_samples
                attn_mask = torch.ones(samples.shape).to(device)
                eval_context_size = samples.shape[-1]
            batch_size = min(self.eval_ppl_batch_size,
                             samples.shape[0])
            num_batches = samples.shape[0] // batch_size
            for i in range(num_batches):
                _samples = torch.split(
                    samples[i * batch_size: (i + 1) * batch_size],
                    eval_context_size,
                    dim=-1)
                _attn_mask = torch.split(
                    attn_mask[i * batch_size: (i + 1) * batch_size],
                    eval_context_size,
                    dim=-1)
                for (sample_chunk, attn_mask_chunk) in zip(_samples,
                                                           _attn_mask):
                    logits = eval_model(sample_chunk.to(device),
                                        attention_mask=attn_mask_chunk.to(device))
                    logits = logits[0].transpose(-1, -2)
                    nlls = F.cross_entropy(logits[..., :-1],
                                           sample_chunk[..., 1:],
                                           reduction='none')
                    first_eos = (
                        sample_chunk
                        == self.tokenizer.eos_token_id).cumsum(-1) == 1
                    token_mask = sample_chunk != self.tokenizer.eos_token_id
                    valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
                    self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
        else:
            eval_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.gen_ppl_eval_model_name_or_path,
                torch_dtype=torch.bfloat16).eval()
            eval_model = eval_model.to(device)
            # Re-tokenize using eval model's tokenizer
            tokenizer_llama = transformers.AutoTokenizer.from_pretrained(
                self.gen_ppl_eval_model_name_or_path)
            if tokenizer_llama.pad_token is None:
                tokenizer_llama.pad_token = tokenizer_llama.eos_token
                tokenizer_llama.pad_token_id = tokenizer_llama.eos_token_id
            tokenizer_gpt = transformers.AutoTokenizer.from_pretrained(
                'gpt2')
            if tokenizer_gpt.pad_token is None:
                tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
                tokenizer_gpt.pad_token_id = tokenizer_gpt.eos_token_id
            num_samples = len(text_samples)
            batch_size = min(16, num_samples)
            num_batches = num_samples // batch_size

            with torch.inference_mode():
                # 1. divide into batches of 16
                # 2. encode each batch
                # 3. eval each batch of 16
                for i in range(num_batches):
                    batch_text_samples = text_samples[i *
                                                      batch_size:(i+1)*batch_size]
                    encoded_inputs = tokenizer_llama(
                        batch_text_samples,
                        return_tensors="pt",
                        padding=True,
                    )

                    input_ids = encoded_inputs['input_ids'].to(device)
                    attention_mask = encoded_inputs['attention_mask'].to(
                        device)

                    labels = input_ids.clone()
                    labels[labels == tokenizer_llama.pad_token_id] = 50000
                    labels = labels.to(device)

                    outputs = eval_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    llama_logits = outputs.logits

                    logits = llama_logits.transpose(-1, -2)
                    nlls = F.cross_entropy(logits[..., :-1],
                                           labels[..., 1:],
                                           reduction='none')
                    valid_tokens = attention_mask[..., 1:].bool()
                    self.gen_ppl.update(nlls * valid_tokens, valid_tokens)

    @torch.no_grad()
    def record_tc(self, noise_index, sample):
        self.tc.update(noise_index, sample)

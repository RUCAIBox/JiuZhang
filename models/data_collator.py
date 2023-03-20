# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Union, Tuple

import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from data_utils import (
    pad_and_truncate,
    get_formula_split,
    get_sentence_split,
    get_word_starts,
    cut_list,
)


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""
#DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


@dataclass
class DataCollatorForSingle:

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch['input_ids'] = torch.tensor([e['input_ids'] for e in examples])
        batch["attention_mask"] = torch.tensor([e['attention_mask'] for e in examples])
        batch["labels"] = torch.tensor([e['labels'] for e in examples])

        return batch


@dataclass
class DataCollatorForPair:

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch['input_ids_1'] = torch.tensor([e['input_ids_1'] for e in examples])
        batch['attention_mask_1'] = torch.tensor([e['attention_mask_1'] for e in examples])
        batch['input_ids_2'] = torch.tensor([e['input_ids_2'] for e in examples])
        batch['attention_mask_2'] = torch.tensor([e['attention_mask_2'] for e in examples])

        batch["labels"] = torch.tensor([e['label'] for e in examples])

        return batch


@dataclass
class DataCollatorForMTP:

    tokenizer: PreTrainedTokenizerBase
    use_linear_mask: bool = False
    mlm_probability: float = 0.15

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([e['input_ids'] for e in examples])
        labels = torch.ones_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        batch_size = input_ids.size(0)
        use_decoder = torch.bernoulli(torch.tensor([0.5]*batch_size)).long()

        for i in range(batch_size):
            if use_decoder[i].item() == 0:
                mlm_or_dae = 'mlm'
                prob = self.mlm_probability
            else:
                mlm_or_dae = 'dae'
                prob = 2 * self.mlm_probability
            source, label = self.get_whole_word_mask(
                input_ids[i],
                examples[i]['is_word_start'],
                examples[i]['word_starts'],
                prob,
                mlm_or_dae
            )
            input_ids[i] = source
            labels[i] = label

        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'use_decoder': use_decoder,
        }

        return batch

    def get_whole_word_mask(self, source, is_word_start, word_starts, prob, mlm_or_dae):
        label = source.clone()

        is_word_start, word_starts = get_word_starts(source.tolist(), self.tokenizer)

        prob = prob * 1.1
        if self.use_linear_mask:
            prob_list = torch.linspace(0.01, 2*prob-0.01, steps=word_starts.size(0))
        else:
            prob_list = torch.full((word_starts.size(0),), prob)
        head_indices = word_starts[torch.bernoulli(prob_list).bool()]

        to_keep = torch.ones(source.size(0), dtype=torch.bool)
        # acts as a long length, so spans don't go over the end of doc
        is_word_start = torch.cat([is_word_start, torch.tensor([255])], dim=0)
        
        temp_indices = head_indices.clone()
        tail_indices = []
        while temp_indices.size(0) > 0:
            uncompleted = is_word_start[temp_indices + 1] == 0
            temp_indices = temp_indices[uncompleted] + 1
            tail_indices.extend(temp_indices.tolist())
        tail_indices = torch.tensor(tail_indices, dtype=torch.long)

        if mlm_or_dae == 'dae':
            to_keep[tail_indices] = 0
            wwm_indices = head_indices
            label[label == self.tokenizer.pad_token_id] = -100
        else:
            wwm_indices = torch.cat([head_indices, tail_indices])
            not_maksed = torch.ones_like(label)
            not_maksed[wwm_indices] = 0
            label[not_maksed.bool()] = -100
        
        mask = torch.FloatTensor(wwm_indices.size()).uniform_() < 0.8
        random = (torch.FloatTensor(wwm_indices.size()).uniform_() < 0.5) & ~mask

        source[wwm_indices[mask]] = self.tokenizer.mask_token_id
        source[wwm_indices[random]] = torch.randint(1, len(self.tokenizer), size=(random.sum(),))

        source = source[to_keep]
        if mlm_or_dae == 'dae':
            padding = torch.tensor([self.tokenizer.pad_token_id] * tail_indices.size(0)).long()
            source = torch.cat([source, padding], dim=0)

        return source, label


@dataclass
class DataCollatorForSC:

    tokenizer: PreTrainedTokenizerBase
    use_linear_mask: bool = False
    mlm_probability: float = 0.15
    decoder_rate: float = 0.5

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([e['input_ids'] for e in examples])
        labels = torch.ones_like(input_ids)
        adv_labels = torch.ones_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        batch_size = input_ids.size(0)
        use_decoder = torch.bernoulli(torch.tensor([self.decoder_rate]*batch_size)).long()

        for i in range(batch_size):
            if use_decoder[i].item() == 0:
                mlm_or_dae = 'mlm'
                prob = self.mlm_probability
            else:
                mlm_or_dae = 'dae'
                prob = 2 * self.mlm_probability
            source, label, adv_label = self.get_whole_word_mask(
                input_ids[i],
                examples[i]['is_word_start'],
                examples[i]['word_starts'],
                prob,
                mlm_or_dae
            )
            input_ids[i] = source
            labels[i] = label
            adv_labels[i] = adv_label

        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'adv_labels': adv_labels,
            'use_decoder': use_decoder,
        }

        return batch

    def get_whole_word_mask(self, source, is_word_start, word_starts, prob, mlm_or_dae):
        label = source.clone()
        adv_label = source.clone()
        adv_label[adv_label == self.tokenizer.pad_token_id] = -100

        is_word_start, word_starts = get_word_starts(source.tolist(), self.tokenizer)

        prob = prob * 1.1
        if self.use_linear_mask:
            prob_list = torch.linspace(0.01, 2*prob-0.01, steps=word_starts.size(0))
        else:
            prob_list = torch.full((word_starts.size(0),), prob)
        head_indices = word_starts[torch.bernoulli(prob_list).bool()]

        to_keep = torch.ones(source.size(0), dtype=torch.bool)
        # acts as a long length, so spans don't go over the end of doc
        is_word_start = torch.cat([is_word_start, torch.tensor([255])], dim=0)
        
        temp_indices = head_indices.clone()
        tail_indices = []
        while temp_indices.size(0) > 0:
            uncompleted = is_word_start[temp_indices + 1] == 0
            temp_indices = temp_indices[uncompleted] + 1
            tail_indices.extend(temp_indices.tolist())
        tail_indices = torch.tensor(tail_indices, dtype=torch.long)

        if mlm_or_dae == 'dae':
            to_keep[tail_indices] = 0
            wwm_indices = head_indices
            label[label == self.tokenizer.pad_token_id] = -100
        else:
            wwm_indices = torch.cat([head_indices, tail_indices])
            not_maksed = torch.ones_like(label)
            not_maksed[wwm_indices] = 0
            label[not_maksed.bool()] = -100
        
        mask = torch.FloatTensor(wwm_indices.size()).uniform_() < 0.8
        random = (torch.FloatTensor(wwm_indices.size()).uniform_() < 0.5) & ~mask

        source[wwm_indices[mask]] = self.tokenizer.mask_token_id
        source[wwm_indices[random]] = torch.randint(1, len(self.tokenizer), size=(random.sum(),))

        source = source[to_keep]
        if mlm_or_dae == 'dae':
            padding = torch.tensor([self.tokenizer.pad_token_id] * tail_indices.size(0)).long()
            source = torch.cat([source, padding], dim=0)

        return source, label, adv_label


@dataclass
class DataCollatorForLogic:

    tokenizer: PreTrainedTokenizerBase
    use_linear_mask: bool = False
    mlm_probability: float = 0.15
    max_len: int = 256

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        input_ids = torch.ones(batch_size, 512, dtype=torch.long)
        labels = torch.ones_like(input_ids)
        attention_mask = torch.ones_like(input_ids)
        use_decoder = torch.bernoulli(torch.tensor([0.5]*batch_size)).long()
        use_shuffle = torch.bernoulli(torch.tensor([0.5]*batch_size)).long()

        for i in range(batch_size):
            if use_decoder[i].item() == 0:
                mlm_or_dae = 'mlm'
                prob = self.mlm_probability
                is_shuffle = False
            else:
                mlm_or_dae = 'dae'
                prob = 2 * self.mlm_probability
                is_shuffle = (use_shuffle[i].item() == 1)
            
            source, dae_label = self.shuffle_sentence(examples[i], is_shuffle)

            if not is_shuffle:
                source, label = self.get_whole_word_mask(source, prob, mlm_or_dae, use_linear_mask=True)

            input_ids[i] = source
            labels[i] = label if mlm_or_dae == 'mlm' else dae_label

        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'use_decoder': use_decoder,
        }

        return batch
  
    def shuffle_sentence(self, inputs, is_shuffle):
        content, analysis = inputs['content'], inputs['analysis']

        source = pad_and_truncate(content, analysis, max_len=self.max_len)
        source = torch.tensor(source, dtype=torch.long)
        dae_label = source.clone()
        dae_label[dae_label == self.tokenizer.pad_token_id] = -100

        if not is_shuffle:
            return source, dae_label
        
        choice = torch.bernoulli(torch.tensor([0.5])).long()
        
        sen = analysis
        s_len, s_idx = get_sentence_split(sen, sep_ids=self.tokenizer.convert_tokens_to_ids([',', '.']))
        f_len, f_idx = get_formula_split(sen, sep_id=self.tokenizer.convert_tokens_to_ids('$'))

        if choice[0] == 1 or len(f_idx) == 0:
            # sentence
            sen_list = cut_list(sen, s_len)

            # shuffle sentence
            shuffle_idx = s_idx.copy()
            random.shuffle(shuffle_idx)
            shuffle_sen = [sen_list[idx] for idx in shuffle_idx]
            for i, idx in enumerate(s_idx):
                sen_list[idx] = shuffle_sen[i]
        else:
            # formula
            sen_list = cut_list(sen, f_len)

            # shuffle formula
            shuffle_idx = f_idx.copy()
            random.shuffle(shuffle_idx)
            shuffle_sen = [sen_list[idx] for idx in shuffle_idx]
            for i, idx in enumerate(f_idx):
                sen_list[idx] = shuffle_sen[i]
        
        source = pad_and_truncate(content, sum(sen_list, []), max_len=self.max_len)
        source = torch.tensor(source, dtype=torch.long)
        return source, dae_label

    def get_whole_word_mask(self, source, prob, mlm_or_dae):
        label = source.clone()
        is_word_start, word_starts = get_word_starts(source.tolist(), self.tokenizer)

        prob = prob * 1.1
        if self.use_linear_mask:
            prob_list = torch.linspace(0.01, 2*prob-0.01, steps=word_starts.size(0))
        else:
            prob_list = torch.full((word_starts.size(0),), prob)
        head_indices = word_starts[torch.bernoulli(prob_list).bool()]

        to_keep = torch.ones(source.size(0), dtype=torch.bool)
        # acts as a long length, so spans don't go over the end of doc
        is_word_start = torch.cat([is_word_start, torch.tensor([255])], dim=0)
        
        temp_indices = head_indices.clone()
        tail_indices = []
        while temp_indices.size(0) > 0:
            uncompleted = is_word_start[temp_indices + 1] == 0
            temp_indices = temp_indices[uncompleted] + 1
            tail_indices.extend(temp_indices.tolist())
        tail_indices = torch.tensor(tail_indices, dtype=torch.long)

        if mlm_or_dae == 'dae':
            to_keep[tail_indices] = 0
            wwm_indices = head_indices
            label[label == self.tokenizer.pad_token_id] = -100
        else:
            wwm_indices = torch.cat([head_indices, tail_indices])
            not_maksed = torch.ones_like(label)
            not_maksed[wwm_indices] = 0
            label[not_maksed.bool()] = -100
        
        mask = torch.FloatTensor(wwm_indices.size()).uniform_() < 0.8
        random = (torch.FloatTensor(wwm_indices.size()).uniform_() < 0.5) & ~mask

        source[wwm_indices[mask]] = self.tokenizer.mask_token_id
        source[wwm_indices[random]] = torch.randint(1, len(self.tokenizer), size=(random.sum(),))

        source = source[to_keep]
        if mlm_or_dae == 'dae':
            padding = torch.tensor([self.tokenizer.pad_token_id] * tail_indices.size(0)).long()
            source = torch.cat([source, padding], dim=0)

        return source, label


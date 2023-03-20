from itertools import accumulate

import torch
import jieba


def get_sentence_split(sen, sep_ids):
    s_len, s_idx = [], []
    curr_len = 0
    for tok in sen:
        if tok in sep_ids:
            if curr_len != 0:
                s_idx.append(len(s_len))
                s_len.append(curr_len)
                curr_len = 0
            s_len.append(1)
        else:
            curr_len += 1
    if curr_len != 0:
        s_idx.append(len(s_len))
        s_len.append(curr_len)
    return s_len, s_idx


def get_formula_split(sen, sep_id):
    f_len, f_idx = [], []
    in_formula = False
    curr_len = 0
    for tok in sen:
        if tok == sep_id:
            if in_formula:
                f_idx.append(len(f_len))
                f_len.append(curr_len+1)
                in_formula = False
                curr_len = 0
            else:
                if curr_len != 0:
                    f_len.append(curr_len)
                curr_len = 1
                in_formula = True
        else:
            curr_len += 1
    if curr_len != 0:
        f_len.append(curr_len)
    assert not in_formula
    return f_len, f_idx


def cut_list(sen, s_len):
    s_len = list(accumulate([0] + s_len))
    return [sen[s_len[i-1] : s_len[i]] for i in range(1, len(s_len))]


def pad_and_truncate(id_1, id_2, max_len=512):
    CLS, SEP, PAD = 101, 102, 0
    if len(id_1) + len(id_2) + 3 <= max_len:
        input_ids = [CLS] + id_1 + [SEP] + id_2 + [SEP]
        return input_ids + [PAD] * (max_len - len(input_ids))
    while len(id_1) + len(id_2) + 3 > max_len:
        if len(id_1) > len(id_2):
            id_1 = id_1[:-1]
        else:
            id_2 = id_2[:-1]
    input_ids = [CLS] + id_1 + [SEP] + id_2 + [SEP]
    assert len(input_ids) == max_len
    return input_ids


def is_chinese(c):
    return 11904 <= ord(c) <= 42191


def jieba_retokenize(words):
    new_words = []
    ch = ''
    for word in words:
        if len(word) == 1 and is_chinese(word):
            ch += word
        else:
            if ch != '':
                new_words.extend(list(jieba.cut(ch, cut_all=False)))
                ch = ''
            new_words.append(word)
    if ch != '':
        new_words.extend(list(jieba.cut(ch, cut_all=False)))
    return new_words


def align_linear(atokens, btokens):
    a2c = []
    c2b = []
    a2b = []
    length = 0
    for tok in atokens:
        a2c.append([length + i for i in range(len(tok))])
        length += len(tok)
    for i, tok in enumerate(btokens):
        c2b.extend([i for _ in range(len(tok))])

    for i, amap in enumerate(a2c):
        bmap = [c2b[ci] for ci in amap]
        a2b.append(list(set(bmap)))
    return a2b


def get_word_starts(source, tokenizer):
    special_token = ['[CLS]', '[PAD]', '[SEP]']
    raw_tokens = tokenizer.convert_ids_to_tokens(source)
    words = jieba_retokenize(raw_tokens)
    raw_to_word_align = align_linear(raw_tokens, words)
    is_word_start = [1] * len(source)
    word_starts = []
    if raw_tokens[0] not in special_token:
        word_starts.append(0)
    for i in range(1, len(raw_to_word_align)):
        if raw_to_word_align[i-1] == raw_to_word_align[i]:
            is_word_start[i] = 0
        elif raw_tokens[i] not in special_token:
            word_starts.append(i)

    return torch.tensor(is_word_start).long(), torch.tensor(word_starts).long()

import os
from typing import Union, List, Tuple

import torch
from torch.utils.data import Dataset

from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


class TextDataset(Dataset):
    def __init__(self, data_file_en: str, data_file_de: str, sp_model_prefix: str = None,
                 vocab_size: int = 3000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 200, langs=['de', 'en']):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file_en: txt file containing texts in English
        :param data_file_de: txt file containing texts in German
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        # init models for two given languages
        self.sp_source = self.get_sp_model(data_file_de, langs[0], sp_model_prefix,
                                           vocab_size, normalization_rule_name, model_type)
        self.sp_target = self.get_sp_model(data_file_en, langs[1], sp_model_prefix,
                                           vocab_size, normalization_rule_name, model_type)

        # read lines of given texts for both source and target languages
        self.source_texts, self.target_texts = [], []
        with open(data_file_de, encoding='utf8') as file:
            self.source_texts = file.readlines()
        with open(data_file_en, encoding='utf8') as file:
            self.target_texts = file.readlines()

        # convert read texts to tokens using models
        self.source_tokens = self.sp_source.encode(self.source_texts)
        self.target_tokens = self.sp_target.encode(self.target_texts)

        # save config informatoion about model
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_source.pad_id(), self.sp_source.unk_id(), \
            self.sp_source.bos_id(), self.sp_source.eos_id()
        self.vocab_size_source = self.sp_source.vocab_size()
        self.vocab_size_target = self.sp_target.vocab_size()
        self.max_length = max_length

    def get_sp_model(self, data_file, lang, sp_model_prefix, vocab_size, normalization_rule_name, model_type):
        if not os.path.isfile(sp_model_prefix + '_' + lang + '.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix + '_' + lang,
                normalization_rule_name=normalization_rule_name,
                unk_id=0, bos_id=1, eos_id=2, pad_id=3
            )
        # load tokenizer from file
        return SentencePieceProcessor(model_file=sp_model_prefix + '_' + lang + '.model')

    def text2ids(self, texts: Union[str, List[str]], text_type: str) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :param text_type: source or target
        :return: encoded indices
        """
        return self.sp_source.encode(texts) if text_type == 'source' else self.sp_target.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]], text_type: str) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :param text_type: source or target
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_source.decode(ids) if text_type == 'source' else self.sp_target.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.source_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded source and target texts
        """
        # get encoded source line and add PAD token
        encoded_source_text = [self.bos_id] + self.source_tokens[item] + [self.eos_id]
        padded_source_text = torch.full((self.max_length,), self.pad_id, dtype=torch.int64)
        padded_source_text[:len(encoded_source_text)] = torch.tensor(encoded_source_text)

        # get encoded target line and add PAD token
        encoded_target_text = [self.bos_id] + self.target_tokens[item] + [self.eos_id]
        padded_target_text = torch.full((self.max_length,), self.pad_id, dtype=torch.int64)
        padded_target_text[:len(encoded_target_text)] = torch.tensor(encoded_target_text)

        return padded_source_text, padded_target_text

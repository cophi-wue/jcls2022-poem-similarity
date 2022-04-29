from typing import List, Iterable, Dict

import numpy as np
from sentence_transformers.models import Pooling
from sentence_transformers.models.tokenizer import WhitespaceTokenizer
import regex
import torch
from torch import Tensor


class WordTokenizer(WhitespaceTokenizer):

    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = []):
        super().__init__(vocab, stop_words)

    def tokenize(self, text: str) -> List[int]:
        # ignore punctuation
        tokens = regex.findall(r'\p{L}+', text)

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered


class DistanceMatrix():

    def __init__(self, id_to_index, matrix):
        self.id_to_index = id_to_index
        self.matrix = matrix

    def dist(self, pairs):
        if type(pairs) == 'tuple': pairs = [pairs]
        res = []
        for pair in pairs:
            res.append(self.matrix[self.id_to_index[pair[0]], self.id_to_index[pair[1]]])

        return torch.stack(res)


def pooling_median(vectors):
    return torch.median(vectors, axis=0).values


def pooling_mean(vectors):
    return torch.mean(vectors, axis=0)


def pooling_meannorm(vectors):
    m = torch.mean(vectors / torch.linalg.vector_norm(vectors, dim=1)[:, np.newaxis], axis=0)
    m = m / torch.norm(m)
    return m


class CustomPooling(Pooling):

    def __init__(self, word_embedding_dimension: int, pooling_method):
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_method = pooling_method
        super(Pooling, self).__init__()

    def save(self, output_path):
        raise NotImplementedError

    @staticmethod
    def load(input_path):
        raise NotImplementedError

    def get_pooling_mode_str(self) -> str:
        return super().get_pooling_mode_str()

    def forward(self, features):
        batch_token_embeddings = features['token_embeddings']
        batch_attention_masks = features['attention_mask']


        ## Pooling strategy
        output_vectors = []
        for token_embeddings, attention_masks in zip(batch_token_embeddings, batch_attention_masks):
            token_embeddings = token_embeddings[attention_masks.bool()]
            sentence_vector = self.pooling_method(token_embeddings)
            assert sentence_vector.shape == (self.word_embedding_dimension,)
            output_vectors.append(sentence_vector)

        output_vector = torch.stack(output_vectors)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return super().get_sentence_embedding_dimension()


class SIFPooling(Pooling):

    def __init__(self, word_embedding_dimension: int, word_weights):
        self.word_embedding_dimension = word_embedding_dimension
        self.word_weights = torch.tensor(word_weights)
        super(Pooling, self).__init__()

    def save(self, output_path):
        raise NotImplementedError

    @staticmethod
    def load(input_path):
        raise NotImplementedError

    def get_pooling_mode_str(self) -> str:
        raise NotImplementedError

    def forward(self, features):
        batch_input_ids = features['input_ids']
        batch_token_embeddings = features['token_embeddings']
        batch_attention_masks = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        for token_ids, token_embeddings, attention_masks in zip(batch_input_ids, batch_token_embeddings, batch_attention_masks):
            token_ids = token_ids[attention_masks.bool()]
            token_embeddings = token_embeddings[attention_masks.bool()]
            token_weights = self.word_weights[token_ids]
            sentence_vector = token_weights @ token_embeddings

            assert sentence_vector.shape == (self.word_embedding_dimension,)
            output_vectors.append(sentence_vector)

        output_vector = torch.stack(output_vectors)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return super().get_sentence_embedding_dimension()


class BERTHiddenPooling(Pooling):

    def __init__(self, word_embedding_dimension: int, pooling_method, layers):
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_method = pooling_method
        self.layers = layers
        super(Pooling, self).__init__()

    def save(self, output_path):
        raise NotImplementedError

    @staticmethod
    def load(input_path):
        raise NotImplementedError

    def get_pooling_mode_str(self) -> str:
        raise NotImplementedError

    def forward(self, features):
        batch_token_embeddings = torch.stack(features['all_layer_embeddings']).permute(1,2,0,3)
        batch_attention_masks = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        for token_embeddings, attention_masks in zip(batch_token_embeddings, batch_attention_masks):
            token_embeddings = token_embeddings[attention_masks.bool()]
            token_layer_mean = torch.mean(token_embeddings[:,self.layers,:], axis=1)
            sentence_vector = self.pooling_method(token_layer_mean)
            assert sentence_vector.shape == (self.word_embedding_dimension,)
            output_vectors.append(sentence_vector)

        output_vector = torch.stack(output_vectors)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return super().get_sentence_embedding_dimension()

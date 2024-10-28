__author__ = 'max'

import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
import gzip

from .io import utils
from gensim.models.keyedvectors import KeyedVectors


def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    """
    Load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimension
    """
    print("Loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'word2vec':
        # loading word2vec
        word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:  # No gzip for plain text GloVe
            for line in file:
                line = line.strip()
                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                assert (embedd_dim + 1 == len(tokens))
                embedd = np.array(tokens[1:], dtype=np.float32)
                word = utils.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        # Adding a default vector for unknown words
        default_vector = np.zeros(embedd_dim)
        embedd_dict['<UNK>'] = default_vector
        return embedd_dict, embedd_dim
    
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.decode('utf-8')
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = utils.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        # loading sskip embeddings
        word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        embedd_dim = word2vec.vector_size
        embedd_dict = {word: word2vec[word] for word in word2vec.vocab}
        # Add a default vector for unknown words (zero vector or random)
        default_vector = np.zeros(embedd_dim)  # Or use np.random.randn(embedd_dim) for a random vector
        embedd_dict['<UNK>'] = default_vector  # Use a specific key like '<UNK>' for unknown words
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'))
        _, embedd_dim = embeddings.shape
        embedd_dict = dict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = utils.DIGIT_RE.sub(b"0", word) if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

    else:
        raise ValueError("embedding should choose from [word2vec, senna, glove, sskip, polyglot]")

# -*- coding: utf-8 -*-

"""Main module."""

from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models import doc2vec

import pickle

import os
import numpy as np
import copy


class TDEC:
    """
    Temporal Document Embeddings in a Compass
    Handles alignment between multiple slices of text
    """
    def __init__(self, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                            min_count=5, workers=2, init_mode="hidden"):
        """

        :param size: Number of dimensions. Default is 100.
        :param mode: Either DM or DBOW document embedding architecture of Doc2Vec. DM is default
            Note: DBOW as presented by Le and Mikolov (2014) does not train word vectors. As a result, gensim's development of DBOW, which trains word vectors in skip-gram fashion in parallel to the DBOW process, will be used
        :param siter: Number of static iterations (epochs). Default is 5.
        :param diter: Number of dynamic iterations (epochs). Default is 5.
        :param ns: Number of negative sampling examples. Default is 10, min is 1.
        :param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
        :param alpha: Initial learning rate. Default is 0.025.
        :param min_count: Min frequency for words over the entire corpus. Default is 5.
        :param workers: Number of worker threads. Default is 2.
        :param test: Folder name of the diachronic corpus files for testing.
        :param opath: Name of the desired output folder. Default is model.
        :param init_mode: If \"hidden\" (default), initialize models with hidden embeddings of the context;'
                            'if \"both\", initilize also the word embeddings;'
                            'if \"copy\", models are initiliazed as a copy of the context model
                            (same vocabulary)
        """
        self.size = size
        self.mode = mode
        self.trained_slices = dict()
        self.gvocab = []
        self.static_iter = siter
        self.dynamic_iter =diter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        self.workers = workers
        self.init_mode = init_mode
        self.compass = None
        self.trained_slices = {}

    def train_compass(self, corpus_file=None, sentences=None):
        if self.mode == "dm":
            self.compass = Word2Vec(sg=0, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "dbow":
            self.compass = Word2Vec(sg=1, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            return Exception('Set "mode" to be "dm" or "dbow"')
        if corpus_file:
            self.compass.build_vocab(corpus_file=corpus_file)
            self.compass.train(corpus_file=corpus_file,
                  total_words=self.compass.corpus_total_words, epochs=self.static_iter, compute_loss=True)
        elif sentences:
            self.compass.build_vocab(sentences=sentences)
            self.compass.train(sentences=sentences,
                  total_words=self.compass.corpus_total_words, epochs=self.static_iter, compute_loss=True)

        self.compass.learn_hidden = False

    def _initialize_model(self):
        model = copy.deepcopy(self.compass)
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
        return model

    def train_slice(self, slice_text, slice_titles=None, out_name = None, csave=False, fsave=False):
        """
        Training a slice of text
        :param slice_text:
        :param slice_titles:
        :param out_name: output name/file path
        :param csave: save to compass
        :param fsave: save to file
        :return: model
        """
        if self.compass == None:
            return Exception("Missing Compass")
        if csave and not out_name:
            return Exception("Specify compass name using 'out_name'")
        if fsave and not out_name:
            return Exception("Specify output file using 'out_name' to save")

        if not csave and not fsave:
            print("Warning: You don't save to anything. Save to compass with 'csave' or to file with 'fsave'")
        sentences = None
        if slice_titles:
            sentences = [doc2vec.TaggedDocument(doc, [title]) for doc, title in zip(slice_text, slice_titles)]
        else:
            sentences = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(slice_text)]

        if self.mode == "dm":
            model = Doc2Vec(vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "dbow":
            model = Doc2Vec(dm=0, dbow_words=1, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            return Exception('Set "mode" to be "dm" or "dbow"')
        model.build_vocab(sentences)

        vocab_m = model.wv.index_to_key
        indices = [self.compass.wv.index_to_key.index(w) for w in vocab_m]
        new_syn1neg = np.array([self.compass.syn1neg[index] for index in indices])
        model.syn1neg = new_syn1neg
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
#         model.dv.index_to_key = []
#         model.dv.vectors = np.zeros([0, self.size])
#         model.dv.resize_vectors()
#         model.dv.vectors_lockf = np.ones(len(sentences), dtype=np.float32)
        model.train(sentences,
              total_words=model.corpus_total_words, epochs=self.dynamic_iter, compute_loss=True)
        if csave:
            model_name = os.path.splitext(os.path.basename(out_name))[0]
            self.trained_slices[model_name] = model

        if save and out_name:
            model.save(out_name)

        return model
def save(obj, fname):
    """
    pickle wrapper for saving CADE
    """
    with open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        pickle.dump(obj, fout)


def load(fname):
    """
    pickle wrapper for loading CADE

    """
    with open(fname, 'rb') as f:
        return pickle.load(f, encoding='latin1')  # needed because loading from S3 doesn't support readline()

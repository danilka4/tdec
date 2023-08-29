# -*- coding: utf-8 -*-

"""Main module."""

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models import doc2vec

import os
import numpy as np
import logging
import copy


class TDEC:
    """
    Handles alignment between multiple slices of text
    """
    def __init__(self, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                            min_count=5, workers=2, test = "test", opath="model", init_mode="hidden"):
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
        self.test = test
        self.opath = opath
        self.init_mode = init_mode
        self.compass = None

        if not os.path.isdir(self.opath):
            os.makedirs(self.opath)
        with open(os.path.join(self.opath, "log.txt"), "w") as f_log:
            f_log.write(str("")) # todo args
            f_log.write('\n')
            logging.basicConfig(filename=os.path.realpath(f_log.name),
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def initialize_from_compass(self, model):

        print("Initializing embeddings from compass.")

        if self.init_mode == "copy":
            model = copy.deepcopy(self.compass)
            model.learn_hidden = False
            model.alpha = self.dynamic_alpha
            model.iter = self.dynamic_iter
            return model

        if self.compass.layer1_size != self.size:
            return Exception("Compass and Slice have different vector sizes")
        vocab_m = model.wv.index2word
        indices = [self.compass.wv.vocab[w].index for w in vocab_m]

        # intialize syn1neg with compass embeddings
        new_syn1neg = np.array([self.compass.syn1neg[index] for index in indices])
        model.syn1neg = new_syn1neg

        if self.init_mode == "both":
            new_syn0 = np.array([self.compass.wv.syn0[index] for index in indices])
            model.wv.syn0 = new_syn0

        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
        return model

    def internal_trimming_rule(self, word, count, min_count):
        """
        Internal rule used to trim words
        :param word:
        :return:
        """
        if word in self.gvocab:
            return gensim.utils.RULE_KEEP
        else:
            return gensim.utils.RULE_DISCARD

    def train_model(self, sentences):
        model = None
        if self.compass == None or self.init_mode != "copy":
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
            model.build_vocab(sentences, trim_rule=self.internal_trimming_rule if self.compass != None else None)
        if self.compass != None:
            model = self.initialize_from_compass(model)
        model.train(sentences, total_words=sum([len(s) for s in sentences]), epochs=model.epochs, compute_loss=True)
        return model

    def train_compass(self, compass_text, keep_dv = False, overwrite=False, save=False, compass_name="compass.model"):
        compass_exists = os.path.isfile(os.path.join(self.opath, compass_name))
        if compass_exists and not overwrite:
            print("Compass is being loaded from file.")
            self.compass = Doc2Vec.load(os.path.join(self.opath, compass_name))
            self.gvocab = self.compass.wv.vocab
            return
        sentences = gensim.models.word2vec.PathLineSentences(compass_text)
        #[doc2vec.TaggedDocument(doc, [key]) for key, doc in texts.items()]
        sentences.input_files = [s for s in sentences.input_files if not os.path.basename(s).startswith('.')]
        print("Training the compass from scratch.")
        if compass_exists:
            print("Current saved compass will be overwritten after training")
        self.compass = self.train_model(sentences)
        if not keep_dv:
            self.compass.dv = gensim.models.KeyedVectors(vector_size=100)
        if save:
            self.compass.save(os.path.join(self.opath, compass_name))
        self.gvocab = self.compass.wv.vocab

    def train_slice(self, slice_text, slice_titles=None, save=False):
        """
        Training a slice of text
        :param slice_text:
        :param save:
        :return:
        """
        if self.compass == None:
            return Exception("Missing Compass")
        print("Training embeddings: slice {}.".format(slice_text))

        sentences = None
        if slice_titles:
            sentences = [doc2vec.TaggedDocument(doc, [title]) for doc, title in zip(slice_text, slice_titles)]
        else:
            sentences = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(slice_text)]
        model = self.train_model(sentences)

        model_name = os.path.splitext(os.path.basename(slice_text))[0]

        self.trained_slices[model_name] = model

        if save:
            model.save(os.path.join(self.opath, model_name + ".model"))

        return self.trained_slices[model_name]

# -*- coding: utf-8 -*-
import logging, gensim
from os.path import basename,splitext
from gensim.models import word2vec, doc2vec
import numpy as np
class TWEC:
    """
    Temporal Word Embeddings in a Compass
    Handles alignment between multiple slices of text
    """
    def __init__(self, size=100, mode="cbow", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                            min_count=5, workers=2, log=False, log_name="log.txt"):
        """

        :param size: Number of dimensions. Default is 100.
        :param mode: Either cbow or sg document embedding architecture of Word2Vec. cbow is default
        :param siter: Number of static iterations (epochs). Default is 5.
        :param diter: Number of dynamic iterations (epochs). Default is 5.
        :param ns: Number of negative sampling examples. Default is 10, min is 1.
        :param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
        :param alpha: Initial learning rate. Default is 0.025.
        :param min_count: Min frequency for words over the entire corpus. Default is 5.
        :param workers: Number of worker threads. Default is 2.
        :param test: Folder name of the diachronic corpus files for testing.
        :param opath: Name of the desired output folder. Default is model.
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
        self.compass = None
        self.trained_slices = {}
        self.learn_hidden = True
        
        # Log good, can tell you what's going on
        if log:
            with open(log_name, "w") as f_log:
                f_log.write(str("")) # todo args
                f_log.write('\n')
                logging.basicConfig(filename=f_log.name,
                                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                
    def internal_trimming_rule(self, word, count, min_count):
        """
        Internal rule used to trim words
        :param word:
        :return:
        """
        if word in self.compass.wv.vocab:
            return gensim.utils.RULE_KEEP
        else:
            return gensim.utils.RULE_DISCARD



    def train_compass(self, corpus_file=None, sentences=None):
        if not corpus_file and not sentences:
            return Exception('Please provide a "corpus_file" or "sentences"')
        if self.mode == "cbow":
            self.compass = word2vec.Word2Vec(sg=0, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "sg":
            self.compass = word2vec.Word2Vec(sg=1, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            return Exception('Set "mode" to be "cbow" or "sg"')
        self.compass.learn_hidden = True
        self.compass.build_vocab(corpus_file=corpus_file, sentences=sentences)
        self.compass.train(corpus_file=corpus_file, sentences=sentences,
              total_words=self.compass.corpus_total_words, epochs=self.static_iter, compute_loss=True)
        self.compass.learn_hidden = False

    def train_slice(self, corpus_file=None, sentences=None, out_name = None, csave=False, fsave=False):
        """
        Training a slice of text
        :param corpus_file: A file path of sentences
        :param sentences: A list of sentences
        :param out_name: output name/file path
        :param csave: save to compass
        :param fsave: save to file
        :return: model
        """
        if not corpus_file and not sentences:
            return Exception('Please provide a "corpus_file" or "sentences"')
        if self.compass == None:
            return Exception("Missing Compass")
        if csave and not out_name:
            return Exception("Specify compass name using 'out_name'")
        if fsave and not out_name:
            return Exception("Specify output file using 'out_name' to save")

        if not csave and not fsave:
            print("Warning: You don't save to anything. Save to compass with 'csave' or to file with 'fsave'")
        
        if self.mode == "cbow":
            model = word2vec.Word2Vec(sg=0, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "sg":
            model = word2vec.Word2Vec(sg=1, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            return Exception('Set "mode" to be "cbow" or "sg"')
        model.build_vocab(corpus_file=corpus_file, sentences=sentences,
                          trim_rule=self.internal_trimming_rule if self.compass != None else None)

        vocab_m = model.wv.index2word
        indices = [self.compass.wv.index2word.index(w) for w in vocab_m]
        new_syn1neg = np.array([self.compass.trainables.syn1neg[index] for index in indices])
        model.trainables.syn1neg = new_syn1neg
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.epochs = self.dynamic_iter
        
        model.train(corpus_file=corpus_file, sentences=sentences,
              total_words=model.corpus_total_words, epochs=self.dynamic_iter, compute_loss=True)
        if csave:
            model_name = splitext(basename(str(out_name)))[0]
            self.trained_slices[model_name] = model

        if fsave and out_name:
            model.save(out_name)

        return model

    
class TDEC(TWEC):
    """
    Temporal Document Embeddings in a Compass
    Handles alignment between multiple slices of text
    """
    def __init__(self, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                            min_count=5, workers=2, log=False, log_name="log.txt"):
        """
        Initialize Temporal Document Embeddings in a Compass
        :param size: Number of dimensions. Default is 100.
        :param mode: Either dm or dbow document embedding architecture of Doc2Vec. dm is default
            Note: DBOW as presented by Le and Mikolov (2014) does not train word vectors.
            Gensim's development of DBOW, which trains word vectors in skip-gram fashion in parallel to the DBOW process, will be used
        :param siter: Number of static iterations (epochs). Default is 5.
        :param diter: Number of dynamic iterations (epochs). Default is 5.
        :param ns: Number of negative sampling examples. Default is 10, min is 1.
        :param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
        :param alpha: Initial learning rate. Default is 0.025.
        :param min_count: Min frequency for words over the entire corpus. Default is 5.
        :param workers: Number of worker threads. Default is 2.
        """
        self.size = size
        if mode == "dm":
            w_mode = "cbow"
        elif mode == "dbow":
            w_mode = "sg"
        else:
            return Exception("Set mode to 'dm' or 'dbow'")
        self.d_mode = mode
        super().__init__(size=size, mode=w_mode, siter=siter, diter=diter,
                                   ns=ns, window=window, alpha=alpha,
                                   min_count=min_count, workers=workers, log=log, log_name=log_name)

    def train_slice(self, corpus_file=None, sentences=None, out_name = None, csave=False, fsave=False):
        """
        Training a slice of text
        :param corpus_file: File path to sentences. Doesn't name documents
        :param sentences: List of gensim.doc2vec.TaggedObject. Can name documents using TaggedObject
        :param out_name: Output name/file path
        :param csave: Save to compass
        :param fsave: Save to file
        :return: model
        """
        
        if not corpus_file and not sentences:
            return Exception('Please provide a "corpus_file" or "sentences"')
        if self.compass == None:
            return Exception("Missing Compass")
        if not csave and not fsave:
            print("Warning: You don't save to anything. Save to compass with 'csave' or to file with 'fsave'")
        if csave and not out_name:
            return Exception("Specify compass name using 'out_name'")
        if fsave and not out_name:
            return Exception("Specify output file using 'out_name' to save")


        if self.d_mode == "dm":
            model = doc2vec.Doc2Vec(size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.d_mode == "dbow":
            model = doc2vec.Doc2Vec(dm=0, dbow_words=1, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            return Exception('Set "mode" to be "dm" or "dbow"')
        model.build_vocab(corpus_file=corpus_file, documents=sentences,
                          trim_rule=self.internal_trimming_rule if self.compass != None else None)

        vocab_m = model.wv.index2word
        indices = [self.compass.wv.index2word.index(w) for w in vocab_m]
        new_syn1neg = np.array([self.compass.trainables.syn1neg[index] for index in indices])
        model.trainables.syn1neg = new_syn1neg
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.epochs = self.dynamic_iter
        model.train(corpus_file=corpus_file, documents=sentences,
              total_words=model.corpus_total_words, epochs=self.dynamic_iter)
        if csave:
            model_name = splitext(basename(str(out_name)))[0]
            self.trained_slices[model_name] = model

        if fsave and out_name:
            model.save(out_name)

        return model

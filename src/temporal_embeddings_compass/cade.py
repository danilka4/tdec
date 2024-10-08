# -*- coding: utf-8 -*-
import sys
import traceback
import logging
import gensim
import os
from os.path import basename, splitext
import re
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from umap import UMAP
import hdbscan
from hdbscan.flat import HDBSCAN_flat, approximate_predict_flat
from hdbscan import HDBSCAN
import pickle
from collections import Counter
import math
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

from scipy.spatial import distance

from numba import set_num_threads, get_num_threads

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    # -*- coding: utf-8 -*-

import sys

from concurrent.futures import ProcessPoolExecutor

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


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
        self.log = log
        
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
            raise Exception('Please provide a "corpus_file" or "sentences"')
        if self.mode == "cbow":
            self.compass = Word2Vec(sg=0, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "sg":
            self.compass = Word2Vec(sg=1, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            raise Exception('Set "mode" to be "cbow" or "sg"')
        self._train_compass(corpus_file, sentences)
    def _train_compass(self, corpus_file, sentences):
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
            raise Exception('Please provide a "corpus_file" or "sentences"')
        if self.compass == None:
            raise Exception("Missing Compass")
        if csave and not out_name:
            raise Exception("Specify compass name using 'out_name'")
        if fsave and not out_name:
            raise Exception("Specify output file using 'out_name' to save")

#         if not csave and not fsave:
#             print("Warning: You don't save to anything. Save to compass with 'csave' or to file with 'fsave'")
        
        if self.mode == "cbow":
            model = Word2Vec(sg=0, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode == "sg":
            model = Word2Vec(sg=1, vector_size=self.size, alpha=self.static_alpha, epochs=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            raise Exception('Set "mode" to be "cbow" or "sg"')
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
            mode_w = "cbow"
        elif mode == "dbow":
            mode_w = "sg"
        else:
            raise Exception("Set mode to 'dm' or 'dbow'")
        self.mode_d = mode
        self.alpha_d = alpha
        super().__init__(size=size, mode=mode_w, siter=siter, diter=diter,
                                   ns=ns, window=window, alpha=alpha,
                                   min_count=min_count, workers=workers, log=log, log_name=log_name)


    def train_compass(self, corpus_file=None, sentences=None, create_documents=True):
        if not corpus_file and not sentences:
            raise Exception('Please provide a "corpus_file" or "sentences"')
        if not create_documents:
            super().train_compass(corpus_file=corpus_file, sentences=sentences)
            return
        if self.mode_d == "dm":
            self.compass = Doc2Vec(vector_size=self.size,
                         alpha=self.static_alpha, alpha_d=self.alpha_d,
                         epochs=self.static_iter, negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode_d == "dbow":
            self.compass = Doc2Vec(dm=0, dbow_words=1, vector_size=self.size,
                             alpha=self.static_alpha, alpha_d=self.alpha_d,
                             epochs=self.static_iter, negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            raise Exception('Set "mode" to be "dm" or "dbow"')
        self._train_compass(corpus_file, sentences)

    def _train_compass(self, corpus_file, sentences):
        self.compass.learn_hidden = True
        self.compass.build_vocab(corpus_file=corpus_file, documents=sentences)
        self.compass.train(corpus_file=corpus_file, documents=sentences,
              total_words=self.compass.corpus_total_words, epochs=self.static_iter)
        self.compass.learn_hidden = False

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
            raise Exception('Please provide a "corpus_file" or "sentences"')
        if self.compass == None:
            raise Exception("Missing Compass")
#         if not csave and not fsave:
#             print("Warning: You don't save to anything. Save to compass with 'csave' or to file with 'fsave'")
        if csave and not out_name:
            raise Exception("Specify compass name using 'out_name'")
        if fsave and not out_name:
            raise Exception("Specify output file using 'out_name' to save")


        if self.mode_d == "dm":
            model = Doc2Vec(vector_size=self.size, alpha=self.dynamic_alpha, alpha_d = self.alpha_d, epochs=self.dynamic_iter,
                         negative=self.negative,
                         window=self.window, min_count=self.min_count, workers=self.workers)
        elif self.mode_d == "dbow":
            model = Doc2Vec(dm=0, dbow_words=1, vector_size=self.size, alpha=self.dynamic_alpha, epochs=self.dynamic_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
        else:
            raise Exception('Set "mode" to be "dm" or "dbow"')
        model.build_vocab(corpus_file=corpus_file, documents=sentences,
                          trim_rule=self.internal_trimming_rule if self.compass != None else None)
#         print(len(sentences), model.docvecs.vectors_docs.shape)
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
class TTEC(TDEC):
    """
    Temporal Topic Embeddings in a Compass
    Handles alignment between multiple slices of text
    """
    def __init__(self, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                    min_count=5, workers=2, log=False, log_name="log.txt",
                    umap_args=None, hdbscan_args=None, n_topics = 10, hdbscan_selection = "nested", n_terms = 10):
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
        self.global_topics = None
        self.umap = None
        self.umap_args=umap_args
        self.hdbscan = None
        self.hdbscan_args=hdbscan_args
        self.hdbscan_selection = hdbscan_selection
        self.n_topics = n_topics
        self.n_terms = n_terms
        self.topic_map = None
        super().__init__(size=size, mode=mode, siter=siter, diter=diter,
                                   ns=ns, window=window, alpha=alpha,
                                   min_count=min_count, workers=workers, log=log, log_name=log_name)
    def train_compass(self, corpus_file=None, sentences=None, create_topics=True, neighbors_2d = 15):
        try:
            super().train_compass(corpus_file=corpus_file, sentences=sentences, create_documents=True)
            self.compass_umap_2d = UMAP(n_neighbors = neighbors_2d).fit(self.compass.docvecs.vectors_docs)
            logging.info("2D umap created")
            logging.info("Compass trained")
            if create_topics:
                self.remake_topic_embeddings(None, None)
                logging.info("Topics made")
        except:
            logging.exception("Compass unable to train")
    def remake_topic_embeddings(self, umap_args=None, hdbscan_args=None):
        """
        Obtains topics using the "flat algorithm."
        The flat algorithm finds the largest HDBSCAN min_cluster_size
        in a binary search. This is because the largest min_cluster_size
        will have all the topics be as large as possible.
        Has performance issues when dealing with big data due to recomputing
        the HDBSCAN space many times.
        """
        if umap_args:
            pass
        elif self.umap_args:
            umap_args = self.umap_args
        else:
            umap_args = {'n_neighbors': 15,
                         'n_components': 5,
                         'metric': 'cosine'}
#         umap_args["verbose"] = True
        self.umap = UMAP(**umap_args)
        self.umap.fit(self.compass.docvecs.vectors_docs)
        self.remake_topics(hdbscan_args)
    def _find_optimal_flat(self):
        logging.info("Started flat")
        self.topic_map = None
        n_left = 2
        
#         max_hdbscan = hdbscan.HDBSCAN(min_cluster_size = 3)
#         max_hdbscan.fit(self.umap.embedding_)
#         n_right = math.floor(len(np.unique(max_hdbscan.labels_)))
        n_right = math.floor(self.compass.docvecs.vectors_docs.shape[0] / self.n_topics)
        if n_right < 2:
            n_right = 2
        n_right = min(n_right, 1000)
        hdbscan_left = None
        while n_left <= n_right:
            n_mid = math.floor((n_left + n_right) / 2)
            logging.info(f"flat {n_left}, {n_mid}, {n_right}")
#             print("flat", HDBSCAN_flat)
#             print("args", self.hdbscan_args)
#             print("umap", self.umap.embedding_)
#             HDBSCAN_flat(self.umap.embedding_, **self.hdbscan_args)

            hdbscan_mid = HDBSCAN_flat(self.umap.embedding_, min_cluster_size=n_mid, **self.hdbscan_args)
            topic_counts = np.unique(hdbscan_mid.labels_, return_counts=True)
#             print("Amount of topics + noise:", len(topic_counts[0]))
            if len(topic_counts[0]) == self.n_topics + 1:
                hdbscan_left = hdbscan_mid
                n_left = n_mid + 1
            elif len(topic_counts[0]) < self.n_topics + 1:
                n_right = n_mid - 1
            else:
                logging.exception("This should be impossible")
                n_right = n_mid - 1 # prevents infinite loop
        if hdbscan_left:
            return hdbscan_left
        else:
            logging.exception("Unable to find a diagram with n topics")
            return None
    def _find_optimal_nested(self):
        """
        Repeatedly places the smallest topic into its closest topic until the
        desired topic count is reached. Could be optimized by not repeatedly
        recreating topic summary.
        """
        logging.info("Started nested")
        hdbscan_overall = hdbscan.HDBSCAN(**self.hdbscan_args)
        hdbscan_overall.fit(self.umap.embedding_)
        # np.argpartition(np.unique(wrapper.compass.hdbscan.labels_, return_counts=True)[1], 1)
        lab_overall = hdbscan_overall.labels_
        lab_stats = np.unique(lab_overall, return_counts=True)
        self.topic_map = {}
        for topic in lab_stats[0]:
            self.topic_map[topic] = topic
        while len(lab_stats[0]) > self.n_topics + 1:
            smallest_index = np.argpartition(lab_stats[1], 1)[0]
            smallest_topic = lab_stats[0][smallest_index]
            topic_spread = self._create_topic_summary(self.compass, lab_overall, "centroid")
            smallest_topvec = topic_spread[smallest_topic]["topvec"]
            topic_possibilities = list(topic_spread.keys())
            if topic_possibilities[0] not in [-1, smallest_topic]:
                closest_topic = topic_possibilities[0]
            elif topic_possibilities[1] not in [-1, smallest_topic]:
                closest_topic = topic_possibilities[1]
            else:
                closest_topic = topic_possibilities[2]
            closest_topvec = topic_spread[closest_topic]["topvec"]
            closest_cos_sim = np.dot(smallest_topvec, closest_topvec) / (np.linalg.norm(smallest_topvec) * np.linalg.norm(closest_topvec))
            for topic_number in topic_spread:
                if topic_number == smallest_topic or topic_number == -1:
                    continue
                comparison_topvec = topic_spread[topic_number]["topvec"]
                comparison_cos_sim = np.dot(smallest_topvec, comparison_topvec) / (np.linalg.norm(smallest_topvec) * np.linalg.norm(comparison_topvec))
                if comparison_cos_sim > closest_cos_sim:
                    closest_topic = topic_number
                    closest_topvec = comparison_topvec
                    closest_cos_sim = comparison_cos_sim
            lab_overall[lab_overall == smallest_topic] = closest_topic
            self.topic_map[smallest_topic] = closest_topic
            for k, v in self.topic_map.items():
                if v == smallest_topic:
                    self.topic_map[k] = closest_topic
            lab_stats = np.unique(lab_overall, return_counts=True)
            logging.info(f"After taking out Amount of topics: {len(lab_stats[0])}. Topics: {lab_stats[0]}, Smallest topic: {smallest_topic}, Closest topic: {closest_topic}")

        hdbscan_overall.labels_ = lab_overall
        final_topics = list(set(self.topic_map.values()))
        for topic in final_topics:
            self.topic_map[topic] = topic
        return hdbscan_overall
    def reduce_topics(self, n_topics):
        """
        Reduces the amount of topics to a specified count.
        """
        if n_topics >= self.n_topics:
            return
        self.n_topics = n_topics
        lab_stats = np.unique(self.global_topics, return_counts=True)
        while len(lab_stats[0]) > self.n_topics + 1:
            smallest_index = np.argpartition(lab_stats[1], 1)[0]
            smallest_topic = lab_stats[0][smallest_index]
            topic_spread = self._create_topic_summary(self.compass, self.global_topics, "centroid")
            smallest_topvec = topic_spread[smallest_topic]["topvec"]
            topic_possibilities = list(topic_spread.keys())
            if topic_possibilities[0] not in [-1, smallest_topic]:
                closest_topic = topic_possibilities[0]
            elif topic_possibilities[1] not in [-1, smallest_topic]:
                closest_topic = topic_possibilities[1]
            else:
                closest_topic = topic_possibilities[2]
            closest_topvec = topic_spread[closest_topic]["topvec"]
            closest_cos_sim = np.dot(smallest_topvec, closest_topvec) / (np.linalg.norm(smallest_topvec) * np.linalg.norm(closest_topvec))
            for topic_number in topic_spread:
                if topic_number == smallest_topic or topic_number == -1:
                    continue
                comparison_topvec = topic_spread[topic_number]["topvec"]
                comparison_cos_sim = np.dot(smallest_topvec, comparison_topvec) / (np.linalg.norm(smallest_topvec) * np.linalg.norm(comparison_topvec))
                if comparison_cos_sim > closest_cos_sim:
                    closest_topic = topic_number
                    closest_topvec = comparison_topvec
                    closest_cos_sim = comparison_cos_sim
            self.global_topics[self.global_topics == smallest_topic] = closest_topic
#             print(self.topic_map, '||||||', smallest_topic, '|||||', closest_topic)
            if self.topic_map:
                self.topic_map[smallest_topic] = closest_topic
                for k, v in self.topic_map.items():
                    if v == smallest_topic:
                        self.topic_map[k] = closest_topic
            lab_stats = np.unique(self.global_topics, return_counts=True)
            logging.info(f"After taking out Amount of topics: {len(lab_stats[0])}. Topics: {lab_stats[0]}, Smallest topic: {smallest_topic}, Closest topic: {closest_topic}")

#         final_topics = list(set(self.topic_map.values()))
#         for topic in final_topics:
#             self.topic_map[topic] = topic
#         return hdbscan_overall

    def remake_topics(self, hdbscan_args = None, hdbscan_selection = None, similarity_method = "vote"):
        """
        Recreates the topics without changing the topic embedding space
        """
        if not hdbscan_selection:
            hdbscan_selection = self.hdbscan_selection
        if hdbscan_args:
            self.hdbscan_args = hdbscan_args
        elif self.n_topics and hdbscan_selection == "flat":
            self.hdbscan_args = {'n_clusters': self.n_topics,
                        'metric': 'euclidean',
                            'cluster_selection_method': 'leaf',
                           'prediction_data': True
                                }
        elif self.n_topics and hdbscan_selection == "nested":
            self.hdbscan_args = {'min_cluster_size': 15,
                            'metric': 'euclidean',
                            'cluster_selection_method': 'leaf',
                           'prediction_data': True}
        elif self.hdbscan_args:
            pass
        else:
            self.hdbscan_args = {'min_cluster_size': 15,
                            'metric': 'euclidean',
                            'cluster_selection_method': 'leaf',
                           'prediction_data': True}
        if self.n_topics and hdbscan_selection == "flat":
            self.hdbscan = self._find_optimal_flat()
            self.global_topics = self.hdbscan.labels_
        elif self.n_topics and hdbscan_selection == "nested":
            self.hdbscan = self._find_optimal_nested()
            self.global_topics = self.hdbscan.labels_
        else:
            self.hdbscan = hdbscan.HDBSCAN(**self.hdbscan_args)
            self.global_topics = self.hdbscan.fit_predict(self.umap.embedding_)

        self.global_topic_summary = self._create_topic_summary(self.compass, self.global_topics, similarity_method)

        return self.global_topics
    def _fit_to_global_topics(self, vecs):
        """
        Takes outside vectors and places them into the topic space
        """
        reduction = self.umap.transform(vecs)
        return reduction, self._approx_predict(reduction)
    def _approx_predict(self, embeddings):
        """
        Predicts HDBSCAN topics given UMAP embeddings
        """
        if self.hdbscan_selection == "flat":
            output = approximate_predict_flat(self.hdbscan, embeddings, n_clusters=self.n_topics)[0]
        else:
            output = hdbscan.approximate_predict(self.hdbscan, embeddings)[0]
        if self.topic_map:
            output = np.array([self.topic_map[i] for i in output])
        return output
    def _create_topic_summary(self, model, topics, similarity_method = "vote"):
        """
        Creates summary of HDBSCAN topics using word2vec
        """
        topic_summary = {}
        topics_in_slice = list(set(topics))
        for topic in topics_in_slice:
            if similarity_method == "weighted":
                topic_vector = self.hdbscan.weighted_cluster_centroid(topic)
            else:
                vecs_in_topic = model.docvecs.vectors_docs[topics == topic]
                topic_vector = vecs_in_topic.mean(axis=0)
            topic_top_words = []
            if similarity_method in ["centroid", "weighted"]:
                topic_top_words = [term for (term, _) in model.wv.most_similar([topic_vector], topn=self.n_terms)]
            elif similarity_method == "vote":
                top_n_per_article = []
                for vec_index in range(vecs_in_topic.shape[0]):
                    vec = vecs_in_topic[vec_index]
                    arts = [term for (term, _) in model.wv.most_similar([vec], topn=self.n_terms)]
                    top_n_per_article += arts
                counter = Counter(top_n_per_article)
                topic_top_words = [term for (term, _) in counter.most_common(self.n_terms)]
            else:
                raise Exception("Please select a valid similarity_method")
            topic_summary[topic] = {"topvec": topic_vector, "topn": topic_top_words}
        return topic_summary

    def train_slice(self, corpus_file=None, sentences=None, out_name = None, csave=False, fsave=False,
                   similarity_method = "vote", create_topics = True):
        """
        Training a slice of text
        :param corpus_file: File path to sentences. Doesn't name documents
        :param sentences: List of gensim.doc2vec.TaggedObject. Can name documents using TaggedObject
        :param n_terms: Number of terms describing each topic
        :param out_name: Output name/file path
        :param csave: Save to compass
        :param fsave: Save to file
        :param similarity_method: Method by which topic-defining words are extracted
            "centroid" - A topic vector is calculated as the mean of a topic, from which similar words are found
            "vote" - Calculate each article's top n terms. The n most common terms are returned from that 
        :param n_terms: Amount of output terms from a topic
        :return: model
        """
#         if slice_number is None:
#             return Exception("Please provide the slice number being worked on using 'slice_number'")
#         if not self.slice_sizes:
#             return Exception("Please provide a 'slice_sizes' array. This is a list of integers containing the amount of numbers per time slice")
#         if slice_number < 0 or slice_number > len (self.slice_sizes):
#             return Exception("Provide a valid slice number that's in the amount of slices you have, zero-indexed")
        
#         start_index = sum(self.slice_sizes[:slice_number])
#         end_index = sum(self.slice_sizes[:(slice_number+1)])
#         local_topics = self.global_topics[start_index:end_index]
        
        # if sentences and len(sentences) != end_index - start_index:
        #     return Exception("The amount of given sentences doesn't equal to the amount of topic labels")
        
        model = super().train_slice(corpus_file=corpus_file, sentences=sentences, out_name=out_name,
                            csave=csave, fsave=fsave)
#         return model
        # Create topics
        if create_topics:
            local_reduction, local_topics = self._fit_to_global_topics(model.docvecs.vectors_docs)
            topic_summary = self._create_topic_summary(model, local_topics, similarity_method = similarity_method)
        else:
            local_reduction, local_topics, topic_summary = None, None, None
        time_slice = TimeSlice(model, local_reduction, local_topics, topic_summary)#, local_reduction, local_topics, topic_summary)
        if fsave:
            save(time_slice, out_name)
        return time_slice
class TimeSlice:
    """
    Time Slice for TTEC
    """
    def __init__(self, model, reduction=None, topics=None, topic_summary=None):
        self.model = model
        self.reduction = reduction
        self.topics = topics
        self.topic_summary = topic_summary
    def new_topic_embeddings(self, ttec_model:TTEC, similarity_method="vote", n_terms=10):
        self.reduction, self.topics = ttec_model._fit_to_global_topics(self.model.docvecs.vectors_docs)
        self.umap_2d = ttec_model.compass_umap_2d.transform(self.model.docvecs.vectors_docs)
        og_n_terms = ttec_model.n_terms
        ttec_model.n_terms = n_terms
        self.topic_summary = ttec_model._create_topic_summary(self.model, self.topics, similarity_method=similarity_method)
        ttec_model.n_terms = og_n_terms
        return self.topics
    def new_topics(self, ttec_model:TTEC, similarity_method="vote", n_terms=10):
        self.topics = ttec_model._approx_predict(self.reduction)
        og_n_terms = ttec_model.n_terms
        ttec_model.n_terms = n_terms
        self.topic_summary = ttec_model._create_topic_summary(self.model, self.topics, similarity_method=similarity_method)
        ttec_model.n_terms = og_n_terms
        return self.topics

class TTEC_wrapper:
    """
    Wrapper that automatically creates time slices using time stamp
    (akin to BERTopic). Might be useful to merge this into TTEC.
    """
    def __init__(self, time_stamps, sentences=None, corpus_file=None, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                    min_count=5, yearly=True, n_slices=-1, similarity_method="vote", hdbscan_selection="flat", n_terms=10,
                    workers=2, log=False, log_name="log.txt", train_compass_now = True, train_slices_now = True,
                    umap_args=None, hdbscan_args=None, n_topics = None):
        if not sentences and not corpus_file:
            raise Exception("Select 'sentences' or 'corpus_file'")
        elif sentences and corpus_file:
            raise Exception("Select 'sentences' xor 'corpus_file'")
        self.sentences = sentences
        self.corpus_file = corpus_file
        self.yearly=yearly
        self.time_stamps = time_stamps
        self.n_slices = n_slices
        self.hdbscan_selection = hdbscan_selection
        self.n_terms = n_terms
        self.workers = workers
        self.similarity_method = similarity_method
        self.mode = mode
        if log:
            with open(log_name, "w") as f_log:
                f_log.write(str("")) # todo args
                f_log.write('\n')
                logging.basicConfig(filename=f_log.name,
                                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.compass = TTEC(size=size, mode=mode, siter=siter, diter=diter, ns=ns, window=window, alpha=alpha,
                    min_count=min_count, workers=workers, log=False, log_name=log_name, # Time stamps... 
                    umap_args=umap_args, hdbscan_args=hdbscan_args, n_topics = n_topics, hdbscan_selection=hdbscan_selection, n_terms=n_terms)
        self.slices = {}

        if yearly:
            self.time_intervals = [datetime.datetime(i, 1, 1) for i in range(min(time_stamps).year, max(time_stamps).year+2)]
        elif n_slices == -1:
            raise Exception("Specify either yearly or a number of slices")
        else:
            self.time_intervals = pd.date_range(start=min(time_stamps), end=max(time_stamps), periods = n_slices).to_pydatetime().tolist()


        if train_compass_now:
            self.train_compass()
        if train_slices_now:
            self.train_slices()
    def train_compass(self, neighbors_2d = 15):
        """
        Trains compass and creates 2D UMAP embedding space.
        """
        if self.sentences:
            sentences = [TaggedDocument(sentence, [i]) for i, sentence in enumerate(self.sentences)]
        else:
            sentences = None
        self.compass.train_compass(sentences=sentences, corpus_file=self.corpus_file, neighbors_2d=neighbors_2d)
        logging.info("Compass made")
    def _obtain_sentence_subset(self, i):
        """
        Obtains the data for a particular time slice
        """
        if self.yearly:
            fax = (np.array(self.time_stamps) >= self.time_intervals[i]) * (np.array(self.time_stamps) < self.time_intervals[i + 1])
        else:
            fax = (np.array(self.time_stamps) >= self.time_intervals[i]) * (np.array(self.time_stamps) < self.time_intervals[i + 1])
        if self.sentences:
            sentences_subset = [sentence for j, sentence in enumerate(self.sentences) if fax[j]]
        else:
            nonzero = np.flatnonzero(fax).tolist()
            sentences_subset = []
            with open(self.corpus_file, 'r') as f:
                for i, line in enumerate(f):
                    if not nonzero:
                        break
                    if i in nonzero:
                        nonzero.remove(i)
                        sentences_subset.append(line.rstrip().split())
        return sentences_subset
    def name(self, i):
        """
        Names the time slice based on whether it is yearly or something else.
        """
        time = self.time_intervals[i]
        if self.yearly:
            return time.year
        else:
            return time.strftime("%Y/%m/%d")
    def _train_slice(self, i, create_topics=True, verbose=False):
        """
        Trains an individual time slice.
        """
        try:
            sentences_subset = self._obtain_sentence_subset(i)
            # This line is here because gensim interacts weirdly with list subsets otherwise
            sentences_in_range = [TaggedDocument(document, [j]) for j,document in enumerate(sentences_subset)]
#             if verbose:
#                 print(len(sentences_in_range))
            slic = self.compass.train_slice(sentences=sentences_in_range, similarity_method = self.similarity_method,
                                            fsave=False, create_topics=create_topics)
            if create_topics:
                slic.umap_2d = self.compass.compass_umap_2d.transform(slic.model.docvecs.vectors_docs)
            slic.i = i
            logging.info(f"Trained time slice {self.name(i)}")
            return slic
        except Exception as exception:
            print(exception, flush=True)
            raise exception
    def train_slices(self, parallel = False, verbose = False):
        """
        Trains time slices. Parallel option uses a ProcessPoolExecutor,
        but does not work as intended. A ProcessPoolExecutor created outside
        this class will work.
        """
        self.slices = {}
        if parallel:
            if __name__ == '__main__':
                with ProcessPoolExecutor(workers=self.workers) as executor:
                    cur_threads = get_num_threads()
                    set_num_threads(1)
                    range_executor = len(self.time_intervals) - 1
                    for result in executor.map(self._train_slice, range(range_executor), [False] * range_executor, [verbose] * range_executor):
#                         print(result.i, flush=True)
                        self.slices[self.name(result.i)] = result
                    set_num_threads(cur_threads)
                for i in self.slices:
                    self.slices[i].new_topic_embeddings(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)
                    self.slices[i].umap_2d = self.compass.compass_umap_2d.transform(self.slices[i].model.docvecs.vectors_docs)
                    logging.info(f"Slice {i} topics made")
        else:
            for i in range(len(self.time_intervals) - 1):
                self.slices[self.name(i)] = self._train_slice(i, verbose=verbose)
    def remake_topic_embeddings(self, umap_args=None, hdbscan_args=None):
        """
        Recreates UMAP and HDBSCAN topics.
        """
        self.compass.remake_topic_embeddings(umap_args, hdbscan_args)
        if self.slices and len(self.slices) > 0 :
            for slic in self.slices:
                self.slices[slic].new_topic_embeddings(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)
    def remake_topics(self, hdbscan_args = None, remake_slices=True, hdbscan_selection=None, similarity_method=None):
        """
        Remakes HDBSCAN topics, with potential to also remake time slice topics.
        """
        topics = self.compass.remake_topics(hdbscan_args, hdbscan_selection, similarity_method)
        logging.info("Compass topics remade")
        if similarity_method:
            self.similarity_method = similarity_method
        if remake_slices:
            for i in self.slices:
                self.slices[i].new_topics(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)
                logging.info(f"Slice {i} topics made")
        return topics
    def remake_slice_embeddings(self):
        for slic in self.slices:
            self.slices[slic].new_topic_embeddings(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)
    def remake_slice_topics(self):
        for i in self.slices:
            self.slices[i].new_topics(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)
            logging.info(f"Slice {i} topics made")
    def reduce_topics(self, n_topics, remake_slices=True):
        """
        Reduces the number of topics. Only works for nested option.
        """
        if n_topics >= self.compass.n_topics:
            return
        self.compass.reduce_topics(n_topics)
        self.compass.global_topic_summary = self.compass._create_topic_summary(self.compass.compass, self.compass.global_topics, self.similarity_method)
        if remake_slices:
            for i in self.slices:
                self.slices[i].new_topics(self.compass, similarity_method=self.similarity_method, n_terms=self.n_terms)

    def plot_compass(self, n_terms = None, alpha=0.5):
        """
        Plots what the compass using plotly express.
        """
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        comp_df = pd.DataFrame({'x': self.compass.compass_umap_2d.embedding_[:,0],
                              'y': self.compass.compass_umap_2d.embedding_[:,1],
                               'topic': [str(self.compass.global_topics[i]) + ': ' + " ".join(self.compass.global_topic_summary[self.compass.global_topics[i]]["topn"][0:n_terms]) for i in range(self.compass.global_topics.shape[0])]
                               })
        fig = px.scatter(comp_df, x='x', y='y', color='topic', opacity=alpha, title="Compass")
        return fig
    def plot_term(self, term = None, n_terms = None, years = None):
        if not term:
            print("Please say what term(s) you want")
            return
        elif type(term) == str:
            term = [term]
        if not years:
            years = self.slices.keys()
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
            
        fig = make_subplots(rows=1, cols=1)
        for tn in np.unique(self.compass.global_topics):
            if tn == -1:
                continue
            sub = self.compass.compass_umap_2d.embedding_[[i for i,topic in enumerate(self.compass.global_topics) if topic == tn],:]
            if n_terms > 0:
                nam = f"{tn} - {','.join(self.compass.global_topic_summary[tn]['topn'][0:n_terms])} "
            else:
                nam = str(tn)
            fig.add_trace(go.Scatter(x=sub[:,0], y=sub[:,1], mode="markers", name=nam))
        for t in term:
            term_id = []
            term_vectors = []
            for year in years:
                slic = self.slices[year]
                if t not in slic.model.wv.index2word:
                    continue
                term_id.append(f"{t} - {year}")
                term_vectors.append(slic.model.wv[t])
            if len(term_vectors) > 0:
                twod = self.compass.compass_umap_2d.transform(term_vectors)
                lis_df = pd.DataFrame({'name': term_id, 'x': twod[:,0] , 'y': twod[:,1]})
                fig.add_trace(go.Scatter(x=lis_df["x"], y=lis_df["y"], text=lis_df["name"], mode="markers+text+lines", line=dict(color="black"), textposition=improve_text_position(lis_df.index), name=t))
        return fig
    def plot_slice(self, slice_num, n_terms = None):
        """
        Plots a single time slice using plotly express.
        """
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        
        wrap_slice = self.slices[slice_num]
        list_of_names = list_of_names = [" ".join([f"{i}:"]+wrap_slice.topic_summary[i]["topn"][0:n_terms]) for i in wrap_slice.topics.tolist()]
        wrap_slice_df = pd.DataFrame({'x': wrap_slice.umap_2d[:,0],
                                      'y': wrap_slice.umap_2d[:,1],
                                      'name': list_of_names
                                     })
        fig = px.scatter(wrap_slice_df, x='x', y='y', color='name', title=slice_num)
        return fig
    def plot_slider_scatter(self, n_terms=5):
        """
        Plots all time slices and compass with a slider
        """
        fig = go.Figure()
        wrap_df = pd.DataFrame({'x': self.compass.compass_umap_2d.embedding_[:,0],
                                          'y': self.compass.compass_umap_2d.embedding_[:,1],
                                          'topic': map(str,self.compass.global_topics.tolist()),#[" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrapper.compass.global_topics.tolist()],
                                          'date': "compass",
                                          'name': [str(self.compass.global_topics[i]) + ': ' + " ".join(self.compass.global_topic_summary[self.compass.global_topics[i]]["topn"][0:n_terms]) for i in range(self.compass.global_topics.shape[0])]
                                         })
        # wrap_df = pd.DataFrame()
        for step in list(self.slices.keys()):
            wrap_slice = self.slices[step]
            wrap_slice_df = pd.DataFrame({'x': wrap_slice.umap_2d[:,0],
                                          'y': wrap_slice.umap_2d[:,1],
                                          'topic': map(str,wrap_slice.topics.tolist()),#[" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrap_slice.topics.tolist()],
                                          'date': step,
                                          'name': [" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrap_slice.topics.tolist()]
                                         })
            wrap_df = pd.concat([wrap_df,wrap_slice_df])
        fig = px.scatter(wrap_df, x="x", y="y", color="topic", animation_frame="date", title="Scatterplots of Compass and Time Slices", hover_data=["name"])
        fig["layout"].pop("updatemenus") # optional, drop animation buttons
        return fig
    def plot_line(self, include_noise = False, n_terms=5):
        """
        Plots a line graph that shows the amount of articles in each topic
        per time period
        """
        df_list = []
        for date, slic in self.slices.items():
            topic_counts = np.unique(slic.topics, return_counts=True)
            for topic in topic_counts[0]:
                if topic == -1 and not include_noise:
                    continue
                words = slic.topic_summary[topic]["topn"][:n_terms]
                count = topic_counts[1][topic_counts[0] == topic][0]
                df_list.append([date, f"{topic} - {' '.join(self.compass.global_topic_summary[topic]['topn'][:n_terms])}", count, words])
        df = pd.DataFrame(df_list, columns=["Date", "Cluster", "Counts", "Words"])
        fig = px.line(df, x="Date", y="Counts", color="Cluster", hover_data="Words")
        return fig
    def topic_distribution(self):
        """
        Obtains compass topic distribution
        """
        return np.unique(self.compass.hdbscan.labels_, return_counts=True)
    def test_coherence(self, return_annual_count = False):
        """
        Tests coherence of topics.
        return_annual_count - Breaks down by time slice
        """
        annual_count = {}
        for idx, date in enumerate(self.slices):
            slic = self.slices[date]
            sentences_subset = self._obtain_sentence_subset(idx)
#         print(sentences_subset[0])
#         sentences_subset = [simple_preprocess(" ".join(doc)) for doc in sentences_subset]
            dic = corpora.Dictionary()
            BoW_corpus = [dic.doc2bow(doc, allow_update=True) for doc in sentences_subset]
            topics = [slic.topic_summary[thing]['topn'] for thing in slic.topic_summary]  
#         texts = [[dictionary.token2id[word] for word in sentence] for sentence in sentences_subset]
            mod = CoherenceModel(topics=topics, texts=sentences_subset, dictionary=dic, topn=10, coherence='c_npmi')
            annual_count[date] = mod.get_coherence()
#         annual_count[date] = np.mean(np.array(local_count))
        if return_annual_count:
            return (annual_count, np.mean(np.array(list(annual_count.values()))))
        else:
            return np.mean(np.array(list(annual_count.values())))

    def test_diversity(self, return_annual_count = False):
        """
        Tests diversity of topics.
        return_annual_count - Breaks down by time slice
        """
        annual_count = {}
        for date in self.slices:
            slic = self.slices[date]
            local_count = {}
            for topic_number in slic.topic_summary:
                if topic_number != -1:
                    topic_words = slic.topic_summary[topic_number]["topn"]
                    for word in topic_words:
                        try:
                            local_count[word] += 1
                        except:
                            local_count[word] = 1
            annual_count[date] = local_count
        sums = []
        for date in annual_count.keys():
            slic = annual_count[date]
            sums.append(np.mean(np.array(list(slic.values())) == 1))
        if return_annual_count:
            return (np.mean(sums), sums)
        else:
            return np.mean(sums)

class TTEC_big:
    """
    Wrapper that automatically creates time slices for big data using time stamp
    (akin to BERTopic). Might be useful to merge this into TTEC.
    """
    def __init__(self, corpus_file=None, size=100, mode="dm", siter=5, diter=5, ns=10, window=5, alpha=0.025,
                    min_count=5, yearly=True, n_slices=-1, similarity_method="vote", hdbscan_selection="flat", n_terms=10,
                    workers=2, log=False, log_name="log.txt", umap_args=None, hdbscan_args=None, n_topics = None,
                    file_path=None, 
                ):
        if not corpus_file:
            raise Exception("Select 'corpus_file'")
        self.corpus_file = corpus_file
        self.yearly=yearly
        self.hdbscan_selection = hdbscan_selection
        self.n_terms = n_terms
        self.workers = workers
        self.similarity_method = similarity_method
        self.mode = mode
        self.umap_args = umap_args
        self.hdbscan_args = hdbscan_args
        self.n_topics = n_topics
        self.slice_names = []
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = os.getcwd()
        if log:
            with open(log_name, "w") as f_log:
                f_log.write(str("")) # todo args
                f_log.write('\n')
                logging.basicConfig(filename=f_log.name,
                                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        tdec = TDEC(size=size, mode=mode, siter=siter, diter=diter, ns=ns, window=window, alpha=alpha,
                    min_count=min_count, workers=2, log=False, log_name=log_name)
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        data_path = os.path.join(self.file_path, "data")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        slices_path = os.path.join(self.file_path, "slices")
        if not os.path.exists(slices_path):
            os.mkdir(slices_path)
        save(tdec, os.path.join(self.file_path, "tdec_og.pkl"))
        terms_path = os.path.join(self.file_path, "terms.csv")
        if not os.path.exists(terms_path):
            with open(terms_path, 'w') as f:
                f.write('term,ts,x,y\n')
        self.slices = {}
    def train_compass(self, neighbors_2d = 15, overwrite = True):
        """
        Trains compass and creates 2D UMAP embedding space.
        """
        tdec = load(os.path.join(self.file_path, "tdec_og.pkl"))
        tdec_path = os.path.join(self.file_path, "tdec.pkl")
        if (overwrite and os.path.exists(tdec_path)) or not os.path.exists(tdec_path):
            tdec.train_compass(corpus_file=self.corpus_file)
            save(tdec, tdec_path)
            logging.info("TDEC made")
        else:
            tdec = load(tdec_path)
            logging.info("TDEC loaded")
        umap_2d_path = os.path.join(self.file_path, "umap_2d.pkl")
        if (overwrite and os.path.exists(umap_2d_path)) or not os.path.exists(umap_2d_path):
            ump_2d = UMAP(n_neighbors = neighbors_2d, n_components=2, metric="cosine")
            ump_2d.fit(tdec.compass.docvecs.vectors_docs)
            save(ump_2d, umap_2d_path)
            logging.info("2D UMAP made")
            del ump_2d
        umap_path = os.path.join(self.file_path, "umap.pkl")
        if (overwrite and os.path.exists(umap_path)) or not os.path.exists(umap_path):
            if self.umap_args:
                umap_args = self.umap_args
            else:
                umap_args = {'n_neighbors': 15,
                             'n_components': 5,
                             'metric': 'cosine'}
    #         umap_args["verbose"] = True
            ump = UMAP(**umap_args)
    #         umap_args["verbose"] = True
            ump.fit(tdec.compass.docvecs.vectors_docs)
            save(ump, umap_path)
            logging.info("UMAP made")
        else:
            ump = load(umap_path)
            logging.info("UMAP loaded")
        del tdec
        hdbscan_path = os.path.join(self.file_path, "hdbscan.pkl")
        if (overwrite and os.path.exists(hdbscan_path)) or not os.path.exists(hdbscan_path):
            if self.hdbscan_args:
                hdbscan_args = self.hdbscan_args
            else:
                hdbscan_args = {'min_cluster_size': 15,
                                'metric': 'euclidean',
                                'cluster_selection_method': 'leaf',
                               'prediction_data': True}
            hdb = HDBSCAN(**hdbscan_args)
            hdb.fit(ump.embedding_)
            save(hdb, hdbscan_path)
            logging.info("HDBSCAN made")
        else:
            hdb = load(hdbscan_path)
            logging.info("HDBSCAN loaded")
        del ump
        global_topic_path = os.path.join(self.file_path, "global_topics.pkl")
        if (overwrite and os.path.exists(global_topic_path)) or not os.path.exists(global_topic_path):
            global_topics = hdb.labels_
            save(global_topics, global_topic_path)
            logging.info("Topics made")
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        if (overwrite and os.path.exists(topic_description_path)) or not os.path.exists(topic_description_path):
            tdec = load(tdec_path)
            description = self._topic_summary(tdec.compass,  hdb.labels_)
            save(description, topic_description_path)
            logging.info("Topic summary made")
            del tdec
            del description
        del hdb
        
    def train_slice(self, name):
        name = str(name)
        file_path = os.path.join(self.file_path, "data", str(name)+".txt")
        corpus_file = None
        try:
            if not os.path.isfile(file_path):
                raise Exception('Please have the text as "model_file_path/data/{name}.txt"')
            tdec = load(os.path.join(self.file_path, "tdec.pkl"))
            if os.path.isfile(os.path.join(self.file_path, "data", name+"_titles.txt")):
                title_path = os.path.join(self.file_path, "data", name+"_titles.txt")
                titles = []
                with open(title_path, 'r') as f:
                    for line in f:
                        titles.append(line.rstrip())
            elif os.path.isfile(os.path.join(self.file_path, "data", name+"_titles.pkl")):
                title_path = os.path.join(self.file_path, "data", name+"_titles.pkl")
                titles = load(title_path)
            else:
                sentences = None
                titles = None
                corpus_file=file_path
            if titles:
                with open(file_path, 'r') as f:
                    sentences = []
                    for i,line in enumerate(f):
                        sentences.append(TaggedDocument(line.rstrip().split(), [titles[i]]))
            temp_name = os.path.join(self.file_path, "slices", f"{name}.doc2vec")
            model = tdec.train_slice(corpus_file=corpus_file, sentences=sentences)#, out_name=temp_name, fsave=True)
            del tdec
            logging.info(f"{name} model made")
            ump_2d = load(os.path.join(self.file_path, "umap_2d.pkl"))
            reduction_2d = ump_2d.transform(model.docvecs.vectors_docs)
            del ump_2d
            logging.info(f"{name} 2D reduction made")
            ump = load(os.path.join(self.file_path, "umap.pkl"))
            reduction = ump.transform(model.docvecs.vectors_docs)
            del ump
            logging.info(f"{name} reduction made")
            hdb = load(os.path.join(self.file_path, "hdbscan.pkl"))
            topics = hdbscan.approximate_predict(hdb, reduction)
            topics = topics[0]
            del hdb
            logging.info(f"{name} topic numbers made")
            topic_summary = self._topic_summary(model, topics)
            logging.info(f"{name} topic summary made")
            ts = TimeSlice(model)
            ts.reduction = reduction
            ts.umap_2d = reduction_2d
            ts.topics = topics
            ts.topic_summary = topic_summary
            save(ts, os.path.join(self.file_path, "slices", f"{name}.pkl"))
            logging.info(f"{name} TimeSlice saved")
            self.slice_names.append(name)
            self.slice_names = list(set(self.slice_names))
            self.slice_names.sort()
            del model
            del reduction_2d
            del topics
            del topic_summary
            del ts
        except Exception:
            logging.error(traceback.format_exc())
    def train_slices(self, names, parallel = False, verbose = False):
        """
        Trains time slices. Parallel option uses a ProcessPoolExecutor,
        but does not work as intended. A ProcessPoolExecutor created outside
        this class will work.
        """
        executor = ProcessPoolExecutor(max_workers=self.workers)
        for name in names:
            executor.submit(self.train_slice, str(name))
            logging.info(f"Slice {name} submitted")
    def _topic_summary(self, model, topics):
        topic_summary = {}
        topics_in_slice = list(set(topics))
        for topic in topics_in_slice:
            vecs_in_topic = model.docvecs.vectors_docs[topics == topic, :]
            topic_summary[topic] = self._summary_one_topic(model, vecs_in_topic)
        return topic_summary
    def _summary_one_topic(self, model, vecs_in_topic):
        topic_vector = vecs_in_topic.mean(axis=0)
        topic_top_words = []
        if self.similarity_method == "centroid":
            topic_top_words = [term for (term, _) in model.wv.most_similar([topic_vector], topn=self.n_terms)]
        elif self.similarity_method == "vote":
            top_n_per_article = []
            for vec_index in range(vecs_in_topic.shape[0]):
                vec = vecs_in_topic[vec_index]
                arts = [term for (term, _) in model.wv.most_similar([vec], topn=self.n_terms)]
                top_n_per_article += arts
            counter = Counter(top_n_per_article)
            topic_top_words = [term for (term, _) in counter.most_common(self.n_terms)]
        else:
            raise Exception("Please select a valid similarity_method")
        return {"topvec": topic_vector, "topn": topic_top_words}

    def reduce_topics(self, n_topics, remake_slices=True):
        """
        Reduces the number of topics. Only works for nested option.
        """
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        topic_description = load(topic_description_path)
        topics_path = os.path.join(self.file_path, "global_topics.pkl")
        global_topics = load(topics_path)
        if n_topics + 2 >= len(topic_description):
            return
        tdec_path = os.path.join(self.file_path, "tdec.pkl")
        tdec = load(tdec_path)
        self.changes = {}
        for t in topic_description:
            self.changes[t] = t
        counter = Counter(global_topics)
        while n_topics + 2 <= len(topic_description):
            f = min(counter, key=counter.get)
            top_matrix = np.array([topic_description[t]["topvec"] for t in topic_description if t not in [-1, f]])
            
            distances = distance.cdist(topic_description[f]["topvec"].reshape(1,-1), top_matrix, "cosine")[0]
            closest_idx = np.argmin(distances)
            t = list(k for k in topic_description if k not in [-1, f])[closest_idx]
            global_topics[global_topics == f] = t
            self.changes[f] = t
            for k, v in self.changes.items():
                if v == f:
                    self.changes[k] = t
#             print(f"Topics left: {len(topic_description)}")
#             print(f"Merging {f} into {t}")
#             print(topic_description[f]["topn"], topic_description[t]["topn"])
            counter[t] += counter[f]
            counter.pop(f)
            topic_description[t] = self._summary_one_topic(tdec.compass,  tdec.compass.docvecs.vectors_docs[global_topics == t])
            topic_description.pop(f)
        save(tdec, tdec_path)
        save(topic_description, topic_description_path)
        save(global_topics, topics_path)
        del tdec
        del topic_description
        del global_topics
        for i in self.obtain_slices():
            slice_path = os.path.join(self.file_path, "slices", f"{i}.pkl")
            ts = load(slice_path)
            ts.topics = [self.changes[t] for t in ts.topics]
            for t in np.unique(ts.topics):
                ts.topic_summary[t] = self._summary_one_topic(ts.model, ts.model.docvecs.vectors_docs[ts.topics == t])
            save(ts, slice_path)
            del ts
    def name(self, i):
        """
        Names the time slice based on whether it is yearly or something else.
        """
        time = self.time_intervals[i]
        if self.yearly:
            return time.year
        else:
            return time.strftime("%Y/%m/%d")
    
    def obtain_term(self, terms):
        if type(terms) == str:
            terms = [terms]
        terms_path = os.path.join(self.file_path, "terms.csv")
        df = pd.read_csv(terms_path)
        for term in terms:
            df = df[df['term'] != term]
        term_terms = []
        term_slices = []
        term_vectors = []
        for slic in self.obtain_slices():
            slice_path = os.path.join(self.file_path, "slices", f"{slic}.pkl")
            ts = load(slice_path)
            for term in terms:
                if term in ts.model.wv.index2word:
                    term_slices.append(slic)
                    term_vectors.append(ts.model.wv[term])
                    term_terms.append(term)
            del ts
        ump_2d = load(os.path.join(self.file_path, "umap_2d.pkl"))
#         print(term_slices, term_vectors)
        red = ump_2d.transform(term_vectors)
        del ump_2d
        addition = pd.DataFrame({'term': term_terms, 'ts': term_slices,
                                 'x': red[:, 0], 'y': red[:, 1]})
        df = pd.concat([df, addition], ignore_index=True)
        df['label'] = df['term'] + ' - ' + df['ts'].astype(str)
        df.to_csv(terms_path, index=False)
        del df
    def plot_term(self, term = None, n_terms = None, years = None, plot="near", n_closest=10, similarity_threshold=0.4):
        """
        Plots the compass/time slice and the selected terms
        :param term: string or list of string terms
        :param n_terms: Number of descriptor terms per topic
        :param years: Pick specific time periods
        :param plot: either "full" (plot everything), "near" (nearest neighbors of terms), or "none" (plot just the terms)
        :param n_closest: if plot is "near" determine how many nearest articles to use
        :param similarity_threshold: "near" parameter to determine the smallest cosine similarity a "closest article" can be to a term
        """
        
        if not term:
            raise Exception("Please say what term(s) you want")
        elif type(term) == str:
            term = [term]
        if not years:
            years = self.obtain_slices()
        elif type(years) == int:
            years = [years]
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        terms_path = os.path.join(self.file_path, "terms.csv")
        df = pd.read_csv(terms_path)
        for t in term:
            if df["term"].str.contains(t).sum() == 0:
                raise Exception(f"The term {t} has not been cached using obtain_terms")
        df = df[df['ts'].isin(years) & df["term"].isin(term)].reset_index()
        fig = make_subplots(rows=1, cols=1)
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        if plot == "full":
            global_topic_summary = load(topic_description_path)
            global_topics = load(os.path.join(self.file_path, "global_topics.pkl"))
            ump_2d = load(os.path.join(self.file_path, "umap_2d.pkl"))
            for tn in np.unique(global_topics):
#                 if tn == -1:
#                     continue
                sub = ump_2d.embedding_[[i for i,topic in enumerate(hdb.labels_) if topic == tn],:]
                if n_terms > 0:
                    nam = f"{tn} - {','.join(global_topic_summary[tn]['topn'][0:n_terms])} "
                else:
                    nam = str(tn)
                fig.add_trace(go.Scatter(x=sub[:,0], y=sub[:,1], mode="markers", name=nam))
            del global_topics
            del ump_2d
        elif plot == "subset":
            doc_titles = []
            doc_coordinates = []
            doc_tops = []
            for year in years:
                slic = self.load_slice(year)
                sub = df[df['ts'] == year].reset_index()
                for t in sub['term']:
                    if t not in term:
                        continue
                    closest_titles = [i for i, sim in slic.model.docvecs.most_similar([slic.model.wv[t]], topn=n_closest) if sim > similarity_threshold]
                    # Need to check if len(slic.model.docvecs.index2entity) == 0
                    if len(slic.model.docvecs.index2entity) == 0:
                        closest_idx = closest_titles
                    else:
                        closest_idx = [slic.model.docvecs.index2entity.index(title) for title in closest_titles]
                    doc_titles += [f"{title} - {year}" for title in slic.model.docvecs.index2entity]
                    doc_coordinates.append(slic.umap_2d)
                    doc_tops.append(slic.topics)
            doc_coordinates = np.concatenate(doc_coordinates)
            doc_tops = np.concatenate(doc_tops)
            if n_terms > 0 and len(years) > 1:
                topic_summary = load(topic_description_path)
            elif n_terms > 0 and len(years) == 1:
                topic_summary = self.load_slice(years[0])
                topic_summary = topic_summary.topic_summary
            for tn in np.unique(doc_tops):
                # if tn == -1:
                #     continue
                sub_idx = [i for i, t in enumerate(doc_tops) if t == tn]
                if tn == -1:
                    nam = "noise"
                elif n_terms > 0:
                    nam = f"{tn} - {','.join(topic_summary[tn]['topn'][0:n_terms])} "
                else:
                    nam = str(tn)
                fig.add_trace(go.Scatter(x=doc_coordinates[sub_idx, 0], y=doc_coordinates[sub_idx, 1], mode="markers", text=[title for i, title in enumerate(doc_titles) if i in sub_idx], name=nam, marker=dict(color=px.colors.qualitative.Plotly[(tn + 1) % 10])))
        elif plot == "near":
            doc_titles = []
            doc_coordinates = []
            doc_tops = []
            for year in years:
                slic = self.load_slice(year)
                sub = df[df['ts'] == year].reset_index()
                for t in sub['term']:
                    if t not in term:
                        continue
                    closest_titles = [i for i, sim in slic.model.docvecs.most_similar([slic.model.wv[t]], topn=n_closest) if sim > similarity_threshold]
                    # Need to check if len(slic.model.docvecs.index2entity) == 0
                    if len(slic.model.docvecs.index2entity) == 0:
                        closest_idx = closest_titles
                    else:
                        closest_idx = [slic.model.docvecs.index2entity.index(title) for title in closest_titles]
                    closest_titles = [f"{title} - {year}" for title in closest_titles]
                    doc_titles += closest_titles
                    doc_coordinates.append(slic.umap_2d[closest_idx, :])
                    doc_tops.append(slic.topics[closest_idx])
            doc_coordinates = np.concatenate(doc_coordinates)
            doc_tops = np.concatenate(doc_tops)
            if n_terms > 0:
                global_topic_summary = load(topic_description_path)
            for tn in np.unique(doc_tops):
                # if tn == -1:
                #     continue
                sub_idx = [i for i, t in enumerate(doc_tops) if t == tn]
                if tn == -1:
                    nam = "noise"
                elif n_terms > 0:
                    nam = f"{tn} - {','.join(global_topic_summary[tn]['topn'][0:n_terms])} "
                else:
                    nam = str(tn)
                fig.add_trace(go.Scatter(x=doc_coordinates[sub_idx, 0], y=doc_coordinates[sub_idx, 1], mode="markers", text=[title for i, title in enumerate(doc_titles) if i in sub_idx], name=nam, marker=dict(color=px.colors.qualitative.Plotly[(tn + 1) % 10])))

        elif plot == "none":
            pass
        else:
            raise Exception("Pick 'full' 'near' or 'none' for plot")
        if len(years) == 1:
            fig.add_trace(go.Scatter(x=df["x"], y=df["y"], text=df["term"], mode="markers+text", marker=dict(color="black"), name="terms"))
        else:
            for t in term:
                sub = df[df['term'] == t].reset_index()
                fig.add_trace(go.Scatter(x=sub["x"], y=sub["y"], text=sub["label"], mode="markers+text+lines", line=dict(color="black"), textposition=improve_text_position(sub.index), name=t))
        return fig
    def plot_compass(self, n_terms = None, alpha=0.5):
        """
        Plots what the compass using plotly express.
        """
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        ump_2d = load(os.path.join(self.file_path, "umap_2d.pkl"))
        comp_df = pd.DataFrame({'x': ump_2d.embedding_[:,0],
                              'y': ump_2d.embedding_[:,1],
                               })
        del ump_2d
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        global_topic_summary = load(topic_description_path)
        global_topics = load(os.path.join(self.file_path, "global_topics.pkl"))
        
        comp_df['topic'] = [str(global_topics[i]) + ': ' + " ".join(global_topic_summary[global_topics[i]]["topn"][0:n_terms]) for i in range(global_topics.shape[0])]
        color_dict = {}
        for i in list(set(global_topics)):
            color_dict[str(i) + ': ' + " ".join(global_topic_summary[i]["topn"][0:n_terms])] = px.colors.qualitative.Plotly[(i + 1) % 10]
        del global_topic_summary
        fig = px.scatter(comp_df, x='x', y='y', color='topic', opacity=alpha, title="Compass", color_discrete_map=color_dict)
        return fig
    def plot_slice(self, slice_num, n_terms = None):
        """
        Plots a single time slice using plotly express.
        """
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        
        wrap_slice = self.load_slice(slice_num)
        list_of_names = [" ".join([f"{i}:"]+wrap_slice.topic_summary[i]["topn"][0:n_terms]) for i in wrap_slice.topics]
        wrap_slice_df = pd.DataFrame({'x': wrap_slice.umap_2d[:,0],
                                      'y': wrap_slice.umap_2d[:,1],
                                      'name': list_of_names
                                     })
        color_dict = {}
        for i in list(set(wrap_slice.topics)):
            color_dict[" ".join([f"{i}:"]+wrap_slice.topic_summary[i]["topn"][0:n_terms])] = px.colors.qualitative.Plotly[(i + 1) % 10]
        fig = px.scatter(wrap_slice_df, x='x', y='y', color='name', title=slice_num, color_discrete_map=color_dict)
        return fig
    def plot_slider_scatter(self, n_terms=5):
        """
        Plots all time slices and compass with a slider
        """
        if n_terms:
            n_terms = min(n_terms, self.n_terms)
        fig = go.Figure()
        ump_2d = load(os.path.join(self.file_path, "umap_2d.pkl"))
        wrap_df = pd.DataFrame({'x': ump_2d.embedding_[:,0],
                              'y': ump_2d.embedding_[:,1],
                               })
        del ump_2d
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        global_topic_summary = load(topic_description_path)
        global_topics = load(os.path.join(self.file_path, "global_topics.pkl"))
        
        wrap_df['name'] = [str(hdb.labels_[i]) + ': ' + " ".join(global_topic_summary[global_topics[i]]["topn"][0:n_terms]) for i in range(hdb.labels_.shape[0])]
        wrap_df["topic"] = list(map(str, global_topics.tolist()))#[" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrapper.compass.global_topic_description.tolist()],
        del global_topics
        del global_topic_summary
        wrap_df['date'] = 'compass'
        fig = px.scatter()
        for step in self.obtain_slices():
            wrap_slice = self.load_slice(step)
            wrap_slice_df = pd.DataFrame({'x': wrap_slice.umap_2d[:,0],
                                          'y': wrap_slice.umap_2d[:,1],
                                          'topic': map(str,wrap_slice.topics.tolist()),#[" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrap_slice.topics.tolist()],
                                          'date': step,
                                          'name': [" ".join([f"{i}:"] + wrap_slice.topic_summary[i]["topn"]) for i in wrap_slice.topics.tolist()]
                                         })
            wrap_df = pd.concat([wrap_df,wrap_slice_df])
        fig = px.scatter(wrap_df, x="x", y="y", color="topic", animation_frame="date", title="Scatterplots of Compass and Time Slices", hover_data=["name"])
        fig["layout"].pop("updatemenus") # optional, drop animation buttons
        return fig
    def plot_line(self, include_noise = False, n_terms=5, drop_unique=True):
        """
        Plots a line graph that shows the amount of articles in each topic
        per time period
        """
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        global_topic_description = load(topic_description_path)
        df_list = []
        for slic in self.obtain_slices():
            slice_path = os.path.join(self.file_path, "slices", f"{slic}.pkl")
            ts = load(slice_path)
            topic_counts = np.unique(ts.topics, return_counts=True)
            for topic in topic_counts[0]:
                if topic == -1 and not include_noise:
                    continue
                words = ts.topic_summary[topic]["topn"][:n_terms]
                count = topic_counts[1][topic_counts[0] == topic][0]
                df_list.append([slic, f"{topic} - {' '.join(global_topic_description[topic]['topn'][:n_terms])}", count, words])
        for date, slic in self.slices.items():
            topic_counts = np.unique(slic.topics, return_counts=True)
        df = pd.DataFrame(df_list, columns=["Date", "Cluster", "Counts", "Words"])
        fig = px.line(df, x="Date", y="Counts", color="Cluster", hover_data="Words")
        return fig
    def plot_line_terms(self, terms, include_noise = False, n_terms=5, drop_unique=True):
        """
        Plots a line graph that shows the amount of articles in each topic
        per time period
        """
        topic_description_path = os.path.join(self.file_path, "global_topic_description.pkl")
        global_topic_description = load(topic_description_path)
        df_list = []
        for slic in self.obtain_slices():
            slice_path = os.path.join(self.file_path, "slices", f"{slic}.pkl")
            ts = load(slice_path)
            topic_counts = np.unique(ts.topics, return_counts=True)
            for topic in topic_counts[0]:
                topic_words = global_topic_description[topic]['topn']
                if (topic == -1 and not include_noise) or \
                (len(set(terms).intersection(set(topic_words))) == 0):
                    continue
                words = ts.topic_summary[topic]["topn"][:n_terms]
                count = topic_counts[1][topic_counts[0] == topic][0]
                topic_words = list(set(topic_words).intersection(set(terms))) +\
                list(set(topic_words) - set(terms)) # Brings terms of interest up front
                df_list.append([slic, f"{topic} - {' '.join(topic_words[:n_terms])}", count, words])
        for date, slic in self.slices.items():
            topic_counts = np.unique(slic.topics, return_counts=True)
        df = pd.DataFrame(df_list, columns=["Date", "Cluster", "Counts", "Words"])
        if drop_unique:
            df = df[df.groupby('Cluster').Cluster.transform('count')>1].copy()
        fig = px.line(df, x="Date", y="Counts", color="Cluster", hover_data="Words")
        return fig
    def obtain_slices(self):
        files = []
        for file in os.listdir(os.path.join(self.file_path, 'slices/')):
            if file.endswith('.pkl'):
                files.append(int(file[:-4]))
        files.sort()
        return files
    
    def load_slice(self, slice_name):
        if slice_name not in self.obtain_slices():
            raise Exception("Slice name does not exist")
        return load(os.path.join(self.file_path, 'slices/', str(slice_name)+'.pkl'))
    
    def topic_distribution(self):
        """
        Obtains compass topic distribution
        """
        return np.unique(self.compass.hdbscan.labels_, return_counts=True)
    def test_coherence(self, return_annual_count = False):
        """
        Tests coherence of topics.
        return_annual_count - Breaks down by time slice
        """
        annual_count = {}
        for idx, date in enumerate(self.slices):
            slic = self.load_slice(date)
            sentences_subset = self._obtain_sentence_subset(idx)
#         print(sentences_subset[0])
#         sentences_subset = [simple_preprocess(" ".join(doc)) for doc in sentences_subset]
            dic = corpora.Dictionary()
            BoW_corpus = [dic.doc2bow(doc, allow_update=True) for doc in sentences_subset]
            topics = [slic.topic_summary[thing]['topn'] for thing in slic.topic_summary]  
#         texts = [[dictionary.token2id[word] for word in sentence] for sentence in sentences_subset]
            mod = CoherenceModel(topics=topics, texts=sentences_subset, dictionary=dic, topn=10, coherence='c_npmi')
            annual_count[date] = mod.get_coherence()
#         annual_count[date] = np.mean(np.array(local_count))
        if return_annual_count:
            return (annual_count, np.mean(np.array(list(annual_count.values()))))
        else:
            return np.mean(np.array(list(annual_count.values())))

    def test_diversity(self, return_annual_count = False):
        """
        Tests diversity of topics.
        return_annual_count - Breaks down by time slice
        """
        annual_count = {}
        for date in self.slices:
            slic = self.load_slice(date)
            local_count = {}
            for topic_number in slic.topic_summary:
                if topic_number != -1:
                    topic_words = slic.topic_summary[topic_number]["topn"]
                    for word in topic_words:
                        try:
                            local_count[word] += 1
                        except:
                            local_count[word] = 1
            annual_count[date] = local_count
        sums = []
        for date in annual_count.keys():
            slic = annual_count[date]
            sums.append(np.mean(np.array(list(slic.values())) == 1))
        if return_annual_count:
            return (np.mean(sums), sums)
        else:
            return np.mean(sums)
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
    
def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation 
    positions = ['top left', 'top center', 'top right', 'middle right', 'bottom right', 'bottom center', 'bottom left', 'middle left']
    return [positions[i % len(positions)] for i in range(len(x))]

def create_df_json(sentences = None, corpus_file=None, timestamps=None,
                   loops_=5, w2v_dim_=[50, 100, 200, 300], topic_embed_dim_=[5,10],
                   topic_num_=[10,20,30,40,50], descriptor_count_=[5,10,15],
                   yearly=True, n_slices=-1, temp_fname=None
                  ):
    """
    Comprehensive testing function that checks out several different
    combinations of parameters for TTEC.
    """
    out_dic = []
    out_list = []
    topic_num_.sort(reverse=True)
    if temp_fname:
        with open(temp_fname, 'w') as f:
            f.write("w2v_dim,topic_dim,topic_amount,cluster_type,summary_type,summary_size,coherence,diversity\n")
    for _ in range(loops_):
        wrapper_dic = {}
        for w2v_dim in w2v_dim_:
            logging.info(f"Started word2vec embedding dimension {w2v_dim}")
            wrapper = TTEC_wrapper(sentences=sentences, corpus_file=corpus_file, time_stamps=timestamps, size=w2v_dim,n_topics=topic_num_[0],
                                  train_compass_now=False, train_slices_now=False, yearly=yearly,n_slices=n_slices, log=True, log_name="log_un.txt",
                                  hdbscan_selection="nested")
#             wrapper.compass.static_alpha = wrapper.compass.dynamic_alpha / 10
            wrapper.compass.dynamic_alpha = wrapper.compass.static_alpha / 10
            wrapper.compass.train_compass(corpus_file=corpus_file,
                                          sentences=[TaggedDocument(sentence, [i]) for i, sentence in enumerate(sentences)] if sentences else None
                                         )
            wrapper.compass.compass_umap_2d = UMAP(n_neighbors = 15).fit(wrapper.compass.compass.docvecs.vectors_docs)
            wrapper.train_slices() #

            embedding_dic = {}
            for topic_embedding_dim in topic_embed_dim_:
                logging.info(f"Started topic embedding dimension {topic_embedding_dim}")
                wrapper.remake_topic_embeddings(umap_args={'n_neighbors': 15,
                         'n_components': topic_embedding_dim,
                         'metric': 'cosine'})
                topic_num_dic = {}
                
                for topic_num in topic_num_:
                    wrapper.reduce_topics(topic_num)
                    logging.info(f"Started topic amount {topic_num}")
#                     print(wrapper.compass.topic_map)
                    ttec_param_dic = {}
                    wrapper.n_topics = topic_num
                    wrapper.compass.n_topics = topic_num
                    for ttec_param in ["nested"]:
                        logging.info(f"Started TTEC parameter {ttec_param}")
                        
#                         wrapper.remake_topics(remake_slices=True, hdbscan_selection=ttec_param, similarity_method="vote")
                        descriptor_count_dic = {}
                        for descriptor_count in descriptor_count_:
                            selection_type_dic = {}
                            wrapper.compass.n_terms = descriptor_count
                            wrapper.n_terms = descriptor_count
                            wrapper.compass.global_topic_summary = wrapper.compass._create_topic_summary(wrapper.compass.compass, wrapper.compass.global_topics, similarity_method = "vote")
                            for slic in wrapper.slices:
                                wrapper.slices[slic].new_topics(wrapper.compass, similarity_method="vote", n_terms=descriptor_count)
                            selection_type_dic["vote"] = {}
                            for slic in wrapper.slices:
                                selection_type_dic["vote"][slic] = [wrapper.slices[slic].topic_summary[i]["topn"] for i in wrapper.slices[slic].topic_summary if i != -1]
                            
                            coh = wrapper.test_coherence()
                            div = wrapper.test_diversity()
                            out_list.append([w2v_dim, topic_embedding_dim, topic_num,
                                             ttec_param, "vote", descriptor_count,
                                             coh, div])
                            tops = [["compass", list(wrapper.compass.global_topic_summary.keys())]]
                            for i in wrapper.slices:
                                tops.append([i, list(wrapper.slices[i].topic_summary.keys())])
                            logging.info(f"Slice topics: {tops}")
                            if temp_fname:
                                with open(temp_fname, 'a') as f:
                                    f.write(f"{w2v_dim},{topic_embedding_dim},{topic_num},{ttec_param},vote,{descriptor_count},{coh},{div}\n")

                            for topic in wrapper.compass.global_topic_summary:
                                wrapper.compass.global_topic_summary[topic]["topn"] = [term for term,_ in wrapper.compass.compass.wv.most_similar([wrapper.compass.global_topic_summary[topic]["topvec"]], topn=wrapper.compass.n_terms)]
                            for year in wrapper.slices:
                                for topic in wrapper.slices[year].topic_summary:
                                    wrapper.slices[year].topic_summary[topic]["topn"] = [term for term,_ in wrapper.slices[year].model.wv.most_similar([wrapper.slices[year].topic_summary[topic]["topvec"]], topn=wrapper.compass.n_terms)]
                            coh = wrapper.test_coherence()
                            div = wrapper.test_diversity()
                            out_list.append([w2v_dim, topic_embedding_dim, topic_num,
                                             ttec_param, "centroid", descriptor_count,
                                             coh, div])
                            if temp_fname:
                                with open(temp_fname, 'a') as f:
                                    f.write(f"{w2v_dim},{topic_embedding_dim},{topic_num},{ttec_param},centroid,{descriptor_count},{coh},{div}\n")

                            selection_type_dic["centroid"] = {}
                            for slic in wrapper.slices:
                                selection_type_dic["centroid"][slic] = [wrapper.slices[slic].topic_summary[i]["topn"] for i in wrapper.slices[slic].topic_summary if i != -1]
                            descriptor_count_dic[descriptor_count] = selection_type_dic
                        ttec_param_dic[ttec_param] = descriptor_count_dic
                    topic_num_dic[topic_num] = ttec_param_dic
                embedding_dic[topic_embedding_dim] = topic_num_dic
            wrapper_dic[w2v_dim] = embedding_dic
        out_dic.append(wrapper_dic)
    return pd.DataFrame(out_list, columns=["w2v_dim", "topic_dim", "topic_amount",
                                          "cluster_type", "summary_type", "summary_size",
                                          "coherence", "diversity"]), out_dic


def gen_coh(wrapper):
    """
    Coherence function that exists outside the wrapper. Deprecated
    return_annual_count - Breaks down by time slice
    """
    out = []
    for idx, date in enumerate(wrapper.slices):
        slic = wrapper.slices[date]
        sentences_subset = wrapper._obtain_sentence_subset(idx)
#         print(sentences_subset[0])
#         sentences_subset = [simple_preprocess(" ".join(doc)) for doc in sentences_subset]
        dic = corpora.Dictionary()
        BoW_corpus = [dic.doc2bow(doc, allow_update=True) for doc in sentences_subset]
        topics = [slic.topic_summary[thing]['topn'] for thing in slic.topic_summary]  
#         texts = [[dictionary.token2id[word] for word in sentence] for sentence in sentences_subset]
        mod = CoherenceModel(topics=topics, texts=sentences_subset, dictionary=dic, topn=10, coherence='c_npmi')
        out.append(mod.get_coherence())
#         annual_count[date] = np.mean(np.array(local_count))
    return np.mean(np.array(out)), out

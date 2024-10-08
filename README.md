# Visualizing Temporal Topic Embeddings with a Compass

This package is a rewritten fork of the original [CADE](https://github.com/vinid/cade) package, which generates Temporal Word Embeddings with a Compass (TWEC).
It expands on the [original paper](https://ojs.aaai.org/index.php/AAAI/article/view/4594) by creating the concept of **Temporal Document Embeddings with a Compass (TDEC)** and then generalizes to a topic space through **Temporal Topic Embeddings with a Compass (TTEC)**.
It takes advantage of CADE's idea of creating a general, atemporal model (either word2vec or doc2vec) and freezing the hidden weights of that model (the "target embedding") to then be used to train individual time slices.
This results in alignment across time slices because similar outputs would require similar input word/document vectors.
As a result, **words, documents, and topics** can be compared across time slices using this alignment method.

# Installation

As with the original implementation of CADE, this package also takes advantage of a [custom implementation of gensim](https://github.com/danilka4/gensim.git).
The [original custom implementation](https://github.com/vinid/gensim.git) added the ability to freeze the word vector hidden layer to Word2vec.
This additional edit extends the freezing ability to Doc2vec:

```
pip install -U temporal-embeddings-compass
pip install git+https://github.com/danilka4/gensim.git
```

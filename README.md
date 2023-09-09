# Temporal Embeddings with a Compass

This package is a rewritten fork of the original [CADE](https://github.com/vinid/cade) package, which generates Temporal Word Embeddings with a Compass (TWEC).
It expands on the [original paper](https://ojs.aaai.org/index.php/AAAI/article/view/4594) by creating the concept of **Temporal Document Embeddings with a Compass (TDEC)**.
It takes advantage of CADE's idea of creating a general, atemporal model (either word2vec or doc2vec) and freezing the hidden weights of that model (the "target embedding") to then be used to train individual time slices.
This results in alignment across time slices because similar outputs would require similar input word/document vectors.
As a result, **words and documents** can be compared across time slices.

There is currently a working representation of TDEC, I will get around to adding an implementation of TWEC.

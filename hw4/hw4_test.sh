#!/bin/bash
wget 'https://www.dropbox.com/s/mpb0pvajtqbqdg2/word2vec256.model?dl=1'
wget 'https://www.dropbox.com/s/pjnc4m3bxogr30h/RNN.pt?dl=1'
wget 'https://www.dropbox.com/s/93uzzamzqbt9my7/word2vec256.model.trainables.syn1neg.npy?dl=1'
wget 'https://www.dropbox.com/s/v5fd1scyfxjzod7/word2vec256.model.wv.vectors.npy?dl=1'
python3 test.py $1 $2 $3
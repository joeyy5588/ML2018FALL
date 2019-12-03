#!/bin/bash
git clone https://github.com/epfml/sent2vec.git
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install . --user
cd ..
cd sent2vec
make
pip install pybind11 --user
pip install cython --user
cd src
python3 setup.py build_ext
pip install . --user
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B6VhzidiLvjSOWdGM0tOX1lUNEk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6VhzidiLvjSOWdGM0tOX1lUNEk" -O torontobooks_unigrams.bin && rm -rf /tmp/cookies.txt

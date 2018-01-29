# SNLI Data Preparation
curl -O https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
python2 snli_preprocessing.py --in_path ./snli_1.0 --out_path ./Data/snli_lm

cp ./Data/snli_lm/* /artifacts/
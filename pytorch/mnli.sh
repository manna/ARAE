# MNLI Data Preparation
curl -O https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
python2 mnli_preprocessing.py --in_path ./multinli_1.0 --out_path ./Data/mnli_lm_{}

cp -r ./Data/* /artifacts/
# Environment Setup
# pytorch:
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 

# cmake
apt-get install software-properties-common
echo | add-apt-repository ppa:george-edison55/cmake-3.x
yes | apt-get update
yes | apt-get install cmake

# KenLM
yes | apt-get install libboost-all-dev
curl -O http://kheafield.com/code/kenlm.tar.gz
tar xvzf kenlm.tar.gz
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
cd ../..
yes | pip install https://github.com/kpu/kenlm/archive/master.zip

cd kenlm/lm/builder
bjam # compile with bjam
cd ../../..

# SNLI Data Preparation
curl -O https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
python2 snli_preprocessing.py --in_path ./snli_1.0 --out_path ./Data/snli_lm

# MNLI Data Preparation
# curl -O https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
# unzip multinli_1.0.zip
# python2 mnli_preprocessing.py --in_path ./multinli_1.0 --out_path ./Data/mnli_lm_{}

# Train
# python2 train.py --data_path ./Data/snli_lm --cuda --no_earlystopping
python2 train.py --data_path ./Data/snli_lm --cuda --kenlm_path ./kenlm --min_epochs 6
# python2 train.py --data_path ./Data/mnli_lm_fiction --cuda --no_earlystopping --epochs 25

# Save results
cp ./output/example/* /artifacts/
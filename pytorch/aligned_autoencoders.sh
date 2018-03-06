# Environment Setup
# pytorch:
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
pip install tensorboard_logger

# cmake:
apt-get install software-properties-common
echo | add-apt-repository ppa:george-edison55/cmake-3.x
yes | apt-get update
yes | apt-get install cmake

# KenLM:
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
# curl -O https://nlp.stanford.edu/projects/snli/snli_1.0.zip
# unzip snli_1.0.zip
# python2 snli_preprocessing.py --in_path ./snli_1.0 --out_path ./Data/snli_lm

# MNLI Data Preparation
# curl -O https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
# unzip multinli_1.0.zip
# python2 mnli_preprocessing.py --in_path ./multinli_1.0 --out_path ./Data/mnli_lm_{}

# Train
echo "Running train.py"
#python2 train.py --cuda --no_earlystopping --epochs 25 ae0 --data_path ./Data/snli_lm ae1 --data_path ./Data/snli_lm 

# Feb 21:

# Job jsj2squzoi6tjo (gov,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 25 \
# ae0 --outf gov_ae0 --data_path ./Data/mnli_lm_government \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job jsvtie90gzzol1 (slate,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 25 \
# ae0 --outf slate_ae0 --data_path ./Data/mnli_lm_slate \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job jsst2d6r7d8tw6 (travel,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 25 \
# ae0 --outf travel_ae0 --data_path ./Data/mnli_lm_travel \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job js6i3lvohryvl7 (gov,slate):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 25 \
# ae0 --outf gov_ae0 --data_path ./Data/mnli_lm_government \
# ae1 --outf slate_ae1 --data_path ./Data/mnli_lm_slate

# Job jq7dth2e0hsy1 (snli,fic): 
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 6 --patience 2 --epochs 25 \
# ae0 --outf snli_ae0 --data_path ./Data/snli_lm \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job js0sijwvj6fvy5 (travel,slate):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 6 --patience 2 --epochs 25 \
# ae0 --outf travel_ae0 --data_path ./Data/mnli_lm_travel \
# ae1 --outf slate_ae1 --data_path ./Data/mnli_lm_slate


# Feb 22:

# Job j5sygoxura0h8 (gov,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 50 \
# ae0 --outf gov_ae0 --data_path ./Data/mnli_lm_government \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job jsd9pekf37rvg (slate,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 50 \
# ae0 --outf slate_ae0 --data_path ./Data/mnli_lm_slate \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job jshfzsrz840v0l (travel,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 50 \
# ae0 --outf travel_ae0 --data_path ./Data/mnli_lm_travel \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# Job jxtpv3mzammnj (telephone,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 50 \
# ae0 --outf telephone_ae0 --data_path ./Data/mnli_lm_telephone \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction

# March 2

# Job jsaryh1g0xqj86 (fic,telephone,slate,gov):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 6 --patience 2 --epochs 25 \
# ae0 --outf fiction_ae0 --data_path ./Data/mnli_lm_fiction \
# ae1 --outf telephone_ae1 --data_path ./Data/mnli_lm_telephone \
# ae2 --outf slate_ae2 --data_path ./Data/mnli_lm_slate \
# ae3 --outf government_ae2 --data_path ./Data/mnli_lm_government

# Job jswitdy3ebr7df and j76wd26nb8h7v (telephone,fic):
# python2 train.py --kenlm_path ./kenlm --cuda --min_epochs 1 --patience 2 --epochs 50 \
# ae0 --outf telephone_ae0 --data_path ./Data/mnli_lm_telephone \
# ae1 --outf fiction_ae1 --data_path ./Data/mnli_lm_fiction


# Save results
cp -r ./output/* /artifacts/

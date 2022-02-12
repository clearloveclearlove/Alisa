# Alisa
ALiSa: Acrostic Linguistic Steganography Based on BERT and Gibbs Sampling

# 1.Generate stego texts
python bert-gibbs.py
python bert-only.py

# 2.PPL test
python ppl_test.py

# 3.Steganalysis
cd Steganalysis

python bert.py

## or 

cd Steganalysis

python rnn.py



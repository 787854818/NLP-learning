import os

if __name__ == '__main__':
    os.system("conda activate py38_torch")
    os.system("python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
            --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
            --valid-niter=100 --max-epoch=101 --no-char-decoder")
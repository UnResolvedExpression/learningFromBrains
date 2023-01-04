import os

if __name__ == '__main__':
    os.system("python scripts/train.py --data data/upstream/HCP --architecture GPT --training-style CSM --training-steps 1 --validation-steps 1 --test-steps 1 --per-device-training-batch-size 256 --per-device-validation-batch-size 256 --learning-rate 1e-4 --log-dir results/models/upstream/HCP --log-every-n-steps 1")
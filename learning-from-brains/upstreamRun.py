import os

if __name__ == '__main__':
    os.system("python scripts/train.py --data data/upstream/HCP --n-train-subjects-per-dataset 6 --n-val-subjects-per-dataset 2 --n-test-subjects-per-dataset 5 --architecture GPT --training-style CSM --training-steps 10000 --per-device-training-batch-size 64 --learning-rate 1e-4 --log-dir results/models/upstream/HCP --log-every-n-steps 1000")
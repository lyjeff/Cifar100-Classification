# ART-Tool-Classification-Cifar100

Author: Jeff Lin

---

## Summary

- [ART-Tool-Classification-Cifar100](#art-tool-classification-cifar100)
  - [Summary](#summary)
  - [Introduciotn](#introduciotn)
  - [Quick Run](#quick-run)
  - [Execution parameters](#execution-parameters)
  - [Experimental Results](#experimental-results)

---

## Introduciotn

---

## Quick Run

Run the following command to run default training.

```python
python train.py
```

Please see execution parameters by following command.
```
python train.py -h
```

---

## Execution parameters

See detail by `python train.py -h` command.

- `-h, --help`
  - show this help message and exit
- `--cuda CUDA`
  - set the model to run on which gpu (default: 0)
- `--holdout-p HOLDOUT_P`
  - set hold out CV probability (default: 0.8)
- `--num-workers NUM_WORKERS`
  - set the number of processes to run (default: 8)
- `--batch-size BATCH_SIZE`
  - set the batch size (default: 1)
- `--epochs EPOCHS`
  - set the epochs (default: 1)
- `--model MODEL_NAME`
  - set model name (default: 'VGG19')
  - The acceptable models are 'VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3'
- `--iteration`
  - set to decrease learning rate each iteration (default: False)
- `--train-all`
  - set to update all parameters of model (default: False)
- `--optim OPTIM`
  - set optimizer (default: SGD)
- `--lr LR`
  - set the learning rate (default: 1e-5)
- `--momentum MOMENTUM`
  - set momentum of SGD (default: 0.9)
- `--scheduler`
  - training with step or multi step scheduler (default: False)
- `--gamma GAMMA`
  - set decreate factor (default: 0.99985)
- `--threshold THRESHOLD`
  - the number thresholds the output answer
  - Float number >= 0 and <=1 (default: 0.99)
- `--output-path OUTPUT_PATH`
  - output file (csv, txt, pth) path (default: ./output)
- `--train-path TRAIN_PATH`
  - training dataset path (default: ./data/train/)
- `--test-path TEST_PATH`
  - evaluating dataset path (default: ./data/test1/)
- `--submit-csv SUBMIT_CSV`
  - submission CSV file (default: ./data/sample_submission.csv)
- `--kaggle Kaggle_Submission_Message`
  - the submission message to upload Kaggle.

---

## Experimental Results

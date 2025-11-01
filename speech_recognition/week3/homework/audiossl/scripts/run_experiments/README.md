# Experiment Running Scripts

Bash scripts to run all experiments for the SSL Audio Homework.

## Quick Start

### Step 1: Test Everything Works (REQUIRED)

Before running full experiments, test that everything works:

```bash
cd /media/vova/BC7CA8E37CA899A2/edu/audiossl/scripts/run_experiments
bash 00_run_all_tests.sh
```

This runs quick tests (5 epochs each) for all tasks. Should complete in ~10-15 minutes.

### Step 2: Run Full Experiments

After tests pass, run full experiments:

```bash
bash 07_run_all_experiments.sh
```

This will run ALL experiments (estimated 3-4 hours).

## Individual Tasks

You can also run tasks individually:

### Task 1: Supervised Learning
```bash
bash 02_train_supervised.sh
```
Trains 3 models:
- 1D encoder (waveform)
- 2D encoder (spectrogram)
- Both encoders

### Task 2: Contrastive Learning (InfoNCE)
```bash
bash 04_train_contrastive.sh
```
Trains multi-format contrastive model.

### Task 3: Non-Contrastive Learning (SimSiam)
```bash
bash 06_train_simsiam.sh
```
Trains SimSiam model.

## File Structure

```
run_experiments/
├── 00_run_all_tests.sh           # Test all scripts (5 epochs)
├── 01_test_supervised.sh         # Test Task 1
├── 02_train_supervised.sh        # Full Task 1 (40 epochs)
├── 03_test_contrastive.sh        # Test Task 2
├── 04_train_contrastive.sh       # Full Task 2 (50 epochs)
├── 05_test_simsiam.sh            # Test Task 3
├── 06_train_simsiam.sh           # Full Task 3 (50 epochs)
├── 07_run_all_experiments.sh     # Run all tasks
└── README.md                     # This file
```

## Training Parameters

### Task 1: Supervised
- Batch size: 64
- Learning rate: 3e-4
- Epochs: 40
- Early stopping: 12 epochs
- Optimizer: Adam with weight decay

### Task 2: Contrastive (InfoNCE)
- Batch size: 128 (larger for better negatives)
- Learning rate: 3e-4
- Epochs: 50
- Early stopping: 15 epochs
- Temperature: 0.07

### Task 3: Non-Contrastive (SimSiam)
- Batch size: 128
- Learning rate: 3e-4
- Epochs: 50
- Early stopping: 15 epochs
- Stop-gradient: Enabled

## Results Location

All results are saved to:
```
/media/vova/BC7CA8E37CA899A2/edu/audiossl/experiments/
├── task1_supervised/
│   ├── supervised_1d/
│   ├── supervised_2d/
│   └── supervised_both/
├── task2_contrastive/
│   └── contrastive_infonce/
└── task3_ncl/
    └── simsiam/
```

Each experiment folder contains:
- `best_model.pt` - Best model checkpoint
- `training_curves.png` - Loss and accuracy plots
- `config.json` - Hyperparameters
- `history.json` - Training history
- Checkpoints every 10 epochs

## Troubleshooting

### Data not found error
Make sure AudioMNIST data is at:
```
/media/vova/BC7CA8E37CA899A2/edu/audiossl/data/AudioMNIST/data
```

### Out of memory
Reduce batch size in the scripts:
```bash
# Edit the .sh file and change:
--batch-size 64  # to smaller value like 32
```

### Scripts not executable
Make them executable:
```bash
chmod +x *.sh
```

## Tips

1. **Always run tests first** - Use `00_run_all_tests.sh` to catch issues early
2. **Monitor GPU usage** - Use `nvidia-smi` or `watch -n 1 nvidia-smi`
3. **Check results incrementally** - Don't wait for all experiments to finish
4. **WandB disabled by default** - Remove `--no-wandb` to enable logging
5. **Reasonable epoch counts** - Scripts use 40-50 epochs with early stopping to prevent overfitting

## Next Steps

After experiments complete:
1. Compare results across all tasks
2. Create comparison plots and tables
3. Analyze which method performs best
4. Fill in homework notebook with findings

# Model integration and operation

The performance page previously ignored all model artifacts and generated random
charts and images. It now reads outputs/evaluation_results.csv and
outputs/training_history.csv and explicitly distinguishes historical records
from results matched to the installed checkpoint. The overview also no longer
claims that every inference is a placeholder.

The workspace contains no model checkpoint or dataset images. The committed log
contains 3 epochs, and the evaluation CSV contains 83 records referencing a
Windows MVTec bottle dataset. These are evidence of recorded prior runs, but
training and results cannot be independently reproduced without their weights
and images. Models and data are excluded by .gitignore.

The existing architecture is a 7,689,539-parameter convolutional VAE with four
encoder/decoder stages, 128x128 RGB input and a 128-dimensional latent vector.
Training uses normal images only, MSE plus beta-weighted KL divergence, Adam,
and best training-loss checkpoint selection. Inference uses the latent mean.
There is no validation-loss log. The current environment has CPU PyTorch and
no available CUDA device. CPU training is supported; batch size 8 is a modest
starting point, with runtime and memory depending on image count and hardware.
No full training has been launched during this repair.

The legacy CSV yields Accuracy 0.2410, Precision/Recall/F1 0, AUROC 0.4087,
TN=20, FP=0, FN=63, TP=0 at threshold 0.5. Precision is undefined with no
positive predictions; the repository reports zero by convention. These are
recomputed historical metrics, not a fresh evaluation. The logged reconstruction
losses are training MSE, not test MSE. No historical test SSIM is available.

## Windows CMD commands

Run from the repository directory. Replace the dataset path with the actual
MVTec category root. Expected directories: train\good and test\good plus
one or more test\defect-category directories. Keep training, calibration and
final test sets disjoint. A bottle-trained model should be evaluated on bottles
with comparable framing and lighting.

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
```

Only if no usable trained weights are available and training is approved:

```bat
python -m src.train --dataset-root "C:\datasets\mvtec_anomaly_detection\bottle" --epochs 50 --batch-size 8 --device cpu
```

Training writes models\vae_bottle.pth and outputs\training_history.csv, replacing
existing outputs. Back up prior artifacts before a new training run. For a CUDA
machine, install the appropriate CUDA PyTorch build and use --device cuda.

With existing weights, evaluate without retraining:

```bat
python -m src.evaluate --dataset-root "C:\datasets\mvtec_anomaly_detection\bottle" --checkpoint "models\vae_bottle.pth" --threshold 0.5 --output-csv "outputs\evaluation_results.csv"
python -m streamlit run app.py
python -m pytest tests -q
```

The 0.5 threshold above reproduces the legacy protocol, not a recommended
validated threshold. Calibrate on held-out known-good data with
`python -m src.calibration --normal-dir "C:\datasets\bottle_calibration\good"`
and pass the resulting threshold explicitly to evaluation. Do not optimize the
threshold on the final test set.

Evaluation now honors the requested checkpoint, saves actual reconstruction
PNGs and per-image MSE/SSIM, and writes checkpoint/scoring provenance alongside
the CSV. MSE and SSIM compare display-sized RGB originals with reconstructions:
SSIM uses an 11x11 Gaussian window, sigma 1.5 and data range 1. These measure
reconstruction similarity, not defect segmentation quality. AUROC requires both
normal and defective labels. The dashboard retains its title, theme, four metric
cards, two chart columns and four sample columns. Missing/invalid files produce
explicit messages, with no synthetic replacements.

Automated integration tests use temporary untrained fixture checkpoints and
controlled images solely to test loading, inference and artifact export. They
are never saved as product models or presented as measured model quality.

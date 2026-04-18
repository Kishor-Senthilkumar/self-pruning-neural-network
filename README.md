# Sparse MLP CIFAR-10

This project trains a sparse multi-layer perceptron on CIFAR-10 using PyTorch. The model uses custom gate layers to learn sparse effective weights, and it saves a distribution plot of gate values after training.

## Files

- `MPL_SPARSITY.py` - Main training script for the sparse neural network.
- `gate_distributions.png` - Output image showing gate-value distributions from training.
- `data/` - CIFAR-10 dataset files (downloaded by the script if missing).
- `.gitignore` - Files and folders that should not be committed to Git.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib

## Setup

1. Create and activate a Python environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install torch torchvision numpy matplotlib
```

## Run

```powershell
python MPL_SPARSITY.py
```

The script will:
- download CIFAR-10 into `./data` if needed
- train the sparse MLP for multiple sparsity regularization strengths
- save `gate_distributions.png`

## GitHub Push Instructions

If you want to push this project to GitHub from this folder, use the following commands.

1. Initialize a git repository in the project folder:

```powershell
git init
```

2. Add files and commit:

```powershell
git add README.md MPL_SPARSITY.py .gitignore
git commit -m "Initial project commit"
```

3. Create a new repository on GitHub, then add the remote. Replace `YOUR_USERNAME` and `YOUR_REPO`:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

4. Push to GitHub:

```powershell
git branch -M main
git push -u origin main
```

## Notes

- The workspace does not currently contain its own `.git` repository, so initializing one here is recommended for this project.
- If you already have a GitHub repository created, use its URL in step 3.

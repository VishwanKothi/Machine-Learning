# Federated Learning Experiments

This repository contains three notebook-based experiments for federated learning on MNIST under non-IID data and concept drift settings. The included zip files are saved result/checkpoint folders for Task 2 and Task 3.

## Project Files

| File | Description |
| --- | --- |
| `Task_1.ipynb` | Baseline federated learning experiment on MNIST with 50 non-IID clients, 100 rounds, 10 clients per round, and greedy/DivFL vs Shapley-style/S-FedAvg client selection. |
| `Task_2.ipynb` | TensorFlow/scikit-learn drift experiment with resumable checkpoints. Runs phase-based label and feature drift using `simple_cnn`, `lenet`, `logreg`, `svm`, and `rf`. |
| `Task_3.ipynb` | PyTorch non-IID federated learning experiment using Random selection vs Sliding-Window Shapley UCB (SW-ShUCB). |
| `federated_learning_task2_checkpoints.zip` | Saved Task 2 result/checkpoint folder. |
| `SW_ShUCB_nonIID_checkpoints.zip` | Saved Task 3 result/checkpoint folder, including results and plots. |
| `Federated_Learning_ppt.pdf` | Presentation slides for the project. |

## Environment

Google Colab is the easiest way to run the notebooks because Task 2 and Task 3 already contain Google Drive checkpoint paths. A local Python environment also works, but Task 2 needs one small path adjustment if you are not using Colab.

Recommended packages:

```bash
pip install notebook jupyterlab numpy pandas matplotlib seaborn scikit-learn scipy joblib tensorflow torch torchvision
```

If you use a GPU locally, install the PyTorch build that matches your CUDA setup.

## Running In Google Colab

1. Upload the notebooks and zip files to Google Drive.
2. Unzip the saved results into these Drive folders:

```text
MyDrive/federated_learning_task2_checkpoints
MyDrive/SW_ShUCB_nonIID_checkpoints
```

3. Open each notebook in Colab.
4. Run cells from top to bottom.
5. When prompted, mount Google Drive so checkpoints and plots can be loaded or saved.

The notebooks download MNIST automatically.

## Running Locally

From this project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install notebook jupyterlab numpy pandas matplotlib seaborn scikit-learn scipy joblib tensorflow torch torchvision
jupyter lab
```

Then open the notebook you want to run.

For saved Task 2 results, unzip:

```powershell
Expand-Archive -Path .\federated_learning_task2_checkpoints.zip -DestinationPath .\federated_learning_task2_checkpoints
```

For saved Task 3 results, unzip:

```powershell
Expand-Archive -Path .\SW_ShUCB_nonIID_checkpoints.zip -DestinationPath .\SW_ShUCB_nonIID_checkpoints
```

If the archive already contains the top-level folder, unzip it into the project root instead.

## Task Notes

### Task 1

Run `Task_1.ipynb` to reproduce the baseline comparison. It uses TensorFlow/Keras for CNN models and scikit-learn for classical models. The main settings are:

- `N_CLIENTS = 50`
- `CLIENTS_PER_ROUND = 10`
- `ROUNDS = 100`
- `LOCAL_EPOCHS = 3`
- models: `simple_cnn`, `logreg`, `svm`, `rf`, `lenet`
- strategies: `greedy`, `shapley`

### Task 2

Run `Task_2.ipynb` for concept drift experiments. It saves and resumes from:

```text
/content/drive/MyDrive/federated_learning_task2_checkpoints
```

For local runs, change `BASE_SAVE_DIR_TASK2` in the notebook to:

```python
BASE_SAVE_DIR_TASK2 = Path("./federated_learning_task2_checkpoints")
```

The configured drift scenarios are:

- `phase4_label_shift`
- `phase4_feature_shift`

Task 2 writes per-experiment histories, model checkpoints, `task2_results.pkl`, `task2_summary.csv`, and `task2_summary.xlsx`.

### Task 3

Run `Task_3.ipynb` for the SW-ShUCB non-IID experiment. It uses PyTorch and torchvision, partitions MNIST across 100 clients, and compares Random selection against SW-ShUCB.

Main settings:

- `NUM_CLIENTS = 100`
- `NUM_ROUNDS = 100`
- `LOCAL_EPOCHS = 3`
- `K_DEFAULT = 10`
- label drift rounds: `[20, 40, 60, 80]`
- feature drift rounds: `[20, 40, 60, 80]`
- SW-ShUCB windows: `[5, 10, 15]`

Task 3 saves outputs under:

```text
SW_ShUCB_nonIID_checkpoints/
  results/
  plots/
```

## Reproducibility

The notebooks set random seeds internally. Results can still vary slightly depending on hardware, TensorFlow/PyTorch versions, and whether the run uses CPU or GPU.

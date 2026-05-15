# Federated Learning Experiments

This repository contains three notebook-based experiments for federated learning on MNIST under non-IID data and concept drift settings. The included zip files are saved result/checkpoint folders for Task 2 and Task 3.

## Project Files

| File | Description |
| --- | --- |
| `Task-1.ipynb` | Baseline federated learning experiment on MNIST with 50 non-IID clients, 100 rounds, 10 clients per round, and greedy/DivFL vs Shapley-style/S-FedAvg client selection. |
| `Task-2.ipynb` | TensorFlow/scikit-learn drift experiment with resumable checkpoints. Runs phase-based label and feature drift using `simple_cnn`, `lenet`, `logreg`, `svm`, and `rf`. |
| `Task-3.ipynb` | PyTorch non-IID federated learning experiment using Random selection vs Sliding-Window Shapley UCB (SW-ShUCB). |
| `federated_learning_task2_checkpoints.zip` | Saved Task 2 result/checkpoint folder. |
| `SW_ShUCB_nonIID_checkpoints.zip` | Saved Task 3 result/checkpoint folder, including results and plots. |
| `Federated_Learning_ppt.pdf` | Presentation slides for the project. |
| `Federated_Learning_report.pdf` | Complete project report containing methodology, experimental setup, architectures, results, and analysis for all tasks. |

## Environment

Google Colab is the easiest way to run the notebooks because Task 2 and Task 3 already contain Google Drive checkpoint paths. A local Python environment also works, but Task 2 needs one small path adjustment if you are not using Colab.

Recommended packages:

```bash
pip install notebook jupyterlab numpy pandas matplotlib seaborn scikit-learn scipy joblib tensorflow torch torchvision
```

If you use a GPU locally, install the PyTorch build that matches your CUDA setup.

## Running In Google Colab

1. Upload the notebooks and zip files to Google Drive (root of `My Drive`).
2. Unzip the saved results inside Colab by running the following cell **before** opening any task notebook:

```python
from google.colab import drive
drive.mount("/content/drive")

import zipfile

# Task 2
with zipfile.ZipFile("/content/drive/MyDrive/federated_learning_task2_checkpoints.zip", "r") as z:
    z.extractall("/content/drive/MyDrive/")
print("Task 2 checkpoints extracted.")

# Task 3
with zipfile.ZipFile("/content/drive/MyDrive/SW_ShUCB_nonIID_checkpoints.zip", "r") as z:
    z.extractall("/content/drive/MyDrive/")
print("Task 3 checkpoints extracted.")
```

After extraction, the following folders must exist in your Drive:

```text
MyDrive/federated_learning_task2_checkpoints/
MyDrive/SW_ShUCB_nonIID_checkpoints/
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

Run `Task-1.ipynb` to reproduce the baseline comparison. It uses TensorFlow/Keras for CNN models and scikit-learn for classical models. The main settings are:

- `N_CLIENTS = 50`
- `CLIENTS_PER_ROUND = 10`
- `ROUNDS = 100`
- `LOCAL_EPOCHS = 3`
- models: `simple_cnn`, `logreg`, `svm`, `rf`, `lenet`
- strategies: `greedy`, `shapley`

### Task 2

Run `Task-2.ipynb` for concept drift experiments. It saves and resumes from:

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

Run `Task-3.ipynb` for the SW-ShUCB non-IID experiment. It uses PyTorch and torchvision, partitions MNIST across 100 clients, and compares Random selection against SW-ShUCB.

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

## References

1. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,  
   *Communication-Efficient Learning of Deep Networks from Decentralized Data*,  
   Proceedings of AISTATS, 2017.

2. R. Balakrishnan, T. Li, T. Zhou, N. Himayat, V. Smith, and J. Bilmes,  
   *Diverse Client Selection for Federated Learning via Submodular Maximization*,  
   Proceedings of ICLR, 2022.

3. L. Nagalapatti and R. Narayanam,  
   *Game of Gradients: Mitigating Irrelevant Clients in Federated Learning*,  
   Proceedings of AAAI, 2021.

4. E. Jothimurugesan, K. Hsieh, J. Wang, G. Joshi, and P. B. Gibbons,  
   *Federated Learning under Distributed Concept Drift*,  
   Proceedings of AISTATS, 2023.

5. S. Guan, Y. Zhou, S. Ding, and S. Ji,  
   *FLASH: Concept Drift Adaptation in Federated Learning*,  
   Proceedings of ICML, 2023.

6. J. Kang, Z. Xiong, D. Niyato, H. Yu, and Y. Zhang,  
   *FedNN: Federated Learning on Concept Drift Data using Weight and Group Normalization*,  
   arXiv preprint arXiv:2409.11973, 2024.

7. W. Chen, L. Wang, H. Zhao, and K. Zheng,  
   *Combinatorial Semi-Bandit in the Non-Stationary Environment*,  
   Proceedings of UAI, 2021.

## Acknowledgements

We sincerely thank **Ms. Shradha Sharma** for her continuous guidance, valuable insights, and support throughout the development of this project. Her suggestions and feedback greatly helped in shaping the experimental design, analysis, and overall presentation of this work.

# A Multimodal Neural Network Model for Non-invasive HER2 Status Assessment in Breast Cancer Patients 

## How to use

Install environment using`pip install -r requirements.txt`

To run, execute `python main-XXX.py`.

## Note

`BP.py`, `SVM.py`, `Transformer.py`, `LSTM.py`: These scripts compare different classifiers.

`main-her2-PUMC.py`: This script performs k-fold cross-validation for HER2 prediction using the PUMC dataset.

`main-her2-PUMC_LF.py`: This script performs k-fold cross-validation for HER2 prediction using both the PUMC and Longfu datasets.

`main-ln-PUMC.py`: This script performs k-fold cross-validation for predicting lymph node categories in HER2-positive cases using the PUMC dataset.

`main-ln-PUMC_LF.py`: This script performs k-fold cross-validation for predicting lymph node categories in HER2-positive cases using both the PUMC and Longfu datasets.

`main-split-her2.py`: This script trains, validates, and tests a model for HER2 category prediction using both the PUMC and Longfu datasets.

`main-ln-split.py`: This script trains, validates, and tests a model for predicting lymph node categories in HER2-positive cases using both the PUMC and Longfu datasets.

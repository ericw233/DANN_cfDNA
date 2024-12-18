
# **DANN-cfDNA**  
A Python implementation of Domain-Adversarial Neural Network (DANN) for correcting batch effects in cfDNA genomic features used for multi-cancer early detection (MCED)

---

## **Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## **Features**
- Loads and preprocesses data for training and validation.
- Identifies the best hyperparameter set by RayTune.
- Fits the DANN model with default hyperparameters or the best hyperparameter set using the training frame and validate it using the validation frame.
- Conducts k-fold cross validations on the training frame.
- Outputs results in .csv format.

---

## **Installation**
### Prerequisites
- Python 3.8 or higher
- Git installed
- Conda installed

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ericw233/DANN_cfDNA.git
   cd DANN_cfDNA

2. Create and activate a conda environment with dependencies
    ```bash
    conda env create -n DANN_cfDNA_env -c conda-forge -f DANN_cfDNA_env.yml
    conda activate DANN_cfDNA_env

## **Usage**
1. Specify the directories and parameters in DANN_run.sh

2. Run the scripts to process data and fit models
    ```bash
    nohup ./DANN_run.sh >log_test.txt 2>&1 &

## **License**
This project is licensed under the MIT License.

## **Contact**
Created by Eric Wu
Email: nanoeric2@gmail.com


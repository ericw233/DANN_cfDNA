import pytest
import torch
import pickle
import tempfile
import sys
import os
from DANN_cfDNA.training.train_and_tune_1D import DANNwithTrainingTuning_1D

def test_dann_initialization():
    config = {
        "out1": 32, "out2": 128, "conv1": 3, "pool1": 2, "drop1": 0.0,
        "conv2": 4, "pool2": 1, "drop2": 0.4, "fc1": 128, "fc2": 32,
        "drop3": 0.2, "batch_size": 128, "num_epochs": 500, "lambda": 0.1
    }
    model = DANNwithTrainingTuning_1D(config=config, input_size=2600, num_class=2, num_domain=2, gamma_r01b=1000.0)
    assert model.batch_size == 128
    assert model.num_epochs == 500


def test_dann_with_feature_example_data(tmp_path):

    # use the temporary directory for output checks

    """
    Tests the DANN model training process using data from the Feature_example folder.
    """
    # Define path to data file
    # The path is relative to the test file's location.
    # Test file is in DANN_cfDNA/tests/
    # Data is in DANN_cfDNA/Feature_example/
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'Feature_example', 'feature_example.pkl')

    # Skip test if data file doesn't exist
    if not os.path.exists(data_dir):
        pytest.skip("Test data file not found: feature_example.pkl")

    test_config = {'out1': 32, 'out2': 128, 'conv1': 3, 'pool1': 2, 'drop1': 0.0, 
            'conv2': 4, 'pool2': 1, 'drop2': 0.4, 'fc1': 128, 'fc2': 32, 'drop3': 0.2, 
            'batch_size': 128, 'num_epochs': 500, 'lambda': 0.1}

    test_model=DANNwithTrainingTuning_1D(config=test_config, input_size=2600,
                                         num_class=2,num_domain=2,
                                         gamma_r01b=1000.0)

    test_model.data_loader(data_dir=data_dir, 
                           input_size=2600,
                           feature_type="Cnv",
                           R01BTuning=False)
    
    test_model.fit(output_path=str(tmp_path),R01BTuning_fit=False)

    tmp_path_raw = tmp_path / "Raw"
    # Check if the model has been trained
    model_file_found = any(f.name.endswith('.pt') for f in tmp_path_raw.iterdir())
    scores_file_found = any(f.name.endswith('.csv') for f in tmp_path_raw.iterdir())

    assert model_file_found, "Model state file (.pt) not found in output path."
    assert scores_file_found, "Scores file (.csv) not found in output path."


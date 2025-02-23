# E-GGAT for resource consumption prediction
- `datapreprocess.py`: perform several preprocessing operations for graph construction, such as format conversion, normalization, and node/edge feature extraction
- `dataset.py`: construct the co-location graph dataset based on preprocessed data
- `layers.py`: contain the implementation of layers described in the manuscript, such as E-GGAT block, E-GAT layer, and GRU layer
- `models.py`: contain the implementation of E-GGAT with different number of E-GGAT blocks for resource consumption prediction
- `train_resource.py`: perform the training process of E-GGAT based on the co-location graph dataset
- `train_resource_without_gpu_features.py`, `train_resource_without_hyperparameter_features.py`, `train_resource_without_model_features.py`: perform the training process of E-GGAT for feature analysis experiments in the manuscript. These 3 trained model are saved in `ablation_trained_models`
- `ablation_experiment.py`: perform the feature analysis experiments in the manuscript
- `inference_experiments.py`: perform the inference process of trained E-GGAT and selected comparison methods. Detailed results can be found in `results/main.py`
- `layer_number_experiments.py`: perform performance evaluation of E-GGAT with different number of E-GGAT blocks. These trained models are saved in `layer_trained_models`

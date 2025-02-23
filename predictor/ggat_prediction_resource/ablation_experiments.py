import torch
import torch.nn as nn
from models import GGAT_with_3_blocks
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from torchmetrics import R2Score
from dataset import InterferenceDataset, InterferenceWithoutGPUFeaturesDataset, InterferenceWithoutModelFeaturesDataset, InterferenceWithoutParameterFeaturesDataset, collate
from datapreprocess import get_max_and_min_in_colocation_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colocation_data_min, colocation_data_max = get_max_and_min_in_colocation_data()

colocation_resource_consumption_min = [colocation_data_min['gpu_memory'],
                                                       colocation_data_min['pcie_band'],
                                                       colocation_data_min['gpu_util']]
colocation_resource_consumption_max = [colocation_data_max['gpu_memory'],
                                                       colocation_data_max['pcie_band'],
                                                       colocation_data_max['gpu_util']]

colocation_resource_consumption_min_tensor = torch.tensor(colocation_resource_consumption_min,
                                                                          dtype=torch.float).reshape(1, -1).to(device)
colocation_resource_consumption_max_tensor = torch.tensor(colocation_resource_consumption_max,
                                                                          dtype=torch.float).reshape(1, -1).to(device)

# 定义 RMSLE LOSS
class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()
        # 初始化 MSELOSS
        self.mse = nn.MSELoss()

    def forward(self, predict, target):
        # 计算 MSELOSS
        mse_loss = self.mse(torch.log1p(predict), torch.log1p(target))
        # 计算 RMSLE LOSS
        rmsle_loss = torch.sqrt(mse_loss)
        return rmsle_loss


class PredictionErr(nn.Module):
    def __init__(self):
        super(PredictionErr, self).__init__()

    def forward(self, predict, target):
        each_error = torch.abs(predict - target) * 100 / target
        error = torch.mean(each_error)

        return error


dataset_dir_path = '/workspace/datasets/graph_datasets'

interference_dataset = InterferenceDataset(dataset_dir_path)
interference_dataset_without_gpu_features = InterferenceWithoutGPUFeaturesDataset(dataset_dir_path)
interference_dataset_without_model_features = InterferenceWithoutModelFeaturesDataset(dataset_dir_path)
interference_dataset_without_hyperparameter_features = InterferenceWithoutParameterFeaturesDataset(dataset_dir_path)

train_subset, val_subset, test_subset = split_dataset(interference_dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True)
train_subset_without_gpu_features, val_subset_without_gpu_features, test_subset_without_gpu_features = split_dataset(interference_dataset_without_gpu_features, frac_list=[0.8, 0.1, 0.1], shuffle=True)
train_subset_without_model_features, val_subset_without_model_features, test_subset_without_model_features = split_dataset(interference_dataset_without_model_features, frac_list=[0.8, 0.1, 0.1], shuffle=True)
train_subset_without_hyperparameter_features, val_subset_without_hyperparameter_features, test_subset_without_hyperparameter_features = split_dataset(interference_dataset_without_hyperparameter_features, frac_list=[0.8, 0.1, 0.1], shuffle=True)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_subset, batch_size=64, shuffle=True, collate_fn=collate)

train_dataloader_without_gpu_features = DataLoader(train_subset_without_gpu_features, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader_without_gpu_features = DataLoader(val_subset_without_gpu_features, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader_without_gpu_features = DataLoader(test_subset_without_gpu_features, batch_size=64, shuffle=True, collate_fn=collate)

train_dataloader_without_model_features = DataLoader(train_subset_without_model_features, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader_without_model_features = DataLoader(val_subset_without_model_features, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader_without_model_features = DataLoader(test_subset_without_model_features, batch_size=64, shuffle=True, collate_fn=collate)

train_dataloader_without_hyperparameter_features = DataLoader(train_subset_without_hyperparameter_features, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader_without_hyperparameter_features = DataLoader(val_subset_without_hyperparameter_features, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader_without_hyperparameter_features = DataLoader(test_subset_without_hyperparameter_features, batch_size=64, shuffle=True, collate_fn=collate)

model = GGAT_with_3_blocks(in_dim=24, hidden_dim=512, out_dim=3, dropout=1, num_heads=3).to(device)
model_without_gpu_features = GGAT_with_3_blocks(in_dim=19, hidden_dim=512, out_dim=3, dropout=1, num_heads=3).to(device)
model_without_model_features = GGAT_with_3_blocks(in_dim=10, hidden_dim=512, out_dim=3, dropout=1, num_heads=3).to(device)
model_without_hyperparameter_features = GGAT_with_3_blocks(in_dim=19, hidden_dim=512, out_dim=3, dropout=1, num_heads=3).to(device)

loss_function = torch.nn.SmoothL1Loss(reduction='mean').to(device)

mse_function = torch.nn.MSELoss(reduction='mean').to(device)
mae_function = torch.nn.L1Loss(reduction='mean').to(device)
rmsle_function = RMSLELoss().to(device)
r2_function = R2Score(num_outputs=3, multioutput='raw_values').to(device)
pred_err_function = PredictionErr().to(device)

print('Loading the best epoch of our prediction')
model.load_state_dict(torch.load('./trained_models/6/24.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor
    denormalized_truth = test_graph.ndata['resource_util_label'] * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    mean_r2_value = torch.mean(r2_value)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += mean_r2_value.detach().item()

    total_pred_err_value += pred_err_value.detach().item()

total_test_loss /= len(test_dataloader)

total_rmse_value /= len(test_dataloader)
total_mae_value /= len(test_dataloader)
total_rmsle_value /= len(test_dataloader)
total_r2_value /= len(test_dataloader)
total_pred_err_value /= len(test_dataloader)

print(f'Test Loss: {total_test_loss:.6f}')

print(f'Mean RMSE Value of Test Dataset: {total_rmse_value:.6f}')
print(f'Mean MAE Value of Test Dataset: {total_mae_value:.6f}')
print(f'Mean RMSLE Value of Test Dataset: {total_rmsle_value:.6f}')
print(f'Mean R-Square Value of Test Dataset: {total_r2_value:.6f}')
print(f'Mean Prediction Error of Test Dataset (%): {total_pred_err_value:.6f}')

print('Loading the best epoch of the prediction model without GPU features')
model_without_gpu_features.load_state_dict(torch.load('./ablation_trained_models/ours_without_gpu_features/26.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader_without_gpu_features:
    model_without_gpu_features.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_without_gpu_features(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds * (
                colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor
    denormalized_truth = test_graph.ndata['resource_util_label'] * (
                colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    mean_r2_value = torch.mean(r2_value)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += mean_r2_value.detach().item()
    total_pred_err_value += pred_err_value.detach().item()

total_test_loss /= len(test_dataloader_without_gpu_features)

total_rmse_value /= len(test_dataloader_without_gpu_features)
total_mae_value /= len(test_dataloader_without_gpu_features)
total_rmsle_value /= len(test_dataloader_without_gpu_features)
total_r2_value /= len(test_dataloader_without_gpu_features)
total_pred_err_value /= len(test_dataloader_without_gpu_features)

print(f'Test Loss: {total_test_loss:.6f}')

print(f'Mean RMSE Value of Test Dataset: {total_rmse_value:.6f}')
print(f'Mean MAE Value of Test Dataset: {total_mae_value:.6f}')
print(f'Mean RMSLE Value of Test Dataset: {total_rmsle_value:.6f}')
print(f'Mean R-Square Value of Test Dataset: {total_r2_value:.6f}')
print(f'Mean Prediction Error of Test Dataset (%): {total_pred_err_value:.6f}')

print('Loading the best epoch of the prediction model without model features')
model_without_model_features.load_state_dict(torch.load('./ablation_trained_models/ours_without_model_features/74.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0


for test_graph in test_dataloader_without_model_features:
    model_without_model_features.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_without_model_features(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor
    denormalized_truth = test_graph.ndata['resource_util_label'] * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    mean_r2_value = torch.mean(r2_value)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += mean_r2_value.detach().item()
    total_pred_err_value += pred_err_value.detach().item()

total_test_loss /= len(test_dataloader_without_model_features)

total_rmse_value /= len(test_dataloader_without_model_features)
total_mae_value /= len(test_dataloader_without_model_features)
total_rmsle_value /= len(test_dataloader_without_model_features)
total_r2_value /= len(test_dataloader_without_model_features)
total_pred_err_value /= len(test_dataloader_without_model_features)

print(f'Test Loss: {total_test_loss:.6f}')

print(f'Mean RMSE Value of Test Dataset: {total_rmse_value:.6f}')
print(f'Mean MAE Value of Test Dataset: {total_mae_value:.6f}')
print(f'Mean RMSLE Value of Test Dataset: {total_rmsle_value:.6f}')
print(f'Mean R-Square Value of Test Dataset: {total_r2_value:.6f}')
print(f'Mean Prediction Error of Test Dataset (%): {total_pred_err_value:.6f}')

print('Loading the best epoch of the prediction model without hyperparameter features')
model_without_hyperparameter_features.load_state_dict(torch.load('./ablation_trained_models/ours_without_hyperparameter_features/33.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0


for test_graph in test_dataloader_without_hyperparameter_features:
    model_without_hyperparameter_features.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_without_hyperparameter_features(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor
    denormalized_truth = test_graph.ndata['resource_util_label'] * (
            colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    mean_r2_value = torch.mean(r2_value)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += mean_r2_value.detach().item()
    total_pred_err_value += pred_err_value.detach().item()

total_test_loss /= len(test_dataloader_without_hyperparameter_features)

total_rmse_value /= len(test_dataloader_without_hyperparameter_features)
total_mae_value /= len(test_dataloader_without_hyperparameter_features)
total_rmsle_value /= len(test_dataloader_without_hyperparameter_features)
total_r2_value /= len(test_dataloader_without_hyperparameter_features)
total_pred_err_value /= len(test_dataloader_without_hyperparameter_features)

print(f'Test Loss: {total_test_loss:.6f}')

print(f'Mean RMSE Value of Test Dataset: {total_rmse_value:.6f}')
print(f'Mean MAE Value of Test Dataset: {total_mae_value:.6f}')
print(f'Mean RMSLE Value of Test Dataset: {total_rmsle_value:.6f}')
print(f'Mean R-Square Value of Test Dataset: {total_r2_value:.6f}')
print(f'Mean Prediction Error of Test Dataset (%): {total_pred_err_value:.6f}')

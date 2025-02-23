import torch
import torch.nn as nn
from models import GGAT_with_1_block, GGAT_with_2_blocks, GGAT_with_3_blocks, GGAT_with_4_blocks, GGAT_with_5_blocks
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from torchmetrics import R2Score
from dataset import InterferenceDataset, collate
from datapreprocess import get_max_and_min_in_colocation_data

colocation_data_min, colocation_data_max = get_max_and_min_in_colocation_data()

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

interference_dataset = InterferenceDataset(dataset_dir_path)

train_subset, val_subset, test_subset = split_dataset(interference_dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_subset, batch_size=64, shuffle=True, collate_fn=collate)

model_1 = GGAT_with_1_block(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)
model_2 = GGAT_with_2_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)
model_3 = GGAT_with_3_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)
model_4 = GGAT_with_4_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)
model_5 = GGAT_with_5_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)

loss_function = torch.nn.SmoothL1Loss(reduction='mean').to(device)

mse_function = torch.nn.MSELoss(reduction='mean').to(device)
mae_function = torch.nn.L1Loss(reduction='mean').to(device)
rmsle_function = RMSLELoss().to(device)
r2_function = R2Score().to(device)
pred_err_function = PredictionErr().to(device)

print('Loading the best epoch of our prediction with 1 E-GGAT block')
model_1.load_state_dict(torch.load('./layer_trained_models/1_E_GGAT/303.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model_1.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_1(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds.reshape(-1) * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']
    denormalized_truth = test_graph.ndata['execution_time_label'] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += r2_value.detach().item()
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

print('Loading the best epoch of our prediction with 2 E-GGAT blocks')
model_2.load_state_dict(torch.load('./layer_trained_models/2_E_GGAT/1239.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model_2.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_2(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds.reshape(-1) * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']
    denormalized_truth = test_graph.ndata['execution_time_label'] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += r2_value.detach().item()
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

print('Loading the best epoch of our prediction with 3 E-GGAT blocks')
model_3.load_state_dict(torch.load('./layer_trained_models/3_E_GGAT/894.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model_3.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_3(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds.reshape(-1) * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']
    denormalized_truth = test_graph.ndata['execution_time_label'] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += r2_value.detach().item()
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

print('Loading the best epoch of our prediction with 4 E-GGAT blocks')
model_4.load_state_dict(torch.load('./layer_trained_models/4_E_GGAT/1586.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model_4.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_4(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds.reshape(-1) * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']
    denormalized_truth = test_graph.ndata['execution_time_label'] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += r2_value.detach().item()
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

print('Loading the best epoch of our prediction with 5 E-GGAT blocks')
model_5.load_state_dict(torch.load('./layer_trained_models/5_E_GGAT/1333.pkl'))

total_test_loss = 0.0

total_rmse_value = 0.0
total_mae_value = 0.0
total_rmsle_value = 0.0
total_r2_value = 0.0
total_pred_err_value = 0.0

for test_graph in test_dataloader:
    model_5.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model_5(test_graph, test_graph.ndata['node_features'])

    denormalized_preds = preds.reshape(-1) * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']
    denormalized_truth = test_graph.ndata['execution_time_label'] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

    test_loss = loss_function(denormalized_preds, denormalized_truth)

    rmse_value = torch.sqrt(mse_function(denormalized_preds, denormalized_truth))
    mae_value = mae_function(denormalized_preds, denormalized_truth)
    rmsle_value = rmsle_function(denormalized_preds, denormalized_truth)
    r2_value = r2_function(denormalized_preds, denormalized_truth)
    pred_err_value = pred_err_function(denormalized_preds, denormalized_truth)

    total_test_loss += test_loss.detach().item()

    total_rmse_value += rmse_value.detach().item()
    total_mae_value += mae_value.detach().item()
    total_rmsle_value += rmsle_value.detach().item()
    total_r2_value += r2_value.detach().item()
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

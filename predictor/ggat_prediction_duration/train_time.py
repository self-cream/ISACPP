import torch
import glob
import os
from torch.utils.data import DataLoader
from dataset import InterferenceDataset, collate
from torch.optim.lr_scheduler import MultiStepLR
from models import GGAT_with_3_blocks
from dgl.data.utils import split_dataset
import torch.nn as nn
from torchmetrics import R2Score


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


def get_total_para_num(model):
    num_count = 0
    num_params = 0
    for para in model.parameters():
        num_count += 1
        num_params += para.numel()
    print("total para num: %d" % num_params)


dataset_dir_path = '/workspace/datasets/graph_datasets'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

interference_dataset = InterferenceDataset(dataset_dir_path)
train_subset, val_subset, test_subset = split_dataset(interference_dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_subset, batch_size=64, shuffle=True, collate_fn=collate)

model = GGAT_with_3_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)

get_total_para_num(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[300], gamma=0.1)

loss_function = torch.nn.SmoothL1Loss(reduction='mean').to(device)

mse_function = torch.nn.MSELoss(reduction='mean').to(device)
mae_function = torch.nn.L1Loss(reduction='mean').to(device)
rmsle_function = RMSLELoss().to(device)
r2_function = R2Score().to(device)
pred_err_function = PredictionErr().to(device)

train_loss_values = []
val_loss_values = []

min_epochs = 10
best_epoch = 0
patience = 1000
bad_counter = 0
best_val_loss = float('inf')

print("Ready to train GGAT model......")
for epoch in range(3000):
    epoch_train_loss = 0.0
    for train_graph in train_dataloader:
        model.train()
        train_graph = train_graph.to(device)
        prediction = model(train_graph, train_graph.ndata['node_features'])
        loss = loss_function(prediction.reshape(-1), train_graph.ndata['execution_time_label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.detach().item()

    epoch_train_loss /= len(train_dataloader)
    train_loss_values.append(epoch_train_loss)

    epoch_val_loss = 0.0
    for val_graph in val_dataloader:
        model.eval()
        val_graph = val_graph.to(device)

        with torch.no_grad():
            preds = model(val_graph, val_graph.ndata['node_features'])

        val_loss = loss_function(preds.reshape(-1), val_graph.ndata['execution_time_label'])
        epoch_val_loss += val_loss.detach().item()

    scheduler.step()

    epoch_val_loss /= len(val_dataloader)
    val_loss_values.append(epoch_val_loss)

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    epoch_train_loss *= 1000
    epoch_val_loss *= 1000

    print(f'Epoch: {epoch:02d}, Train Loss: {epoch_train_loss:.4f},'
          f'Val Loss: {epoch_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')

    if epoch > min_epochs and bad_counter >= patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

torch.save(model.state_dict(), 'the-last.pkl')

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

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

    test_loss = loss_function(preds.reshape(-1), test_graph.ndata['execution_time_label'])

    rmse_value = torch.sqrt(mse_function(preds.reshape(-1), test_graph.ndata['execution_time_label']))
    mae_value = mae_function(preds.reshape(-1), test_graph.ndata['execution_time_label'])
    rmsle_value = rmsle_function(preds.reshape(-1), test_graph.ndata['execution_time_label'])
    r2_value = r2_function(preds.reshape(-1), test_graph.ndata['execution_time_label'])
    pred_err_value = pred_err_function(preds.reshape(-1), test_graph.ndata['execution_time_label'])

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

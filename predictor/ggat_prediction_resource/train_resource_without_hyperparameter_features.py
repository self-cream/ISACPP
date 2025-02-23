import torch
import glob
import os
from torch.utils.data import DataLoader
from dataset import InterferenceWithoutParameterFeaturesDataset, collate
from torch.optim.lr_scheduler import MultiStepLR
from models import GGAT_with_3_blocks
from dgl.data.utils import split_dataset


def get_total_para_num(model):
    num_count = 0
    num_params = 0
    for para in model.parameters():
        num_count += 1
        num_params += para.numel()
    print("total para num: %d" % num_params)


dataset_dir_path = '/workspace/datasets/graph_datasets'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

interference_dataset = InterferenceWithoutParameterFeaturesDataset(dataset_dir_path)
train_subset, val_subset, test_subset = split_dataset(interference_dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_subset, batch_size=64, shuffle=True, collate_fn=collate)

model = GGAT_with_3_blocks(in_dim=19, hidden_dim=512, out_dim=3, dropout=0, num_heads=3).to(device)

get_total_para_num(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[300], gamma=0.1)

loss_function = torch.nn.SmoothL1Loss(reduction='mean').to(device)

train_loss_values = []
val_loss_values = []

min_epochs = 10
best_epoch = 0
patience = 1000
bad_counter = 0
best_val_loss = float('inf')

print("Ready to train GGAT model......")
for epoch in range(5000):
    epoch_train_loss = 0.0
    for train_graph in train_dataloader:
        model.train()
        train_graph = train_graph.to(device)
        prediction = model(train_graph, train_graph.ndata['node_features'])
        loss = loss_function(prediction, train_graph.ndata['resource_util_label'])

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

        val_loss = loss_function(preds, val_graph.ndata['resource_util_label'])
        epoch_val_loss += val_loss.detach().item()

    scheduler.step()

    epoch_val_loss /= len(val_dataloader)
    val_loss_values.append(epoch_val_loss)

    torch.save(model.state_dict(), './ablation_trained_models/ours_without_hyperparameter_features/{}.pkl'.format(epoch))

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

    files = glob.glob('./ablation_trained_models/ours_without_hyperparameter_features/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[-1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./ablation_trained_models/ours_without_hyperparameter_features/*.pkl')
for file in files:
    epoch_nb = int(file.split('/')[-1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

torch.save(model.state_dict(), './ablation_trained_modelss/ours_without_hyperparameter_features/the-last.pkl')

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./ablation_trained_models/ours_without_hyperparameter_features/{}.pkl'.format(best_epoch)))

total_test_loss = 0.0
for test_graph in test_dataloader:
    model.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        preds = model(test_graph, test_graph.ndata['node_features'])

    test_loss = loss_function(preds, test_graph.ndata['resource_util_label'])
    total_test_loss += test_loss.detach().item()

total_test_loss /= len(test_dataloader)

print(f'Test Loss: {total_test_loss:.4f}')

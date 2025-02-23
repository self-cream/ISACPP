import torch
import pandas as pd
from datapreprocess import get_max_and_min_in_solo_data, get_max_and_min_in_colocation_data
from models import GGAT_with_3_blocks
import dgl

rtx_4090_solo_csv_file = '/workspace/datasets/rtx_4090.csv'
rtx_3090_solo_csv_file = '/workspace/datasets/rtx_3090.csv'
titan_xp_solo_csv_file = '/workspace/datasets/titan_xp.csv'

rtx_4090_colocation_csv_file = '/workspace/datasets/rtx_4090_colocation.csv'
rtx_3090_colocation_csv_file = '/workspace/datasets/rtx_3090_colocation.csv'
titan_xp_colocation_csv_file = '/workspace/datasets/titan_xp_colocation.csv'

rtx_4090_workloads = ['vgg19', 'densenet121', 'inceptionv3', 'resnet50', 'moflow']
rtx_4090_workload_batch_sizes = ['512', '256', '256', '256', '256']

rtx_4090_colocation_workloads = 'wideresnet-depth-52-width-5'
rtx_4090_colocation_workload_batch_sizes = ['512', '256', '256', '512', '256']

rtx_3090_workloads = ['vgg19', 'densenet161', 'inceptionv3', 'resnet50', 'moflow']
rtx_3090_workload_batch_sizes = ['512', '128', '256', '256', '256']

rtx_3090_colocation_workloads = 'wideresnet-depth-52-width-5'
rtx_3090_colocation_workload_batch_sizes = ['512', '256', '256', '512', '256']

titan_xp_workloads = ['vgg19', 'densenet201', 'inceptionv3', 'resnet50', 'moflow']
titan_xp_workload_batch_sizes = ['512', '64', '128', '128', '128']

titan_xp_colocation_workloads = 'wideresnet-depth-52-width-5'
titan_xp_colocation_workload_batch_sizes = ['256', '256', '128', '256', '128']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
execution_time_model = GGAT_with_3_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)

execution_time_model.load_state_dict(torch.load('./trained_models/6/261.pkl'))
execution_time_model.eval()


def create_graph(graph_features, graph_edges):
    print(graph_features)
    print(graph_edges)
    graph_nodes_list = graph_features['Model name'].tolist()

    idx_map = {j: i for i, j in enumerate(graph_nodes_list)}

    graph_features_copy = graph_features.copy()
    graph_edges_copy = graph_edges.copy()

    graph_features_copy['Model name'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                              inplace=True)

    graph_edges_copy['src'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                    inplace=True)
    graph_edges_copy['dst'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                    inplace=True)

    src = graph_edges_copy['src'].to_numpy()
    dst = graph_edges_copy['dst'].to_numpy()

    g = dgl.graph((src, dst))

    node_features = torch.tensor(graph_features_copy.loc[:, 'Batch size':'Tensor cores'].to_numpy(), dtype=torch.float)
    performance_degradation = torch.tensor(graph_edges_copy.loc[:, 'performance degradation'].to_numpy(),
                                           dtype=torch.float).reshape(-1, 1)

    g.ndata['node_features'] = node_features
    g.edata['performance_degradation'] = performance_degradation

    return g


def get_graph_features_and_edges(solo_csv_file_path, colocation_csv_file_path, scheduling_task_name, scheduling_task_batch_size, colocation_task_name, colocation_task_batch_size):
    numeric_columns, solo_data_max, solo_data_min = get_max_and_min_in_solo_data()

    all_tasks_data = pd.read_csv(solo_csv_file_path)
    all_colocation_tasks_data = pd.read_csv(colocation_csv_file_path)

    scheduling_task_data = all_tasks_data[(all_tasks_data['Model name'] == scheduling_task_name) & (
            all_tasks_data['Batch size'] == int(scheduling_task_batch_size))]

    colocation_task_data = all_tasks_data[(all_tasks_data['Model name'] == colocation_task_name) & (
            all_tasks_data['Batch size'] == int(colocation_task_batch_size))]

    scheduling_task_solo_execution_time = scheduling_task_data.loc[:, 'Execution time (s)'].values[0]
    colocation_task_solo_execution_time = colocation_task_data.loc[:, 'Execution time (s)'].values[0]

    graph_nodes_solo_execution_time = {}

    graph_nodes_solo_execution_time[scheduling_task_name] = scheduling_task_solo_execution_time
    graph_nodes_solo_execution_time[colocation_task_name] = colocation_task_solo_execution_time

    normalized_scheduling_task_data = scheduling_task_data.copy()
    normalized_scheduling_task_data[numeric_columns] = (normalized_scheduling_task_data[
                                                            numeric_columns] - solo_data_min) / (
                                                               solo_data_max - solo_data_min)

    normalized_colocation_task_data = colocation_task_data.copy()
    normalized_colocation_task_data[numeric_columns] = (normalized_colocation_task_data[
                                                            numeric_columns] - solo_data_min) / (
                                                               solo_data_max - solo_data_min)

    normalized_scheduling_task_features = normalized_scheduling_task_data.loc[:, 'Model name':'Tensor cores']
    normalized_colocation_task_features = normalized_colocation_task_data.loc[:, 'Model name':'Tensor cores']

    gpu_graph_features = pd.DataFrame()

    gpu_graph_features = pd.concat([gpu_graph_features, normalized_scheduling_task_features], ignore_index=True)
    gpu_graph_features = pd.concat([gpu_graph_features, normalized_colocation_task_features], ignore_index=True)

    if (colocation_task_name + ' / ' + scheduling_task_name) in all_colocation_tasks_data.set_index(
                        'Model name').index:
        task_name_search_cond = colocation_task_name + ' / ' + scheduling_task_name
        task_batch_size_search_cond = colocation_task_batch_size + ' / ' + scheduling_task_batch_size
    elif (scheduling_task_name + ' / ' + colocation_task_name) in all_colocation_tasks_data.set_index(
            'Model name').index:
        task_name_search_cond = scheduling_task_name + ' / ' + colocation_task_name
        task_batch_size_search_cond = scheduling_task_batch_size + ' / ' + colocation_task_batch_size
    else:
        task_name_search_cond = ''
        task_batch_size_search_cond = ''
        print("Cannot find the colocation combination in the colocation csv file")

    colocation_combination_data = all_colocation_tasks_data[
        (all_colocation_tasks_data['Model name'] == task_name_search_cond) & (
                all_colocation_tasks_data['Batch size'] == task_batch_size_search_cond)]

    colocation_combination_execution_time = colocation_combination_data.loc[:,
                                            'Execution time (s)'].values.tolist()
    colocation_combination_execution_time_str = colocation_combination_execution_time[0]

    colocation_combination_name_list = task_name_search_cond.split(' / ')
    colocation_combination_execution_time_list = colocation_combination_execution_time_str.split(' / ')

    colocation_task_one_performance_degradation = float(colocation_combination_execution_time_list[0]) / (
        graph_nodes_solo_execution_time[colocation_combination_name_list[0]])
    colocation_task_other_performance_degradation = float(colocation_combination_execution_time_list[1]) / (
        graph_nodes_solo_execution_time[colocation_combination_name_list[1]])

    colocation_dict_src = []
    colocation_dict_dst = []
    colocation_dict_performance_degradation = []

    colocation_dict_src.append(colocation_combination_name_list[1])
    colocation_dict_dst.append(colocation_combination_name_list[0])
    colocation_dict_performance_degradation.append(colocation_task_one_performance_degradation)

    colocation_dict_src.append(colocation_combination_name_list[0])
    colocation_dict_dst.append(colocation_combination_name_list[1])
    colocation_dict_performance_degradation.append(colocation_task_other_performance_degradation)

    colcoation_dict = {'src': colocation_dict_src, 'dst': colocation_dict_dst,
                       'performance degradation': colocation_dict_performance_degradation}

    gpu_graph_edges = pd.DataFrame(colcoation_dict)

    return gpu_graph_features, gpu_graph_edges


def get_inference_results(solo_csv_file_path, colocation_csv_file_path, workload_list, workload_batch_size_list, colocation_workload_name, colocation_workload_batch_size_list):
    inference_results_list = []

    for idx in range(len(workload_list)):
        current_workload_name = workload_list[idx]
        current_workload_batch_size = workload_batch_size_list[idx]
        current_colocation_workload_name = colocation_workload_name
        current_colocation_workload_batch_size = colocation_workload_batch_size_list[idx]

        current_graph_features, current_graph_edges = get_graph_features_and_edges(solo_csv_file_path, colocation_csv_file_path, current_workload_name, current_workload_batch_size, current_colocation_workload_name, current_colocation_workload_batch_size)

        current_graph = create_graph(current_graph_features, current_graph_edges)
        current_graph = current_graph.to(device)

        with torch.no_grad():
            normalized_pred_time_per_epoch = execution_time_model(current_graph, current_graph.ndata['node_features'])

        colocation_data_min, colocation_data_max = get_max_and_min_in_colocation_data()

        colocation_execution_time_min = [colocation_data_min['execution_time']]
        colocation_execution_time_max = [colocation_data_max['execution_time']]

        colocation_execution_time_min_tensor = torch.tensor(colocation_execution_time_min,
                                                            dtype=torch.float).to(
            device)
        colocation_execution_time_max_tensor = torch.tensor(colocation_execution_time_max,
                                                            dtype=torch.float).to(
            device)

        pred_time_per_epoch = normalized_pred_time_per_epoch * (
                colocation_execution_time_max_tensor - colocation_execution_time_min_tensor) + colocation_execution_time_min_tensor

        inference_results_list.append(pred_time_per_epoch)

    return inference_results_list


if __name__ == '__main__':
    inference_results_list = get_inference_results(titan_xp_solo_csv_file, titan_xp_colocation_csv_file, titan_xp_workloads, titan_xp_workload_batch_sizes, titan_xp_colocation_workloads, titan_xp_colocation_workload_batch_sizes)
    print(inference_results_list)











from flask import Flask, request, jsonify
import pandas as pd
import torch
import dgl
from datapreprocess import get_max_and_min_in_solo_data, get_max_and_min_in_colocation_data
from models import GGAT_with_3_blocks
from model_resource import GGAT_RES

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

execution_time_model = GGAT_with_3_blocks(in_dim=24, hidden_dim=512, out_dim=1, dropout=1, num_heads=3).to(device)
resource_consumption_model = GGAT_RES(in_dim=24, hidden_dim=512, out_dim=3, dropout=1, num_heads=3).to(device)

execution_time_model.load_state_dict(torch.load('./trained_models/6/261.pkl'))
resource_consumption_model.load_state_dict(torch.load('./resource_trained_models/5/24.pkl'))

execution_time_model.eval()
resource_consumption_model.eval()


def create_graph(graph_features, graph_edges):
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


def get_graph_features_and_edges(solo_csv_file_path, colocation_csv_file_path, scheduling_task_name,
                                 scheduling_task_batch_size, scheduling_task_epochs, all_gpus_info):
    numeric_columns, solo_data_max, solo_data_min = get_max_and_min_in_solo_data()

    all_tasks_data = pd.read_csv(solo_csv_file_path)
    all_colocation_tasks_data = pd.read_csv(colocation_csv_file_path)

    scheduling_task_data = all_tasks_data[(all_tasks_data['Model name'] == scheduling_task_name) & (
            all_tasks_data['Batch size'] == int(scheduling_task_batch_size))]

    normalized_scheduling_task_data = scheduling_task_data.copy()
    normalized_scheduling_task_data[numeric_columns] = (normalized_scheduling_task_data[
                                                            numeric_columns] - solo_data_min) / (
                                                               solo_data_max - solo_data_min)

    normalized_scheduling_task_features = normalized_scheduling_task_data.loc[:, 'Model name':'Tensor cores']

    scheduling_task_solo_execution_time = scheduling_task_data.loc[:, 'Execution time (s)'].values[0]
    scheduling_task_solo_resource_consumption = scheduling_task_data.loc[:, ['Model name', 'GPU memory (MB)',
                                                                             'PCIe bandwidth',
                                                                             'GPU utilization']]

    all_gpus_graph_features = {}
    all_gpus_graph_edges = {}
    all_gpus_graph_node_info = {}
    all_gpus_graph_node_resource_consumption = {}
    graph_nodes_solo_execution_time = {}
    graph_node_list = []
    graph_node_batch_size_list = []

    for gpu in all_gpus_info:
        gpu_index = gpu['index']
        running_tasks_on_gpu = gpu['taskinfo']

        if len(running_tasks_on_gpu) == 0:
            print("There is no running task on this GPU")
            all_gpus_graph_features[gpu_index] = pd.DataFrame()
            all_gpus_graph_edges[gpu_index] = pd.DataFrame()
            all_gpus_graph_node_resource_consumption[gpu_index] = pd.DataFrame()
            all_gpus_graph_node_info[gpu_index] = pd.DataFrame()

            continue

        gpu_graph_features = pd.DataFrame()
        gpu_graph_resource_consumption = pd.DataFrame()

        scheduling_task_remaining_iters = int(scheduling_task_epochs) * (50000 // int(scheduling_task_batch_size) + 1)

        graph_node_info = pd.DataFrame(
            columns=['name', 'batch size', 'remaining iters', 'colocation period', 'remaining execution time',
                     'time per epoch',
                     'resource contention degree'])

        graph_nodes_solo_execution_time[scheduling_task_name] = scheduling_task_solo_execution_time
        graph_node_list.append(scheduling_task_name)
        graph_node_batch_size_list.append(scheduling_task_batch_size)
        gpu_graph_features = pd.concat([gpu_graph_features, normalized_scheduling_task_features], ignore_index=True)
        gpu_graph_resource_consumption = pd.concat(
            [gpu_graph_resource_consumption, scheduling_task_solo_resource_consumption], ignore_index=True)

        graph_node_info = pd.concat([graph_node_info, pd.DataFrame(
            {'name': [scheduling_task_name], 'batch size': [int(scheduling_task_batch_size)],
             'remaining iters': [scheduling_task_remaining_iters], 'colocation period': [0.0],
             'remaining execution time': [0.0], 'time per epoch': ['0'], 'resource contention degree': ['0']})], axis=0,
                                    ignore_index=True)

        for running_task in running_tasks_on_gpu:
            running_task_name = running_task['name']
            running_task_batch_size = running_task['batchsize']
            running_task_elapsed_iters = running_task['elapsediters']
            running_task_epochs = running_task['totalepochs']
            running_task_iters_per_epoch = running_task['itersperepoch']

            running_task_remaining_iters = int(running_task_epochs) * int(running_task_iters_per_epoch) - int(
                running_task_elapsed_iters)

            graph_node_info = pd.concat([graph_node_info, pd.DataFrame(
                {'name': [running_task_name], 'batch size': [int(running_task_batch_size)],
                 'remaining iters': [running_task_remaining_iters],
                 'colocation period': [0.0], 'remaining execution time': [0.0], 'time per epoch': ['0'],
                 'resource contention degree': ['0']})], axis=0, ignore_index=True)

            running_task_data = all_tasks_data[(all_tasks_data['Model name'] == running_task_name) & (
                    all_tasks_data['Batch size'] == int(running_task_batch_size))]

            normalized_running_task_data = running_task_data.copy()
            normalized_running_task_data[numeric_columns] = (normalized_running_task_data[
                                                                 numeric_columns] - solo_data_min) / (
                                                                    solo_data_max - solo_data_min)

            normalized_running_task_features = normalized_running_task_data.loc[:, 'Model name':'Tensor cores']

            running_task_solo_execution_time = running_task_data.loc[:, 'Execution time (s)'].values[0]
            running_task_solo_resource_consumption = running_task_data.loc[:,
                                                     ['Model name', 'GPU memory (MB)',
                                                      'PCIe bandwidth', 'GPU utilization']]

            graph_nodes_solo_execution_time[running_task_name] = running_task_solo_execution_time
            graph_node_list.append(running_task_name)
            graph_node_batch_size_list.append(running_task_batch_size)
            gpu_graph_features = pd.concat([gpu_graph_features, normalized_running_task_features], ignore_index=True)
            gpu_graph_resource_consumption = pd.concat(
                [gpu_graph_resource_consumption, running_task_solo_resource_consumption], ignore_index=True)

        all_gpus_graph_features[gpu_index] = gpu_graph_features
        all_gpus_graph_node_info[gpu_index] = graph_node_info
        all_gpus_graph_node_resource_consumption[gpu_index] = gpu_graph_resource_consumption

        colocation_dict_src = []
        colocation_dict_dst = []
        colocation_dict_performance_degradation = []

        for index in range(len(graph_node_list)):
            colocation_task_name = graph_node_list[index]
            colocation_task_batch_size = graph_node_batch_size_list[index]

            for neigh_index in range(index + 1, len(graph_node_list)):
                neighbor_task_name = graph_node_list[neigh_index]
                neighbor_task_batch_size = graph_node_batch_size_list[neigh_index]

                if (colocation_task_name + ' / ' + neighbor_task_name) in all_colocation_tasks_data.set_index(
                        'Model name').index:
                    task_name_search_cond = colocation_task_name + ' / ' + neighbor_task_name
                    task_batch_size_search_cond = colocation_task_batch_size + ' / ' + neighbor_task_batch_size
                elif (neighbor_task_name + ' / ' + colocation_task_name) in all_colocation_tasks_data.set_index(
                        'Model name').index:
                    task_name_search_cond = neighbor_task_name + ' / ' + colocation_task_name
                    task_batch_size_search_cond = neighbor_task_batch_size + ' / ' + colocation_task_batch_size
                else:
                    task_name_search_cond = ''
                    task_batch_size_search_cond = ''
                    print("Cannot find the colocation combination in the colocation csv file")

                colocation_combination_data = all_colocation_tasks_data[
                    (all_colocation_tasks_data['Model name'] == task_name_search_cond) & (
                            all_colocation_tasks_data['Batch size'] == task_batch_size_search_cond)]

                if colocation_combination_data.empty:
                    print("Can not find the co-location data matching the batch size")
                    alternative_colocation_combination_data = all_colocation_tasks_data[all_colocation_tasks_data['Model name'] == task_name_search_cond]
                    alternative_colocation_combination_data_batch_size_list = alternative_colocation_combination_data.loc[:, 'Batch size'].values.tolist()
                    colocation_combination_data = all_colocation_tasks_data[
                    (all_colocation_tasks_data['Model name'] == task_name_search_cond) & (
                            all_colocation_tasks_data['Batch size'] == alternative_colocation_combination_data_batch_size_list[0])]


                colocation_combination_execution_time = colocation_combination_data.loc[:,
                                                        'Execution time (s)'].values.tolist()
                colocation_combination_execution_time_str = colocation_combination_execution_time[0]

                colocation_combination_name_list = task_name_search_cond.split(' / ')
                colocation_combination_execution_time_list = colocation_combination_execution_time_str.split(' / ')

                colocation_task_one_performance_degradation = float(colocation_combination_execution_time_list[0]) / (
                    graph_nodes_solo_execution_time[colocation_combination_name_list[0]])
                colocation_task_other_performance_degradation = float(colocation_combination_execution_time_list[1]) / (
                    graph_nodes_solo_execution_time[colocation_combination_name_list[1]])

                colocation_dict_src.append(colocation_combination_name_list[1])
                colocation_dict_dst.append(colocation_combination_name_list[0])
                colocation_dict_performance_degradation.append(colocation_task_one_performance_degradation)

                colocation_dict_src.append(colocation_combination_name_list[0])
                colocation_dict_dst.append(colocation_combination_name_list[1])
                colocation_dict_performance_degradation.append(colocation_task_other_performance_degradation)

        colcoation_dict = {'src': colocation_dict_src, 'dst': colocation_dict_dst, 'performance degradation': colocation_dict_performance_degradation}
        gpu_graph_edges = pd.DataFrame(colcoation_dict)
        all_gpus_graph_edges[gpu_index] = gpu_graph_edges

    return all_gpus_graph_features, all_gpus_graph_edges, all_gpus_graph_node_resource_consumption, all_gpus_graph_node_info


@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Get the feature data from the request
        request_data = request.json

        scheduling_task_name = request_data['name']
        scheduling_task_gpu_model = request_data['gpumodel']
        scheduling_task_batch_size = request_data['batchsize']
        scheduling_task_epochs = request_data['totalepochs']
        all_gpus_info = request_data['gpuinfo']

        print("task name: %s" % scheduling_task_name)
        print("task gpu model: %s" % scheduling_task_gpu_model)
        print("task batch size: %s" % scheduling_task_batch_size)
        print("task total epochs: %s" % scheduling_task_epochs)


        csv_file_path = ''
        if scheduling_task_gpu_model == 'NVIDIA GeForce RTX 4090':
            solo_csv_file_path = '/workspace/datasets/rtx_4090.csv'
            colocation_csv_file_path = '/workspace/datasets/rtx_4090_colocation.csv'
        elif scheduling_task_gpu_model == 'NVIDIA GeForce RTX 3090':
            solo_csv_file_path = '/workspace/datasets/rtx_3090.csv'
            colocation_csv_file_path = '/workspace/datasets/rtx_3090_colocation.csv'
        elif scheduling_task_gpu_model == 'NVIDIA TITAN Xp':
            solo_csv_file_path = '/workspace/datasets/titan_xp.csv'
            colocation_csv_file_path = '/workspace/datasets/titan_xp_colocation.csv'
        else:
            solo_csv_file_path = ''
            colocation_csv_file_path = ''
            print("can't find the gpu model for this task")

        all_gpus_graph_features, all_gpus_graph_edges, all_gpus_graph_node_resource_consumption, all_gpus_graph_node_info = get_graph_features_and_edges(
            solo_csv_file_path,
            colocation_csv_file_path,
            scheduling_task_name,
            scheduling_task_batch_size,
            scheduling_task_epochs,
            all_gpus_info)

        all_gpus_graph_interference = {}
        for index, graph_features in all_gpus_graph_features.items():
            if graph_features.empty:
                print("There is no node in the graph")
                all_gpus_graph_interference[index] = 0.0
                continue

            graph_node_end_sort_list = []
            graph_edges = all_gpus_graph_edges[index]
            graph_node_info = all_gpus_graph_node_info[index]
            graph_node_resource_consumption = all_gpus_graph_node_resource_consumption[index]
            graph_nodes_list = graph_features['Model name'].tolist()
            remaining_graph_nodes_list = graph_nodes_list.copy()
            remaining_graph_node_info = graph_node_info.copy()
            graph_node_number = len(graph_nodes_list)

            colocation_data_min, colocation_data_max = get_max_and_min_in_colocation_data()

            while (len(graph_node_end_sort_list) + 1) < graph_node_number:
                graph = create_graph(graph_features, graph_edges)
                graph = graph.to(device)

                with torch.no_grad():
                    normalized_pred_time_per_epoch = execution_time_model(graph, graph.ndata['node_features'])
                    normalized_pred_resource = resource_consumption_model(graph, graph.ndata['node_features'])

                colocation_execution_time_min = [colocation_data_min['execution_time']]
                colocation_execution_time_max = [colocation_data_max['execution_time']]

                colocation_resource_consumption_min = [colocation_data_min['gpu_memory'],
                                                       colocation_data_min['pcie_band'],
                                                       colocation_data_min['gpu_util']]
                colocation_resource_consumption_max = [colocation_data_max['gpu_memory'],
                                                       colocation_data_max['pcie_band'],
                                                       colocation_data_max['gpu_util']]

                colocation_execution_time_min_tensor = torch.tensor(colocation_execution_time_min,
                                                                    dtype=torch.float).to(
                    device)
                colocation_execution_time_max_tensor = torch.tensor(colocation_execution_time_max,
                                                                    dtype=torch.float).to(
                    device)

                colocation_resource_consumption_min_tensor = torch.tensor(colocation_resource_consumption_min,
                                                                          dtype=torch.float).reshape(1, -1).to(device)
                colocation_resource_consumption_max_tensor = torch.tensor(colocation_resource_consumption_max,
                                                                          dtype=torch.float).reshape(1, -1).to(device)

                pred_time_per_epoch = normalized_pred_time_per_epoch * (
                        colocation_execution_time_max_tensor - colocation_execution_time_min_tensor) + colocation_execution_time_min_tensor

                pred_resource = normalized_pred_resource * (
                        colocation_resource_consumption_max_tensor - colocation_resource_consumption_min_tensor) + colocation_resource_consumption_min_tensor

                all_tasks_remaining_iters = remaining_graph_node_info.loc[:, 'remaining iters'].to_numpy().astype(float)
                all_tasks_batch_size = remaining_graph_node_info.loc[:, 'batch size'].to_numpy().astype(int)

                all_tasks_remaining_iters_tensor = torch.tensor(all_tasks_remaining_iters, dtype=torch.float).reshape(
                    -1,
                    1).to(
                    device)
                all_tasks_batch_size_tensor = torch.tensor(all_tasks_batch_size, dtype=torch.int).reshape(-1, 1).to(
                    device)

                estimated_colocation_period_tensor = all_tasks_remaining_iters_tensor * (
                        pred_time_per_epoch / (50000 // all_tasks_batch_size_tensor + 1))

                colocation_period_tensor, min_index_tensor = torch.min(estimated_colocation_period_tensor, dim=0)

                colocation_period = colocation_period_tensor.item()
                min_index = min_index_tensor.item()

                end_task_name = remaining_graph_nodes_list[min_index]

                graph_node_info.loc[graph_node_info['name'] == end_task_name, 'colocation period'] = colocation_period

                all_tasks_remaining_iters_tensor = all_tasks_remaining_iters_tensor - torch.floor(
                    colocation_period / (pred_time_per_epoch / (torch.floor(50000 / all_tasks_batch_size_tensor) + 1)))

                remaining_graph_node_info.loc[remaining_graph_node_info['name'] == end_task_name, 'remaining iters'] = \
                    all_tasks_remaining_iters_tensor[min_index].item()
                graph_node_info.loc[graph_node_info['name'] == end_task_name, 'remaining iters'] = \
                    all_tasks_remaining_iters_tensor[min_index].item()

                prev_colocation_period = 0.0
                for task_name in graph_node_end_sort_list:
                    prev_colocation_period += \
                        graph_node_info.loc[graph_node_info['name'] == task_name, 'colocation period'].values[0]

                graph_node_info.loc[graph_node_info[
                                        'name'] == end_task_name, 'remaining execution time'] = prev_colocation_period + colocation_period

                end_task_solo_resource_consumption = graph_node_resource_consumption.loc[
                                                     graph_node_resource_consumption['Model name'] == end_task_name,
                                                     'GPU memory (MB)':'GPU utilization'].to_numpy().astype(
                    float)

                end_task_solo_resource_consumption_tensor = torch.tensor(end_task_solo_resource_consumption,
                                                                         dtype=torch.float).reshape(1, -1).to(device)

                end_task_pred_resource = pred_resource[min_index]

                end_task_resource_contention_degree = end_task_solo_resource_consumption_tensor / end_task_pred_resource

                sum_end_task_resource_contention_degree = torch.sum(end_task_resource_contention_degree, dim=1)

                sum_end_task_resource_contention_degree_str = format(sum_end_task_resource_contention_degree.item(),
                                                                    '.4f')

                prev_end_task_resource_contention_degree = graph_node_info.loc[
                    graph_node_info['name'] == end_task_name, 'resource contention degree']

                if prev_end_task_resource_contention_degree.values[0] == '0':
                    graph_node_info.loc[graph_node_info[
                                            'name'] == end_task_name, 'resource contention degree'] = sum_end_task_resource_contention_degree_str
                else:
                    graph_node_info.loc[graph_node_info['name'] == end_task_name, 'resource contention degree'] = \
                        prev_end_task_resource_contention_degree.values[
                            0] + ' / ' + sum_end_task_resource_contention_degree_str

                for task_index in range(len(remaining_graph_nodes_list)):
                    if task_index == min_index:
                        continue

                    other_task_name = remaining_graph_nodes_list[task_index]

                    remaining_graph_node_info.loc[
                        remaining_graph_node_info['name'] == other_task_name, 'remaining iters'] = \
                        all_tasks_remaining_iters_tensor[task_index].item()
                    graph_node_info.loc[graph_node_info['name'] == other_task_name, 'remaining iters'] = \
                        all_tasks_remaining_iters_tensor[task_index].item()

                    other_task_solo_resource_consumption = graph_node_resource_consumption.loc[
                                                           graph_node_resource_consumption[
                                                               'Model name'] == other_task_name,
                                                           'GPU memory (MB)':'GPU utilization'].to_numpy().astype(
                        float)
                    other_task_solo_resource_consumption_tensor = torch.tensor(other_task_solo_resource_consumption,
                                                                               dtype=torch.float).reshape(1, -1).to(
                        device)

                    other_task_pred_resource = pred_resource[task_index]
                    other_task_resource_contention_degree = other_task_solo_resource_consumption_tensor / other_task_pred_resource

                    sum_other_task_resource_contention_degree = torch.sum(other_task_resource_contention_degree, dim=1)
                    sum_other_task_resource_contention_degree_str = format(
                        sum_other_task_resource_contention_degree.item(),
                        '.4f')

                    prev_other_task_resource_contention_degree = graph_node_info.loc[
                        graph_node_info['name'] == other_task_name, 'resource contention degree']

                    if prev_other_task_resource_contention_degree.values[0] == '0':
                        graph_node_info.loc[graph_node_info[
                                                'name'] == other_task_name, 'resource contention degree'] = sum_other_task_resource_contention_degree_str
                    else:
                        graph_node_info.loc[graph_node_info['name'] == other_task_name, 'resource contention degree'] = \
                            prev_other_task_resource_contention_degree.values[
                                0] + ' / ' + sum_other_task_resource_contention_degree_str

                graph_node_end_sort_list.append(end_task_name)
                remaining_graph_nodes_list.remove(end_task_name)

                graph_features = graph_features[~(graph_features['Model name'] == end_task_name)]
                graph_edges = graph_edges[
                    ~((graph_edges['src'] == end_task_name) | (graph_edges['dst'] == end_task_name))]

                remaining_graph_node_info = remaining_graph_node_info[
                    ~(remaining_graph_node_info['name'] == end_task_name)]

            last_remaining_task_name = remaining_graph_nodes_list[0]

            last_remaining_task_batch_size = \
                graph_node_info.loc[graph_node_info['name'] == last_remaining_task_name, 'batch size'].values[0]
            last_remaining_task_remaining_iters = \
                graph_node_info.loc[graph_node_info['name'] == last_remaining_task_name, 'remaining iters'].values[0]

            all_tasks_data = pd.read_csv(solo_csv_file_path)
            last_remaining_task_data = all_tasks_data[(all_tasks_data['Model name'] == last_remaining_task_name) & (
                    all_tasks_data['Batch size'] == last_remaining_task_batch_size)]

            last_remaining_task_solo_time_per_epoch = last_remaining_task_data.loc[:, 'Execution time (s)'].values[0]

            last_remaining_task_solo_execution_time = last_remaining_task_remaining_iters * (
                    last_remaining_task_solo_time_per_epoch / (50000 // last_remaining_task_batch_size + 1))

            prev_colocation_period = 0.0
            for task_name in graph_node_end_sort_list:
                prev_colocation_period += \
                    graph_node_info.loc[graph_node_info['name'] == task_name, 'colocation period'].values[0]

            graph_node_info.loc[graph_node_info[
                                    'name'] == last_remaining_task_name, 'remaining execution time'] = prev_colocation_period + last_remaining_task_solo_execution_time

            prev_last_remaining_task_resource_contention_degree = graph_node_info.loc[
                graph_node_info['name'] == last_remaining_task_name, 'resource contention degree']

            if prev_last_remaining_task_resource_contention_degree.values[0] == '0':
                graph_node_info.loc[graph_node_info[
                                        'name'] == last_remaining_task_name, 'resource contention degree'] = '0.0'
            else:
                graph_node_info.loc[graph_node_info['name'] == last_remaining_task_name, 'resource contention degree'] = \
                    prev_last_remaining_task_resource_contention_degree.values[0] + ' / ' + '0.0'

            graph_node_end_sort_list.append(last_remaining_task_name)

            scheduling_task_remaining_execution_time = \
                graph_node_info.loc[graph_node_info['name'] == scheduling_task_name, 'remaining execution time'].values[
                    0]

            colocation_period_list = []
            for task_name in graph_node_end_sort_list:
                if task_name == scheduling_task_name:
                    scheduling_task_colocation_period = \
                        graph_node_info.loc[graph_node_info['name'] == task_name, 'colocation period'].values[0]
                    colocation_period_list.append(scheduling_task_colocation_period)
                    break

                prev_colocation_period = \
                    graph_node_info.loc[graph_node_info['name'] == task_name, 'colocation period'].values[0]
                colocation_period_list.append(prev_colocation_period)

            scheduling_task_resource_contention_degree = \
                graph_node_info.loc[
                    graph_node_info['name'] == scheduling_task_name, 'resource contention degree'].values[0]

            scheduling_task_resource_contention_degree_str_list = scheduling_task_resource_contention_degree.split(' / ')
            scheduling_task_resource_contention_degree_list = [float(item) for item in
                                                              scheduling_task_resource_contention_degree_str_list]

            interference_on_scheduling_task = 0.0
            for idx, cp in enumerate(colocation_period_list):
                scheduling_task_rcd = scheduling_task_resource_contention_degree_list[idx]
                interference_on_scheduling_task += (cp / scheduling_task_remaining_execution_time) * scheduling_task_rcd

            interference_on_all_running_tasks_list = []
            for task_name in graph_nodes_list:
                if task_name == scheduling_task_name:
                    continue

                current_task_remaining_execution_time = \
                    graph_node_info.loc[graph_node_info['name'] == task_name, 'remaining execution time'].values[0]

                current_colocation_period_list = []
                if graph_node_end_sort_list.index(task_name) < graph_node_end_sort_list.index(scheduling_task_name):
                    for task in graph_node_end_sort_list:
                        if task == task_name:
                            scheduling_task_colocation_period = \
                                graph_node_info.loc[graph_node_info['name'] == task, 'colocation period'].values[0]
                            current_colocation_period_list.append(scheduling_task_colocation_period)
                            break

                        prev_colocation_period = \
                            graph_node_info.loc[graph_node_info['name'] == task, 'colocation period'].values[0]
                        current_colocation_period_list.append(prev_colocation_period)
                else:
                    for task in graph_node_end_sort_list:
                        if task == scheduling_task_name:
                            scheduling_task_colocation_period = \
                                graph_node_info.loc[graph_node_info['name'] == task, 'colocation period'].values[0]
                            current_colocation_period_list.append(scheduling_task_colocation_period)
                            break

                        prev_colocation_period = \
                            graph_node_info.loc[graph_node_info['name'] == task, 'colocation period'].values[0]
                        current_colocation_period_list.append(prev_colocation_period)

                current_task_resource_contention_degree = \
                    graph_node_info.loc[graph_node_info['name'] == task_name, 'resource contention degree'].values[0]

                current_task_resource_contention_degree_str_list = current_task_resource_contention_degree.split(' / ')
                current_task_resource_contention_degree_list = [float(item) for item in
                                                               current_task_resource_contention_degree_str_list]

                interference_on_current_task = 0.0
                for idx, cp in enumerate(current_colocation_period_list):
                    current_task_rcd = current_task_resource_contention_degree_list[idx]
                    interference_on_current_task += (cp / current_task_remaining_execution_time) * current_task_rcd

                interference_on_all_running_tasks_list.append(interference_on_current_task)

            interference_on_all_running_tasks_tensor = torch.tensor(interference_on_all_running_tasks_list,
                                                                    dtype=torch.float).reshape(1, -1)
            normalized_interference_on_all_running_tasks_tensor = interference_on_all_running_tasks_tensor / len(
                interference_on_all_running_tasks_list)

            total_interference_on_all_running_tasks = torch.sum(normalized_interference_on_all_running_tasks_tensor,
                                                                dim=1).item()

            total_interference = interference_on_scheduling_task + total_interference_on_all_running_tasks

            all_gpus_graph_interference[index] = total_interference

        print("start to prepare the response body...")

        return jsonify(all_gpus_graph_interference), 200

    except Exception as e:
        print("server error: %s" % (str(e)))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)




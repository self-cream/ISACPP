#import dgl
import pandas as pd
import os
import torch
#import matplotlib.pyplot as plt
#import networkx as nx

dataset_dir_path = '/workspace/datasets/graph_datasets'
input_excel_file = '/workspace/datasets/data.xlsx'

rtx_4090_solo_csv_file = '/workspace/datasets/rtx_4090.csv'
rtx_3090_solo_csv_file = '/workspace/datasets/rtx_3090.csv'
titan_xp_solo_csv_file = '/workspace/datasets/titan_xp.csv'

rtx_4090_colocation_csv_file = '/workspace/datasets/rtx_4090_colocation.csv'
rtx_3090_colocation_csv_file = '/workspace/datasets/rtx_3090_colocation.csv'
titan_xp_colocation_csv_file = '/workspace/datasets/titan_xp_colocation.csv'


def convert_xlsx_to_csv():
    # Replace 'input.xlsx' with the path to your Excel file and 'output.csv' with the desired CSV file name

    # Read the Excel file into a pandas DataFrame
    df_rtx_4090 = pd.read_excel(input_excel_file, sheet_name='RTX 4090')
    df_rtx_3090 = pd.read_excel(input_excel_file, sheet_name='RTX 3090')
    df_titan_xp = pd.read_excel(input_excel_file, sheet_name='Titan Xp')

    df_rtx_4090_colocation = pd.read_excel(input_excel_file, sheet_name='RTX 4090 colocation')
    df_rtx_3090_colocation = pd.read_excel(input_excel_file, sheet_name='RTX 3090 colocation')
    df_titan_xp_colocation = pd.read_excel(input_excel_file, sheet_name='Titan Xp colocation')

    # Write the DataFrame to a CSV file
    df_rtx_4090.to_csv(rtx_4090_solo_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{rtx_4090_solo_csv_file}' in CSV format.")
    df_rtx_3090.to_csv(rtx_3090_solo_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{rtx_3090_solo_csv_file}' in CSV format.")
    df_titan_xp.to_csv(titan_xp_solo_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{titan_xp_solo_csv_file}' in CSV format.")

    df_rtx_4090_colocation.to_csv(rtx_4090_colocation_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{rtx_4090_colocation_csv_file}' in CSV format.")
    df_rtx_3090_colocation.to_csv(rtx_3090_colocation_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{rtx_3090_colocation_csv_file}' in CSV format.")
    df_titan_xp_colocation.to_csv(titan_xp_colocation_csv_file, index=False)
    print(
        f"Conversion complete. Excel file '{input_excel_file}' has been converted to '{titan_xp_colocation_csv_file}' in CSV format.")


def get_max_and_min_in_solo_data():
    rtx_4090_solo_data = pd.read_csv(rtx_4090_solo_csv_file)
    rtx_3090_solo_data = pd.read_csv(rtx_3090_solo_csv_file)
    titan_xp_solo_data = pd.read_csv(titan_xp_solo_csv_file)

    solo_data = pd.concat([rtx_4090_solo_data, rtx_3090_solo_data, titan_xp_solo_data], axis=0, join='inner',
                          ignore_index=True)

    numeric_columns = solo_data.select_dtypes(include=['number']).columns

    return numeric_columns, solo_data[numeric_columns].max(), solo_data[numeric_columns].min()


def get_max_and_min_in_colocation_data():
    rtx_4090_workload_colocation_data = pd.read_csv(rtx_4090_colocation_csv_file)
    rtx_3090_workload_colocation_data = pd.read_csv(rtx_3090_colocation_csv_file)
    titan_xp_workload_colocation_data = pd.read_csv(titan_xp_colocation_csv_file)

    colocation_data_max = {}
    colocation_data_min = {}

    workload_colocation_data = pd.concat(
        [rtx_4090_workload_colocation_data, rtx_3090_workload_colocation_data, titan_xp_workload_colocation_data],
        axis=0, join='inner',
        ignore_index=True)

    workload_colocation_data_remove_nil = workload_colocation_data.dropna(axis=1)

    execution_time_values = workload_colocation_data_remove_nil['Execution time (s)'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    gpu_memory_values = workload_colocation_data_remove_nil['GPU memory (MB)'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    pcie_band_values = workload_colocation_data_remove_nil['PCIe bandwidth'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    gpu_util_values = workload_colocation_data_remove_nil['GPU utilization'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])

    execution_time_values_list = execution_time_values.tolist()
    min_execution_time_value = min(min(sublist) for sublist in execution_time_values_list)
    max_execution_time_value = max(max(sublist) for sublist in execution_time_values_list)
    colocation_data_min['execution_time'] = min_execution_time_value
    colocation_data_max['execution_time'] = max_execution_time_value

    gpu_memory_values_list = gpu_memory_values.tolist()
    min_gpu_memory_value = min(min(sublist) for sublist in gpu_memory_values_list)
    max_gpu_memory_value = max(max(sublist) for sublist in gpu_memory_values_list)
    colocation_data_min['gpu_memory'] = min_gpu_memory_value
    colocation_data_max['gpu_memory'] = max_gpu_memory_value

    pcie_band_values_list = pcie_band_values.tolist()
    min_pcie_band_value = min(min(sublist) for sublist in pcie_band_values_list)
    max_pcie_band_value = max(max(sublist) for sublist in pcie_band_values_list)
    colocation_data_min['pcie_band'] = min_pcie_band_value
    colocation_data_max['pcie_band'] = max_pcie_band_value

    gpu_util_values_list = gpu_util_values.tolist()
    min_gpu_util_value = min(min(sublist) for sublist in gpu_util_values_list)
    max_gpu_util_value = max(max(sublist) for sublist in gpu_util_values_list)
    colocation_data_min['gpu_util'] = min_gpu_util_value
    colocation_data_max['gpu_util'] = max_gpu_util_value

    return colocation_data_min, colocation_data_max


def load_normalized_csv_and_process_data_for_graph(csv_file_path, numeric_columns, solo_data_max, solo_data_min, colocation_data_min, colocation_data_max):
    workload_solo_info_csv_file_path = ''
    gpu_model = ''

    if 'rtx_4090' in csv_file_path:
        gpu_model = 'RTX_4090'
        workload_solo_info_csv_file_path = rtx_4090_solo_csv_file
    elif 'rtx_3090' in csv_file_path:
        gpu_model = 'RTX_3090'
        workload_solo_info_csv_file_path = rtx_3090_solo_csv_file
    elif 'titan_xp' in csv_file_path:
        gpu_model = 'Titan_Xp'
        workload_solo_info_csv_file_path = titan_xp_solo_csv_file
    else:
        print(f"Cannot resolve gpu model for given csv file path '{csv_file_path}'.")

    workload_solo_data = pd.read_csv(workload_solo_info_csv_file_path)
    workload_colocation_data = pd.read_csv(csv_file_path)
    workload_colocation_data_remove_nil = workload_colocation_data.dropna(axis=1)

    execution_time_values = workload_colocation_data_remove_nil['Execution time (s)'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    gpu_memory_values = workload_colocation_data_remove_nil['GPU memory (MB)'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    pcie_band_values = workload_colocation_data_remove_nil['PCIe bandwidth'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])
    gpu_util_values = workload_colocation_data_remove_nil['GPU utilization'].str.split(' / ').apply(
        lambda x: [float(i) for i in x])

    normalized_execution_time_values = [
        [(value - colocation_data_min['execution_time']) / (
                    colocation_data_max['execution_time'] - colocation_data_min['execution_time']) for value in
         sublist] for sublist in execution_time_values]
    normalized_gpu_memory_values = [
        [(value - colocation_data_min['gpu_memory']) / (
                colocation_data_max['gpu_memory'] - colocation_data_min['gpu_memory']) for value in
         sublist] for sublist in gpu_memory_values]
    normalized_pcie_band_values = [
        [(value - colocation_data_min['pcie_band']) / (
                colocation_data_max['pcie_band'] - colocation_data_min['pcie_band']) for value in
         sublist] for sublist in pcie_band_values]
    normalized_gpu_util_values = [
        [(value - colocation_data_min['gpu_util']) / (
                colocation_data_max['gpu_util'] - colocation_data_min['gpu_util']) for value in
         sublist] for sublist in gpu_util_values]

    workload_colocation_data_remove_nil.loc[:, 'Execution time (s)'] = normalized_execution_time_values
    workload_colocation_data_remove_nil.loc[:, 'GPU memory (MB)'] = normalized_gpu_memory_values
    workload_colocation_data_remove_nil.loc[:, 'PCIe bandwidth'] = normalized_pcie_band_values
    workload_colocation_data_remove_nil.loc[:, 'GPU utilization'] = normalized_gpu_util_values

    workload_colocation_data_list = workload_colocation_data_remove_nil.values.tolist()

    for workload_colocation_data_each_row in workload_colocation_data_list:
        colocation_pair_name_str = workload_colocation_data_each_row[0]
        colocation_pair_batch_size_str = workload_colocation_data_each_row[1]
        colocation_pair_execution_time_list = workload_colocation_data_each_row[-4]
        colocation_pair_gpu_memory_list = workload_colocation_data_each_row[-3]
        colocation_pair_pcie_band_list = workload_colocation_data_each_row[-2]
        colocation_pair_gpu_util_list = workload_colocation_data_each_row[-1]

        colocation_pair_name_list = colocation_pair_name_str.split(' / ')
        colocation_pair_batch_size_list = colocation_pair_batch_size_str.split(' / ')

        if len(colocation_pair_name_list) == 2:
            graph_csv_dir_name = f"{colocation_pair_name_list[0]}_{colocation_pair_batch_size_list[0]}_and_{colocation_pair_name_list[1]}_{colocation_pair_batch_size_list[1]}_in_{gpu_model}"
        elif len(colocation_pair_name_list) == 3:
            graph_csv_dir_name = f"{colocation_pair_name_list[0]}_{colocation_pair_batch_size_list[0]}_{colocation_pair_name_list[1]}_{colocation_pair_batch_size_list[1]}_and_{colocation_pair_name_list[2]}_{colocation_pair_batch_size_list[2]}_in_{gpu_model}"
        else:
            break

        colocation_pair_features = pd.DataFrame()
        colocation_pair_normalized_performance = pd.DataFrame(columns=['src', 'dst', 'performance degradation'])
        colocation_pair_labels = pd.DataFrame({'execution_time': colocation_pair_execution_time_list, 'gpu_memory': colocation_pair_gpu_memory_list, 'pcie_band': colocation_pair_pcie_band_list, 'gpu_util': colocation_pair_gpu_util_list})

        for index in range(len(colocation_pair_name_list)):
            colocation_workload_name = colocation_pair_name_list[index]
            colocation_workload_batch_size = colocation_pair_batch_size_list[index]
            colocation_workload_execution_time = colocation_pair_execution_time_list[index] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

            colocation_workload_solo_info = workload_solo_data[
                (workload_solo_data['Model name'] == colocation_workload_name) & (
                        workload_solo_data['Batch size'] == int(colocation_workload_batch_size))]

            normalized_colocation_workload_solo_info = colocation_workload_solo_info.copy()
            normalized_colocation_workload_solo_info[numeric_columns] = (colocation_workload_solo_info[numeric_columns] - solo_data_min) / (solo_data_max - solo_data_min)

            colocation_workload_solo_features = normalized_colocation_workload_solo_info.loc[:, 'Model name':'Tensor cores']
            colocation_pair_features = colocation_pair_features.append(colocation_workload_solo_features,
                                                                       ignore_index=True)

            for neigh_index in range(index + 1, len(colocation_pair_name_list)):
                neighbor_workload_name = colocation_pair_name_list[neigh_index]
                neighbor_workload_batch_size = colocation_pair_batch_size_list[neigh_index]
                neighbor_workload_execution_time = colocation_pair_execution_time_list[neigh_index] * (colocation_data_max['execution_time'] - colocation_data_min['execution_time']) + colocation_data_min['execution_time']

                neighbor_workload_solo_info = workload_solo_data[
                    (workload_solo_data['Model name'] == neighbor_workload_name) & (
                            workload_solo_data['Batch size'] == int(neighbor_workload_batch_size))]

                colocation_workload_solo_execution_time = colocation_workload_solo_info.loc[:,
                                                          ['Model name', 'Batch size', 'Execution time (s)']]

                neighbor_workload_solo_execution_time = neighbor_workload_solo_info.loc[:,
                                                          ['Model name', 'Batch size', 'Execution time (s)']]

                neighbor_workload_performance_degradation = float(neighbor_workload_execution_time) / \
                                                              neighbor_workload_solo_execution_time.values.tolist()[
                                                                  0][-1]

                colocation_workload_performance_degradation = float(colocation_workload_execution_time) / \
                                                              colocation_workload_solo_execution_time.values.tolist()[
                                                                  0][-1]

                colocation_pair_normalized_performance = colocation_pair_normalized_performance.append(
                    {'src': neighbor_workload_name, 'dst': colocation_workload_name,
                     'performance degradation': colocation_workload_performance_degradation}, ignore_index=True)

                colocation_pair_normalized_performance = colocation_pair_normalized_performance.append(
                    {'src': colocation_workload_name, 'dst': neighbor_workload_name,
                     'performance degradation': neighbor_workload_performance_degradation}, ignore_index=True)

        graph_full_dir_path = os.path.join(dataset_dir_path, graph_csv_dir_name)
        os.makedirs(graph_full_dir_path, exist_ok=True)

        graph_features_csv_file_path = os.path.join(graph_full_dir_path, 'node_features.csv')
        graph_edges_csv_file_path = os.path.join(graph_full_dir_path, 'edges.csv')
        graph_labels_csv_file_path = os.path.join(graph_full_dir_path, 'labels.csv')

        colocation_pair_features.to_csv(graph_features_csv_file_path, index=False)
        colocation_pair_normalized_performance.to_csv(graph_edges_csv_file_path, index=False)
        colocation_pair_labels.to_csv(graph_labels_csv_file_path, index=False)


if __name__ == '__main__':
    colocation_data_min, colocation_data_max = get_max_and_min_in_colocation_data()
    numeric_columns, solo_data_max, solo_data_min = get_max_and_min_in_solo_data()
    load_normalized_csv_and_process_data_for_graph(rtx_4090_colocation_csv_file, numeric_columns, solo_data_max, solo_data_min, colocation_data_min, colocation_data_max)
    load_normalized_csv_and_process_data_for_graph(rtx_3090_colocation_csv_file, numeric_columns, solo_data_max,
                                                   solo_data_min, colocation_data_min, colocation_data_max)
    load_normalized_csv_and_process_data_for_graph(titan_xp_colocation_csv_file, numeric_columns, solo_data_max,
                                                   solo_data_min, colocation_data_min, colocation_data_max)









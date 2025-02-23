import torch
import dgl
import os
import pandas as pd
from dgl.data import DGLDataset


class InterferenceDataset(DGLDataset):
    def __init__(self, dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        super(InterferenceDataset, self).__init__(name='interference_dataset')

    def process(self):
        self.graphs = []
        self.labels = []

        print("Start to process datasets.......")

        for graph_dir_name in os.listdir(self.dataset_dir_path):
            graph_dir_path = os.path.join(self.dataset_dir_path, graph_dir_name)

            node_features_path = os.path.join(graph_dir_path, 'node_features.csv')
            edges_path = os.path.join(graph_dir_path, 'edges.csv')
            labels_path = os.path.join(graph_dir_path, 'labels.csv')

            node_data = pd.read_csv(node_features_path)
            edge_data = pd.read_csv(edges_path)
            label_data = pd.read_csv(labels_path)

            node_name_data = node_data['Model name'].tolist()

            idx_map = {j: i for i, j in enumerate(node_name_data)}

            node_data['Model name'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                            inplace=True)

            edge_data['src'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)
            edge_data['dst'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)

            src = edge_data['src'].to_numpy()
            dst = edge_data['dst'].to_numpy()

            print()

            g = dgl.graph((src, dst))

            node_features = torch.tensor(node_data.loc[:, 'Batch size':'Tensor cores'].to_numpy(), dtype=torch.float)
            execution_time_label = label_data.loc[:, 'execution_time'].to_numpy()
            resource_util_label = label_data.loc[:, 'gpu_memory':'gpu_util'].to_numpy()
            labels = label_data.loc[:, 'execution_time':'gpu_util'].to_numpy()
            performance_degradation = torch.tensor(edge_data.loc[:, 'performance degradation'].to_numpy(), dtype=torch.float).reshape(-1, 1)

            g.ndata['node_features'] = node_features
            g.ndata['execution_time_label'] = torch.tensor(execution_time_label, dtype=torch.float)
            g.ndata['resource_util_label'] = torch.tensor(resource_util_label, dtype=torch.float)
            g.edata['performance_degradation'] = performance_degradation

            self.graphs.append(g)
            self.labels.append(labels)

        print("Dataset process completed")

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.graphs)


class InterferenceWithoutGPUFeaturesDataset(DGLDataset):
    def __init__(self, dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        super(InterferenceWithoutGPUFeaturesDataset, self).__init__(name='interference_dataset_without_gpu_features')

    def process(self):
        self.graphs = []
        self.labels = []

        print("Start to process datasets without GPU features.......")

        for graph_dir_name in os.listdir(self.dataset_dir_path):
            graph_dir_path = os.path.join(self.dataset_dir_path, graph_dir_name)

            node_features_path = os.path.join(graph_dir_path, 'node_features.csv')
            edges_path = os.path.join(graph_dir_path, 'edges.csv')
            labels_path = os.path.join(graph_dir_path, 'labels.csv')

            node_data = pd.read_csv(node_features_path)
            edge_data = pd.read_csv(edges_path)
            label_data = pd.read_csv(labels_path)

            node_name_data = node_data['Model name'].tolist()

            idx_map = {j: i for i, j in enumerate(node_name_data)}

            node_data['Model name'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                            inplace=True)

            edge_data['src'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)
            edge_data['dst'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)

            src = edge_data['src'].to_numpy()
            dst = edge_data['dst'].to_numpy()

            print()

            g = dgl.graph((src, dst))

            node_features = torch.tensor(node_data.loc[:, 'Batch size':'Tanh number'].to_numpy(), dtype=torch.float)
            execution_time_label = label_data.loc[:, 'execution_time'].to_numpy()
            resource_util_label = label_data.loc[:, 'gpu_memory':'gpu_util'].to_numpy()
            labels = label_data.loc[:, 'execution_time':'gpu_util'].to_numpy()
            performance_degradation = torch.tensor(edge_data.loc[:, 'performance degradation'].to_numpy(), dtype=torch.float).reshape(-1, 1)

            g.ndata['node_features'] = node_features
            g.ndata['execution_time_label'] = torch.tensor(execution_time_label, dtype=torch.float)
            g.ndata['resource_util_label'] = torch.tensor(resource_util_label, dtype=torch.float)
            g.edata['performance_degradation'] = performance_degradation

            self.graphs.append(g)
            self.labels.append(labels)

        print("Dataset process completed")

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.graphs)


class InterferenceWithoutModelFeaturesDataset(DGLDataset):
    def __init__(self, dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        super(InterferenceWithoutModelFeaturesDataset, self).__init__(name='interference_dataset_without_model_features')

    def process(self):
        self.graphs = []
        self.labels = []

        print("Start to process datasets without model features.......")

        for graph_dir_name in os.listdir(self.dataset_dir_path):
            graph_dir_path = os.path.join(self.dataset_dir_path, graph_dir_name)

            node_features_path = os.path.join(graph_dir_path, 'node_features.csv')
            edges_path = os.path.join(graph_dir_path, 'edges.csv')
            labels_path = os.path.join(graph_dir_path, 'labels.csv')

            node_data = pd.read_csv(node_features_path)
            edge_data = pd.read_csv(edges_path)
            label_data = pd.read_csv(labels_path)

            node_name_data = node_data['Model name'].tolist()

            idx_map = {j: i for i, j in enumerate(node_name_data)}

            node_data['Model name'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                            inplace=True)

            edge_data['src'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)
            edge_data['dst'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)

            src = edge_data['src'].to_numpy()
            dst = edge_data['dst'].to_numpy()

            print()

            g = dgl.graph((src, dst))

            node_features = torch.tensor(node_data.loc[:, ['Batch size', 'FLOPs (G)', 'Parmameters size (MB)', 'Activations size (MB)', 'Input size (MB)', 'Memory bandwidth (GB/s)', 'CUDA cores', 'SM count', 'Memory clock speed (MHz)', 'Tensor cores']].to_numpy(), dtype=torch.float)
            execution_time_label = label_data.loc[:, 'execution_time'].to_numpy()
            resource_util_label = label_data.loc[:, 'gpu_memory':'gpu_util'].to_numpy()
            labels = label_data.loc[:, 'execution_time':'gpu_util'].to_numpy()
            performance_degradation = torch.tensor(edge_data.loc[:, 'performance degradation'].to_numpy(), dtype=torch.float).reshape(-1, 1)

            g.ndata['node_features'] = node_features
            g.ndata['execution_time_label'] = torch.tensor(execution_time_label, dtype=torch.float)
            g.ndata['resource_util_label'] = torch.tensor(resource_util_label, dtype=torch.float)
            g.edata['performance_degradation'] = performance_degradation

            self.graphs.append(g)
            self.labels.append(labels)

        print("Dataset process completed")

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.graphs)


class InterferenceWithoutParameterFeaturesDataset(DGLDataset):
    def __init__(self, dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        super(InterferenceWithoutParameterFeaturesDataset, self).__init__(name='interference_dataset_without_hyperparameter_features')

    def process(self):
        self.graphs = []
        self.labels = []

        print("Start to process datasets without hyperparameter features.......")

        for graph_dir_name in os.listdir(self.dataset_dir_path):
            graph_dir_path = os.path.join(self.dataset_dir_path, graph_dir_name)

            node_features_path = os.path.join(graph_dir_path, 'node_features.csv')
            edges_path = os.path.join(graph_dir_path, 'edges.csv')
            labels_path = os.path.join(graph_dir_path, 'labels.csv')

            node_data = pd.read_csv(node_features_path)
            edge_data = pd.read_csv(edges_path)
            label_data = pd.read_csv(labels_path)

            node_name_data = node_data['Model name'].tolist()

            idx_map = {j: i for i, j in enumerate(node_name_data)}

            node_data['Model name'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                            inplace=True)

            edge_data['src'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)
            edge_data['dst'].replace([j for _, j in enumerate(idx_map)], [i for i, _ in enumerate(idx_map)],
                                     inplace=True)

            src = edge_data['src'].to_numpy()
            dst = edge_data['dst'].to_numpy()

            print()

            g = dgl.graph((src, dst))

            node_features = torch.tensor(node_data.loc[:, 'Mul number':'Tensor cores'].to_numpy(), dtype=torch.float)
            execution_time_label = label_data.loc[:, 'execution_time'].to_numpy()
            resource_util_label = label_data.loc[:, 'gpu_memory':'gpu_util'].to_numpy()
            labels = label_data.loc[:, 'execution_time':'gpu_util'].to_numpy()
            performance_degradation = torch.tensor(edge_data.loc[:, 'performance degradation'].to_numpy(), dtype=torch.float).reshape(-1, 1)

            g.ndata['node_features'] = node_features
            g.ndata['execution_time_label'] = torch.tensor(execution_time_label, dtype=torch.float)
            g.ndata['resource_util_label'] = torch.tensor(resource_util_label, dtype=torch.float)
            g.edata['performance_degradation'] = performance_degradation

            self.graphs.append(g)
            self.labels.append(labels)

        print("Dataset process completed")

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.graphs)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph

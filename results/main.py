import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
# from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

params = {
    'axes.labelsize':'22',
    'xtick.labelsize':'22',
    'ytick.labelsize':'22',
    'lines.linewidth':'2',
    'legend.fontsize':'22'
}

pylab.rcParams.update(params)


# Function to format the y-axis ticks as powers of 2
def format_func(y, _):
    return f"${np.power(2.0, int(np.log2(y)))}$" if y > 0 else "0"


def get_manuscript_fig_2a():
    fig_params = {
        'axes.labelsize': '32',
        'xtick.labelsize': '32',
        'ytick.labelsize': '32',
        'lines.linewidth': '1',
        'legend.fontsize': '32',
        'xtick.direction': 'in',
        'ytick.direction': 'in'

    }

    pylab.rcParams.update(fig_params)

    schedtune_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-util/schedtune-Shufflenet-gpu-util-data.csv'
    horus_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-util/horus-Shufflenet-gpu-util-data.csv'
    expected_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-util/expected-Shufflenet-gpu-util-data.csv'

    schedtune_gpuutil_workload_data = pd.read_csv(schedtune_csv_file_path, header=None)
    horus_gpuutil_workload_data = pd.read_csv(horus_csv_file_path, header=None)
    expected_gpuutil_workload_data = pd.read_csv(expected_csv_file_path, header=None)

    schedtune_gpuutil_workload_data_timestamp = schedtune_gpuutil_workload_data.loc[:, 0].values
    schedtune_workload_data_gpuutil = schedtune_gpuutil_workload_data.loc[:, 1].values

    horus_gpuutil_workload_data_timestamp = horus_gpuutil_workload_data.loc[:, 0].values
    horus_workload_data_gpuutil = horus_gpuutil_workload_data.loc[:, 1].values

    expected_gpuutil_workload_data_timestamp = expected_gpuutil_workload_data.loc[:, 0].values
    expected_workload_data_gpuutil = expected_gpuutil_workload_data.loc[:, 1].values

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    ax1.plot(schedtune_gpuutil_workload_data_timestamp, schedtune_workload_data_gpuutil,'b-', linewidth=4)
    ax1.plot(horus_gpuutil_workload_data_timestamp, horus_workload_data_gpuutil, 'r-', linewidth=4)
    ax1.plot(expected_gpuutil_workload_data_timestamp, expected_workload_data_gpuutil, 'g-', linewidth=4)

    plt.xlim([0, 4400])
    plt.ylim([0, 100])

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Horus', 'Expected'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax1.set_xlabel('Timestamp (s)', fontweight='bold')
    ax1.set_ylabel('GPU Utilization (%)', fontweight='bold')

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.savefig('Figure 2(a).jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def get_manuscript_fig_2b():
    fig_params = {
        'axes.labelsize': '32',
        'xtick.labelsize': '32',
        'ytick.labelsize': '32',
        'lines.linewidth': '1',
        'legend.fontsize': '32',
        'xtick.direction': 'in',
        'ytick.direction': 'in'

    }

    pylab.rcParams.update(fig_params)

    schedtune_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-mem/schedtune-Shufflenet-gpu-mem-data.csv'
    horus_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-mem/horus-Shufflenet-gpu-mem-data.csv'
    expected_csv_file_path = '/home/lzj/result2image/benchmarking_workload_data/gpu-mem/expected-Shufflenet-gpu-mem-data.csv'

    schedtune_gpumem_workload_data = pd.read_csv(schedtune_csv_file_path, header=None)
    horus_gpumem_workload_data = pd.read_csv(horus_csv_file_path, header=None)
    expected_gpumem_workload_data = pd.read_csv(expected_csv_file_path, header=None)

    schedtune_gpumem_workload_data_timestamp = schedtune_gpumem_workload_data.loc[:, 0].values
    schedtune_workload_data_gpumem = schedtune_gpumem_workload_data.loc[:, 1].values

    horus_gpumem_workload_data_timestamp = horus_gpumem_workload_data.loc[:, 0].values
    horus_workload_data_gpumem = horus_gpumem_workload_data.loc[:, 1].values

    expected_gpumem_workload_data_timestamp = expected_gpumem_workload_data.loc[:, 0].values
    expected_workload_data_gpumem = expected_gpumem_workload_data.loc[:, 1].values

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    ax1.plot(schedtune_gpumem_workload_data_timestamp, schedtune_workload_data_gpumem, 'b-',
             linewidth=4)
    ax1.plot(horus_gpumem_workload_data_timestamp, horus_workload_data_gpumem, 'r-', linewidth=4)
    ax1.plot(expected_gpumem_workload_data_timestamp, expected_workload_data_gpumem, 'g-',
             linewidth=4)

    plt.xlim([0, 4400])
    plt.ylim([0, 100])

    ax1.set_xlabel('Timestamp (s)', fontweight='bold')
    ax1.set_ylabel('GPU Memory Utilization (%)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='upper right', labels=['Schedtune', 'Horus', 'Expected'])
    plt.setp(legend.get_texts(), fontweight='bold')

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.savefig('Figure 2(b).jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def get_manuscript_fig_3():
    fig_params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '13',
        'ytick.labelsize': '25',
        'lines.linewidth': '2',
        'legend.fontsize': '18',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    x1 = [1.996, 1.069, 1.225, 1.83, 1.184, 2.033, 1.828, 2.12, 1.468]
    err1 = [0.021, 0.046, 0.042, 0.09, 0.026, 0.095, 0.073, 0.083, 0.054]
    x2 = [1.984, 1.017, 1.337, 1.835, 1.562, 1.654, 1.688, 1.514, 1.508]
    err2 = [0.019, 0.024, 0.032, 0.076, 0.042, 0.068, 0.059, 0.043, 0.058]

    x = np.arange(len(x1))

    width = 0.45

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    ax.bar(x - width / 2, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x + width / 2, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)

    ax.errorbar(x - width / 2, x1, yerr=err1, capsize=5, elinewidth=5, color='k', linestyle='None')
    ax.errorbar(x + width / 2, x2, yerr=err2, capsize=5, elinewidth=5, color='k', linestyle='None')

    ax.set_xticks(x,
               ['ShuffleNet', 'MobileNetV2', 'VGG19', 'InceptionV4', 'ResNet50','RNN-T', 'BERT', 'GNMT', 'MoFlow'])

    ax.set_ylabel('Performance Degradation', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='upper left', labels=['Co-located workload', 'WideResNet'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 3.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_4():
    fig_params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '13',
        'ytick.labelsize': '25',
        'lines.linewidth': '2',
        'legend.fontsize': '20',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    x1 = [1.59, 1.069, 1.225, 1.046, 1.155, 1.83, 1.184, 1.004, 1.732]
    err1 = [0.06, 0.046, 0.042, 0.05, 0.042, 0.09, 0.026, 0.022, 0.085]
    x2 = [1.152, 1.033, 1.021, 1.007, 1.018, 1.207, 1.032, 1.005, 1.215]
    err2 = [0.032, 0.028, 0.013, 0.011, 0.013, 0.015, 0.023, 0.01, 0.012]

    x = np.arange(len(x1))

    width = 0.45

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    ax.bar(x - width / 2, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x + width / 2, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)


    plt.errorbar(x - width / 2, x1, yerr=err1, capsize=5, elinewidth=5, color='k', linestyle='None')
    plt.errorbar(x + width / 2, x2, yerr=err2, capsize=5, elinewidth=5, color='k', linestyle='None')

    ax.set_xticks(x,
               ['ShuffleNet', 'MobilenetV2', 'VGG19', 'GoogleNet', 'DenseNet201', 'InceptionV4', 'ResNet50',
                'Attention92', 'Self'])

    ax.set_ylabel('Performance Degradation', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best', labels=['NVIDIA RTX 3090', 'NVIDIA RTX 4090'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 4.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_5():
    fig_params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '25',
        'ytick.labelsize': '25',
        'lines.linewidth': '3',
        'legend.fontsize': '20',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    x1 = [1.005, 1.231, 1.537, 1.777, 1.996, 2.064]
    err1 = [0.002, 0.013, 0.042, 0.032, 0.021, 0.013]
    x2 = [1.472, 1.678, 1.855, 1.957, 1.984, 1.987]
    err2 = [0.034, 0.056, 0.045, 0.023, 0.019, 0.016]

    x = np.arange(len(x1))

    width = 0.35

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    ax.bar(x - width / 2, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3, label='ShuffleNet')
    ax.bar(x + width / 2, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3, label='WideResNet')

    ax.errorbar(x - width / 2, x1, yerr=err1, capsize=5, elinewidth=5, color='k', linestyle='None')
    ax.errorbar(x + width / 2, x2, yerr=err2, capsize=5, elinewidth=5, color='k', linestyle='None')

    ax.set_xticks(x, ['16', '32', '64', '128', '256', '512'])

    ax.set_ylabel('Performance Degradation', fontweight='bold')
    ax.set_xlabel('Batch Size', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best')
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 5.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_6():
    fig_params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '25',
        'ytick.labelsize': '25',
        'lines.linewidth': '3',
        'legend.fontsize': '20',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    x1 = [2092.27, 1981.61, 1747.99, 1527.99, 1315.08, 1122.11]
    x2 = [2224.14, 2097.12, 1867.44, 1640.4, 1458.26, 1301.06]

    x = np.arange(len(x1))

    width = 0.35

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    ax.bar(x - width / 2, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3, label='ShuffleNet')
    ax.bar(x + width / 2, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3, label='WideResNet')

    ax.set_xticks(x,
               ['0', '200', '400', '600', '800', '1000'])

    ax.set_ylabel('Workload Completion Time (s)', fontweight='bold')
    ax.set_xlabel('Time Interval (s)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best')
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 6.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_7():
    fig_params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '25',
        'ytick.labelsize': '25',
        'lines.linewidth': '3',
        'legend.fontsize': '22',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    x = [1.59, 1.588, 1.069, 1.017, 1.225, 1.337, 1.046, 1.183, 1.155, 1.298, 1.83, 1.835, 1.184, 1.562, 1.004, 1.214,
         1.732, 1.735, 1.002, 1.01, 1.008, 1.003, 1.017, 1.019, 1.315, 1.283, 1.063, 1.423, 1.079, 1.529, 1.145, 1.527,
         1.205, 1.29, 1.057, 1.412, 1.056, 1.402, 1.846, 1.6, 1.816, 1.574, 1.75, 1.569]
    y = [1.463, 1.461, 1.03, 1.037, 1.095, 1.271, 1.034, 1.096, 1.042, 1.177, 1.627, 1.652, 1.076, 1.359, 1.01, 1.078,
         1.507, 1.476, 1.011, 1.06, 1.032, 1.049, 1.023, 1.046, 1.227, 1.177, 1.024, 1.421, 1.073, 1.436, 1.092, 1.494,
         1.154, 1.169, 1.038, 1.312, 1.029, 1.271, 1.619, 1.54, 1.667, 1.523, 1.61, 1.496]

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=300, marker='o', c='g', alpha=0.8)

    ax.set_ylabel('Resource Contention Degree', fontweight='bold')
    ax.set_xlabel('Performance Degradation', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best', labels=['DLT workload co-located with WideResNet'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.xlim(right=1.9)
    plt.ylim(top=1.8)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 7.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_11a():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [262.45, 473.16, 638.27, 296.79, 315.47, 430.21]
    # x2 = [330.61, 497.45, 784.8, 419.29, 228.66, 649.42]
    # x3 = [443.57, 584.8, 877.4, 488.42, 291.07, 740.65]
    # x4 = [349.69, 527.5, 737.45, 425, 248.44, 486.74]
    # x5 = [359.18, 522.96, 730.2, 426.73, 249.34, 484.52]

    x1 = [26.93, 9.52, 12.59, 30.45, 26.52, 11.21]
    x2 = [7.95, 4.88, 7.48, 1.74, 8.29, 34.03]
    x3 = [23.5, 11.82, 20.16, 14.46, 16.74, 52.86]
    x4 = [2.64, 0.87, 0.99, 0.41, 0.36, 0.46]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet121', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='upper left', labels=['Schedtune', 'Hydra', 'Optimus', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 11(a).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_11b():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [224.18, 421.15, 786.33, 330.41, 373.64, 411.9]
    # x2 = [357.76, 754.55, 1652.96, 563.01, 421.39, 456.49]
    # x3 = [398.37, 822.02, 1731.38, 708.98, 546.7, 564.06]
    # x4 = [388.06, 755.91, 1524.44, 644.9, 567.3, 460.68]
    # x5 = [394.39, 753.09, 1502.4, 653.27, 573.32, 444.98]

    x1 = [43.16, 44.08, 47.66, 49.42, 34.83, 7.43]
    x2 = [9.29, 0.19, 10.02, 13.82, 26.5, 2.59]
    x3 = [1.01, 9.15, 15.24, 8.53, 4.64, 26.76]
    x4 = [1.61, 0.37, 1.47, 1.28, 1.05, 3.53]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet161', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Hydra', 'Optimus', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 11(b).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_11c():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [252.04, 298.61, 582.58, 295.32, 495.87, 298.18]
    # x2 = [476.22, 771.39, 1309.41, 567.7, 922.25, 1043.18]
    # x3 = [510.51, 744.71, 1311.02, 573.02, 1028.86, 982.09]
    # x4 = [561.43, 721.91, 1332.38, 574.6, 1006.02, 717.2]
    # x5 = [527.24, 730.01, 1337.21, 574.32, 1051, 659.7]

    x1 = [52.2, 59.1, 56.43, 48.58, 52.82, 54.8]
    x2 = [9.68, 5.67, 2.08, 1.15, 12.25, 58.13]
    x3 = [3.17, 2.01, 1.96, 0.23, 2.11, 48.87]
    x4 = [6.48, 1.11, 0.36, 0.05, 4.28, 8.72]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet201', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Hydra', 'Optimus', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 11(c).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_12a():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [3693.3, 11658.85, 12922.52, 9302.49, 10623.78, 6269.22]
    # x2 = [2516.19, 14151.79, 15126.7, 14059.51, 12454.28, 8890.94]
    # x3 = [3683.5, 11556.78, 13374.91, 11048.17, 10446.42, 7729.23]
    # x4 = [3669.59, 12172.82, 13802.14, 8751.77, 10256.89, 6137.62]
    # x5 = [3724, 12160, 13586, 8714, 10388, 6030]

    x1 = [0.82, 4.12, 4.88, 6.75, 2.27, 3.97]
    x2 = [32.43, 16.38, 11.34, 61.34, 19.89, 47.45]
    x3 = [1.09, 4.96, 1.55, 26.79, 0.56, 28.18]
    x4 = [1.46, 0.11, 1.59, 0.43, 1.26, 1.78]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet121', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 12(a).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_12b():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [3616.13, 11492.31, 12854.03, 9276.28, 10548.71, 6354.43]
    # x2 = [2516.19, 12829.95, 15126.7, 14059.51, 12454.28, 8890.94]
    # x3 = [3479.81, 10671.53, 13232.86, 10267.68, 10529.2, 7385.89]
    # x4 = [3621.6, 11679.13, 13764.41, 8550.57, 10738.34, 6738.47]
    # x5 = [3633, 11533, 13495, 8623, 10851, 6866]

    x1 = [0.46, 0.35, 4.75, 7.58, 2.79, 7.45]
    x2 = [30.74, 11.25, 12.09, 63.05, 14.78, 29.49]
    x3 = [4.22, 7.47, 1.94, 19.07, 2.97, 7.57]
    x4 = [0.31, 1.27, 2, 0.84, 1.04, 1.86]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet161', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 12(b).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_12c():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [3146.25, 4715.92, 7126.43, 4732.8, 9248.06, 4830.32]
    # x2 = [2516.19, 5208.33, 7563.35, 7029.76, 6227.14, 4445.47]
    # x3 = [2890.62, 4635.01, 6901.39, 5276.88, 9190.35, 3822.01]
    # x4 = [3052.92, 5113.76, 7178.92, 4568, 8024.82, 5237.45]
    # x5 = [3081, 5175, 7139, 4473, 8145, 5382]

    x1 = [2.12, 8.87, 0.18, 5.81, 13.54, 10.25]
    x2 = [18.33, 0.64, 5.94, 57.16, 23.55, 17.4]
    x3 = [6.18, 10.43, 3.33, 17.97, 12.83, 28.99]
    x4 = [0.91, 1.18, 0.56, 2.12, 1.48, 2.69]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet201', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 12(c).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_13a():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [27.736, 32.563, 58.628, 56.94, 45.683, 19.494]
    # x2 = [32.952, 44.069, 46.667, 55.825, 42.164, 25.825]
    # x3 = [29.871, 37.568, 61.776, 58.254, 41.027, 43.997]
    # x4 = [25.548, 30.276, 49.872, 41.552, 40.238, 10.543]
    # x5 = [24.364, 28, 51.597, 42.676, 41.757, 11.029]

    x1 = [13.84, 16.3, 13.63, 33.42, 9.4, 76.75]
    x2 = [35.25, 57.39, 9.55, 30.81, 0.97, 134.16]
    x3 = [22.6, 34.17, 19.73, 36.5, 1.75, 298.92]
    x4 = [4.86, 8.13, 3.34, 2.63, 3.64, 4.41]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet121', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 13(a).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_13b():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [50.992, 68.398, 85.309, 82.746, 52.531, 56.835]
    # x2 = [50.592, 61.993, 73.595, 71.595, 50.255, 52.054]
    # x3 = [46.683, 75.16, 85.496, 82.403, 66.669, 64.719]
    # x4 = [40.685, 44.473, 48.605, 46.967, 48.529, 23.702]
    # x5 = [41.154, 46.629, 50.792, 45.745, 49.963, 23.843]

    x1 = [23.91, 46.69, 67.96, 80.89, 5.14, 138.37]
    x2 = [22.93, 32.95, 44.89, 60.88, 0.58, 118.32]
    x3 = [13.43, 61.19, 68.33, 80.14, 33.44, 171.44]
    x4 = [1.14, 4.62, 4.31, 2.67, 2.87, 0.59]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet161', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 13(b).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def get_manuscript_fig_13c():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '25',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '25',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    # x1 = [77.423, 75.563, 94.662, 93.129, 95.286, 79.197]
    # x2 = [66.615, 76.354, 97.983, 95.983, 94.337, 76.368]
    # x3 = [76.789, 79.49, 93.844, 93.886, 93.463, 87.831]
    # x4 = [46.638, 30.841, 50.715, 46.605, 45.328, 40.625]
    # x5 = [44.522, 31.996, 49.506, 46.676, 47.53, 41.354]

    x1 = [73.9, 136.16, 91.21, 99.52, 100.48, 91.51]
    x2 = [49.62, 138.64, 97.92, 109.92, 98.48, 84.67]
    x3 = [72.47, 148.44, 89.56, 101.14, 96.64, 112.39]
    x4 = [4.75, 3.61, 2.44, 0.15, 4.63, 1.76]

    x = np.arange(len(x1))

    width = 0.18

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    ax.bar(x - 1.5 * width, x1, width=width, color='c', edgecolor='black', hatch='/', linewidth=3)
    ax.bar(x - 0.5 * width, x2, width=width, color='g', edgecolor='black', hatch='\\', linewidth=3)
    ax.bar(x + 0.5 * width, x3, width=width, color='r', edgecolor='black', hatch='.', linewidth=3)
    ax.bar(x + 1.5 * width, x4, width=width, color='y', edgecolor='black', hatch='o', linewidth=3)
    # ax.bar(x + width * 2, x5, width=width, color='y', edgecolor='black', hatch='x', linewidth=3)

    ax.set_xticks(x,
               ['VGG19', 'DenseNet201', 'InceptionV3', 'ResNet50', 'BERT', 'MoFlow'], fontweight='bold')

    ax.set_ylabel('Prediction Error (%)', fontweight='bold')

    # 设置对数纵坐标，并指定间隔
    ax.set_yscale('log', base=2)

    # Add a legend
    legend = plt.legend(loc='lower right', labels=['Schedtune', 'Horus', 'Liquid', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 13(c).jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


# def get_manuscript_fig_13():
#     fig_params = {
#         'axes.labelsize': '20',
#         'xtick.labelsize': '20',
#         'ytick.labelsize': '20',
#         'lines.linewidth': '3',
#         'legend.fontsize': '20',
#         'xtick.direction': 'in',
#         'ytick.direction': 'in'
#     }
#
#     pylab.rcParams.update(fig_params)
#
#     schedtune_list = [119.78, 188.73, 476.85, 586.53, 995.67, 506, 484.73, 1264.33, 30.96,
#                       68.3, 77.48, 90.46, 92.9, 28.87, 222.92, 155.79, 41.64, 71.52, 113.45,
#                       106.08, 181.92, 117.67, 124.72, 138.25, 1461.99, 611.53, 4681, 3000.02,
#                       42.65, 68.42, 87.81, 103.87, 228.01, 37.67, 235.39, 204.07, 42.57, 80.67,
#                       124.07, 123.74, 147.4, 122.94, 136.99, 209.31, 3185.99, 3706.7, 78.52, 276.48,
#                       399.31, 259.28, 670.02]
#
#     horus_list = [166.82, 194, 416.93, 519.39, 650.39, 414.78, 881.28, 1266.79, 28.84, 72.18, 102.21,
#                   73.72, 103.3, 29.04, 146.66, 155.48, 43.26, 91.86, 66.57, 81.11, 93.57, 163.29, 163.02,
#                   139.02, 814.01, 1243.72, 4111, 3684.48, 43.64, 69.21, 117.63, 134.32, 142.24,
#                   33.86, 236.97, 218.12, 48.64, 83.65, 128.65, 124.03, 162.31, 159.2, 188.47, 143.12,
#                   2863.49, 3898.86, 175.54, 332.3, 193.75, 267.6, 759.3]
#
#     liquid_list = [111.83, 416.38, 305.72, 370.67, 486.83, 253.9, 498.97, 2173.49, 35.06, 54.1, 61.98,
#                    90.7, 91.93, 29.13, 152.27, 152.57, 40.46, 74.58, 88.89, 85.42, 114.94, 117.78, 114.5,
#                    137.89, 1616.15, 726.92, 5251, 2910.28, 50.23, 50.01, 93.51, 106.1, 127.95, 26.34,
#                    218.25, 183.64, 56.53, 68.04, 98.83, 81.45, 144.41, 132.69, 147.86, 202.55, 3604.55,
#                    2748.63, 135.93, 187.12, 290.71, 298.55, 1213.07]
#
#     volcano_list = [104.19, 281.81, 279.77, 520.67, 750.82, 417.41, 468.2, 1334.41, 28.81, 54.83, 61.96,
#                     81.83, 128.81, 35.78, 239.19, 262.36, 60.65, 72.68, 94.93, 80.1, 134.59, 109.72, 188.23,
#                     205.99, 796.95, 1646.71, 3957.7, 3656.64, 37.88, 60.09, 87.08, 100.01, 163.69, 48.6,
#                     235.57, 220.55, 62.13, 101.64, 113.32, 142.56, 146.51, 146.66, 210.77, 213.98, 2877.52,
#                     4861, 237.74, 383.59, 512.67, 843.08, 1071.97]
#
#     ours_list = [134.33, 192.45, 341.27, 401.43, 458.55, 259.24, 551.18, 960.08, 37.32, 53.65, 79.3, 79.45,
#                  119.99, 29.16, 167.73, 164.27, 53.23, 59.57, 79.53, 119.84, 101.64, 105.93, 147.95, 150.57,
#                  319.75, 548.78, 3421, 2899.83, 45.92, 49.79, 72.95, 123.75, 126.96, 42.85, 248.12, 182.52,
#                  42.79, 58.34, 126.59, 82.51, 118.98, 113.92, 133.88, 179.19, 3271, 3181.01, 92.59, 330.23,
#                  465.62, 670.72, 391.82]
#
#     # Function to compute the CDF
#     def compute_cdf(data):
#         sorted_data = np.sort(data)
#         cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#         return sorted_data, cdf
#
#     schedtune_nd = np.array(schedtune_list)
#     horus_nd = np.array(horus_list)
#     liquid_nd = np.array(liquid_list)
#     volcano_nd = np.array(volcano_list)
#     ours_nd = np.array(ours_list)
#
#     # Compute the CDF for both data sets
#     schedtune_sorted, cdf_schedtune = compute_cdf(schedtune_nd)
#     horus_sorted, cdf_horus = compute_cdf(horus_nd)
#     liquid_sorted, cdf_liquid = compute_cdf(liquid_nd)
#     volcano_sorted, cdf_volcano = compute_cdf(volcano_nd)
#     ours_sorted, cdf_ours = compute_cdf(ours_nd)
#
#     # schedtune_percentile_90 = np.percentile(schedtune_sorted, 90)
#     # horus_percentile_90 = np.percentile(horus_sorted, 90)
#     # liquid_percentile_90 = np.percentile(liquid_sorted, 90)
#     # volcano_percentile_90 = np.percentile(volcano_sorted, 90)
#     # ours_percentile_90 = np.percentile(ours_sorted, 90)
#
#     # print(schedtune_percentile_90)
#     # print(horus_percentile_90)
#     # print(liquid_percentile_90)
#     # print(volcano_percentile_90)
#     # print(ours_percentile_90)
#
#     # Create the plot
#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(111)
#
#     # Plot the CDF with different line styles
#     ax.plot(schedtune_sorted, cdf_schedtune, 'c-', label='Schedtune', linewidth=4)
#     ax.plot(horus_sorted, cdf_horus, 'g-', label='Horus', linewidth=4)
#     ax.plot(liquid_sorted, cdf_liquid, 'r-', label='Liquid', linewidth=4)
#     ax.plot(volcano_sorted, cdf_volcano, 'm-', label='Volcano', linewidth=4)
#     ax.plot(ours_sorted, cdf_ours, 'y--', label='Ours', linewidth=4)
#
#     ax.set_ylabel('累积分布函数', fontweight='bold', fontdict={'family': 'SimSun'})
#     ax.set_xlabel('深度学习训练工作负载完成时间（秒）', fontweight='bold', fontdict={'family': 'SimSun'})
#
#     # Add a legend
#     legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Volcano', '本方法'], prop={'family': 'SimSun'})
#     plt.setp(legend.get_texts(), fontweight='bold')
#
#     # legend = ax.legend(loc='best')
#     # plt.setp(legend.get_texts(), fontweight='bold')
#
#     ax.grid(True, linestyle='--', linewidth=0.5)
#
#     # Set bold font for the x and y axis ticks
#     # plt.xticks(fontweight='bold')
#     # plt.yticks(fontweight='bold')
#
#     # Save the figure with high resolution
#     plt.savefig('Figure 13.jpg', dpi=300, bbox_inches='tight')
#
#     # Show the plot
#     plt.show()


def get_manuscript_fig_14():
    fig_params = {
        'axes.labelsize': '30',
        'xtick.labelsize': '30',
        'ytick.labelsize': '30',
        'lines.linewidth': '5',
        'legend.fontsize': '30',
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    pylab.rcParams.update(fig_params)

    schedtune_list = [119.78, 188.73, 476.85, 586.53, 995.67, 506, 484.73, 1264.33, 30.96,
                      68.3, 77.48, 90.46, 92.9, 28.87, 222.92, 155.79, 41.64, 71.52, 113.45,
                      106.08, 181.92, 117.67, 124.72, 138.25, 1461.99, 611.53, 4681, 3000.02,
                      42.65, 68.42, 87.81, 103.87, 228.01, 37.67, 235.39, 204.07, 42.57, 80.67,
                      124.07, 123.74, 147.4, 122.94, 136.99, 209.31, 3185.99, 3706.7, 78.52, 276.48,
                      399.31, 259.28, 670.02]

    horus_list = [166.82, 194, 416.93, 519.39, 650.39, 414.78, 881.28, 1266.79, 28.84, 72.18, 102.21,
                  73.72, 103.3, 29.04, 146.66, 155.48, 43.26, 91.86, 66.57, 81.11, 93.57, 163.29, 163.02,
                  139.02, 814.01, 1243.72, 4111, 3684.48, 43.64, 69.21, 117.63, 134.32, 142.24,
                  33.86, 236.97, 218.12, 48.64, 83.65, 128.65, 124.03, 162.31, 159.2, 188.47, 143.12,
                  2863.49, 3898.86, 175.54, 332.3, 193.75, 267.6, 759.3]

    liquid_list = [111.83, 416.38, 305.72, 370.67, 486.83, 253.9, 498.97, 2173.49, 35.06, 54.1, 61.98,
                   90.7, 91.93, 29.13, 152.27, 152.57, 40.46, 74.58, 88.89, 85.42, 114.94, 117.78, 114.5,
                   137.89, 1616.15, 726.92, 5251, 2910.28, 50.23, 50.01, 93.51, 106.1, 127.95, 26.34,
                   218.25, 183.64, 56.53, 68.04, 98.83, 81.45, 144.41, 132.69, 147.86, 202.55, 3604.55,
                   2748.63, 135.93, 187.12, 290.71, 298.55, 1213.07]

    volcano_list = [104.19, 281.81, 279.77, 520.67, 750.82, 417.41, 468.2, 1334.41, 28.81, 54.83, 61.96,
                    81.83, 128.81, 35.78, 239.19, 262.36, 60.65, 72.68, 94.93, 80.1, 134.59, 109.72, 188.23,
                    205.99, 796.95, 1646.71, 3957.7, 3656.64, 37.88, 60.09, 87.08, 100.01, 163.69, 48.6,
                    235.57, 220.55, 62.13, 101.64, 113.32, 142.56, 146.51, 146.66, 210.77, 213.98, 2877.52,
                    4861, 237.74, 383.59, 512.67, 843.08, 1071.97]

    ours_list = [134.33, 192.45, 341.27, 401.43, 458.55, 259.24, 551.18, 960.08, 37.32, 53.65, 79.3, 79.45,
                 119.99, 29.16, 167.73, 164.27, 53.23, 59.57, 79.53, 119.84, 101.64, 105.93, 147.95, 150.57,
                 319.75, 548.78, 3421, 2899.83, 45.92, 49.79, 72.95, 123.75, 126.96, 42.85, 248.12, 182.52,
                 42.79, 58.34, 126.59, 82.51, 118.98, 113.92, 133.88, 179.19, 3271, 3181.01, 92.59, 330.23,
                 465.62, 670.72, 391.82]

    # Create the plot
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)

    sns.kdeplot(data=schedtune_list, ax=ax, fill=True, color="c", label='Schedtune', linewidth=8)
    sns.kdeplot(data=horus_list, ax=ax, fill=True, color="g", label='Horus', linewidth=8)
    sns.kdeplot(data=liquid_list, ax=ax, fill=True, color="r", label='Liquid', linewidth=8)
    sns.kdeplot(data=volcano_list, ax=ax, fill=True, color="m", label='Volcano', linewidth=8)
    sns.kdeplot(data=ours_list, ax=ax, fill=True, color="y", label='Ours', linewidth=8)

    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_xlabel('Workload Completion Time (s)', fontweight='bold')

    plt.xlim([0, 5500])

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Volcano', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax.grid(True, linestyle='--', linewidth=0.5)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    # Save the figure with high resolution
    plt.savefig('Figure 14.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()


def get_manuscript_fig_15a():
    fig_params = {
        'axes.labelsize': '23',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        'lines.linewidth': '1',
        'legend.fontsize': '20',
        'xtick.direction': 'in',
        'ytick.direction': 'in'

    }

    pylab.rcParams.update(fig_params)

    schedtune_gpu_mem_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/schedtune/schedtune-gpu-mem-data-3090-node-1and-2.csv'
    schedtune_gpu_mem_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/schedtune/schedtune-gpu-mem-data-4090-node.csv'
    schedtune_gpu_mem_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/schedtune/schedtune-gpu-mem-data-3090-node-3and-4.csv'
    schedtune_gpu_mem_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/schedtune/schedtune-gpu-mem-data-titan-node.csv'

    schedtune_gpu_mem_3090_node_part_1_df = pd.read_csv(schedtune_gpu_mem_3090_node_part_1_csv_file_path, header=None)
    schedtune_gpu_mem_4090_node_df = pd.read_csv(schedtune_gpu_mem_4090_node_csv_file_path, header=None)
    schedtune_gpu_mem_3090_node_part_2_df = pd.read_csv(schedtune_gpu_mem_3090_node_part_2_csv_file_path, header=None)
    schedtune_gpu_mem_titan_node_df = pd.read_csv(schedtune_gpu_mem_titan_node_csv_file_path, header=None)

    schedtune_gpu_mem_3090_node_part_1_timestamp = schedtune_gpu_mem_3090_node_part_1_df.loc[:, 0].values
    schedtune_gpu_mem_3090_node_part_1_data = schedtune_gpu_mem_3090_node_part_1_df.loc[:, 1].values

    schedtune_gpu_mem_4090_node_timestamp = schedtune_gpu_mem_4090_node_df.loc[:, 0].values
    schedtune_gpu_mem_4090_node_data = schedtune_gpu_mem_4090_node_df.loc[:, 1].values

    schedtune_gpu_mem_3090_node_part_2_timestamp = schedtune_gpu_mem_3090_node_part_2_df.loc[:, 0].values
    schedtune_gpu_mem_3090_node_part_2_data = schedtune_gpu_mem_3090_node_part_2_df.loc[:, 1].values

    schedtune_gpu_mem_titan_node_timestamp = schedtune_gpu_mem_titan_node_df.loc[:, 0].values[:-1]
    schedtune_gpu_mem_titan_node_data = schedtune_gpu_mem_titan_node_df.loc[:, 1].values[:-1]

    schedtune_gpu_mem_ave_data = (schedtune_gpu_mem_3090_node_part_1_data + schedtune_gpu_mem_4090_node_data + schedtune_gpu_mem_3090_node_part_2_data + schedtune_gpu_mem_titan_node_data) / 4

    horus_gpu_mem_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/horus/horus-gpu-mem-data-3090-node-1-and-2.csv'
    horus_gpu_mem_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/horus/horus-gpu-mem-data-4090-node.csv'
    horus_gpu_mem_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/horus/horus-gpu-mem-data-3090-node-3-and-4.csv'
    horus_gpu_mem_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/horus/horus-gpu-mem-data-titan-node.csv'

    horus_gpu_mem_3090_node_part_1_df = pd.read_csv(horus_gpu_mem_3090_node_part_1_csv_file_path, header=None)
    horus_gpu_mem_4090_node_df = pd.read_csv(horus_gpu_mem_4090_node_csv_file_path, header=None)
    horus_gpu_mem_3090_node_part_2_df = pd.read_csv(horus_gpu_mem_3090_node_part_2_csv_file_path, header=None)
    horus_gpu_mem_titan_node_df = pd.read_csv(horus_gpu_mem_titan_node_csv_file_path, header=None)

    horus_gpu_mem_3090_node_part_1_timestamp = horus_gpu_mem_3090_node_part_1_df.loc[:, 0].values
    horus_gpu_mem_3090_node_part_1_data = horus_gpu_mem_3090_node_part_1_df.loc[:, 1].values

    horus_gpu_mem_4090_node_timestamp = horus_gpu_mem_4090_node_df.loc[:, 0].values
    horus_gpu_mem_4090_node_data = horus_gpu_mem_4090_node_df.loc[:, 1].values

    horus_gpu_mem_3090_node_part_2_timestamp = horus_gpu_mem_3090_node_part_2_df.loc[:, 0].values[:-1]
    horus_gpu_mem_3090_node_part_2_data = horus_gpu_mem_3090_node_part_2_df.loc[:, 1].values[:-1]

    horus_gpu_mem_titan_node_timestamp = horus_gpu_mem_titan_node_df.loc[:, 0].values[:-1]
    horus_gpu_mem_titan_node_data = horus_gpu_mem_titan_node_df.loc[:, 1].values[:-1]

    horus_gpu_mem_ave_data = (horus_gpu_mem_3090_node_part_1_data + horus_gpu_mem_4090_node_data + horus_gpu_mem_3090_node_part_2_data + horus_gpu_mem_titan_node_data) / 4

    liquid_gpu_mem_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/liquid/liquid-gpu-mem-data-3090-node-1-and-2.csv'
    liquid_gpu_mem_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/liquid/liquid-gpu-mem-data-4090-node.csv'
    liquid_gpu_mem_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/liquid/liquid-gpu-mem-data-3090-node-3-and-4.csv'
    liquid_gpu_mem_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/liquid/liquid-gpu-mem-data-titan-node.csv'

    liquid_gpu_mem_3090_node_part_1_df = pd.read_csv(liquid_gpu_mem_3090_node_part_1_csv_file_path, header=None)
    liquid_gpu_mem_4090_node_df = pd.read_csv(liquid_gpu_mem_4090_node_csv_file_path, header=None)
    liquid_gpu_mem_3090_node_part_2_df = pd.read_csv(liquid_gpu_mem_3090_node_part_2_csv_file_path, header=None)
    liquid_gpu_mem_titan_node_df = pd.read_csv(liquid_gpu_mem_titan_node_csv_file_path, header=None)

    liquid_gpu_mem_3090_node_part_1_timestamp = liquid_gpu_mem_3090_node_part_1_df.loc[:, 0].values
    liquid_gpu_mem_3090_node_part_1_data = liquid_gpu_mem_3090_node_part_1_df.loc[:, 1].values

    liquid_gpu_mem_4090_node_timestamp = liquid_gpu_mem_4090_node_df.loc[:, 0].values[:-1]
    liquid_gpu_mem_4090_node_data = liquid_gpu_mem_4090_node_df.loc[:, 1].values[:-1]

    liquid_gpu_mem_3090_node_part_2_timestamp = liquid_gpu_mem_3090_node_part_2_df.loc[:, 0].values[:-1]
    liquid_gpu_mem_3090_node_part_2_data = liquid_gpu_mem_3090_node_part_2_df.loc[:, 1].values[:-1]

    liquid_gpu_mem_titan_node_timestamp = liquid_gpu_mem_titan_node_df.loc[:, 0].values[:-2]
    liquid_gpu_mem_titan_node_data = liquid_gpu_mem_titan_node_df.loc[:, 1].values[:-2]

    liquid_gpu_mem_ave_data = (liquid_gpu_mem_3090_node_part_1_data + liquid_gpu_mem_4090_node_data + liquid_gpu_mem_3090_node_part_2_data + liquid_gpu_mem_titan_node_data) / 4

    volcano_gpu_mem_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/volcano/volcano-gpu-mem-data-3090-node-1-and-2.csv'
    volcano_gpu_mem_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/volcano/volcano-gpu-mem-data-4090-node.csv'
    volcano_gpu_mem_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/volcano/volcano-gpu-mem-data-3090-node-3-and-4.csv'
    volcano_gpu_mem_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/volcano/volcano-gpu-mem-data-titan-node.csv'

    volcano_gpu_mem_3090_node_part_1_df = pd.read_csv(volcano_gpu_mem_3090_node_part_1_csv_file_path, header=None)
    volcano_gpu_mem_4090_node_df = pd.read_csv(volcano_gpu_mem_4090_node_csv_file_path, header=None)
    volcano_gpu_mem_3090_node_part_2_df = pd.read_csv(volcano_gpu_mem_3090_node_part_2_csv_file_path, header=None)
    volcano_gpu_mem_titan_node_df = pd.read_csv(volcano_gpu_mem_titan_node_csv_file_path, header=None)

    volcano_gpu_mem_3090_node_part_1_timestamp = volcano_gpu_mem_3090_node_part_1_df.loc[:, 0].values
    volcano_gpu_mem_3090_node_part_1_data = volcano_gpu_mem_3090_node_part_1_df.loc[:, 1].values

    volcano_gpu_mem_4090_node_timestamp = volcano_gpu_mem_4090_node_df.loc[:, 0].values[:-1]
    volcano_gpu_mem_4090_node_data = volcano_gpu_mem_4090_node_df.loc[:, 1].values[:-1]

    volcano_gpu_mem_3090_node_part_2_timestamp = volcano_gpu_mem_3090_node_part_2_df.loc[:, 0].values[:-2]
    volcano_gpu_mem_3090_node_part_2_data = volcano_gpu_mem_3090_node_part_2_df.loc[:, 1].values[:-2]

    volcano_gpu_mem_titan_node_timestamp = volcano_gpu_mem_titan_node_df.loc[:, 0].values[:-3]
    volcano_gpu_mem_titan_node_data = volcano_gpu_mem_titan_node_df.loc[:, 1].values[:-3]

    volcano_gpu_mem_ave_data = (volcano_gpu_mem_3090_node_part_1_data + volcano_gpu_mem_4090_node_data + volcano_gpu_mem_3090_node_part_2_data + volcano_gpu_mem_titan_node_data) / 4

    ours_gpu_mem_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/ours/isacpp-gpu-mem-data-3090-node-1-and-2.csv'
    ours_gpu_mem_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/ours/isacpp-gpu-mem-data-4090-node.csv'
    ours_gpu_mem_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/ours/isacpp-gpu-mem-data-3090-node-3-and-4.csv'
    ours_gpu_mem_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-mem/ours/isacpp-gpu-mem-data-titan-node.csv'

    ours_gpu_mem_3090_node_part_1_df = pd.read_csv(ours_gpu_mem_3090_node_part_1_csv_file_path, header=None)
    ours_gpu_mem_4090_node_df = pd.read_csv(ours_gpu_mem_4090_node_csv_file_path, header=None)
    ours_gpu_mem_3090_node_part_2_df = pd.read_csv(ours_gpu_mem_3090_node_part_2_csv_file_path, header=None)
    ours_gpu_mem_titan_node_df = pd.read_csv(ours_gpu_mem_titan_node_csv_file_path, header=None)

    ours_gpu_mem_3090_node_part_1_timestamp = ours_gpu_mem_3090_node_part_1_df.loc[:, 0].values
    ours_gpu_mem_3090_node_part_1_data = ours_gpu_mem_3090_node_part_1_df.loc[:, 1].values

    ours_gpu_mem_4090_node_timestamp = ours_gpu_mem_4090_node_df.loc[:, 0].values
    ours_gpu_mem_4090_node_data = ours_gpu_mem_4090_node_df.loc[:, 1].values

    ours_gpu_mem_3090_node_part_2_timestamp = ours_gpu_mem_3090_node_part_2_df.loc[:, 0].values
    ours_gpu_mem_3090_node_part_2_data = ours_gpu_mem_3090_node_part_2_df.loc[:, 1].values

    ours_gpu_mem_titan_node_timestamp = ours_gpu_mem_titan_node_df.loc[:, 0].values
    ours_gpu_mem_titan_node_data = ours_gpu_mem_titan_node_df.loc[:, 1].values

    ours_gpu_mem_ave_data = (ours_gpu_mem_3090_node_part_1_data + ours_gpu_mem_4090_node_data + ours_gpu_mem_3090_node_part_2_data + ours_gpu_mem_titan_node_data) / 4

    # schedtune_max = np.max(schedtune_gpu_mem_ave_data)
    # horus_max = np.max(horus_gpu_mem_ave_data)
    # liquid_max = np.max(liquid_gpu_mem_ave_data)
    # volcano_max = np.max(volcano_gpu_mem_ave_data)
    # ours_max = np.max(ours_gpu_mem_ave_data)
    #
    # schedtune_mean = np.mean(schedtune_gpu_mem_ave_data)
    # horus_mean = np.mean(horus_gpu_mem_ave_data)
    # liquid_mean = np.mean(liquid_gpu_mem_ave_data)
    # volcano_mean = np.mean(volcano_gpu_mem_ave_data)
    # ours_mean = np.mean(ours_gpu_mem_ave_data)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(schedtune_gpu_mem_3090_node_part_1_timestamp, schedtune_gpu_mem_ave_data, 'c-', linewidth=2)
    ax1.plot(horus_gpu_mem_3090_node_part_1_timestamp, horus_gpu_mem_ave_data, 'g-', linewidth=2)
    ax1.plot(liquid_gpu_mem_3090_node_part_1_timestamp, liquid_gpu_mem_ave_data, 'r-', linewidth=2)
    ax1.plot(volcano_gpu_mem_3090_node_part_1_timestamp, volcano_gpu_mem_ave_data, 'm-', linewidth=2)
    ax1.plot(ours_gpu_mem_3090_node_part_1_timestamp, ours_gpu_mem_ave_data, 'y--', linewidth=2)

    ax1.set_xlabel('Timestamp (s)', fontweight='bold')
    ax1.set_ylabel('Resource Utilization (%)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Volcano', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.xlim([0, 5400])
    plt.ylim([0, 65])

    plt.savefig('Figure 15(a).jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def get_manuscript_fig_15b():
    fig_params = {
        'axes.labelsize': '23',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        'lines.linewidth': '1',
        'legend.fontsize': '20',
        'xtick.direction': 'in',
        'ytick.direction': 'in'

    }

    pylab.rcParams.update(fig_params)

    schedtune_gpu_util_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/schedtune/schedtune-gpu-util-data-3090-node-1-and-2.csv'
    schedtune_gpu_util_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/schedtune/schedtune-gpu-util-data-4090-node.csv'
    schedtune_gpu_util_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/schedtune/schedtune-gpu-util-data-3090-node-3-and-4.csv'
    schedtune_gpu_util_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/schedtune/schedtune-gpu-util-data-titan-node.csv'

    schedtune_gpu_util_3090_node_part_1_df = pd.read_csv(schedtune_gpu_util_3090_node_part_1_csv_file_path, header=None)
    schedtune_gpu_util_4090_node_df = pd.read_csv(schedtune_gpu_util_4090_node_csv_file_path, header=None)
    schedtune_gpu_util_3090_node_part_2_df = pd.read_csv(schedtune_gpu_util_3090_node_part_2_csv_file_path, header=None)
    schedtune_gpu_util_titan_node_df = pd.read_csv(schedtune_gpu_util_titan_node_csv_file_path, header=None)

    schedtune_gpu_util_3090_node_part_1_timestamp = schedtune_gpu_util_3090_node_part_1_df.loc[:, 0].values
    schedtune_gpu_util_3090_node_part_1_data = schedtune_gpu_util_3090_node_part_1_df.loc[:, 1].values

    schedtune_gpu_util_4090_node_timestamp = schedtune_gpu_util_4090_node_df.loc[:, 0].values
    schedtune_gpu_util_4090_node_data = schedtune_gpu_util_4090_node_df.loc[:, 1].values

    schedtune_gpu_util_3090_node_part_2_timestamp = schedtune_gpu_util_3090_node_part_2_df.loc[:, 0].values
    schedtune_gpu_util_3090_node_part_2_data = schedtune_gpu_util_3090_node_part_2_df.loc[:, 1].values

    schedtune_gpu_util_titan_node_timestamp = schedtune_gpu_util_titan_node_df.loc[:, 0].values[:-1]
    schedtune_gpu_util_titan_node_data = schedtune_gpu_util_titan_node_df.loc[:, 1].values[:-1]

    schedtune_gpu_util_ave_data = (schedtune_gpu_util_3090_node_part_1_data + schedtune_gpu_util_4090_node_data + schedtune_gpu_util_3090_node_part_2_data + schedtune_gpu_util_titan_node_data) / 4

    horus_gpu_util_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/horus/horus-gpu-util-data-3090-node-1-and-2.csv'
    horus_gpu_util_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/horus/horus-gpu-util-data-4090-node.csv'
    horus_gpu_util_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/horus/horus-gpu-util-data-3090-node-3-and-4.csv'
    horus_gpu_util_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/horus/horus-gpu-util-data-titan-node.csv'

    horus_gpu_util_3090_node_part_1_df = pd.read_csv(horus_gpu_util_3090_node_part_1_csv_file_path, header=None)
    horus_gpu_util_4090_node_df = pd.read_csv(horus_gpu_util_4090_node_csv_file_path, header=None)
    horus_gpu_util_3090_node_part_2_df = pd.read_csv(horus_gpu_util_3090_node_part_2_csv_file_path, header=None)
    horus_gpu_util_titan_node_df = pd.read_csv(horus_gpu_util_titan_node_csv_file_path, header=None)

    horus_gpu_util_3090_node_part_1_timestamp = horus_gpu_util_3090_node_part_1_df.loc[:, 0].values
    horus_gpu_util_3090_node_part_1_data = horus_gpu_util_3090_node_part_1_df.loc[:, 1].values

    horus_gpu_util_4090_node_timestamp = horus_gpu_util_4090_node_df.loc[:, 0].values
    horus_gpu_util_4090_node_data = horus_gpu_util_4090_node_df.loc[:, 1].values

    horus_gpu_util_3090_node_part_2_timestamp = horus_gpu_util_3090_node_part_2_df.loc[:, 0].values[:-1]
    horus_gpu_util_3090_node_part_2_data = horus_gpu_util_3090_node_part_2_df.loc[:, 1].values[:-1]

    horus_gpu_util_titan_node_timestamp = horus_gpu_util_titan_node_df.loc[:, 0].values[:-1]
    horus_gpu_util_titan_node_data = horus_gpu_util_titan_node_df.loc[:, 1].values[:-1]

    horus_gpu_util_ave_data = (horus_gpu_util_3090_node_part_1_data + horus_gpu_util_4090_node_data + horus_gpu_util_3090_node_part_2_data + horus_gpu_util_titan_node_data) / 4

    liquid_gpu_util_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/liquid/liquid-gpu-util-data-3090-node-1-and-2.csv'
    liquid_gpu_util_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/liquid/liquid-gpu-util-data-4090-node.csv'
    liquid_gpu_util_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/liquid/liquid-gpu-util-data-3090-node-3-and-4.csv'
    liquid_gpu_util_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/liquid/liquid-gpu-util-data-titan-node.csv'

    liquid_gpu_util_3090_node_part_1_df = pd.read_csv(liquid_gpu_util_3090_node_part_1_csv_file_path, header=None)
    liquid_gpu_util_4090_node_df = pd.read_csv(liquid_gpu_util_4090_node_csv_file_path, header=None)
    liquid_gpu_util_3090_node_part_2_df = pd.read_csv(liquid_gpu_util_3090_node_part_2_csv_file_path, header=None)
    liquid_gpu_util_titan_node_df = pd.read_csv(liquid_gpu_util_titan_node_csv_file_path, header=None)

    liquid_gpu_util_3090_node_part_1_timestamp = liquid_gpu_util_3090_node_part_1_df.loc[:, 0].values
    liquid_gpu_util_3090_node_part_1_data = liquid_gpu_util_3090_node_part_1_df.loc[:, 1].values

    liquid_gpu_util_4090_node_timestamp = liquid_gpu_util_4090_node_df.loc[:, 0].values[:-1]
    liquid_gpu_util_4090_node_data = liquid_gpu_util_4090_node_df.loc[:, 1].values[:-1]

    liquid_gpu_util_3090_node_part_2_timestamp = liquid_gpu_util_3090_node_part_2_df.loc[:, 0].values[:-1]
    liquid_gpu_util_3090_node_part_2_data = liquid_gpu_util_3090_node_part_2_df.loc[:, 1].values[:-1]

    liquid_gpu_util_titan_node_timestamp = liquid_gpu_util_titan_node_df.loc[:, 0].values[:-2]
    liquid_gpu_util_titan_node_data = liquid_gpu_util_titan_node_df.loc[:, 1].values[:-2]

    liquid_gpu_util_ave_data = (liquid_gpu_util_3090_node_part_1_data + liquid_gpu_util_4090_node_data + liquid_gpu_util_3090_node_part_2_data + liquid_gpu_util_titan_node_data) / 4

    volcano_gpu_util_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/volcano/volcano-gpu-util-data-3090-node-1-and-2.csv'
    volcano_gpu_util_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/volcano/volcano-gpu-util-data-4090-node.csv'
    volcano_gpu_util_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/volcano/volcano-gpu-util-data-3090-node-3-and-4.csv'
    volcano_gpu_util_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/volcano/volcano-gpu-util-data-titan-node.csv'

    volcano_gpu_util_3090_node_part_1_df = pd.read_csv(volcano_gpu_util_3090_node_part_1_csv_file_path, header=None)
    volcano_gpu_util_4090_node_df = pd.read_csv(volcano_gpu_util_4090_node_csv_file_path, header=None)
    volcano_gpu_util_3090_node_part_2_df = pd.read_csv(volcano_gpu_util_3090_node_part_2_csv_file_path, header=None)
    volcano_gpu_util_titan_node_df = pd.read_csv(volcano_gpu_util_titan_node_csv_file_path, header=None)

    volcano_gpu_util_3090_node_part_1_timestamp = volcano_gpu_util_3090_node_part_1_df.loc[:, 0].values
    volcano_gpu_util_3090_node_part_1_data = volcano_gpu_util_3090_node_part_1_df.loc[:, 1].values

    volcano_gpu_util_4090_node_timestamp = volcano_gpu_util_4090_node_df.loc[:, 0].values[:-1]
    volcano_gpu_util_4090_node_data = volcano_gpu_util_4090_node_df.loc[:, 1].values[:-1]

    volcano_gpu_util_3090_node_part_2_timestamp = volcano_gpu_util_3090_node_part_2_df.loc[:, 0].values[:-2]
    volcano_gpu_util_3090_node_part_2_data = volcano_gpu_util_3090_node_part_2_df.loc[:, 1].values[:-2]

    volcano_gpu_util_titan_node_timestamp = volcano_gpu_util_titan_node_df.loc[:, 0].values[:-3]
    volcano_gpu_util_titan_node_data = volcano_gpu_util_titan_node_df.loc[:, 1].values[:-3]

    volcano_gpu_util_ave_data = (volcano_gpu_util_3090_node_part_1_data + volcano_gpu_util_4090_node_data + volcano_gpu_util_3090_node_part_2_data + volcano_gpu_util_titan_node_data) / 4

    ours_gpu_util_3090_node_part_1_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/ours/isacpp-gpu-util-data-3090-node-1-and-2.csv'
    ours_gpu_util_4090_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/ours/isacpp-gpu-util-data-4090-node.csv'
    ours_gpu_util_3090_node_part_2_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/ours/isacpp-gpu-util-data-3090-node-3-and-4.csv'
    ours_gpu_util_titan_node_csv_file_path = '/home/lzj/result2image/device_monitor_data/gpu-util/ours/isacpp-gpu-util-data-titan-node.csv'

    ours_gpu_util_3090_node_part_1_df = pd.read_csv(ours_gpu_util_3090_node_part_1_csv_file_path, header=None)
    ours_gpu_util_4090_node_df = pd.read_csv(ours_gpu_util_4090_node_csv_file_path, header=None)
    ours_gpu_util_3090_node_part_2_df = pd.read_csv(ours_gpu_util_3090_node_part_2_csv_file_path, header=None)
    ours_gpu_util_titan_node_df = pd.read_csv(ours_gpu_util_titan_node_csv_file_path, header=None)

    ours_gpu_util_3090_node_part_1_timestamp = ours_gpu_util_3090_node_part_1_df.loc[:, 0].values
    ours_gpu_util_3090_node_part_1_data = ours_gpu_util_3090_node_part_1_df.loc[:, 1].values

    ours_gpu_util_4090_node_timestamp = ours_gpu_util_4090_node_df.loc[:, 0].values
    ours_gpu_util_4090_node_data = ours_gpu_util_4090_node_df.loc[:, 1].values

    ours_gpu_util_3090_node_part_2_timestamp = ours_gpu_util_3090_node_part_2_df.loc[:, 0].values
    ours_gpu_util_3090_node_part_2_data = ours_gpu_util_3090_node_part_2_df.loc[:, 1].values

    ours_gpu_util_titan_node_timestamp = ours_gpu_util_titan_node_df.loc[:, 0].values
    ours_gpu_util_titan_node_data = ours_gpu_util_titan_node_df.loc[:, 1].values

    ours_gpu_util_ave_data = (ours_gpu_util_3090_node_part_1_data + ours_gpu_util_4090_node_data + ours_gpu_util_3090_node_part_2_data + ours_gpu_util_titan_node_data) / 4

    # schedtune_count = np.count_nonzero(schedtune_gpu_util_ave_data >= 95)
    # horus_count = np.count_nonzero(horus_gpu_util_ave_data >= 95)
    # liquid_count = np.count_nonzero(liquid_gpu_util_ave_data >= 95)
    # volcano_count = np.count_nonzero(volcano_gpu_util_ave_data >= 95)
    # ours_count = np.count_nonzero(ours_gpu_util_ave_data >= 95)
    #
    # print(schedtune_count / len(schedtune_gpu_util_ave_data))
    # print(horus_count / len(horus_gpu_util_ave_data))
    # print(liquid_count / len(liquid_gpu_util_ave_data))
    # print(volcano_count / len(volcano_gpu_util_ave_data))
    # print(ours_count / len(ours_gpu_util_ave_data))

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(schedtune_gpu_util_3090_node_part_1_timestamp, schedtune_gpu_util_ave_data, 'c-', linewidth=2)
    ax1.plot(horus_gpu_util_3090_node_part_1_timestamp, horus_gpu_util_ave_data, 'g-', linewidth=2)
    ax1.plot(liquid_gpu_util_3090_node_part_1_timestamp, liquid_gpu_util_ave_data, 'r-', linewidth=2)
    ax1.plot(volcano_gpu_util_3090_node_part_1_timestamp, volcano_gpu_util_ave_data, 'm-', linewidth=2)
    ax1.plot(ours_gpu_util_3090_node_part_1_timestamp, ours_gpu_util_ave_data, 'y--', linewidth=2)

    ax1.set_xlabel('Timestamp (s)', fontweight='bold')
    ax1.set_ylabel('Resource Utilization (%)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Volcano', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.xlim([0, 5400])
    plt.ylim([0, 100])

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 15(b).jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def get_manuscript_fig_16():
    fig_params = {
        'axes.labelsize': '18',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
        'lines.linewidth': '1',
        'legend.fontsize': '18',
        'xtick.direction': 'in',
        'ytick.direction': 'in'

    }

    pylab.rcParams.update(fig_params)

    workload_number_list = [20, 40, 60, 80, 100]
    schedtune_delay_list = [3.12, 7.76, 12.26, 20.88, 27.4]
    horus_delay_list = [1.88, 5.36, 8.16, 11.88, 15.3]
    liquid_delay_list = [1.66, 4.84, 6.94, 9.48, 13.1]
    volcano_delay_list = [1.61, 3.66, 4.72, 6.68, 9.29]
    ours_delay_list = [4.43, 9.84, 16.04, 24.52, 35.42]

    workload_number_nd = np.array(workload_number_list)
    schedtune_delay_nd = np.array(schedtune_delay_list)
    horus_delay_nd = np.array(horus_delay_list)
    liquid_delay_nd = np.array(liquid_delay_list)
    volcano_delay_nd = np.array(volcano_delay_list)
    ours_delay_nd = np.array(ours_delay_list)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(workload_number_nd, schedtune_delay_nd, 'c-',
             linewidth=4)
    ax1.plot(workload_number_nd, horus_delay_nd, 'g-', linewidth=4)
    ax1.plot(workload_number_nd, liquid_delay_nd, 'r-', linewidth=4)
    ax1.plot(workload_number_nd, volcano_delay_nd, 'm-', linewidth=4)
    ax1.plot(workload_number_nd, ours_delay_nd, 'y--', linewidth=4)

    ax1.set_xlabel('Workload Number', fontweight='bold')
    ax1.set_ylabel('Scheduling Delay (s)', fontweight='bold')

    # Add a legend
    legend = plt.legend(loc='best', labels=['Schedtune', 'Horus', 'Liquid', 'Volcano', 'Ours'])
    plt.setp(legend.get_texts(), fontweight='bold')

    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Set bold font for the x and y axis ticks
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.savefig('Figure 16.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    get_manuscript_fig_16()
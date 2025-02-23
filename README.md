# Interference-aware Scheduling Approach for Deep Learning Training Workloads Based on Co-location Performance Prediction
This is the code repository for the manuscript titled 'ISACPP: Interference-aware Scheduling Approach for Deep Learning Training Workloads Based on Co-location Performance Prediction'.

This project first builds an edge-fusion gated graph attention network (E-GGAT) that incorporates DL model structures, underlying GPU types, and hyper-parameter settings to predict co-location performance. On this basis, a multi-stage co-location interference quantification model is proposed to obtain accurate interference quantification. The incoming workload is assigned to the GPU device with the minimum overall interference aggregated from all stages.

# Prerequisites
- OS Centos Linux release 7.9
- Nvidia Driver 535.54.03
- CUDA 12.2
- Docker 19.03
- Kubernetes 1.23
- Volcano 1.6.0
- Volcano device plugin 1.0.0
- Pytorch 2.0.0
- DGL 2.1.0+cu118

# Hierarchy
- `workloads`: include all deep learning training (DLT) workloads listed in the manuscript implemented by Pytorch
- `profiler`: obtain runtime infomation of DLT workloads in standalone and co-location scenarios
- `predictor`: estimate the co-location performance of DLT workloads based on E-GGAT
- `scheduler`: achieve accurate interference quantification based on the predicted co-location performance and assign the incoming DLT workload to the GPU device with the minimum interference
- `results`: include all figures and experimental data in the manuscript 

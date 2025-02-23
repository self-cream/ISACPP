ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
FROM ${FROM_IMAGE_NAME}

RUN pip install flask torchinfo thop tensorboard onnx -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

COPY . .

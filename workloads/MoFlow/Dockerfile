# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install git+https://github.com/NVIDIA/dllogger@v1.0.0#egg=dllogger

RUN python3 -m pip install rdkit-pypi
RUN pip install flask torchinfo thop -i https://pypi.tuna.tsinghua.edu.cn/simple

ARG WORKSPACE=/workspace/moflow_pyt
WORKDIR ${WORKSPACE}
ADD . ${WORKSPACE}
RUN python3 -m pip install .

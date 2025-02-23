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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import argparse
import functools
import json
import logging
import time
import datetime
import os
import signal
import onnx

from typing import Dict
from torchinfo import summary
from thop import profile

from apex.contrib.clip_grad import clip_grad_norm_
from apex.optimizers import FusedAdam as Adam
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from moflow.config import CONFIGS, Config
from moflow.data.data_loader import NumpyTupleDataset
from moflow.data import transform
from moflow.model.model import MoFlow, MoFlowLoss
from moflow.model.utils import initialize
from moflow.runtime.logger import MetricsLogger, PerformanceLogger, setup_logging
from moflow.runtime.arguments import PARSER
from moflow.runtime.common import get_newest_checkpoint, load_state, save_state
from moflow.runtime.distributed_utils import (
    get_device, get_rank, get_world_size, init_distributed, reduce_tensor
)
from moflow.runtime.generate import infer
from moflow.utils import check_validity, convert_predictions_to_mols


torch._C._jit_set_autocast_mode(True)


workload_data = {
    "ELAPSED_ITERATIONS": "0",
    "TOTAL_ITERATIONS": "0",
    "ITERATIONS_PER_EPOCH": "0",
    "TOTAL_EPOCHS": "0"
}


def update_shared_data_file(data):
    with open("shared_data.txt", "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")


def run_validation(model: MoFlow, config: Config, ln_var: float, args: argparse.Namespace,
                         is_distributed: bool, world_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    if is_distributed:
        model_callable = model.module
    else:
        model_callable = model
    result = infer(model_callable, config, device=device, ln_var=ln_var, batch_size=args.val_batch_size,
                   temp=args.temperature)
    mols = convert_predictions_to_mols(*result, correct_validity=args.correct_validity)
    validity_info = check_validity(mols)
    valid_ratio = torch.tensor(validity_info['valid_ratio'], dtype=torch.float32, device=device)
    unique_ratio = torch.tensor(validity_info['unique_ratio'], dtype=torch.float32, device=device)
    valid_value = reduce_tensor(valid_ratio, world_size).detach().cpu().numpy()
    unique_value = reduce_tensor(unique_ratio, world_size).detach().cpu().numpy()
    model.train()
    return {'valid': valid_value, 'unique': unique_value}


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    # Device configuration
    device = get_device(args.local_rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()

    logger = setup_logging(args)
    if local_rank == 0:
        perf_logger = PerformanceLogger(logger, args.batch_size * world_size, args.warmup_steps)
        acc_logger = MetricsLogger(logger)

    if local_rank == 0:
        logging.info('Input args:')
        logging.info(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # Model configuration
    assert args.config_name in CONFIGS
    config = CONFIGS[args.config_name]
    data_file = config.dataset_config.dataset_file
    transform_fn = functools.partial(transform.transform_fn, config=config)
    valid_idx = transform.get_val_ids(config, args.data_dir)

    if local_rank == 0:
        logging.info('Config:')
        logging.info(str(config))
    model = MoFlow(config)

    model.to(device)
    loss_module = MoFlowLoss(config)
    loss_module.to(device)

    # Datasets:
    dataset = NumpyTupleDataset.load(
        os.path.join(args.data_dir, data_file),
        transform=transform_fn,
    )
    if len(valid_idx) == 0:
        raise ValueError('Empty validation set!')
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, valid_idx)

    if world_size > 1:
        sampler = DistributedSampler(train, seed=args.seed, drop_last=False)
    else:
        sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if local_rank == 0:
        logging.info(f'Using {world_size} GPUs')
        logging.info(f'Num training samples: {len(train)}')
        logging.info(f'Minibatch-size: {args.batch_size}')
        logging.info(f'Num Iter/Epoch: {len(train_dataloader)}')
        logging.info(f'Num epoch: {args.epochs}')

    if is_distributed:
        train_dataloader.sampler.set_epoch(-1)
    x, adj, *_ = next(iter(train_dataloader))
    x = x.to(device)
    adj = adj.to(device)
    with autocast(enabled=args.amp):
        initialize(model, (adj, x))

    model.to(memory_format=torch.channels_last)
    adj.to(memory_format=torch.channels_last)

    if args.jit:
        model.bond_model = torch.jit.script(model.bond_model)
        model.atom_model = torch.jit.script(model.atom_model)

    # make one pass in both directions to make sure that model works
    with torch.no_grad():
        _ = model(adj, x)
        _ = model.reverse(torch.randn(args.batch_size, config.z_dim, device=device))

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        loss_module = torch.nn.parallel.DistributedDataParallel(
            loss_module,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        model_callable = model.module
        loss_callable = loss_module.module
    else:
        model_callable = model
        loss_callable = loss_module

    # Loss and optimizer
    optimizer = Adam((*model.parameters(), *loss_module.parameters()), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    scaler = GradScaler()

    if args.save_epochs == -1:
        args.save_epochs = args.epochs
    if args.eval_epochs == -1:
        args.eval_epochs = args.epochs
    if args.steps == -1:
        args.steps = args.epochs * len(train_dataloader)

    snapshot_path = get_newest_checkpoint(args.results_dir)
    if snapshot_path is not None:
        snapshot_epoch, ln_var = load_state(snapshot_path, model_callable, optimizer=optimizer, device=device)
        loss_callable.ln_var = torch.nn.Parameter(torch.tensor(ln_var))
        first_epoch = snapshot_epoch + 1
        step = first_epoch * len(train_dataloader)
    else:
        first_epoch = 0
        step = 0

    if first_epoch >= args.epochs:
        logging.info(f'Model was already trained for {first_epoch} epochs')
        exit(0)

    max_iterations = args.epochs * len(train_dataloader)


    example_x = torch.tensor([])
    example_adj = torch.tensor([])

    for epoch in range(first_epoch, args.epochs):
        if local_rank == 0:
            acc_logger.reset()
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)

#        start = time.time()
        epoch_start_time = time.time()
#        prev_iteration_duration = time.time()
#        eplison = 0.005
#        elapsed_iteration = 0
#        hydra_flag = False
#        optimus_flag = False

        for i, batch in enumerate(train_dataloader):
#            iteration_start = time.time()

            if local_rank == 0:
                perf_logger.update()
            step += 1

            workload_data['ELAPSED_ITERATIONS'] = str(step)
            workload_data['TOTAL_ITERATIONS'] = str(max_iterations)
            workload_data['ITERATIONS_PER_EPOCH'] = str(len(train_dataloader))
            workload_data['TOTAL_EPOCHS'] = str(args.epochs)
            update_shared_data_file(workload_data)

            optimizer.zero_grad()
            x = batch[0].to(device)
            adj = batch[1].to(device=device,memory_format=torch.channels_last)

            example_x = x
            example_adj = adj

            # Forward, backward and optimize
            with_cuda_graph = (
                args.cuda_graph
                and step >= args.warmup_steps
                and x.size(0) == args.batch_size
            )
            with autocast(enabled=args.amp, cache_enabled=not with_cuda_graph):
                output = model(adj, x, with_cuda_graph=with_cuda_graph)
                nll_x, nll_adj = loss_module(*output)
                loss = nll_x + nll_adj

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            # Print log info
            if (i + 1) % args.log_interval == 0:
                nll_x_value = reduce_tensor(nll_x, world_size).item()
                nll_adj_value = reduce_tensor(nll_adj, world_size).item()
                loss_value = nll_x_value + nll_adj_value

                if local_rank == 0:
                    acc_logger.update({
                        'loglik': loss_value,
                        'nll_x': nll_x_value,
                        'nll_adj': nll_adj_value
                    })

                    acc_logger.summarize(step=(epoch, i, i))
                    perf_logger.summarize(step=(epoch, i, i))

 #           iteration_end = time.time()
#            current_iteration_duration = iteration_end - iteration_start
#
#            abs_iteration_differ = abs(current_iteration_duration - prev_iteration_duration)
#            elapsed_iteration += 1
#
#            if ((abs_iteration_differ / current_iteration_duration) < eplison) and (hydra_flag == False):
#                elapsed_duration_since_start = iteration_end - start
#                hydra_estimated_epoch_duration = elapsed_duration_since_start + (len(train_dataloader) - elapsed_iteration) * current_iteration_duration
#                print('hydra estimated epoch time: {:.3f}s'.format(hydra_estimated_epoch_duration))
#                hydra_flag = True
#            if (elapsed_iteration == 30) and (optimus_flag == False):
#                optimus_estimated_epoch_duration = iters_per_epoch * ((iteration_end - start) / elapsed_iteration)
#                print('optimus estimated epoch time: {:.3f}s'.format(optimus_estimated_epoch_duration))
#                optimus_flag = True
#
#            prev_iteration_duration = current_iteration_duration

            if step >= args.steps:
                break

        total_training_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        print("-------Epoch training time: %s" %total_time_str)

        if (epoch + 1) % args.eval_epochs == 0:
            with autocast(enabled=args.amp):
                metrics = run_validation(model, config, loss_callable.ln_var.item(), args, is_distributed, world_size, device)
            if local_rank == 0:
                acc_logger.update(metrics)

        # The same report for each epoch
        if local_rank == 0:
            acc_logger.summarize(step=(epoch,))
            perf_logger.summarize(step=(epoch,))

        # Save the model checkpoints
        if (epoch + 1) % args.save_epochs == 0:
            if local_rank == 0 or not is_distributed:
                save_state(args.results_dir, model_callable, optimizer, loss_callable.ln_var.item(), epoch, keep=5)

        if step >= args.steps:
            break

    summary(model, input_data=[example_adj, example_x])
    flops, params = profile(model, (example_adj, example_x))
    print('the flops is {}G, the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))

    model.eval()
    onnx_model_path = "model.onnx"  # Path to save the ONNX model
    torch.onnx.export(model, (example_adj, example_x), onnx_model_path, verbose=False)

    print(f"Model exported to {onnx_model_path}")

    onnx_model = onnx.load("model.onnx")

    conv_count = 0
    normalization_count = 0
    maxpool_count = 0
    averagepool_count = 0
    mul_count = 0
    sigmoid_count = 0
    lstm_count = 0
    div_count = 0
    sqrt_count = 0
    softmax_count = 0
    tanh_count = 0
    pow_count = 0
    relu_count = 0
    gemm_count = 0

    # Iterate through all nodes in the ONNX model's graph
    for node in onnx_model.graph.node:
       # print(node.op_type)
        if node.op_type == "Gemm":
            gemm_count += 1
        if "Conv" in node.op_type:
            conv_count += 1
        if node.op_type == "Relu":
            relu_count += 1
        if "Normalization" in node.op_type:
            normalization_count += 1
        if node.op_type == "MaxPool":
            maxpool_count += 1
        if "AveragePool" in node.op_type:
            averagepool_count += 1
        if "Mul" in node.op_type:
            mul_count += 1
        if node.op_type == "Sigmoid":
            sigmoid_count += 1
        if node.op_type == "LSTM":
            lstm_count += 1
        if node.op_type == "Div":
            div_count += 1
        if node.op_type == "Sqrt":
            sqrt_count += 1
        if node.op_type == "Softmax":
            softmax_count += 1
        if node.op_type == "Tanh":
            tanh_count += 1
        if node.op_type == "Pow":
            pow_count += 1

    # Print the total number of GEMM operations
    print(f"Total number of Mul operations: {mul_count}")
    print(f"Total number of Div operations: {div_count}")
    print(f"Total number of Pow operations: {pow_count}")
    print(f"Total number of Sqrt operations: {sqrt_count}")
    print(f"Total number of Conv operations: {conv_count}")
    print(f"Total number of Gemm operations: {gemm_count}")
    print(f"Total number of MaxPool operations: {maxpool_count}")
    print(f"Total number of AveragePool operations: {averagepool_count}")
    print(f"Total number of Normalization operations: {normalization_count}")
    print(f"Total number of LSTM operations: {lstm_count}")
    print(f"Total number of Relu operations: {relu_count}")
    print(f"Total number of Sigmoid operations: {sigmoid_count}")
    print(f"Total number of Softmax operations: {softmax_count}")
    print(f"Total number of Tanh operations: {tanh_count}")

    if local_rank == 0:
        acc_logger.summarize(step=tuple())
        perf_logger.summarize(step=tuple())


if __name__ == '__main__':
    start_time = time.time()
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    args = PARSER.parse_args()
    train(args)
    job_makespan_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=job_makespan_time))
    print("------Total Job Makespan: %s" %total_time_str)


# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
import onnx
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
from thop import profile

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

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


def train(epoch, example_input, flag, total_epochs):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        if not flag:
            example_input = images
            flag = True

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        max_iterations = total_epochs * len(cifar100_training_loader)
        iters_per_epoch = len(cifar100_training_loader)

        workload_data['ELAPSED_ITERATIONS'] = str(n_iter)
        workload_data['TOTAL_ITERATIONS'] = str(max_iterations)
        workload_data['ITERATIONS_PER_EPOCH'] = str(iters_per_epoch)
        workload_data['TOTAL_EPOCHS'] = str(total_epochs)

        update_shared_data_file(workload_data)

        last_layer = list(net.children())[-1]
 #       for name, para in last_layer.named_parameters():
 #           if 'weight' in name:
 #               writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
 #           if 'bias' in name:
 #               writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
 #       writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

 #   for name, param in net.named_parameters():
 #       layer, attr = os.path.splitext(name)
 #       attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return example_input, flag

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
 #   if tb:
 #       writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
 #       writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-epochs', type=int, default=1, help='the number of training epochs')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
   # writer = SummaryWriter(log_dir=os.path.join(
   #         settings.LOG_DIR, args.net, settings.TIME_NOW))
   # input_tensor = torch.Tensor(1, 3, 32, 32)
   # if args.gpu:
   #     input_tensor = input_tensor.cuda()
   # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    example_input = torch.tensor([])

    flag = False

    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        example_input, flag = train(epoch, example_input, flag, args.epochs)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    job_makespan_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=job_makespan_time))
    print("------Total Job Makespan: %s" %total_time_str)

    summary(net, input_data=example_input)

    flops, params = profile(net, (example_input,))
    print('the flops is {}G, the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))

    net.eval()
    onnx_model_path = "model.onnx"  # Path to save the ONNX model
    torch.onnx.export(net, example_input, onnx_model_path, verbose=False)

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
        #print(node.op_type)
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

 #   writer.close()

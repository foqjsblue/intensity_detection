#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import KITTIDataset
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, classification_report
import psutil  # 추가

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints/' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def custom_collate_fn(batch):
    max_size = max([s[0].shape[0] for s in batch])

    padded_data = []
    labels = []

    for item in batch:
        data, label = item
        data_tensor = torch.tensor(data, dtype=torch.float)
        padding_size = max_size - data_tensor.shape[0]
        padded_sample = F.pad(data_tensor, (0, 0, 0, padding_size), 'constant', 0)
        padded_data.append(padded_sample)
        labels.append(label)

    padded_data = torch.stack(padded_data, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_data, labels


def train(args, io):
    print("Training mode")
    train_directories = [
        '/training/box_car_train',
        '/training/box_car_train_rotate',
        '/training/box_ped_train',
        '/training/box_cyc_train',
        '/training/box_car_train_rd02710',
        '/training/box_ped_train_rd02710',
        '/training/box_cyc_train_rd02710',
        '/opt_training/fake_car_pgd_sc',
        '/opt_training/fake_car_pgd_pp',
        '/opt_training/fake_car_pgd_sc_rotate',
        '/opt_training/fake_car_pgd_pp_rotate'
    ]

    train_loader = DataLoader(
        KITTIDataset(directories=train_directories, num_points=args.num_points, labels=[
            'Normal Object', 'Normal Object', 'Normal Object', 'Normal Object', 'Fake Object', 'Fake Object',
            'Fake Object', 'Fake Object', 'Fake Object', 'Fake Object', 'Fake Object']), num_workers=4,
        batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

    test_directories = [
        '/testing/box_car_test',
        '/testing/box_ped_test',
        '/testing/box_cyc_test',
        '/testing/box_car_test_rd02710',
        '/testing/box_ped_test_rd02710',
        '/testing/box_cyc_test_rd02710'
    ]

    test_loader = DataLoader(KITTIDataset(directories=test_directories, num_points=args.num_points,
                                          labels=['Normal Object', 'Normal Object', 'Normal Object', 'Fake Object',
                                                  'Fake Object', 'Fake Object']), num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False,
                             collate_fn=custom_collate_fn)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise ValueError("The requested model is not supported: {}".format(args.model))
    model.to(device)

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.clone().detach().to(device)
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()

        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    print("Evaluation mode")
    test_directories = ['/mnt/d/materials/sn2/3/rd_sn2_sc_bg_']
    #test_directories = ['../../box_car_fp_sc', '../../box_ped_fp_sc', '../../box_cyc_fp_sc',
                        #'/mnt/d/materials/fgsm/4.save_box_fake/box_car_fgsm_s1_sc']
    test_labels = ['Normal Object']

    test_loader = DataLoader(KITTIDataset(directories=test_directories, num_points=args.num_points, labels=test_labels),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False,
                             collate_fn=custom_collate_fn)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    num_batches = len(test_loader)

    with torch.no_grad():
        for batch in test_loader:

            # Data loading
            data, label = batch

            # Model inference
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1]

            # Post-processing
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())


    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)

    conf_matrix = confusion_matrix(test_true, test_pred)
    class_counts = np.sum(conf_matrix, axis=1)

    io.cprint('=== Class counts ===')
    for i, count in enumerate(class_counts):
        io.cprint(f"Class {i} count: {count}")

    io.cprint('Confusion Matrix:')
    io.cprint(np.array2string(conf_matrix))

    class_report = classification_report(test_true, test_pred, labels=[0, 1], target_names=['Normal Object', 'Fake Object'], digits=2)
    io.cprint(class_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='kitti_box', metavar='N',
                        choices=['kitti_box'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use ADAM')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=256,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

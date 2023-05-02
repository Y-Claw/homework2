import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import MyData
from algorithm import pca_algorithm, resnet_classification
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils.data import DataLoader, random_split, Subset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dataset_path',
        default= '../../Dataset/face',
        type=str,
        help='dataset name')
    parser.add_argument('--dataset',
        default= 'faces94',
        type=str,
        help='dataset name')

    parser.add_argument('--algorithm',
        default= 'pca',
        type=str,
        help='algorithm name')
    parser.add_argument('--pca_dim',
        default= 150,
        type=int,
        help='pca dim')

    parser.add_argument('--model_name',
        default= 'resnet18',
        type=str,
        help='resnet model name')
    parser.add_argument('--gpu',
        default=False,
        type=bool,
        help='gpu')
    parser.add_argument('--epoch',
        default= 20,
        type=int,
        help='epoch')
    parser.add_argument('--batch_size',
        default= 32,
        type=int,
        help='epoch')
    parser.add_argument('--lr',
        default=0.0001,
        type=float,
        help='lr')
    parser.add_argument('--wd',
        default=0,
        type=float,
        help='weight decay')
    parser.add_argument('--loss',
        default='cross_entropy',
        type=str,
        help='loss function')
    

    parser.add_argument('--fold_num',
        default= 5,
        type=int,
        help='fold number')

    return parser.parse_args()

def main():
    args = get_args()
    dataset = MyData(args)

    if args.algorithm == 'pca':
        results = []
        for fold_idx in range(args.fold_num):
            train_data, train_labels = dataset.get_all_train_data(fold_idx)
            test_data, test_labels = dataset.get_all_test_data(fold_idx)
            accuracy = pca_algorithm(train_data, train_labels, test_data, test_labels)
            results.append(accuracy)
        print('Mean Accuracy', sum(results) / len(results))

    elif args.algorithm == 'resnet':
        results = []
        for fold_idx in range(args.fold_num):
            class_num = dataset.get_class_num()
            train_idx, test_idx = dataset.get_idx(fold_idx)

            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
            model = eval(args.model_name)(class_num)
            if args.gpu:
                model.cuda()
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            accuracy, epoch = resnet_classification(model, train_loader, test_loader, optimizer, loss_function, args)
            results.append(accuracy)
        print('Mean Accuracy', sum(results) / len(results))
            



if __name__ == "__main__":
    main()
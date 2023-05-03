import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import torch


def pca_algorithm(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    pca = PCA(n_components=100)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)

    X_test_pca = pca.transform(X_test)

    # 训练SVM分类器
    clf = SVC(kernel='linear')
    clf.fit(X_train_pca, y_train)

    # 测试分类器准确率
    accuracy = clf.score(X_test_pca, y_test)
    print('Accuracy:', accuracy)
    return accuracy

def train(model, train_loader, optimizer, loss_function, epoch, args):

    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(train_loader.dataset)
        ))

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(model, eval_loader, optimizer, loss_function, epoch, args, tb=True):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in eval_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(eval_loader.dataset),
        correct.float() / len(eval_loader.dataset),
        finish - start
    ))
    print()

    return correct.float() / len(eval_loader.dataset)

def resnet_classification(model, train_loader, test_loader, optimizer, loss_function, args):
    max_epoch = 0
    max_accuracy = 0
    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, loss_function, epoch, args)
        accuracy = eval_training(model, test_loader, optimizer, loss_function, epoch, args)
        if accuracy > max_accuracy:
            max_epoch = epoch
            max_accuracy = accuracy
    return max_accuracy, max_epoch
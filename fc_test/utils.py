from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import os
import torch

import torch.nn as nn
from torch.utils.data import Dataset


class ChestMNIST(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        npz_file = np.load(self.root)

        self.split = split
        self.transform = transform

        if self.split == 'train':
            self.img = npz_file['X_train']
            self.label = npz_file['y_train']
        elif self.split == 'val':
            self.img = npz_file['X_val']
            self.label = npz_file['y_val']
        elif self.split == 'test':
            self.img = npz_file['X_test']
            self.label = npz_file['y_test']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.img.shape[0]


def get_auc(y_true, y_score):
    auc = 0
    for i in range(y_score.shape[1]):
        label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        auc += label_auc
    return auc / y_score.shape[1]


def get_acc(y_true, y_score, threshold=0.5):
    zero = np.zeros_like(y_score)
    one = np.ones_like(y_score)
    y_pre = np.where(y_score < threshold, zero, one)
    acc = 0
    for label in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
        acc += label_acc
    return acc / y_true.shape[1]


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        # if task == 'multi-label, binary-class':
        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets).to(device)
        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            # if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = get_auc(y_true, y_score)
        val_auc_list.append(auc)

    state = {'net': model.state_dict(), 'auc': auc, 'epoch': epoch}
    path = os.path.join(dir_path, f'ckpt_{epoch}_auc_{auc:%.5f}.pth')
    torch.save(state, path)
    return state['auc']


def test(model, data_loader, device):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            # if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = get_auc(y_true, y_score)
        acc = get_acc(y_true, y_score)

        return auc, acc

# You are not allowed to import any other libraries or modules.

import torch
import torch.nn as nn
import numpy as np


def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'\nEpoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss, batch_train_acc = [], []

    #TODO: train the given model for only one batch and store accuracy and loss in batch_train_acc, batch_train_loss repectively.
    
    return batch_train_loss, batch_train_acc



def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []

    #TODO: Test the model on the given test dataset and store accuracy in batch_test_acc. This function return nothing.
    # Remember you should disable gradient computation during testing.

    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")

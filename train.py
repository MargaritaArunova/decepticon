import math
import time
from typing import Optional, Any

from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import TranslationModel


def print_metrics(epoch, time, train_loss, val_loss):
    print(f'Epoch: {epoch}')
    print(f'\tTraining epoch time: {time} seconds')
    print(f'\tTrain: loss - {train_loss:.3f}, perplexity - {math.exp(train_loss):10.3f}')
    print(f'\tValidation: loss - {val_loss:.3f}, perplexity - {math.exp(val_loss):10.3f}')


def training_epoch(model: TranslationModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: translation model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    model.train()
    device = next(model.parameters()).device

    train_loss = 0.0
    for source, target in tqdm(loader, desc=tqdm_desc):
        source = source.to(device)
        target = target.to(device)

        # standard training process
        optimizer.zero_grad()
        output, _ = model(source, target[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        loss = criterion(output, target[:, 1:].contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # to prevent exploding gradient effect
        optimizer.step()
        # add loss to total loss
        train_loss += loss.item() * source.shape[0]

    return train_loss / len(loader.dataset)


@torch.no_grad()
def validation_epoch(model: TranslationModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: translation model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    model.eval()
    device = next(model.parameters()).device

    val_loss = 0.0
    for source, target in tqdm(loader, desc=tqdm_desc):
        source = source.to(device)
        target = target.to(device)

        # standard training process
        output, _ = model(source, target[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        loss = criterion(output, target[:, 1:].contiguous().view(-1))
        # add loss to total loss
        val_loss += loss.item() * source.shape[0]

    return val_loss / len(loader.dataset)


def train(model: TranslationModel,
          optimizer: torch.optim.Optimizer, criterion: nn.Module, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
    """
    Train language model for several epochs
    :param model: translation model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )
        end_time = time.time()

        if scheduler is not None:
            scheduler.step()

        # save loss
        train_losses += [train_loss]
        val_losses += [val_loss]

        # save best model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'translation-model.pt')
            best_val_loss = val_loss

        # save model after each 10 epoch
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'translation-model_' + str(epoch) + '.pt')

        # print metrics
        print_metrics(epoch, end_time - start_time, train_loss, val_loss)


def set_model_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)

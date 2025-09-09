from torch import nn, optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import deque

from src.datautils import load_data, save_to_file
from src.cerberus import MatFact
from src.losses import WMSE

from configs import torch_config as train_config

def run(train_config):
    # read
    C_train, C_val, C_shape, T_C = load_data(train_config['C_path'],
                                             batch_size=train_config['batch_size'], source='C')
    A_train, A_val, A_shape, T_A = load_data(train_config['A_path'],
                                             batch_size=train_config['batch_size'], source='A')

    # define
    device = train_config['device']
    model = MatFact(N=A_shape[0], M=A_shape[1],
                    D=C_shape[1], K=train_config['K'], T=T_C).to(device)
    opt = optim.AdamW(model.parameters(), lr=train_config['lr'])
    loss_fn = WMSE(train_config['c0_scaler'])

    # capturing vars
    epoch_train_losses, epoch_val_losses = [], []
    early_stopping = deque([1000] * train_config['early_stopping_p'])
    best_loss = 1000000

    # train loop
    print('Starting training')
    for i in range(train_config['num_epochs']):
        # train step
        train_losses = []
        dloaders = [iter(A_train), iter(C_train)]

        model.train()
        for batch in tqdm(range(max(len(A_train), len(C_train)))):
            opt.zero_grad()

            losses = {'A': 0, 'C': 0}
            for dl in dloaders:
                try:
                    t, user_ind, target, source = next(dl)
                except StopIteration:
                    continue

                t, user_ind, target = t.to(device), user_ind.to(device), target.to(device)
                preds = model(t, user_ind, source)
                losses[source] = loss_fn(preds, target)

            loss = train_config['weight_A'] * losses['A'] + \
                   train_config['weight_C'] * losses['C'] + \
                   train_config['lambda_1'] * model.weight_regularization() + \
                   train_config['lambda_2'] * model.alignment_regularization()

            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        epoch_train_losses.append(np.mean(train_losses))

        # val step
        val_losses = []
        dl = {'A': iter(A_val), 'C': iter(C_val)}

        model.eval()
        for batch in tqdm(range(max(len(A_val), len(A_val)))):
            opt.zero_grad()

            losses = {'A': 0, 'C': 0}
            for dl in dloaders:
                try:
                    t, user_ind, target, source = next(dl)
                except StopIteration:
                    continue

                t, user_ind, target = t.to(device), user_ind.to(device), target.to(device)
                preds = model(t, user_ind, source)
                losses[source] = loss_fn(preds, target)

            loss = train_config['weight_A'] * losses['A'] + \
                   train_config['weight_C'] * losses['C'] + \
                   train_config['lambda_1'] * model.weight_regularization() + \
                   train_config['lambda_2'] * model.alignment_regularization()

            val_losses.append(loss.item())
        epoch_val_losses.append(np.mean(val_losses))
        # epoch stats

        print(f'Epoch: {i}, Train Loss: {epoch_train_losses[-1]:0.5f}, Val Loss: {epoch_val_losses[-1]:0.5f}')

        # save best model
        if epoch_val_losses[-1] < best_loss:
            best_loss = epoch_val_losses[-1]
            print(f'New best model found: {name}_{i}')
            save_to_file(model, f'{name}_{i}', train_config, {'epoch_val_losses': epoch_val_losses,
                                                              'epoch_train_losses': epoch_train_losses})

        # dump to file incrementally
        if (i % 100 == 0) or (i == train_config['num_epochs'] - 1):
            save_to_file(model, f'{name}_{i}', train_config, {'epoch_val_losses': epoch_val_losses,
                                                              'epoch_train_losses': epoch_train_losses})

        # early stopping
        early_stopping.append(epoch_val_losses[-1])
        if early_stopping.popleft() - min(early_stopping) < train_config['early_stopping_tol']:
            print('Early stopping triggered')
            break


if __name__=='__main__':
    print('Starting up')
    name = datetime.now().strftime('%H%M%d%m%y')
    train_config['name'] = name
    print('**** Hyperparams: ****')
    _ = [print(k, v) for k, v in train_config.items()]
    print('**********************')

    run(train_config)




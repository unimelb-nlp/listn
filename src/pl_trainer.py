import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .datautils import SparseDataset, Reader
from .losses import WMSE
from .cerberus import MatFact


class DataModulePL(pl.LightningDataModule):
    def __init__(self, train_path_A, train_path_C, batch_size, val_frac=0.1):
        super().__init__()
        self.train_path_A = train_path_A
        self.train_path_C = train_path_C
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.shape_A = []
        self.shape_C = []
        self.T = 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print('loading data')
            reader = Reader(self.train_path_C, self.val_frac)
            reader.run()
            self.train_ds_C = SparseDataset(reader.train_samples, 'C', 'train')
            self.val_ds_C = SparseDataset(reader.val_samples, 'C', 'val')
            self.shape_C = reader.shape
            self.T = reader.num_timesteps

            reader = Reader(self.train_path_A, self.val_frac)
            reader.run()
            self.train_ds_A = SparseDataset(reader.train_samples, 'A', 'train')
            self.val_ds_A = SparseDataset(reader.val_samples, 'A', 'val')
            self.shape_A = reader.shape

    def train_dataloader(self):
        # Create separate dataloaders for A and C
        train_dl_A = DataLoader(self.train_ds_A, batch_size=self.batch_size, drop_last=True,
                                num_workers=2, persistent_workers=True)
        train_dl_C = DataLoader(self.train_ds_C, batch_size=self.batch_size, drop_last=True,
                                num_workers=2, persistent_workers=True)
        return train_dl_A, train_dl_C

    def val_dataloader(self):
        # Create separate dataloaders for A and C
        val_dl_A = DataLoader(self.val_ds_A, batch_size=self.batch_size, drop_last=True,
                              num_workers=2, persistent_workers=True)
        val_dl_C = DataLoader(self.val_ds_C, batch_size=self.batch_size, drop_last=True,
                              num_workers=2, persistent_workers=True)
        return val_dl_A, val_dl_C

class MatFactPL(pl.LightningModule):
    def __init__(self, **hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        print('**** Hyperparams: ****')
        _ = [print(k,v) for k,v in self.hparams.items()]
        print('**********************')

        self.MatFact = MatFact(N=self.hparams.N, M=self.hparams.M,
                    D=self.hparams.D, K=self.hparams.K, T=self.hparams.T)

        self.loss_fn = WMSE(self.hparams.c0_scaler)
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.train_losses = []

    def forward(self, t, user_ind, source, item_inds=None):
        return self.MatFact(t, user_ind, source, item_inds)

    def training_step(self, batches, batch_idx):
        losses = {'A': 0, 'C': 0}

        for batch in batches: # batches = 2 dataloaders
            t, user_ind, target, source = batch
            preds = self(t, user_ind, source)
            losses[source[0]] = self.loss_fn(preds, target)

        loss_A = self.hparams.weight_A * losses['A']
        loss_C = self.hparams.weight_C * losses['C']
        weight_reg = self.hparams.lambda_1 * self.MatFact.weight_regularization()
        align_reg = self.hparams.lambda_2 * self.MatFact.alignment_regularization()

        # Total loss = weighted loss for both sources + regularization
        loss = loss_A + loss_C + weight_reg + align_reg

        self.train_step_outputs.append(torch.tensor([loss_A, loss_C, weight_reg, align_reg], device=self.device))
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        losses = torch.mean(torch.stack(self.train_step_outputs),axis=0)

        for key,val in zip(['loss_A', 'loss_C', 'weight_reg', 'align_reg'],losses):
            self.log(key, val,  prog_bar=False, logger=True, sync_dist=True,
                 on_step=False, on_epoch=True)

        self.log('train_loss', torch.mean(torch.stack(self.train_losses)), sync_dist=True,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.train_step_outputs.clear()
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        t, user_ind, target, source, item_inds = batch
        pred = self(t, user_ind, source, item_inds)
        loss = self.loss_fn(pred, target)

        # Return loss inside a dictionary
        self.validation_step_outputs.append((source[0],loss))
        return loss

    def on_validation_epoch_end(self):
        # Aggregate the losses from each dataloader
        losses = {'A':[],'C':[]}

        for source,loss in self.validation_step_outputs:
            losses[source].append(loss)

        # Combine the losses with regularization
        val_error = self.hparams.weight_A * torch.mean(torch.stack(losses['A'])) + \
                       self.hparams.weight_C * torch.mean(torch.stack(losses['C']))

        # val_loss = val_error + self.hparams.lambda_1 * self.MatFact.weight_regularization() + \
        #                self.hparams.lambda_2 * self.MatFact.alignment_regularization()

        self.log("val_error", val_error, sync_dist=True, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()
        return {'val_error': val_error}

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return opt
from scipy import sparse, io
import glob
import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader

def load_data(path, val_frac, batch_size, source):
    reader = Reader(path, val_frac)
    reader.run()

    train_ds = SparseDataset(reader.train_samples, source, 'train')
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, drop_last=True, num_workers=4)

    val_ds = SparseDataset(reader.val_samples, source, 'val')
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=4)

    return train_dl, val_dl, reader.shape, reader.num_timesteps

def save_to_file(model, name, train_config, loss_history):
    params = {
        'U': model.U, 'W': model.W, 'V': model.V,
        'U_bias': model.U_bias, 'W_bias': model.W_bias, 'V_bias': model.V_bias,
        'offset': model.offset, 'config': train_config, 'loss_history': loss_history
    }

    print(f'Saving to models/{name}.pkl')
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(params, f)


class Reader(object):

    def __init__(self, path, validation_frac):
        self.files = list(sorted(glob.glob(path)))
        self.train_samples, self.val_samples = [], []
        self.validation_frac = validation_frac
        self.shape = None
        self.num_timesteps = len(self.files)

    def run(self):
        _ = [self.load_and_split(filename, timestep)
                for timestep, filename in enumerate(self.files)]

    def load_and_split(self, filename, timestep):
        mat = sparse.load_npz(filename).tolil()
        empty = np.ravel(mat.sum(axis=1) == 0)
        valid_users = np.arange(mat.shape[0])[~empty]
        n_items = mat.shape[1]

        for user in tqdm(valid_users):
            user_items = mat[user]
            n_val = round(self.validation_frac * n_items)
            val_inds = np.random.choice(np.arange(n_items), n_val)
            val_targets = user_items[0, val_inds]
            user_items[0, val_inds] = 0

            self.train_samples.append((timestep, user, user_items))
            self.val_samples.append((timestep, user, val_targets, val_inds))

        self.shape = mat.shape

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, samples, source, split):
        self.samples = samples
        self.source = source
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'train':
            timestep, user, user_items = self.samples[idx]
            return timestep, user, user_items.toarray()[0], self.source
        else:
            timestep, user, val_targets, val_inds = self.samples[idx]
            return timestep, user, val_targets.toarray()[0], self.source, val_inds



import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
    
class DataLoader_cluster(object):
    def __init__(self, xs, ys,xc,yc, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            num_padding = (batch_size - (len(xc) % batch_size)) % batch_size
            x_padding = np.repeat(xc[-1:], num_padding, axis=0)
            y_padding = np.repeat(yc[-1:], num_padding, axis=0)
            xc = np.concatenate([xc, x_padding], axis=0)
            yc = np.concatenate([yc, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, xc, yc = self.xs[permutation], self.ys[permutation], self.xc[permutation], self.yc[permutation]
        self.xs = xs
        self.ys = ys
        self.xc = xc
        self.yc = yc

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_c = self.xc[start_ind: end_ind, ...]
                y_c = self.yc[start_ind: end_ind, ...]
                yield (x_i, y_i, x_c, y_c)
                self.current_ind += 1

        return _wrapper()
    
class DataLoader_multi_modal(object):
    def __init__(self,x1,y1,x2,y2,x3,y3, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x1) % batch_size)) % batch_size
            x_padding = np.repeat(x1[-1:], num_padding, axis=0)
            y_padding = np.repeat(y1[-1:], num_padding, axis=0)
            x1 = np.concatenate([x1, x_padding], axis=0)
            y1 = np.concatenate([y1, y_padding], axis=0)
            num_padding = (batch_size - (len(x2) % batch_size)) % batch_size
            x_padding = np.repeat(x2[-1:], num_padding, axis=0)
            y_padding = np.repeat(y2[-1:], num_padding, axis=0)
            x2 = np.concatenate([x2, x_padding], axis=0)
            y2 = np.concatenate([y2, y_padding], axis=0)
            num_padding = (batch_size - (len(x3) % batch_size)) % batch_size
            x_padding = np.repeat(x3[-1:], num_padding, axis=0)
            y_padding = np.repeat(y3[-1:], num_padding, axis=0)
            x3 = np.concatenate([x3, x_padding], axis=0)
            y3 = np.concatenate([y3, y_padding], axis=0)
        self.size = len(x1)
        self.num_batch = int(self.size // self.batch_size)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x1, y1, x2, y2, x3, y3 = self.x1[permutation], self.y1[permutation], self.x2[permutation], self.y2[permutation], self.x3[permutation], self.y3[permutation]
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_1 = self.x1[start_ind: end_ind, ...]
                y_1 = self.y1[start_ind: end_ind, ...]
                x_2 = self.x2[start_ind: end_ind, ...]
                y_2 = self.y2[start_ind: end_ind, ...]
                x_3 = self.x3[start_ind: end_ind, ...]
                y_3 = self.y3[start_ind: end_ind, ...]
                yield (x_1, y_1, x_2, y_2, x_3, y_3)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return  adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    for category in ['train_Trend', 'val_Trend', 'test_Trend']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    data['x_train'] = np.concatenate((data['x_train'], data['x_train_Trend']), axis=-1)
    data['y_train'] = np.concatenate((data['y_train'], data['y_train_Trend']), axis=-1)
    data['x_val'] = np.concatenate((data['x_val'], data['x_val_Trend']), axis=-1)
    data['y_val'] = np.concatenate((data['y_val'], data['y_val_Trend']), axis=-1)
    data['x_test'] = np.concatenate((data['x_test'], data['x_test_Trend']), axis=-1)
    data['y_test'] = np.concatenate((data['y_test'], data['y_test_Trend']), axis=-1)

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)

    return data


def load_dataset_multi_modal(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = np.float32(cat_data['x'])
        data['y_' + category] = np.float32(cat_data['y'])
    for category in ['train_text', 'val_text', 'test_text']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = np.float32(cat_data['x'])
        data['y_' + category] = np.float32(cat_data['y'])
    for category in ['train_index', 'val_index', 'test_index']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = np.float32(cat_data['x'])
        data['y_' + category] = np.float32(cat_data['y'])
    
    data['train_multi_modal_loader'] = DataLoader_multi_modal(data['x_train'], data['y_train'],data['x_train_text'], data['y_train_text'],data['x_train_index'], data['y_train_index'], batch_size)
    data['val_multi_modal_loader'] = DataLoader_multi_modal(data['x_val'], data['y_val'],data['x_val_text'], data['y_val_text'],data['x_val_index'], data['y_val_index'], valid_batch_size)
    data['test_multi_modal_loader'] = DataLoader_multi_modal(data['x_test'], data['y_test'],data['x_test_text'], data['y_test_text'],data['x_test_index'], data['y_test_index'], test_batch_size)
       
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse



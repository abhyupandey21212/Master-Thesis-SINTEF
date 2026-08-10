import os
import numpy as np
import pandas as pd
import torch
import scipy
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
import utils.utilities as utilities
from  sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_raw_data(folder, filter3D = None, filter2D = None, skip_2D = True, pyro_cutoff = 2500, mu_keys = ['Power 3D (kW)', 'Input Energy 3D (MWh)', 'Cooling Energy 3D (MWh)', 'Pyro 3D (°C)']
):
    dt = 0.5
    _3D_all = {}
    _2D_all = {}
    res_all = {}
    mu_dict = {}

    for file in os.listdir(folder):
        name = file.split('_')[0]
        if '3D' in file:
            df = pd.read_csv(folder+file)
            if not filter3D is None:
                df = df.merge(
                        filter3D, 
                        on=['X', 'Y', 'Z'], how='inner',
                        )
            _3D_all['columns'] = df.columns
            _3D_all[name] = df.to_numpy()

        if '2D' in file:
            if skip_2D:
                continue
            df = pd.read_csv(folder+file)
            if not filter2D is None:
                df = df.merge(
                        filter2D, 
                        on=['R', 'Z'], how='inner',
                        )
            _2D_all['columns'] = df.columns
            _2D_all[name] = df.to_numpy()

        if 'Results' in file:
            res = pd.read_csv(folder+file)
            temp = res['Pyro 3D (°C)']*res['Pyro 3D (°C)'].le(pyro_cutoff)
            res['Pyro 3D (°C)'] = temp.replace(0, pyro_cutoff)
            res['Cooling Power 3D (kW)'] = 1000*np.gradient(res['Cooling Energy 3D (MWh)'].to_numpy(), dt)
            res['Net Energy 3D (kWh)'] = 1000*(res['Input Energy 3D (MWh)'] - res['Cooling Energy 3D (MWh)'])
            res['Input Energy 3D (kWh)'] = 1000*res['Input Energy 3D (MWh)']
            res['Cooling Energy 3D (kWh)'] = 1000*res['Cooling Energy 3D (MWh)']

            res_all[name] = res
            res_all['columns'] = res.columns
            mu_ = res[mu_keys]
            mu_dict[name] = {'mu': mu_.to_numpy(), 'columns': mu_.columns}


        _3D_dict = {}
        for k in _3D_all.keys():
            if k == 'columns':
                continue
            # name = k.split('_')[0]
            # res_k = ''.join([k, '_Resultsabridged.csv'])
            if 'Unnamed' in _3D_all['columns'][-1]:
                _3D_dict[k] = {'temps': _3D_all[k][:,3:-1], 'res': res_all[k]}
            else:
                _3D_dict[k] = {'temps': _3D_all[k][:,3:], 'res': res_all[k]}
            nodes = _3D_all[k][:,:3]
        


    return _3D_dict, mu_dict, nodes, [_3D_all, _2D_all, res_all]

def calc_POD_basis(_3D_training, POD_dim):
    all_data = []
    for k in _3D_training.keys():
        if k == 'columns':
            continue
        data = _3D_training[k]#[:,3:]
        all_data.append(data)
    all_data = np.hstack(all_data)
    U,_,_ = scipy.linalg.svd(all_data)
    Ur = U[:,:POD_dim]
    return Ur

class DatasetManager:
    def __init__(self, data_dict, mu_dict, train_share, valid_share, POD_dim, seq_len, padding_key):
        self.POD_dim = POD_dim
        self.seq_len = seq_len
        self.data_dict = data_dict
        self.mu_dict = mu_dict
        n_sets = len(data_dict)
        n_train = int(np.floor(train_share*n_sets))
        n_valid = int(np.ceil(valid_share*n_sets))
        self.padding_key = padding_key

        np.random.seed(9)

        self.train_keys = np.random.choice([_ for _ in data_dict.keys()], n_train, replace = False)
        self.valid_keys = np.random.choice([_ for _ in data_dict.keys() if _ not in self.train_keys], n_valid, replace = False)
        self.test_keys = [_ for _ in data_dict.keys() if (_ not in self.train_keys and _ not in self.valid_keys)]

        self.train_data = {k:data_dict[k] for k in self.train_keys}
        self.valid_data = {k:data_dict[k] for k in self.valid_keys}
        self.test_data  = {k:data_dict[k] for k in self.test_keys}
        self.train_mu   = {k:mu_dict[k] for k in self.train_keys}
        self.valid_mu   = {k:mu_dict[k] for k in self.valid_keys}
        self.test_mu    = {k:mu_dict[k] for k in self.test_keys}

        self.trainY_unif = np.hstack([_['temps'] for _ in self.train_data.values()]).T #time in first dim for scaler

        self.YscalerPre = StandardScaler().fit(self.trainY_unif)
        self.trainY_unif_s = self.YscalerPre.transform(self.trainY_unif)
        self.Ur,self.Sr,self.Vtr = randomized_svd(self.trainY_unif_s, n_components=self.POD_dim)
        self.YscalerPost = MinMaxScaler().fit(self.trainY_unif_s @ self.Vtr.T)


        self.Yscaler = MinMaxScaler().fit(self.trainY_unif)
 

        trainX_unif = np.vstack([mu_dict[name]['mu'] for name in self.train_mu.keys()])
        self.trainX_unif = trainX_unif
        power_unif = trainX_unif[:,0].reshape(-1,1)
        trainU_unif = trainX_unif[:,:-1]
        self.Xscaler = MinMaxScaler().fit(trainX_unif)
        self.Uscaler = MinMaxScaler().fit(trainU_unif)
        self.Pscaler = MinMaxScaler().fit(power_unif)


        for k in self.train_keys:
            dummy = self.train_data[k]['temps'].T
            dummy1 = self.YscalerPre.transform(dummy)
            dummy2 = self.Yscaler.transform(dummy)
            self.train_data[k]['scaled_temps'] = torch.tensor(dummy2, dtype=torch.float32)
            dummy1 = dummy1 @ self.Vtr.T
            dummy1 = self.YscalerPost.transform(dummy1)
            self.train_data[k]['POD_coefs'] = torch.tensor(dummy1, dtype=torch.float32)
            # self.train_data[k]['POD_seqs'] = 

            power = self.Uscaler.transform(self.train_mu[k]['mu'][:,:-1])
            self.train_mu[k]['scaled_p'] = torch.tensor(power, dtype=torch.float32)
            dummy = self.train_mu[k]['mu']
            dummy = self.Xscaler.transform(dummy)
            self.train_mu[k]['scaled_mu'] = torch.tensor(dummy, dtype=torch.float32)
            self.train_mu[k]['scaled_mu_seqs'] = self.make_Xseqs(self.train_mu[k]['scaled_mu'], self.seq_len, self.padding_key)
            

        for k in self.valid_keys:
            dummy = self.valid_data[k]['temps'].T
            dummy1 = self.YscalerPre.transform(dummy)
            dummy2 = self.Yscaler.transform(dummy)
            self.valid_data[k]['scaled_temps'] = torch.tensor(dummy2, dtype=torch.float32)
            dummy = dummy1 @ self.Vtr.T
            dummy = self.YscalerPost.transform(dummy)
            self.valid_data[k]['POD_coefs'] = torch.tensor(dummy, dtype=torch.float32)

            dummy = self.valid_mu[k]['mu']
            dummy = self.Xscaler.transform(dummy)
            power = self.Uscaler.transform(self.valid_mu[k]['mu'][:,:-1])
            self.valid_mu[k]['scaled_p'] = torch.tensor(power, dtype=torch.float32)
            self.valid_mu[k]['scaled_mu'] = torch.tensor(dummy, dtype=torch.float32)
            self.valid_mu[k]['scaled_mu_seqs'] = self.make_Xseqs(self.valid_mu[k]['scaled_mu'], self.seq_len, self.padding_key)

        for k in self.test_keys:
            dummy = self.test_data[k]['temps'].T
            dummy1 = self.YscalerPre.transform(dummy)
            dummy2 = self.Yscaler.transform(dummy)
            self.test_data[k]['scaled_temps'] = torch.tensor(dummy2, dtype=torch.float32)
            dummy = dummy1 @ self.Vtr.T
            dummy = self.YscalerPost.transform(dummy)
            self.test_data[k]['POD_coefs'] = torch.tensor(dummy, dtype=torch.float32)

            power = self.Uscaler.transform(self.test_mu[k]['mu'][:,:-1])
            self.test_mu[k]['scaled_p'] = torch.tensor(power, dtype=torch.float32)            
            dummy = self.test_mu[k]['mu']
            dummy = self.Xscaler.transform(dummy)
            self.test_mu[k]['scaled_mu'] = torch.tensor(dummy, dtype=torch.float32)
            self.test_mu[k]['scaled_mu_seqs'] = self.make_Xseqs(self.test_mu[k]['scaled_mu'], self.seq_len, self.padding_key)

        #CONVERTING ALL TO TORCH
        self.YscalerPre = utilities.TorchStandardScaler(self.YscalerPre)
        self.YscalerPost = utilities.TorchMinMaxScaler(self.YscalerPost)
        self.Yscaler = utilities.TorchMinMaxScaler(self.Yscaler)
        self.Xscaler = utilities.TorchMinMaxScaler(self.Xscaler)
        self.Pscaler = utilities.TorchMinMaxScaler(self.Pscaler)
        self.Uscaler = utilities.TorchMinMaxScaler(self.Uscaler)
        self.Vtr = torch.tensor(self.Vtr, dtype=torch.float32)

    def make_Xseqs(self, series, seq_len, padding_key = ['zero', 'zero', 'init', 'init']):
        num_timesteps = series.shape[0]
        num_sensors = series.shape[1]
        if type(padding_key) == list:
            assert len(padding_key) == num_sensors
        elif type(padding_key) == str:
            padding_key = [padding_key]*num_sensors
        # concatenate zeros padding at beginning of sensor data along axis 0
        padding = torch.zeros((seq_len, num_sensors))
        for j in range(num_sensors):
            if padding_key[j] == 'init':
                padding[:,j] = series[0,j]
            elif padding_key[j] == 'zero':
                padding[:,j] = 0
        padded_series = torch.cat((padding, series), dim=0)

        lagged_sequences = torch.zeros((num_timesteps, seq_len, num_sensors))
        for i in range(lagged_sequences.shape[0]):
            lagged_sequences[i] = padded_series[i+1:i+seq_len+1, :]
        return lagged_sequences
    
    def make_SHRED_datasets(self, xkey = 'scaled_mu_seqs', ykey = 'POD_coefs'):
        train_datasetX = torch.cat([self.train_mu[k][xkey] for k in self.train_keys])
        valid_datasetX = torch.cat([self.valid_mu[k][xkey] for k in self.valid_keys])
        test_datasetX  = torch.cat([self.test_mu[k][xkey] for k in self.test_keys])

        train_datasetY = torch.cat([self.train_data[k][ykey] for k in self.train_keys])
        valid_datasetY = torch.cat([self.valid_data[k][ykey] for k in self.valid_keys])
        test_datasetY  = torch.cat([self.test_data[k][ykey] for k in self.test_keys])


        train_dataset = utilities.TimeSeriesDataset(train_datasetX, train_datasetY)
        valid_dataset = utilities.TimeSeriesDataset(valid_datasetX, valid_datasetY)
        test_dataset = utilities.TimeSeriesDataset(test_datasetX, test_datasetY)

        return train_dataset, valid_dataset, test_dataset

    def make_FORECASTER_datasets(self, shred, padding_key = ['zero', 'zero', 'init']):
        padding_key = padding_key + ['init' for _ in range(shred.lstm.hidden_dim)]
        # print(len(padding_key))
        train_datasetX = []
        train_datasetY = []
        for k in self.train_keys:
            x1 = self.train_mu[k]['scaled_mu'].detach()
            # print(x1.shape)
            x2 = shred.lstm(self.train_mu[k]['scaled_mu_seqs']).detach()
            x = torch.cat([x1,x2],dim=1)[:-1,:]
            x2next = x2[1:,:]
            # print(x.shape)
            xprev = self.make_Xseqs(x, self.seq_len, padding_key=padding_key)
            train_datasetX.append(xprev)
            train_datasetY.append(x2next)
        train_datasetX = torch.cat(train_datasetX)
        train_datasetY = torch.cat(train_datasetY)
        train_dataset =  utilities.TimeSeriesDataset(train_datasetX, train_datasetY)

        valid_datasetX = []
        valid_datasetY = []
        for k in self.valid_keys:
            x1 = self.valid_mu[k]['scaled_mu']
            x2 = shred.lstm(self.valid_mu[k]['scaled_mu_seqs'])
            x = torch.cat([x1,x2],dim=1)[:-1,:]
            x2next = x2[1:,:]
            xprev = self.make_Xseqs(x, self.seq_len, padding_key=padding_key)
            valid_datasetX.append(xprev)
            valid_datasetY.append(x2next)
        valid_datasetX = torch.cat(valid_datasetX)
        valid_datasetY = torch.cat(valid_datasetY)
        valid_dataset =  utilities.TimeSeriesDataset(valid_datasetX, valid_datasetY)

        test_datasetX = []
        test_datasetY = []
        for k in self.test_keys:
            x1 = self.test_mu[k]['scaled_mu']
            x2 = shred.lstm(self.test_mu[k]['scaled_mu_seqs'])
            x = torch.cat([x1,x2],dim=1)[:-1,:]
            x2next = x2[1:,:]
            xprev = self.make_Xseqs(x, self.seq_len, padding_key=padding_key)
            test_datasetX.append(xprev)
            test_datasetY.append(x2next)
        test_datasetX = torch.cat(test_datasetX)
        test_datasetY = torch.cat(test_datasetY)
        test_dataset =  utilities.TimeSeriesDataset(test_datasetX, test_datasetY)
        return train_dataset, valid_dataset, test_dataset



if __name__ == '__main__':
    cwd = Path.cwd()
    parent = str(cwd)
    print(parent)
    x,y,z = load_raw_data(parent+'\\data\\raw_data\\', filter3D=pd.concat([pd.read_csv(parent+'\\data\\nodes\\c1.csv'),pd.read_csv(parent+'\\data\\nodes\\c2.csv'),pd.read_csv(parent+'\\data\\nodes\\sus.csv')]))    
    data = {}
    for k in x.keys():
        if k == 'columns':
            continue
        name = k.split('_')[0]
        res_k = ''.join([name, '_Resultsabridged.csv'])
        data[name] = {'temps': x[k], 'res': z[res_k]}
    n_sets = len(data)
    n_train = int(np.floor(0.8*n_sets))
    n_valid = int(np.ceil(0.1*n_sets))
    n_test  = int(n_sets - n_train - n_valid)
    train_set = np.random.choice([_ for _ in data.keys()], n_train, replace = False)
    valid_set = np.random.choice([_ for _ in data.keys() if _ not in train_set], n_valid, replace = False)
    test_set = [_ for _ in data.keys() if (_ not in train_set and _ not in valid_set)]
    print([len(_) for _ in [train_set, valid_set, test_set]])
    print(len(data))





        

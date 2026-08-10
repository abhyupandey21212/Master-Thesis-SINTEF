import torch
import numpy as np

class TorchMinMaxScaler:
    def __init__(self, minmax_scaler):
        self.min = torch.tensor(minmax_scaler.data_min_, dtype=torch.float32)
        self.max = torch.tensor(minmax_scaler.data_max_, dtype=torch.float32)
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform(self, Z):
        return Z * (self.max - self.min) + self.min

class TorchStandardScaler:
    def __init__(self, std_scaler):
        self.mean  = torch.tensor(std_scaler.mean_, dtype  =torch.float32)
        self.var   = torch.tensor(std_scaler.var_  , dtype =torch.float32)
        self.stdev = torch.tensor(np.sqrt(self.var), dtype =torch.float32)

    def transform(self, X):
        return (X - self.mean) / self.stdev 
    
    def inverse_transform(self, Z):
        return (Z * self.stdev) + self.mean 
    
class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __getitemU__(self, index):
        return self.U[index]
    
    def __len__(self):
        return self.len
    
def integrate_series(series, dt = 0.5):
    if type(series) == torch.Tensor:
        out = torch.zeros_like(series)
    elif type(series) == np.ndarray:
        out = np.zeros_like(series)
    else:
        raise
    for j in range(1,series.shape[0]):
        out[j,:] = out[j-1,:] + series[j,:]*dt
    return out

def make_Xseqs(series, seq_len, padding_key = ['zero', 'init', 'init', 'init']):
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

def make_seqs_batch(batch_series, seq_len, padding_key = ['zero', 'init', 'init', 'init']):
        out = []
        for series in batch_series:
            out.append(make_Xseqs(series, seq_len, padding_key))
        return torch.cat(out)
        
def cooling_power(T_wall, cooling_coef = 1.07, Cv = 10, Tw = 20):
    return -cooling_coef*Cv*(Tw - T_wall.mean(dim=0))
from utils.utilities import *
from  sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
import numpy as np
import pandas as pd
import os

class DatasetManager:
    def __init__(self, train_share: float, valid_share: float, test_share: float, seq_len: int, POD_dim: int, Xdata = None, Ydata = None, Y_LFdata = None, Uidx = None, rd_seed = None):
        self.train_share = train_share
        self.valid_share = valid_share
        self.test_share = test_share
        self.seq_len = seq_len
        self.POD_dim = POD_dim
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.Y_LFdata = Y_LFdata
        self.Uidx = Uidx
        self.rd_seed = rd_seed

    def prep(self, Xdata = None, Ydata = None, debugging = False, Y_LFdata = None):
        """
        Xdata, Ydata : np.ndarray or torch.tensor
            Of shape (nfiles, ntimesteps, nfeatures)
        """
        if not self.rd_seed is None:
            np.random.seed(self.rd_seed)
 
        Xdata = self.Xdata if Xdata is None else Xdata
        self.Xdata = Xdata
        if type(Xdata) == np.ndarray:
            Xdata = torch.tensor(Xdata, dtype=torch.float32)
        Ydata = self.Ydata if Ydata is None else Ydata
        self.Ydata = Ydata
        if type(Ydata) == np.ndarray:
            Ydata = torch.tensor(Ydata, dtype=torch.float32)
        Y_LFdata = self.Y_LFdata if Y_LFdata is None else Y_LFdata
        if type(Y_LFdata) == np.ndarray:
            Y_LFdata = torch.tensor(Y_LFdata, dtype=torch.float32)

        if len(Ydata.shape) == 3:
            nfiles, ntimesteps, nfeatures = Ydata.shape
        elif len(Ydata.shape) == 2:
            ntimesteps, nfeatures = Ydata.shape
            nfiles = 1
        else:
            raise
        nsensors = Xdata.shape[-1]
        Xseq = make_seqs_batch(Xdata.reshape(nfiles, ntimesteps, nsensors), self.seq_len, ['init', 'init', 'init', 'zero'])


        self.Xseq = Xseq
        ntimesteps = Ydata.shape[0]
        
        nseq = Xseq.shape[0]

        n_trainval = int(nseq*(self.train_share+self.valid_share))
        print(n_trainval, nseq)
        idx_train_val = np.random.choice(range(nseq), n_trainval, replace = False)
        self.idx_train_val = idx_train_val


        ntrain = int(nseq * self.train_share)
        idx_train = np.random.choice(idx_train_val, ntrain, replace = False)
        idx_valid = [_ for _ in idx_train_val if not _ in idx_train]

        idx_test = [_ for _ in range(nseq) if not _ in idx_train_val]

        nvalid = len(idx_valid)
        ntest = len(idx_test)

        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test  = idx_test


        Ytrain = Ydata.reshape(-1, nfeatures)[idx_train]
        Ytest  = Ydata.reshape(-1, nfeatures)[idx_test]
        Yvalid = Ydata.reshape(-1, nfeatures)[idx_valid]

        self.trainY_unif = Ytrain
        Xtrain = Xseq[idx_train].reshape(-1, nsensors)
        Xtest  = Xseq[idx_test].reshape(-1, nsensors)
        Xvalid = Xseq[idx_valid].reshape(-1, nsensors)

        if not Y_LFdata is None:
            Y_LFtrain = Y_LFdata[idx_train].reshape(-1, nfeatures)
            Y_LFtest  = Y_LFdata[idx_test].reshape(-1, nfeatures)
            Y_LFvalid = Y_LFdata[idx_valid].reshape(-1, nfeatures)
        if debugging:
            print(f'Before POD, the shapes are: \n\tYtrain.shape = : {Ytrain.shape} \n\tYtest.shape = : {Ytest.shape} \n\tYvaild.shape = : {Yvalid.shape}')

        Yscaler_prePOD = StandardScaler().fit(Ytrain)
        Yscaler_prePOD = TorchStandardScaler(Yscaler_prePOD)
        self.Yscaler_prePOD = Yscaler_prePOD
        Ytest_s        = Yscaler_prePOD.transform(Ytest)#.reshape(ntrain*n_timesteps_per_file, n_features))
        Ytrain_s       = Yscaler_prePOD.transform(Ytrain)
        Yvalid_s       = Yscaler_prePOD.transform(Yvalid)
        if not Y_LFdata is None:
            Y_LFtest_s        = Yscaler_prePOD.transform(Y_LFtest)
            Y_LFtrain_s       = Yscaler_prePOD.transform(Y_LFtrain)
            Y_LFvalid_s       = Yscaler_prePOD.transform(Y_LFvalid)

        #U_full,S,Vt_full = sp.linalg.svd(Ytrain_s, full_matrices = False)
        #U = U_full[:,:POD_dim]
        #Vt = Vt_full[:POD_dim,:]
        U,S,Vt     = randomized_svd(Ytrain_s, n_components=self.POD_dim)
        Vt = torch.tensor(Vt, dtype=torch.float32)
        Ytest_POD  =  Ytest_s @ Vt.T
        Ytrain_POD = Ytrain_s @ Vt.T 
        Yvalid_POD = Yvalid_s @ Vt.T

        if not Y_LFdata is None:
            Y_LFtest_POD  =  Y_LFtest_s @ Vt.T
            Y_LFtrain_POD = Y_LFtrain_s @ Vt.T 
            Y_LFvalid_POD = Y_LFvalid_s @ Vt.T


        self.Vt = Vt
        if debugging:
            print(f'Compressed with POD to {self.POD_dim} modes \nNow the shapes are: \n\tYtrain_POD.shape = : {Ytrain_POD.shape} \n\tYtest_POD.shape = : {Ytest_POD.shape} \n\tYvaild_POD.shape = : {Yvalid_POD.shape}')

        #Normalize into [0,1]
        Yscaler    = MinMaxScaler().fit(Ytrain_POD)
        Yscaler    = TorchMinMaxScaler(Yscaler)
        self.Yscaler = Yscaler
        Ytest_POD  = Yscaler.transform(Ytest_POD)#.reshape(ntrain*n_timesteps_per_file, n_features))
        Ytrain_POD = Yscaler.transform(Ytrain_POD)
        Yvalid_POD = Yscaler.transform(Yvalid_POD)
        if not Y_LFdata is None:
            Y_LFtest_POD  = Yscaler.transform(Y_LFtest_POD)#.reshape(ntrain*n_timesteps_per_file, n_features))
            Y_LFtrain_POD = Yscaler.transform(Y_LFtrain_POD)
            Y_LFvalid_POD = Yscaler.transform(Y_LFvalid_POD)

        Xscaler = MinMaxScaler().fit(Xtrain)
        print(type(Xtrain))
        Xscaler = TorchMinMaxScaler(Xscaler)
        self.Xscaler = Xscaler
        Xtest   = Xscaler.transform(Xtest)
        Xtrain  = Xscaler.transform(Xtrain)
        print(type(Xtrain))
        Xvalid  = Xscaler.transform(Xvalid)

        if not Y_LFdata is None:
            print('CONVERTS TO NP, need to port to torch')
            Xtest  = np.hstack([Xtest, Y_LFtest_POD])
            Xtrain = np.hstack([Xtrain, Y_LFtrain_POD])
            Xvalid = np.hstack([Xvalid, Y_LFvalid_POD])
            nsensors = Xtest.shape[-1]

        #Reshaping Xdata back to (nfiles, ntimesteps nsensors) so that seq wont mix files
        Xtrain_seq = Xtrain.reshape(ntrain, self.seq_len, nsensors)
        Xtest_seq  = Xtest.reshape(ntest, self.seq_len, nsensors)
        Xvalid_seq = Xvalid.reshape(nvalid, self.seq_len, nsensors)


        if not self.Uidx is None:
            Utrain_seq = Xtrain_seq[:,:,self.Uidx]
            Utest_seq = Xtest_seq[:,:,self.Uidx]
            Uvalid_seq = Xvalid_seq[:,:,self.Uidx]

            train_dataset = TimeSeriesDataset(Xtrain_seq, Ytrain_POD)#, U=Utrain_seq)#, Xnoseq=Xtrain, Unoseq=Xtrain[:,:,self.Uidx])
            test_dataset  = TimeSeriesDataset(Xtest_seq, Ytest_POD)#, U=Utest_seq)#, Xnoseq=Xtest, Unoseq=Xtest[:,:,self.Uidx])
            valid_dataset = TimeSeriesDataset(Xvalid_seq, Yvalid_POD)#, U=Uvalid_seq)#, Xnoseq=Xvalid, Unoseq=Xvalid[:,:,self.Uidx])
        else:
            train_dataset = TimeSeriesDataset(Xtrain_seq, Ytrain_POD, Xtrain)
            test_dataset  = TimeSeriesDataset(Xtest_seq, Ytest_POD, Xtest)
            valid_dataset = TimeSeriesDataset(Xvalid_seq, Yvalid_POD, Xvalid)

        return train_dataset, test_dataset, valid_dataset
    
    def prep_forecaster(self, seq_len, model, train_share = None, test_share = None, randomize=True):
        if test_share is None and train_share is None:
            randomize = False
            train_share = 1
        elif test_share is None:
            assert not train_share is None
            test_share = 1 - train_share
        elif train_share is None:
            assert not test_share is None
            train_share = 1 - test_share
        Xforecaster = []
        Yforecaster = []
        Zforecaster = []
        for i, run in enumerate(self.Xdata):
            seq,_ = generate_lagged_sensor_measurements3(torch.tensor(run), 0, seq_len)
            latent0 = model.lstm(self.Xscaler.transform(seq[0]))
            Z0 = self.Ydata[i,0].reshape(1,-1)
            # print(model.post(model.sdn(latent0))[:,self.nodes_of_interest].detach().numpy())
            latents = [latent0 for _ in range(seq_len)]
            Z = [Z0 for _ in range(seq_len)]
            for j, s in enumerate(seq):
                s = self.Xscaler.transform(s)
                with torch.no_grad():
                    l = model.lstm(s)
                latents.append(l)
                # print(self.Ydata[i,j].shape)
                Z.append(self.Ydata[i,j].reshape(1,-1))
            latents = torch.cat(latents)
            Z = np.vstack(Z)
            assert Z.shape[0] == latents.shape[0]

            for i in range(seq.shape[0]):
                x = torch.cat([self.Xscaler.transform(seq[i]),latents[i:i+seq_len]],dim=1)
                y = latents[i+seq_len]
                Xforecaster.append(x)
                Yforecaster.append(y)
                Zforecaster.append(Z[i])
        Xforecaster = torch.stack(Xforecaster)
        Yforecaster = torch.stack(Yforecaster)
        Zforecaster = np.stack(Zforecaster)#,axis=2)

        # print(Xforecaster.shape, Yforecaster.shape)
        if randomize:
            idx_train = np.random.choice(range(Xforecaster.shape[0]),int(Xforecaster.shape[0]*self.train_share))
            idx_test = [_ for _ in range(Xforecaster.shape[0]) if not _ in idx_train]
        else:
            idx_train = np.arange(0,int(Xforecaster.shape[0]*self.train_share))
            idx_test = [_ for _ in range(Xforecaster.shape[0]) if not _ in idx_train]

        Xforecaster_train = Xforecaster[idx_train].detach()
        Xforecaster_test  = Xforecaster[idx_test].detach()
        Yforecaster_train = Yforecaster[idx_train].detach()
        Yforecaster_test  = Yforecaster[idx_test].detach()
        train_dataset = TimeSeriesDataset(Xforecaster_train, Yforecaster_train)
        test_dataset = TimeSeriesDataset(Xforecaster_test, Yforecaster_test)
        return train_dataset, test_dataset, Zforecaster[idx_train]
    
    def prep_forecaster_POD_as_state(self, seq_len, model, train_share = None, test_share = None, randomize=True):
        if test_share is None and train_share is None:
            randomize = False
            train_share = 1
        elif test_share is None:
            assert not train_share is None
            test_share = 1 - train_share
        elif train_share is None:
            assert not test_share is None
            train_share = 1 - test_share
        Xforecaster = []
        Yforecaster = []
        Zforecaster = []
        for i, run in enumerate(self.Xdata):
            seq,_ = generate_lagged_sensor_measurements3(torch.tensor(run), 0, seq_len)
            latent0 = model(self.Xscaler.transform(seq[0]))
            Z0 = self.Ydata[i,0].reshape(1,-1)
            # print(model.post(model.sdn(latent0))[:,self.nodes_of_interest].detach().numpy())
            latents = [latent0 for _ in range(seq_len)]
            Z = [Z0 for _ in range(seq_len)]
            for j, s in enumerate(seq):
                s = self.Xscaler.transform(s)
                with torch.no_grad():
                    l = model(s) #THIS IS THE ONLY LINE CHANGED
                latents.append(l)
                # print(self.Ydata[i,j].shape)
                Z.append(self.Ydata[i,j].reshape(1,-1))
            latents = torch.cat(latents)
            Z = np.vstack(Z)
            assert Z.shape[0] == latents.shape[0]

            for i in range(seq.shape[0]):
                x = torch.cat([self.Xscaler.transform(seq[i]),latents[i:i+seq_len]],dim=1)
                y = latents[i+seq_len]
                Xforecaster.append(x)
                Yforecaster.append(y)
                Zforecaster.append(Z[i])
        Xforecaster = torch.stack(Xforecaster)
        Yforecaster = torch.stack(Yforecaster)
        Zforecaster = np.stack(Zforecaster)#,axis=2)

        # print(Xforecaster.shape, Yforecaster.shape)
        if randomize:
            idx_train = np.random.choice(range(Xforecaster.shape[0]),int(Xforecaster.shape[0]*self.train_share))
            idx_test = [_ for _ in range(Xforecaster.shape[0]) if not _ in idx_train]
        else:
            idx_train = np.arange(0,int(Xforecaster.shape[0]*self.train_share))
            idx_test = [_ for _ in range(Xforecaster.shape[0]) if not _ in idx_train]

        Xforecaster_train = Xforecaster[idx_train].detach()
        Xforecaster_test  = Xforecaster[idx_test].detach()
        Yforecaster_train = Yforecaster[idx_train].detach()
        Yforecaster_test  = Yforecaster[idx_test].detach()
        train_dataset = TimeSeriesDataset(Xforecaster_train, Yforecaster_train)
        test_dataset = TimeSeriesDataset(Xforecaster_test, Yforecaster_test)
        return train_dataset, test_dataset, Zforecaster[idx_train]

    def prep_excl(self, Xexcl, Yexcl):
        X = self.Xscaler.transform(torch.tensor(Xexcl.T,dtype=torch.float32))
        Y = self.Yscaler.transform(self.Yscaler_prePOD.transform(torch.tensor(Yexcl.T,dtype=torch.float32)) @ self.Vt.T)
        Xseq, Y = generate_lagged_sensor_measurements3(X, Y, self.seq_len)
        return X, Y, TimeSeriesDataset(Xseq, Y)

    
    def handoff(self, model):
        model.POD_basis = self.Vt
        model.Yscaler = self.Yscaler
        model.Xscaler = self.Xscaler
        model.Yscaler_prePOD = self.Yscaler_prePOD
        model.manager = self
        if not model.forecaster is None:
            model.forecaster.manager = self

    def post(self, pred):
        unscaled = self.Yscaler.inverse_transform(pred.detach().numpy())
        unPODed = unscaled @ self.Vt
        prePODunscaled = self.Yscaler_prePOD.inverse_transform(unPODed)
        return prePODunscaled

#Fidning all relevant files in the data folder
def load_dataset(data_folder = 'complete_data\\abridged\\', coords_folder = 'complete_data\\crucible_nodes\\', POD_dim = 50, seq_len=10, pyro_saturation=True):
        #Fidning all relevant files in the data folder
    folder = data_folder
    coords_list3D = []
    for file in os.listdir(coords_folder):
        coords_list3D.append(pd.read_csv(coords_folder + file))

    def load_data(folder, coords_list3D = None):
        _3Dsamples = []
        resfiles = []
        musamples = []
        idx_to_file = {}
        idx = 0
        for file in os.listdir(folder):
            if '3D' in file:
                _3Ddf = pd.read_csv(folder + file)
                if not coords_list3D is None:
                    filtered = []
                    for coords in coords_list3D:
                        c = _3Ddf.merge(coords, on=['X', 'Y', 'Z'], how='inner')
                        filtered.append(c)
                    _3Ddf = pd.concat(filtered)            
                _3Dnodes = _3Ddf[['X', 'Y', 'Z']]
                try:
                    _3Ddf_samples_only = _3Ddf.drop(['X', 'Y', 'Z', 'Unnamed: 0'],axis=1)
                except:
                    _3Ddf_samples_only = _3Ddf.drop(['X', 'Y', 'Z',],axis=1)
                print(f'{file.split("_")[0]}: {_3Ddf_samples_only.shape}')

                _3Dsamples.append(_3Ddf_samples_only.to_numpy())

            if 'Results' in file:
                print(file)
                resfiles.append(file)
                idx_to_file[idx] = file
                idx+=1
                res = pd.read_csv(folder + file)   
                res['Net Energy 3D (MWh)'] = res['Input Energy 3D (MWh)'] - res['Cooling Energy 3D (MWh)']
                mu = res[['Time (h)', 'Power 3D (kW)', 'Pyro 3D (°C)', 'Net Energy 3D (MWh)']]
                musamples.append(mu.to_numpy().T)
        return np.stack(_3Dsamples, axis=2), np.stack(musamples, axis=2), _3Dnodes, idx_to_file
    Ysamples, Musamples, nodes3D, idx_to_file = load_data(folder, coords_list3D)
    timeseries = Musamples[0,:,0]
    Psamples = Musamples[1]
    Pyrosamples = Musamples[2]
    Esamples = Musamples[3]
    n_features, n_timesteps_per_file, n_files = Ysamples.shape #assumes all files have the same number]]

    # nodes = nodes3D.to_numpy()
    # sensor_nodes = pd.read_csv('labelled_nodes_3D.csv')
    # idxes = {}

    # for name in sensor_nodes['name']:
    #     temp = sensor_nodes[sensor_nodes['name'] == name][['X', 'Y', 'Z']].to_numpy()
    #     temp_idx = np.where((nodes[:,0] == temp[:,0]) & (nodes[:,1] == temp[:,1]) & (nodes[:,2] == temp[:,2]))
    #     idxes[name] = temp_idx[0][0]
    
    # pyro_idx, c1core_idx, c1surf_idx, c2core_idx, c2surf_idx, sus_idx = idxes.values()
    # nodes_of_interest = [c1core_idx, c1surf_idx, c2core_idx, c2surf_idx, sus_idx]

    if pyro_saturation:
        pyro_cutoff = 2500
        pyro_unsat = Pyrosamples.copy()
        pyro_flag = np.ones_like(Pyrosamples)
        idx1, idx2 = np.where(pyro_unsat > pyro_cutoff)
        Pyrosamples[idx1,idx2] = pyro_cutoff
        pyro_flag[idx1,idx2] = 0

    Esamples = Musamples[3]


    np.random.seed(13)
    input_x = np.stack([Pyrosamples, pyro_flag, Esamples]) #ADD ENERGY LATER AFTER DECIDING IF ITS SENSOR OR INPUT
    input_u = np.stack([Psamples])
    sensor_in_dim = input_x.shape[0]
    ctrl_in_dim = input_u.shape[0]
    input_all = np.vstack([input_x, input_u])
    ctrl_idx = -1
    out_dim = POD_dim

    excl = np.random.choice(range(Ysamples.shape[-1]))
    rest = [_ for _ in range(Ysamples.shape[-1]) if not _ == excl]
    Yexcl = Ysamples[:,:,excl]
    Ysamples = Ysamples[:,:,rest]
    Xexcl = input_all[:,:,excl]
    input_all = input_all[:,:,rest]

    tail_thinning = False
    if tail_thinning:
        X_heatup = input_all[:,:31,:]
        Y_heatup = Ysamples[:,:31,:]

        X_cooldown = np.concatenate([input_all[:,31:,:],input_all[:,-1,:].reshape(input_all.shape[0],1,-1)],axis=1)
        Y_cooldown = np.concatenate([Ysamples[:,31:,:],Ysamples[:,-1,:].reshape(Ysamples.shape[0],1,-1)],axis=1)
        cooldown_idx = np.random.choice(range(Y_cooldown.shape[-1]), Y_cooldown.shape[-1]//2)
        Y_cooldown = Y_cooldown[:,:,cooldown_idx]
        X_cooldown = X_cooldown[:,:,cooldown_idx]

        Y_final = np.concatenate([Y_heatup,Y_cooldown],axis=2)
        X_final = np.concatenate([X_heatup,X_cooldown],axis=2)
    else:
        Y_final = Ysamples
        X_final = input_all


    rd_seed = 9
    manager = DatasetManager(0.8, 0.1, 0.1, seq_len=seq_len, POD_dim=POD_dim, Xdata=X_final.T, Ydata=Y_final.T, Uidx = ctrl_idx, rd_seed=rd_seed)

    train_dataset, test_dataset, valid_dataset = manager.prep(debugging=False)

    return Musamples, Ysamples, X_final, Y_final, Xexcl, Yexcl, train_dataset, test_dataset, valid_dataset, manager




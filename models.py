import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


class SDN(nn.Module):
    def __init__(
                 self,
                 in_size,
                 out_size,
                 inner_layers,
                 dropout,
                 batch_norm,
                ):
        super(SDN, self).__init__()
        self.layers = nn.ModuleList()
        sdn_sizes = [in_size] + inner_layers + [out_size]
        for j in range(len(sdn_sizes) - 1):
            self.layers.append(nn.Linear(sdn_sizes[j], sdn_sizes[j+1]))
            if not j == len(sdn_sizes) - 2: #All but last layer
                self.layers.append(nn.ReLU(True))
                if dropout[0]:
                    self.layers.append(nn.Dropout(dropout[1]))
                if batch_norm[0]:
                    self.layers.append(nn.BatchNorm1d())#ADD PARAMS))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, batch_first, seq_len):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=batch_first)

    def forward(self,u):
        udim = len(u.shape)
        #------LSTM------
        if udim == 1: #Single 
            raise ValueError('SHRED needs an input sequence, not a single input. u.shape == 1, so either this is a single input or wrong shape. Shape must be (seq_len, in_dim)')
        elif udim == 2: #Unbatched
            # print('LSTM unbatched')
            assert u.shape[0] == self.seq_len, f'Input sequence length must be shred.seq_len = {self.seq_len}'
            assert u.shape[1] == self.in_dim, f'Each element of the input sequnece must have dim = shred.in_dim = {self.in_dim}'

            c0 = torch.zeros((self.n_layers, self.hidden_dim), dtype=torch.float32)
            h0 = torch.zeros((self.n_layers, self.hidden_dim), dtype=torch.float32)

            out, (h0,c0) = self.lstm(u,(h0,c0))
            # print(out.shape)
            out = out[-1,:].reshape(1,-1)
            # print('lstm out.dtype', out.dtype)


        elif udim ==    3: #Batched
            batch_size = u.shape[0]
            assert u.shape[1] == self.seq_len, f'Input sequence length must be shred.seq_len = {self.seq_len}'
            assert u.shape[2] == self.in_dim, f'Each element of the input sequnece must have dim = shred.in_dim = {self.in_dim}'
            c0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float32)
            h0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float32)
            # print('c0, h0', c0.dtype, h0.dtype)
            # print('LSTM batched')
            for j in range(u.shape[1]):
                uj = u[:,j,:].reshape(batch_size,1,-1)
                # print('uj.dtype',uj.dtype)
                out, (h0,c0) = self.lstm(uj,(h0,c0)) #ROLLED
                out = out[:,-1,:]
                # print('lstm out.dtype', out.dtype)

        else:
            raise NotImplementedError('Unfamiliar input size')
        return out

class SHRED(nn.Module):
    def __init__(
                 self, 
                 in_dim, 
                 out_dim, 
                 lstm_params = {'seq_len': 10, 
                                'n_layers': 1,
                                'hidden_dim': 64,
                                },
                 sdn_params = {'inner_layers': [64,64],
                              'dropout': [True, 0.1],
                              'batch_norm': [False],
                              },
                 batch_first=True,
                 data_manager=None,
                 ):
        super(SHRED, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.seq_len=lstm_params['seq_len']
        self.batch_first=batch_first
        self.data_manager = data_manager

        self.lstm = LSTM(in_dim=in_dim, 
                         hidden_dim=lstm_params['hidden_dim'], 
                         n_layers=lstm_params['n_layers'], 
                         batch_first=batch_first,
                         seq_len=self.seq_len
                         )

        self.sdn = SDN(
                        in_size=lstm_params['hidden_dim'], out_size=out_dim, inner_layers=sdn_params['inner_layers'], 
                        dropout=sdn_params['dropout'],
                        batch_norm=sdn_params['batch_norm']
                        )

    def forward(self,u):
        out = self.lstm(u)
        #------SDN------
        return self.sdn(out)
    
    
    def fit(self, train_dataset, valid_dataset, optimizer, loss, batch_size, max_epoch, patience, plotting=False):
        validation_error = []
        training_error = []
        min_val_error = 1e12

        init_params = self.state_dict()
        best_epoch = epoch = 0
        best_params = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        disappointment = 0

        while epoch < max_epoch and disappointment <= patience:
            te = 0
            print(f'-----Epoch {epoch}-----')
            epoch_train_error = 0
            self.train(True)

            #Epoch training
            for i, (x,y) in enumerate(train_loader):
                pred = self(x)
                optimizer.zero_grad() #Making sure optimizer starts with no grad from last
                l = loss(pred, y) 
                l.backward() #calulating loss and grad of loss
                optimizer.step() #using grad of loss to make optimizer step
                te+=l.item()
            te/=i #averaging the sum of all batches' training loss over no. batches
            training_error.append(te)

            #Epoch validation
            self.eval()
            with torch.no_grad(): #no grad here to avoid pointless calc, coz we will not step optimizer
                pred_validation = self(valid_dataset.X)
                ve = loss(pred_validation, valid_dataset.Y)
                validation_error.append(ve)
                if ve < min_val_error: #Was this step an improvement on the validation dataset?
                    min_val_error = ve
                    best_epoch = epoch
                    disappointment = 0
                    best_params = self.state_dict()
                
                else:
                    disappointment+=1
            
            epoch+=1
            print(f'Training loss  : {te}')
            print(f'Validation loss: {ve}')
        if not best_params is None:
            self.load_state_dict(best_params)
        else:
            print('TRAINING NEVER IMPROVED VALIDATIN ERROR, resetting to inital state')
            self.load_state_dict(init_params)
        if plotting:
            plt.figure()
            plt.title('Loss per epoch')
            plt.semilogy(np.arange(len(training_error)), training_error, label = 'Training')
            plt.semilogy(np.arange(len(validation_error)), validation_error, label='Validation')
            plt.vlines(best_epoch, min([min(validation_error), min(training_error)]), max([max(validation_error), max(training_error)]))
            plt.xlabel('Epoch')
            plt.ylabel('MSE loss')
            plt.legend()
            plt.show()
        return training_error, validation_error
    
    def post(self, pred):
        if self.data_manager is None:
            raise NotImplementedError('data_manager is not provided.')
        unscaled = self.data_manager.YscalerPost.inverse_transform(pred)
        unPODed = unscaled @ self.data_manager.Vtr
        prePODunscaled = self.data_manager.YscalerPre.inverse_transform(unPODed)
        return prePODunscaled

class SHREDForecaster(nn.Module):
    def __init__(
                 self, 
                 u_dim, 
                 out_dim, 
                 lstm_params = {'seq_len': 10, 
                                'n_layers': 1,
                                'hidden_dim': 64,
                                },
                 batch_first=True,
                 data_manager=None,
                 ):
        super(SHREDForecaster, self).__init__()
        self.u_dim = u_dim
        self.out_dim=out_dim
        self.seq_len=lstm_params['seq_len']
        self.hidden_dim = lstm_params['hidden_dim']
        self.batch_first=batch_first
        self.data_manager = data_manager

        self.lstm = LSTM(u_dim, self.hidden_dim, lstm_params['n_layers'], batch_first=batch_first, seq_len=self.seq_len)
        self.proj = nn.Linear(self.hidden_dim, out_dim)

    def forward(self,u):
        # print('Called forecaster')
        # print(u.dtype)
        #------LSTM------
        out = self.lstm(u)
        # print('lstm worked')
        #------SDN------
        out = self.proj(out)
        # print('forecaster_out',out.dtype)
        return out
    
    
    def fit(self, train_dataset, valid_dataset, optimizer, loss, batch_size, max_epoch, patience, plotting=False):
        validation_error = []
        training_error = []
        min_val_error = 1e12

        init_params = self.state_dict()
        best_epoch = epoch = 0
        best_params = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        disappointment = 0

        while epoch < max_epoch and disappointment <= patience:
            te = 0
            print(f'-----Epoch {epoch}-----')
            epoch_train_error = 0
            self.train()

            #Epoch training
            for i, (x,y) in enumerate(train_loader):
                pred = self(x)
                optimizer.zero_grad() #Making sure optimizer starts with no grad from last
                l = loss(pred, y) 
                l.backward() #calulating loss and grad of loss
                optimizer.step() #using grad of loss to make optimizer step
                te+=l.item()
            te/=i #averaging the sum of all batches' training loss over no. batches
            training_error.append(te)

            #Epoch validation
            self.eval()
            with torch.no_grad(): #no grad here to avoid pointless calc, coz we will not step optimizer
                pred_validation = self(valid_dataset.X)
                ve = loss(pred_validation, valid_dataset.Y)
                validation_error.append(ve)
                if ve < min_val_error: #Was this step an improvement on the validation dataset?
                    min_val_error = ve
                    best_epoch = epoch
                    disappointment = 0
                    best_params = self.state_dict()
                
                else:
                    disappointment+=1
            
            epoch+=1
            print(f'Training loss  : {te}')
            print(f'Validation loss: {ve}')
        if not best_params is None:
            self.load_state_dict(best_params)
        else:
            print('TRAINING NEVER IMPROVED VALIDATIN ERROR, resetting to inital state')
            self.load_state_dict(init_params)
        if plotting:
            plt.figure()
            plt.title('Loss per epoch')
            plt.semilogy(np.arange(len(training_error)), training_error, label = 'Training')
            plt.semilogy(np.arange(len(validation_error)), validation_error, label='Validation')
            plt.vlines(best_epoch, min([min(validation_error), min(training_error)]), max([max(validation_error), max(training_error)]))
            plt.xlabel('Epoch')
            plt.ylabel('MSE loss')
            plt.legend()
            plt.show()
        return training_error, validation_error
    
class WeightedLoss(nn.Module):
    def __init__(self, weight_diag):
        super().__init__()
        self.weight_diag = weight_diag / weight_diag.mean()  # (r,)
    def forward(self, input, target):
        diff = input - target # (B, r), both in MinMax-scaled space
        return (diff**2 * self.weight_diag).sum(dim=1).mean()

class CustomLoss(nn.Module):
    def __init__(self, manager, c1_picker, c2_picker, sus_picker = None, coefs = [1,1,1,1]):
        super(CustomLoss, self).__init__()
        assert len(coefs) == 4
        self.mse = nn.MSELoss()
        self.c1_picker = c1_picker.T
        self.c2_picker = c2_picker.T
        self.sus_picker = sus_picker.T
        self.data_manager = manager
        self.a1, self.a2, self.a3, self.a4 = coefs

    def post(self, pred):
        if self.data_manager is None:
            raise NotImplementedError('data_manager is not provided.')
        unscaled = self.data_manager.YscalerPost.inverse_transform(pred)
        unPODed = unscaled @ self.data_manager.Vtr
        prePODunscaled = self.data_manager.YscalerPre.inverse_transform(unPODed)
        return prePODunscaled

    def forward(self, input, target):
        # Compute the loss
        l1 = self.mse(input, target)
        input_post = self.data_manager.Yscaler.inverse_transform(input)
        # target_post = self.data_manager.YscalerPre.inverse_transform(target)
        l2 = (torch.relu(-input_post)**2).mean()
        # target_post_c1 = target_post@self.c1_picker
        # target_post_c2 = target_post@self.c2_picker
        # target_post_sus = target_post@self.sus_picker
        # l2 = self.mse(input_post@self.c1_picker, target_post_c1) / target_post_c1.pow(2).mean()
        # l3 = self.mse(input_post@self.c2_picker, target_post_c2) / target_post_c2.pow(2).mean()
        # l4 = self.mse(input_post@self.sus_picker, target_post_sus) / target_post_sus.pow(2).mean()
        # print(self.a1*l1.item(), self.a2*l2.item(), self.a3*l3.item(), self.a4*l4.item())
        return self.a1*l1 + self.a2*l2# + self.a3*l3 + self.a4*l4
    
class CostWrapperMultipleShooting(nn.Module):
    def __init__(self, shred, f, u_dim, target, picker, P, x_tilde, u_tilde):
        super(CostWrapperMultipleShooting, self).__init__()
        self.f = f
        self.u_dim = u_dim
        self.shred = shred
        self.target, self.picker = target, picker
        self.P = P.detach()
        self.x_tilde = x_tilde.reshape(-1,1)
        self.u_tilde = u_tilde.reshape(-1,1)
        self.x_dim = self.x_tilde.shape[0]


    def Z(self, x):
        j = x.shape[0]
        return self.shred.post(self.shred.sdn(x)).reshape(j,-1)

    def cost(self, u_plan, X_plan, Vf_toggle):
        Z_plan = self.Z(X_plan)
        xf = X_plan[-1,:].reshape(-1,1)        
        exf = self.x_tilde - xf
        Vf = 0.5* exf.T@self.P@exf #torch.einsum('ij,jk,ik->i', Xf, self.P, Xf)
        ez = (self.target - Z_plan @ self.picker.T) / torch.sqrt(self.target**2)
        l = (torch.nn.Softplus(beta=10)(ez)).mean()
        cst = l+Vf_toggle*Vf
        return cst.reshape(1,1)
    
    def forward(self, w):
        f = self.f
        x_dim = self.shred.lstm.hidden_dim
        #Descision variables
        #p_plan     1*f
        #x_plan     out_dim*f
        #Params
        #E_plan     1*f
        #C_plan     1*f
        #Vf_toggle  1*1
        p_plan_idx = 1*f
        x_plan_idx = p_plan_idx + x_dim*f
        E_plan_idx = x_plan_idx + 1*f
        C_plan_idx = E_plan_idx + 1*f

        p_plan = w[:p_plan_idx].reshape(f,1)
        x_plan = w[p_plan_idx:x_plan_idx].reshape(f,x_dim)
        E_plan = w[x_plan_idx:E_plan_idx].reshape(f,1)
        C_plan = w[E_plan_idx:C_plan_idx].reshape(f,1)
        Vf_toggle=w[-1].reshape(1,1)
        u_plan = torch.cat([p_plan, E_plan, C_plan])
        return self.cost(u_plan,x_plan,Vf_toggle)
    

class ForecasterWrapperdo_mpc(nn.Module):
    def __init__(self, forecaster):
        super(ForecasterWrapperdo_mpc, self).__init__()
        self.forecaster = forecaster
        self.seq_len = forecaster.seq_len
        self.x_dim = forecaster.out_dim
        self.u_dim = forecaster.u_dim - self.x_dim
    def forward(self,ux):
        u = ux[:self.u_dim*self.seq_len].reshape(self.seq_len,self.u_dim)
        x = ux[self.u_dim*self.seq_len:].reshape(self.seq_len,self.x_dim)
        x_next = self.forecaster(torch.cat([u,x],dim=1))
        return x_next.reshape(-1,1)

class UnivForecasterWrapper(nn.Module):
    def __init__(self, forecaster, u_picker):
        super(UnivForecasterWrapper, self).__init__()
        self.forecaster = forecaster
        self.seq_len = forecaster.seq_len
        self.x_dim = forecaster.out_dim
        self.u_dim = forecaster.u_dim - self.x_dim
        self.u_picker = u_picker
    def forward(self,ux):
        u = ux[:-self.x_dim*self.seq_len].reshape(self.seq_len,-1)
        # print(u.shape, self.u_picker.T.shape)
        u = u @ self.u_picker.T
        x = ux[-self.x_dim*self.seq_len:].reshape(self.seq_len,self.x_dim)
        x_next = self.forecaster(torch.cat([u,x],dim=1))
        return x_next.reshape(-1,1)
    
class CoolingPowerWrapper(nn.Module):
    def __init__(self, shred, outer_picker):
        super(CoolingPowerWrapper, self).__init__()
        self.sdn = shred.sdn
        self.post = shred.post
        self.outer_picker_avg = outer_picker.mean(dim=0, keepdim=True)  # shape (1, 4184), same as pyro_picker
    def forward(self,a):
        Z = self.post(self.sdn(a.reshape(1,-1)))
        T_wall = (Z@self.outer_picker_avg.T)
        P_cool = 1.07*10*(T_wall - 20)
        return P_cool.reshape(1,1)
    
class PyroWrapper(nn.Module):
    def __init__(self, shred, pyro_picker):
        super(PyroWrapper, self).__init__()
        self.sdn = shred.sdn
        self.post = shred.post
        self.pyro_picker = pyro_picker
    def forward(self,a):
        Z = self.post(self.sdn(a.reshape(1,-1)))
        return (Z@self.pyro_picker.T).reshape(1,1)

    
class StageCost(nn.Module):
    def __init__(self, shred, f, u_dim, x_dim, target, picker):
        super(StageCost, self).__init__()
        self.f = f
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.post = shred.post
        self.sdn = shred.sdn
        self.target, self.picker = target, picker
    def forward(self,w):
        assert w.shape[0] == self.u_dim + self.x_dim, f'The input must have dim (self.u_dim + self.x_dim, 1) = ({self.u_dim + self.x_dim},1), got {w.shape[0]}'
        u = w[:self.u_dim].reshape(1, self.u_dim)
        x = w[self.u_dim:].reshape(1, self.x_dim)
        Z = self.post(self.sdn(x))
        e_Z = (self.target - Z @ self.picker.T) / torch.sqrt(self.target**2)

        #Terms of the stage cost
        l_x = 0
        l_u = 0
        l_Z = (torch.nn.Softplus(beta=10)(e_Z)).mean()

        return (l_x + l_u + l_Z).reshape(1,1)
    
class CombinedWrapper(nn.Module):
    def __init__(self, shred, f, u_dim, x_dim, picker_avg):
        super(CombinedWrapper, self).__init__()
        self.f = f
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.post = shred.post
        self.sdn = shred.sdn
        self.picker_avg = picker_avg

    def forward(self,x):
        assert x.shape[0] == self.x_dim#, f'The input must have dim (self.u_dim + self.x_dim, 1) = ({self.u_dim + self.x_dim},1), got {x.shape[0]}'
        x = x.reshape(1, self.x_dim)
        Z = self.post(self.sdn(x))
        return (Z @ self.picker_avg.T).reshape(-1,1)

class HorizonWrapper(nn.Module):
    def __init__(
            self, 
            forecaster, 
            shred, 
            target, 
            c1_picker,
            c2_picker,
            sus_picker, 
            outer_picker_avg, 
            pyro_picker, 
            dt, 
            f,
            seq_len, 
            x_dim, 
            u_dim, 
            u_min, 
            u_max,
            Vf,
            x_tilde,
            sus_coef=10
        ):
        super(HorizonWrapper, self).__init__()
        self.forecaster = forecaster
        self.post = shred.post
        self.sdn = shred.sdn
        self.picker = torch.cat([c1_picker, c2_picker, -1*sus_picker])
        self.target = target
        self.outer_picker_avg = outer_picker_avg
        self.pyro_picker = pyro_picker
        self.dt = dt
        self.f = f
        self.seq_len = seq_len
        self.x_dim = x_dim
        self.u_dim = u_dim  # dim of (p,E,C,pyro) block, i.e. 4
        self.scaler = torch.sqrt(self.target**2)
        self.pmin, self.Emin, self.Cmin, self.pyromin, self.p_coolmin, self.E_netmin = u_min
        self.pmax, self.Emax, self.Cmax, self.pyromax, self.p_coolmax, self.E_netmax = u_max
        error_weight = torch.ones(self.picker.shape[0])
        error_weight[-sus_picker.shape[0]:] = sus_coef
        self.error_weight = torch.diag(error_weight)
        self.Vf = Vf
        self.x_tilde = x_tilde

    def stage_cost_j(self, Z):
        e_Z = (self.target - Z @ self.picker.T)# @ self.error_weight.T
        return (torch.nn.Softplus(beta=10)(e_Z)).mean()
        return e_Z.mean()

    def predict_custom(self, p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle):
        # print('------CALL------')
        old_f = self.f
        self.f = p_guess.shape[0]
        # print('p_guess',p_guess.shape)
        # print('uprev0',uprev0.shape)
        # print('xhist0',xhist0.shape)
        # print('E0',E0.shape)
        # print('C0',C0.shape)
        # print('pyro0',pyro0.shape)
        # print('p_cool0',p_cool0.shape)
        # print('E_net0',E_net0.shape)
        cost,info = self.predict(p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle)
        self.f = old_f
        return cost,info

    def predict(self, p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle):
        Ej, Cj, pyroj, p_coolj, E_netj = E0, C0, pyro0, p_cool0, E_net0
        xhist_j, uprev_j = xhist0, uprev0
        Z_list, x_list, cost = [], [], torch.zeros((1,1),dtype=torch.float32)
        for j in range(self.f):
            pj = p_guess[j:j+1]
            pj_us = pj * (self.pmax - self.pmin) + self.pmin
            Ej_us = Ej * (self.Emax - self.Emin) + self.Emin
            Cj_us = Cj * (self.Cmax - self.Cmin) + self.Cmin

            #Forecasting
            uj = torch.cat([pj, Ej, Cj, pyroj, p_coolj, E_netj], dim=1)
            uhist_j = torch.cat([uprev_j, uj], dim=0)
            uxhist = torch.cat([uhist_j.flatten(), xhist_j.flatten()], dim=0)
            x_next = self.forecaster(uxhist).reshape(1,-1)
            Z_next = self.post(self.sdn(x_next))

            #Updating
            Ej_us = Ej_us + pj_us*self.dt
            T_wall = Z_next @ self.outer_picker_avg.T
            p_coolj_us = 1.07*20*(T_wall - 20)
            # print(T_wall, p_coolj_us)
            Cj_us = Cj_us + p_coolj_us*self.dt
            pyroj_us = Z_next @ self.pyro_picker.T
            E_netj_us = Ej_us - Cj_us

            #Scaling
            Ej = (Ej_us - self.Emin) / (self.Emax - self.Emin)
            # print(Ej)
            Cj = (Cj_us - self.Cmin) / (self.Cmax - self.Cmin)
            pyroj = (pyroj_us - self.pyromin) / (self.pyromax - self.pyromin)
            p_coolj = (p_coolj_us - self.p_coolmin) / (self.p_coolmax - self.p_coolmin)
            E_netj = (E_netj_us - self.E_netmin) / (self.E_netmax - self.E_netmin)

            # accumulate stage cost HERE -- this is the key structural change

            cost = cost + self.stage_cost_j(Z_next) # your l_x/l_u/l_Z logic, inline

            Z_list.append(Z_next)
            x_list.append(x_next)
            xhist_j = torch.cat([xhist_j[1:], x_next], dim=0)
            uprev_j = torch.cat([uprev_j[1:], uj], dim=0)
    
        xN = x_next.reshape(-1,1)
        eN = xN - self.x_tilde.reshape(-1,1)
        term_cost = Vf_toggle*(eN.T @ self.Vf @ eN)
        cost = cost + term_cost

        return cost.reshape(1,1), [torch.cat(Z_list), torch.cat(x_list), Ej, Cj, pyroj, p_coolj, E_netj, xhist_j,uprev_j]

    def forward(self, flat_in):
        # unpack the single flat vector -- pick ONE consistent order and stick to it
        idx = 0
        p_guess = flat_in[idx:idx+self.f].reshape(self.f, 1)
        idx += self.f
        uprev0 = flat_in[idx:idx+(self.seq_len-1)*self.u_dim].reshape(self.seq_len-1, self.u_dim)
        idx += (self.seq_len-1)*self.u_dim
        xhist0 = flat_in[idx:idx+self.seq_len*self.x_dim].reshape(self.seq_len, self.x_dim)
        idx += self.seq_len*self.x_dim
        E0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        C0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        pyro0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        p_cool0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        E_net0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        Vf_toggle = flat_in[-1].reshape(1,1)

        cost,_ = self.predict(p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle)
        return cost


class HorizonWrapperQcost(nn.Module):
    def __init__(
            self, 
            forecaster, 
            shred, 
            target, 
            c1_picker,
            c2_picker,
            sus_picker, 
            outer_picker_avg, 
            pyro_picker, 
            dt, 
            f,
            seq_len, 
            x_dim, 
            u_dim, 
            u_min, 
            u_max,
            Vf,
            x_tilde,
            sus_coef=10
        ):
        super(HorizonWrapperQcost, self).__init__()
        self.forecaster = forecaster
        self.post = shred.post
        self.sdn = shred.sdn
        self.picker = torch.cat([c1_picker, c2_picker, -1*sus_picker])
        self.target = target
        self.outer_picker_avg = outer_picker_avg
        self.pyro_picker = pyro_picker
        self.dt = dt
        self.f = f
        self.seq_len = seq_len
        self.x_dim = x_dim
        self.u_dim = u_dim  # dim of (p,E,C,pyro) block, i.e. 4
        self.scaler = torch.sqrt(self.target**2)
        self.pmin, self.Emin, self.Cmin, self.pyromin, self.p_coolmin, self.E_netmin = u_min
        self.pmax, self.Emax, self.Cmax, self.pyromax, self.p_coolmax, self.E_netmax = u_max
        error_weight = torch.ones(self.picker.shape[0])
        error_weight[-sus_picker.shape[0]:] = sus_coef
        self.error_weight = torch.diag(error_weight)
        self.Vf = Vf
        self.x_tilde = x_tilde

    def stage_cost_j(self, Z):
        e_Z = (self.target - Z @ self.picker.T)
        # print(e_Z.shape)
        return e_Z@e_Z.T

    def predict_custom(self, p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle):
        # print('------CALL------')
        old_f = self.f
        self.f = p_guess.shape[0]
        # print('p_guess',p_guess.shape)
        # print('uprev0',uprev0.shape)
        # print('xhist0',xhist0.shape)
        # print('E0',E0.shape)
        # print('C0',C0.shape)
        # print('pyro0',pyro0.shape)
        # print('p_cool0',p_cool0.shape)
        # print('E_net0',E_net0.shape)
        cost,info = self.predict(p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle)
        self.f = old_f
        return cost,info

    def predict(self, p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle):
        Ej, Cj, pyroj, p_coolj, E_netj = E0, C0, pyro0, p_cool0, E_net0
        xhist_j, uprev_j = xhist0, uprev0
        Z_list, x_list, cost = [], [], torch.zeros((1,1),dtype=torch.float32)
        for j in range(self.f):
            pj = p_guess[j:j+1]
            pj_us = pj * (self.pmax - self.pmin) + self.pmin
            Ej_us = Ej * (self.Emax - self.Emin) + self.Emin
            Cj_us = Cj * (self.Cmax - self.Cmin) + self.Cmin

            #Forecasting
            uj = torch.cat([pj, Ej, Cj, pyroj, p_coolj, E_netj], dim=1)
            uhist_j = torch.cat([uprev_j, uj], dim=0)
            uxhist = torch.cat([uhist_j.flatten(), xhist_j.flatten()], dim=0)
            x_next = self.forecaster(uxhist).reshape(1,-1)
            Z_next = self.post(self.sdn(x_next))

            #Updating
            Ej_us = Ej_us + pj_us*self.dt
            T_wall = Z_next @ self.outer_picker_avg.T
            p_coolj_us = 1.07*20*(T_wall - 20)
            # print(T_wall, p_coolj_us)
            Cj_us = Cj_us + p_coolj_us*self.dt
            pyroj_us = Z_next @ self.pyro_picker.T
            E_netj_us = Ej_us - Cj_us

            #Scaling
            Ej = (Ej_us - self.Emin) / (self.Emax - self.Emin)
            # print(Ej)
            Cj = (Cj_us - self.Cmin) / (self.Cmax - self.Cmin)
            pyroj = (pyroj_us - self.pyromin) / (self.pyromax - self.pyromin)
            p_coolj = (p_coolj_us - self.p_coolmin) / (self.p_coolmax - self.p_coolmin)
            E_netj = (E_netj_us - self.E_netmin) / (self.E_netmax - self.E_netmin)

            # accumulate stage cost HERE -- this is the key structural change

            cost = cost + self.stage_cost_j(Z_next) # your l_x/l_u/l_Z logic, inline

            Z_list.append(Z_next)
            x_list.append(x_next)
            xhist_j = torch.cat([xhist_j[1:], x_next], dim=0)
            uprev_j = torch.cat([uprev_j[1:], uj], dim=0)
    
        xN = x_next.reshape(-1,1)
        eN = xN - self.x_tilde.reshape(-1,1)
        term_cost = Vf_toggle*(eN.T @ self.Vf @ eN)
        cost = cost + term_cost

        return cost.reshape(1,1), [torch.cat(Z_list), torch.cat(x_list), Ej, Cj, pyroj, p_coolj, E_netj, xhist_j,uprev_j]

    def forward(self, flat_in):
        # unpack the single flat vector -- pick ONE consistent order and stick to it
        idx = 0
        p_guess = flat_in[idx:idx+self.f].reshape(self.f, 1)
        idx += self.f
        uprev0 = flat_in[idx:idx+(self.seq_len-1)*self.u_dim].reshape(self.seq_len-1, self.u_dim)
        idx += (self.seq_len-1)*self.u_dim
        xhist0 = flat_in[idx:idx+self.seq_len*self.x_dim].reshape(self.seq_len, self.x_dim)
        idx += self.seq_len*self.x_dim
        E0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        C0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        pyro0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        p_cool0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        E_net0 = flat_in[idx:idx+1].reshape(1,1)
        idx += 1
        Vf_toggle = flat_in[-1].reshape(1,1)

        cost,_ = self.predict(p_guess, uprev0, xhist0, E0, C0, pyro0, p_cool0, E_net0, Vf_toggle)
        return cost
    
#DEFUNCT
class ForecasterWrapperMultipleShooting(nn.Module):
    def __init__(self, forecaster):
        super(ForecasterWrapperMultipleShooting, self).__init__()
        self.forecaster = forecaster
        self.seq_len = forecaster.seq_len
        self.out_dim = forecaster.out_dim
    def forward(self,w):
        seq_len = self.seq_len
        #Descision variables
        #the first 1*seq_len is p_prev
        #then out_dim*seq_len is x_prev
        #Parameters  
        #then 1*seq_len is E_prev
        #then 1*seq_len is C_prev
        p_prev_idx = 1*seq_len
        x_prev_idx = p_prev_idx + self.out_dim*seq_len
        E_prev_idx = x_prev_idx + 1*seq_len
        C_prev_idx = E_prev_idx + 1*seq_len

        p_prev = w[:p_prev_idx].reshape(seq_len,1)
        x_prev = w[p_prev_idx:x_prev_idx].reshape(seq_len,self.out_dim)
        E_prev = w[x_prev_idx:E_prev_idx].reshape(seq_len,1)
        C_prev = w[E_prev_idx:C_prev_idx].reshape(seq_len,1)
        ux_prev = torch.cat([p_prev, E_prev, C_prev, x_prev])
        return self.forecaster(ux_prev)


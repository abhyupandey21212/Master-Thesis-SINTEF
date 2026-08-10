import numpy as np
from utils.data_loader import *
from models import *
import torch.optim as optim
import torch
import torch.nn as nn
import l4casadi
import casadi as cas
import plotly.graph_objects as go
from shred_config_tester import make_shred_config
from utils.utilities import TimeSeriesDataset
torch.backends.mkldnn.flags.rnn = False
torch.backends.mkldnn.enabled = False
import time

def make_forecaster_datasets(shred1, shred2, manager1, manager2, padding_key1, padding_key2):


    train_forecaster_1, valid_forecaster_1, test_forecaster_1 = manager1.make_FORECASTER_datasets(shred1, padding_key=padding_key1)

    train_forecaster_2, valid_forecaster_2, test_forecaster_2 = manager2.make_FORECASTER_datasets(shred2, padding_key=padding_key2)

    train_forecasterX = torch.cat([train_forecaster_1.X[:,:,:4], train_forecaster_2.X[:,:,1:3], train_forecaster_1.X[:,:,4:]],dim=2)
    train_forecaster = TimeSeriesDataset(train_forecasterX, train_forecaster_1.Y)

    valid_forecasterX = torch.cat([valid_forecaster_1.X[:,:,:4], valid_forecaster_2.X[:,:,1:3], valid_forecaster_1.X[:,:,4:]],dim=2)
    valid_forecaster = TimeSeriesDataset(valid_forecasterX, valid_forecaster_1.Y)

    test_forecasterX = torch.cat([test_forecaster_1.X[:,:,:4], test_forecaster_2.X[:,:,1:3], test_forecaster_1.X[:,:,4:]],dim=2)
    test_forecaster = TimeSeriesDataset(test_forecasterX, test_forecaster_1.Y)

    return train_forecaster, valid_forecaster, test_forecaster

def remove_sensor(train_dataset_forecaster, valid_dataset_forecaster, test_dataset_forecaster, remove_idx = None, keep_idx = None):
    train_dataset_forecasterX = train_dataset_forecaster.X.clone()
    valid_dataset_forecasterX = valid_dataset_forecaster.X.clone()
    test_dataset_forecasterX = test_dataset_forecaster.X.clone()
    if not remove_idx is None:
        mask_rm = torch.ones(train_dataset_forecaster.X.shape[-1],dtype=torch.bool)
        mask_rm[remove_idx] = False
        train_dataset_forecasterX = train_dataset_forecasterX[:,:,mask_rm]
        valid_dataset_forecasterX = valid_dataset_forecasterX[:,:,mask_rm]
        test_dataset_forecasterX = test_dataset_forecasterX[:,:,mask_rm]
    elif not keep_idx is None:
        mask_kp = torch.zeros(train_dataset_forecaster.X.shape[-1],dtype=torch.bool)
        mask_kp[keep_idx] = True
        train_dataset_forecasterX = train_dataset_forecasterX[:,:,mask_kp]
        valid_dataset_forecasterX = valid_dataset_forecasterX[:,:,mask_kp]
        test_dataset_forecasterX = test_dataset_forecasterX[:,:,mask_kp]
    return TimeSeriesDataset(train_dataset_forecasterX, train_dataset_forecaster.Y), TimeSeriesDataset(valid_dataset_forecasterX, valid_dataset_forecaster.Y), TimeSeriesDataset(test_dataset_forecasterX, test_dataset_forecaster.Y)

def train(forecaster, train_forecaster, valid_forecaster):
    learning_rate = 1e-3
    weight_decay  = 0.1
    optimizer = optim.AdamW(forecaster.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()
    # loss = nn.MSELoss()
    train_error, valid_error = forecaster.fit(
        train_forecaster, 
        valid_forecaster, 
        optimizer, 
        loss, 
        batch_size = 64, 
        max_epoch = 10000, 
        patience = 100, 
        plotting = True
        )
    forecaster.eval()
    return forecaster


def make_forecaster(train_forecaster, 
                     valid_forecaster, 
                     lstm_params = {
                        'seq_len': 10, 
                        'n_layers': 1,
                        'hidden_dim': 64,
                        },
                     state_dict_file = None,
                    ):
    forecaster = SHREDForecaster(
        u_dim=train_forecaster.X.shape[-1], 
        out_dim=train_forecaster.Y.shape[-1], 
        lstm_params = lstm_params,
        batch_first=True,
        data_manager=None,
    )

    if state_dict_file is None:
        forecaster = train(forecaster, train_forecaster, valid_forecaster)
    else:
        try:
            forecaster_state_dict = torch.load(state_dict_file, weights_only=True)
            forecaster.load_state_dict(forecaster_state_dict)
        except:
            forecaster = train(forecaster, train_forecaster, valid_forecaster)
            torch.save(forecaster.state_dict(), state_dict_file)
    return forecaster

def testing_forecaster(forecaster, shred, manager, train_dataset, valid_dataset, test_dataset, c1_picker, c2_picker, sus_picker):
    sets = {'Training':train_dataset,'Validation':valid_dataset,'Testing':test_dataset}
    for name in sets:
        if not name == 'Testing':
            continue
        set = sets[name]
        print(f'-----{name}-----')
        latent_pred = forecaster(set.X)
        a_pred = shred.sdn(latent_pred)
        a_pred = manager.YscalerPost.inverse_transform(a_pred)
        latent_true = set.Y
        a_true = shred.sdn(latent_true)
        a_true = manager.YscalerPost.inverse_transform(a_true)
        Z_POD_pred = a_pred @ manager.Vtr
        Z_pred = manager.YscalerPre.inverse_transform(Z_POD_pred)
        Z_POD_true = a_true @ manager.Vtr
        Z_true = manager.YscalerPre.inverse_transform(Z_POD_true)
        #''Parseval check'
        lhs = ((a_pred - a_true)**2).sum(dim=1).mean() #/ (a_true**2).sum(dim=1).mean()
        print('POD coef MS error       : ', lhs.item())
        # lhs2 = custom_loss(a_pred, a_)
        rhs = ((Z_POD_pred - Z_POD_true)**2).sum(dim=1).mean()
        if (rhs - lhs) < 1e-6:
            pars_check = 'PASSED'
        else:
            pars_check = 'FAILED'
        print(f'Parseval check          :  {pars_check}')
        print( 'Temperature MS error    : ',((Z_pred - Z_true)**2 / Z_true**2).sum(dim=1).mean().item())
        c1_error = (Z_pred - Z_true)@c1_picker.T
        c2_error = (Z_pred - Z_true)@c2_picker.T
        sus_error= (Z_pred - Z_true)@sus_picker.T
        # plt.plot(c1_error.mean(dim=1).detach())
        # plt.plot(c2_error.mean(dim=1).detach())
        # plt.plot(sus_error.mean(dim=1).detach())
        # plt.show()

def make_sym(torch_model, l4c_build_dir, input_shape, output_shape, model_name = None, wrapper=None, jacjac=False,folder=''):

                        # 'shred':[(),()]}
    model_name = 'func_sym' if model_name is None else model_name

    torch_model = wrapper(torch_model) if not wrapper is None else torch_model

    l4c_model = l4casadi.L4CasADi(torch_model, device='cpu', build_dir=folder+l4c_build_dir, generate_jac_jac=jacjac)
    if l4c_build_dir in os.listdir(folder):
        print('FOUND EXISTING l4casadi build. Loading...')
        l4c_model._built = True
        l4c_model._input_shape  = input_shape
        l4c_model._output_shape = output_shape
    else:
        print('No such directory, building new model...')

    print('Defining the casadi function')
    X_sym = cas.MX.sym('X', input_shape[0], input_shape[1]) #(u_plan, u_prev, x_prev, Vf_toggle)
    y_sym = l4c_model(X_sym)
    print('In shape : ', X_sym.shape)
    print('Out shape: ', y_sym.shape)
    model_sym = cas.Function(model_name, [X_sym], [y_sym])
    grad_model_sym = None
    for grad in [cas.gradient, cas.jacobian]:
        try:
            grad_model_sym = cas.Function('grad_'+model_name, [X_sym], [grad(model_sym(X_sym), X_sym)])
        except:
            continue
    return l4c_model, model_sym, grad_model_sym



def single_shooting_mpc(cost_l4c, cost_torch, exact_hess, n_max, iter_max, xhist0, uprev0, u0, Z0, p_guess_warmstart, E_limit, Vf_toggle, Kf = None, x_tilde=None, u_tilde = None):
    f = p_guess_warmstart.shape[0]
    seq_len = xhist0.shape[0]
    x_dim = xhist0.shape[1]
    u_dim = uprev0.shape[1]

    hessian_modes = {True:'exact',False:'limited-memory'}
    p_guess = cas.MX.sym('p_guess', f, 1)
    params  = cas.MX.sym('params', x_dim*seq_len + u_dim*(seq_len-1) + (u_dim-1) + 1, 1)  # xhist0, uprev0, E0, C0, pyro0, flattened

    flat_in = cas.vertcat(p_guess, params)
    out = cost_l4c(flat_in)
    cost = out[0]
    # x_traj = out[1:]  # if you need bounds/constraints on the state trajectory

    nlp = {
        'x': p_guess, 
        'p': params, 
        'f': cost, 
        # 'g': x_traj}  # g optional, only if constraining x
    }
    opts = {'ipopt': {
            'max_iter': iter_max,
            'delta': 0.5,
            'bound_push': 1e-6,
            'bound_frac': 1e-6,
            'print_level': 1,
            # 'check_derivatives_for_naninf': 'yes',
            'hessian_approximation': hessian_modes[exact_hess],  # avoid exact Hessian computation
            }}
    solver = cas.nlpsol('solver', 'ipopt', nlp, opts)

    #INITIALIZING
    E0      = u0[1].reshape(1,1)
    C0      = u0[2].reshape(1,1)
    pyro0   = u0[3].reshape(1,1)
    p_cool0 = u0[4].reshape(1,1)
    E_net0  = u0[5].reshape(1,1)

    p_traj = []
    E_traj = []
    C_traj = []
    pyro_traj = []
    p_cool_traj = []
    E_net_traj = []
    x_traj = []
    Z_traj = [Z0]
    cost_traj = []
    for i in range(n_max):
        if E0 > E_limit:
            print('MAX ENERGY LIMIT')
            # break
        print(f'-----{i+1}-----')

        params0 = np.vstack([uprev0.reshape(-1,1), xhist0.reshape(-1,1), E0, C0, pyro0, p_cool0, E_net0, Vf_toggle])
        # each MPC step:
        t0 = time.time()
        sol = solver(
            x0=p_guess_warmstart, 
            p=params0,
            lbx=np.zeros_like(p_guess_warmstart), 
            ubx=np.ones_like(p_guess_warmstart), 
            # lbg=x_lb, 
            # ubg=x_ub
            )
        print('time   : ', time.time()-t0)
        print('iters  : ', solver.stats()['iter_count'])
        print('status : ',solver.stats()['return_status'])

        p_opt = np.array(sol['x']).reshape(f,1)
        cost_traj.append(np.array(sol['f']))
        print('p_guess: ',p_guess_warmstart.T)
        print('p_opt  : ',p_opt.T)

        p_traj.append(p_opt[0])
        E_traj.append(E0)
        C_traj.append(C0)
        pyro_traj.append(pyro0)
        p_cool_traj.append(p_cool0)
        E_net_traj.append(E_net0)

        #RUNNING WITH FIRST INPUT
        _,info = cost_torch.predict_custom(
            torch.tensor(p_opt[0],dtype=torch.float32).reshape(-1,1),
            torch.tensor(uprev0,dtype=torch.float32).reshape(seq_len-1,u_dim),
            torch.tensor(xhist0,dtype=torch.float32).reshape(seq_len,x_dim),
            torch.tensor(E0,dtype=torch.float32).reshape(1,1),
            torch.tensor(C0,dtype=torch.float32).reshape(1,1),
            torch.tensor(pyro0,dtype=torch.float32).reshape(1,1),
            torch.tensor(p_cool0,dtype=torch.float32).reshape(1,1),
            torch.tensor(E_net0,dtype=torch.float32).reshape(1,1),
            torch.tensor(Vf_toggle,dtype=torch.float32)
            )
        _,info2 = cost_torch.predict(
            torch.tensor(p_opt,dtype=torch.float32).reshape(-1,1),
            torch.tensor(uprev0,dtype=torch.float32).reshape(seq_len-1,u_dim),
            torch.tensor(xhist0,dtype=torch.float32).reshape(seq_len,x_dim),
            torch.tensor(E0,dtype=torch.float32).reshape(1,1),
            torch.tensor(C0,dtype=torch.float32).reshape(1,1),
            torch.tensor(pyro0,dtype=torch.float32).reshape(1,1),
            torch.tensor(p_cool0,dtype=torch.float32).reshape(1,1),
            torch.tensor(E_net0,dtype=torch.float32).reshape(1,1),
            torch.tensor(Vf_toggle,dtype=torch.float32)
            )
        Z_next, x_next, E0, C0, pyro0, p_cool0, E_net0, xhist0, uprev0 = info
        xN = info2[1][-1].detach().numpy()
        Z_next = Z_next.detach().numpy()
        x_next = x_next.detach().numpy()
        E0 = E0.detach().numpy()
        C0 = C0.detach().numpy()
        pyro0 = pyro0.detach().numpy()
        xhist0 = xhist0.detach().numpy()
        uprev0 = uprev0.detach().numpy()
        Z_traj.append(Z_next)
        x_traj.append(x_next)

        #UPDATING guess with optimal
        p_guess_warmstart = np.roll(p_opt,-1)
        if Kf is None:
            p_guess_warmstart[-1] = 0.5 #ADD TERMINAL CONTROLLER HERE
        else:
            e = xN - x_tilde
            uf = Kf @ e.T + u_tilde
            p_guess_warmstart[-1] = uf[0]


    p_opt = np.concat(p_traj)
    E_traj = np.concat(E_traj)
    C_traj = np.concat(C_traj)
    pyro_traj = np.concat(pyro_traj)
    p_cool_traj = np.concat(p_cool_traj)
    E_net_traj = np.concat(E_net_traj)
    Z_traj = np.concat(Z_traj)
    x_traj = np.concat(x_traj)
    cost_traj = np.concat(cost_traj)
    return p_opt, x_traj, E_traj, C_traj, pyro_traj, p_cool_traj, E_net_traj, Z_traj, cost_traj


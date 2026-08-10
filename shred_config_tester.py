import numpy as np
from utils.data_loader import *
from models import *
import torch.optim as optim
import torch
import torch.nn as nn
import l4casadi
import plotly.graph_objects as go

def train(shred, train_dataset, valid_dataset, loss_weight):
    learning_rate = 1e-3
    weight_decay  = 0.1
    optimizer = optim.AdamW(shred.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # loss = nn.MSELoss(reduction='sum')
    loss = WeightedLoss(loss_weight)
    train_error, valid_error = shred.fit(
    train_dataset, 
    valid_dataset, 
    optimizer, 
    loss, 
    batch_size = 64, 
    max_epoch = 10000, 
    patience = 100, 
    plotting = True
    )
    return shred

def make_shred_config(mu_keys, POD_dim=10, padding_key=['zero', 'zero','init'], state_dict_file = None):
    parent = ''
    filter = pd.concat([pd.read_csv(parent+'data/nodes/c1.csv'),pd.read_csv(parent+'data/nodes/c2.csv'),pd.read_csv(parent+'data/nodes/sus.csv'),pd.read_csv(parent+'data/nodes/outer_wall.csv')])
    data_dict,mu_dict,nodes,_ = load_raw_data(parent+'data/raw_data/', filter3D=filter, mu_keys=mu_keys)    

    manager = DatasetManager(
    data_dict=data_dict, 
    mu_dict=mu_dict, 
    train_share=0.8, 
    valid_share=0.1, 
    POD_dim=POD_dim, 
    seq_len=10,
    padding_key = padding_key
    )
    train_dataset, valid_dataset, test_dataset = manager.make_SHRED_datasets(ykey='POD_coefs')  

    loss_weight = (manager.YscalerPost.max - manager.YscalerPost.min)**2

    in_dim = train_dataset.X.shape[-1]
    out_dim = train_dataset.Y.shape[-1]

    shred = SHRED(
        in_dim=train_dataset.X.shape[-1],
        out_dim=train_dataset.Y.shape[-1],
        batch_first=True,
        data_manager=manager,
        lstm_params= {
                        'seq_len': 10, 
                        'n_layers': 2,
                        'hidden_dim': 64,
                    },
        sdn_params = {
                        'inner_layers': [200,100],
                        'dropout': [True, 0.1],
                        'batch_norm': [False],
                    },
            )
    if state_dict_file is None:
        shred = train(shred, train_dataset, valid_dataset, loss_weight)
    else:
        try:
            shred_state_dict = torch.load(state_dict_file, weights_only=True)
            shred.load_state_dict(shred_state_dict)
        except:
            shred = train(shred, train_dataset, valid_dataset, loss_weight)
            torch.save(shred.state_dict(), state_dict_file)
    shred.eval()
    return shred, train_dataset, valid_dataset, test_dataset, manager

def testing(shred, train_dataset, valid_dataset, test_dataset, manager, custom_loss=None):
    sets = {'Training':train_dataset,'Validation':valid_dataset,'Testing':test_dataset}
    for name in sets:
        set = sets[name]
        print(f'-----{name}-----')
        a_pred = shred(set.X)
        a_pred = manager.YscalerPost.inverse_transform(a_pred)
        a_true = set.Y
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
        rhs2 = ((Z_pred - Z_true)**2).sum(dim=1).mean() / (Z_true**2).sum(dim=1).mean() 
        if not name == 'Testing':
            continue
        else:
            continue
        for k in range(3):
            plt.title(f'Relative square error, test set {k+1}' )
            plt.semilogy(np.arange(61),100*(((Z_pred[k*61:(k+1)*61] - Z_true[k*61:(k+1)*61])**2).sum(dim=1) / ((Z_true[k*61:(k+1)*61])**2).sum(dim=1)).detach())
            plt.xlabel('Timestep')
            plt.ylabel('Percentage mean square error')
            plt.show()
            print('Relative MS recon error : ',rhs2.item())

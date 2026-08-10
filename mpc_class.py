import utils.utilities as utilities
import numpy as np
import torch 
import casadi as cas
import l4casadi as l4c
import os
import control as ctrl
from scipy.linalg import solve_discrete_lyapunov
torch.backends.mkldnn.flags.rnn = False
torch.backends.mkldnn.enabled = False
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from models import *
from utils.utilities import cooling_power


class MPC:
    def __init__(self, 
                 shred, 
                 forecaster, 
                 manager,
                 f,
                 seq_len,
                 u_dim,
                 out_dim,
                 K_f,
                 P,
                 u_tilde,
                 x_tilde,
                 cooling_plan,
                 c1_picker,
                 c2_picker,
                 sus_picker,
                 outer_wall_picker,
                 pyro_picker,
                 c1_target,
                 c2_target,
                 sus_target,
                 cooling_coef,
                 ):
        self.shred = shred
        self.forecaster = forecaster
        self.manager = manager
        self.f = f
        self.seq_len = seq_len
        self.u_dim = u_dim
        self.out_dim = out_dim
        self.K_f = K_f
        self.P = P
        self.u_tilde = u_tilde
        self.x_tilde = x_tilde
        self.cooling_plan = cooling_plan
        self.c1_picker = c1_picker
        self.c2_picker = c2_picker
        self.sus_picker = sus_picker
        self.outer_wall_picker = outer_wall_picker
        self.c1_target = c1_target
        self.c2_target = c2_target
        self.sus_target = sus_target
        self.cooling_coef = torch.tensor(cooling_coef, dtype=torch.float32)
        self.pyro_picker = pyro_picker
        self.cooling_power = lambda T_wall: cooling_power(T_wall, self.cooling_coef)

    def make_cost_model(self):
        target = torch.cat([self.c1_target*torch.ones((self.c1_picker.shape[0],1)), self.c2_target*torch.ones((self.c2_picker.shape[0],1)), self.sus_target*torch.ones((self.sus_picker.shape[0],1))]).T
        picker = torch.cat([self.c1_picker, self.c2_picker, -1*self.sus_picker])

        cost_single_shooting = CostWrapperSingleShootingNoCooling(
            forecaster=self.forecaster, 
            shred=self.shred, 
            manager=self.manager, 
            f=self.f, 
            u_dim = self.u_dim,
            c1_picker=self.c1_picker, 
            c2_picker=self.c2_picker, 
            sus_picker=self.sus_picker, 
            outer_wall_picker=self.outer_wall_picker,
            pyro_picker=self.pyro_picker,
            c1_target=self.c1_target, 
            c2_target=self.c2_target, 
            sus_target=self.sus_target, 
            P = self.P,
            x_tilde=self.x_tilde,
            u_tilde=self.u_tilde,
            cooling_coef=self.cooling_coef
            )
        
        cost_multiple_shooting = CostWrapperMultipleShooting(
            shred=self.shred, 
            f=self.f, 
            u_dim = self.u_dim,
            target=target, 
            picker=picker, 
            P = self.P,
            x_tilde=self.x_tilde,
            u_tilde=self.u_tilde,
            )
        self.cost_single_shooting = cost_single_shooting
        self.cost_multiple_shooting = cost_multiple_shooting
        return cost_single_shooting, cost_multiple_shooting

    def update_window(self, prev, new, des_len = None):
        des_len = self.seq_len if des_len is None else des_len
        if not type(prev) == type(new):
            raise TypeError('both inputs must have the same type')

        if prev.shape[0] == des_len - 1:
            l = 0
        elif prev.shape[0] == des_len:
            l = 1
        else:
            print(prev.shape, new.shape)
            raise

        if type(prev) == torch.Tensor:
            out = torch.cat([prev[l:].contiguous(), new.reshape(1,-1)])
        elif type(prev) == np.ndarray:
            out = np.vstack([prev[l:].copy(), new.reshape(1,-1)])
        else:
            raise TypeError(f'type(prev)={type(prev)} is not support')
        return out
    
    def create_u0x0(self,u_prev,x_prev,u0=None,x0=None):
        if (x0 is None) and (u0 is None):
            raise ValueError('At least one of u0 or x0 must not be None')
        elif x0 is None:
            x_plan = []
            x_prev__ = x_prev
            u_prev__ = u_prev
            for uj in u0:
                u_prev__ = self.update_window(u_prev__, uj)
                ux_prev = torch.cat([u_prev__,x_prev__],dim=1)
                xj = self.forecaster(ux_prev)
                x_prev__ = self.update_window(x_prev__,xj)
                x_plan.append(xj)
            x0 = torch.cat(x_plan)
        elif u0 is None:
            u_plan = []
            for xj in x0:
                uj = (self.K_f @ (xj - self.x_tilde)) + self.u_tilde
                u_plan.append(uj)
            u0 = torch.cat(u_plan)
        return u0,x0
        
    def create_sym(self, torch_model, l4c_build_dir, model_name = None, input_shape = None, output_shape = None):
        model_shape_dict = {
            'cost_single_shooting':[(self.seq_len*self.out_dim + (self.seq_len-1)*self.u_dim + self.f*1 + (self.u_dim - 1)*1 + 1, 1),(1,1)],
            'forecaster':[(self.seq_len, self.u_dim+self.out_dim),(1,self.out_dim)],
            'cost_multiple_shooting':[(self.f*self.out_dim + self.f*self.u_dim + 1, 1),(1,1)],}
                            # 'shred':[(),()]}
        if not model_name is None:
            assert model_name in model_shape_dict.keys(), f'model_name must be one of {model_shape_dict.keys()}'
            input_shape,output_shape = model_shape_dict[model_name]
            print(input_shape, output_shape)
        else:
            model_name = 'func_sym'
        if input_shape is None:
            raise 
        

        torch_model = torch_model.float()

        l4c_model = l4c.L4CasADi(torch_model, device='cpu', build_dir=l4c_build_dir, generate_jac_jac=False)
        if l4c_build_dir in os.listdir():
            print('FOUND EXISTING l4casadi build. Loading...')
            l4c_model._built = True
            l4c_model._input_shape  = input_shape
            l4c_model._output_shape = output_shape
        else:
            print('No such directory, building new model...')

        print('Defining the casadi function')
        X_sym = cas.MX.sym('X', input_shape[0], input_shape[1]) #(u_plan, u_prev, x_prev, Vf_toggle)

        y_sym = l4c_model(X_sym)
        model_sym = cas.Function(model_name, [X_sym], [y_sym])
        try:
            grad_model_sym = cas.Function('grad_'+model_name, [X_sym], [cas.gradient(model_sym(X_sym), X_sym)])
        except:
            grad_model_sym = None
        return model_sym, grad_model_sym
    
    def linearize_LQR(self,forecaster_sym, grad_forecaster_sym, x_tilde_prev, u_tilde_prev, mu):
        def B_func(X): # Dependence on last input, not full seq
            return np.array(grad_forecaster_sym(X)[:,3*(self.seq_len-1):3*(self.seq_len)])

        def A_func(X): #Dependence on last state, not full seq
            return np.array(grad_forecaster_sym(X)[:,3*(self.seq_len)+64*(self.seq_len-1):])

        X_prev = np.hstack([u_tilde_prev, x_tilde_prev])
        A = A_func(X_prev)
        B = B_func(X_prev)
        print(A.shape, B.shape)

        Q = np.diag(np.ones(self.out_dim))
        R = np.diag(np.ones(self.u_dim))

        # PBH test — check each eigenvalue
        n = A.shape[0]
        eigenvalues = np.linalg.eigvals(A)
        all_ctrb = True
        for lam in eigenvalues:
            PBH = np.hstack([-A + lam * np.eye(n), B])
            r = np.linalg.matrix_rank(PBH)
            
            c = r == n
            if not c:
                print(f"λ = {lam:.2f} → {'Controllable' if c else 'UNCONTROLLABLE'}")
                all_ctrb = False
        if all_ctrb:
            print('All modes are controllable, so the system is controllable.')
        K_f, S, eigs = ctrl.dlqr(A,B,Q,R)
        Ak = A - B@K_f
        Qk = Q + K_f.T @ R @ K_f
        P = solve_discrete_lyapunov(Ak, mu*Qk)
        return A,B,K_f,P


        
    def set_up_single_shooting(
            self, 
            cost_sym, 
            opts = 
                {'ipopt': {
                    'max_iter': 100,
                    'delta': 0.5,
                    'bound_push': 1e-6,
                    'bound_frac': 1e-6,
                    'print_level': 1,
                    'check_derivatives_for_naninf': 'yes',
                    'hessian_approximation': 'limited-memory',  # avoid exact Hessian computation
                    }
                },
            ):
        p_sym = cas.MX.sym('u', self.f*1, 1)
        params_sym = cas.MX.sym('params', (self.u_dim - 1)*1 + self.seq_len*self.out_dim + (self.seq_len-1)*self.u_dim + 1, 1)

        nlp = {
            'x': cas.vec(p_sym),
            'p': cas.vec(params_sym),
            'f': cost_sym(cas.vertcat(p_sym, params_sym)),
        }
        
        solver = cas.nlpsol('solver', 'ipopt', nlp, opts)
        return solver

    def set_up_multiple_shooting(
            self, 
            forecaster_sym, 
            cost_sym,
            opts = 
                {'ipopt': {
                    'max_iter': 100,
                    'delta': 0.5,
                    'bound_push': 1e-6,
                    'bound_frac': 1e-6,
                    'print_level': 2,
                    'check_derivatives_for_naninf': 'yes',
                    'hessian_approximation': 'limited-memory',  # avoid exact Hessian computation
                    }
                }
            ):
        seq_len=self.seq_len
        out_dim=self.out_dim
        u_dim=self.u_dim
        f = self.f
        w = []      # decision variables
        g = []      # constraints
        lbg = []
        ubg = []

        #ADDING E_plan, C_plan, u_prev, x_prev, Vf_toggle to parameters
        E_plan_list = [cas.MX.sym(f'E_{seq_len+j}', 1) for j in range(f)]
        E_plan = cas.vertcat(*E_plan_list)
        C_plan_list = [cas.MX.sym(f'C_{seq_len+j}', 1) for j in range(f)]
        C_plan = cas.vertcat(*C_plan_list)
        x_syms = [cas.MX.sym(f'x_{j}', out_dim) for j in range(seq_len)]
        x_prev = cas.vertcat(*x_syms)
        u_syms = [cas.MX.sym(f'u_{j}', u_dim) for j in range(seq_len-1)]
        u_prev = cas.vertcat(*u_syms)
        Vf_toggle = cas.MX.sym('Vf_toggle', 1)
        print(u_prev.shape, x_prev.shape, E_plan.shape, C_plan.shape)
        params = cas.vertcat(u_prev, x_prev, E_plan, C_plan, Vf_toggle) #parameters

        #ADDING u_next, x_next to decision variables
        for j in range(f):
            pj = cas.MX.sym(f'p_{seq_len-1+j}', 1)
            Ej,Cj = E_plan_list[j],C_plan_list[j]
            uj = cas.vertcat(pj,Ej,Cj)
            u_syms.append(uj)
            w.append(pj)
            
        for j in range(f):
            xj = cas.MX.sym(f'x_{seq_len+j}', out_dim)
            x_syms.append(xj)
            w.append(xj)

        #IMPOSING state dyanmics as equality constraints
        for j in range(f):
            u_window = [_.reshape((1,-1)) for _ in u_syms[j:j+seq_len]]
            u_window = cas.vertcat(*u_window)
            x_window = [_.reshape((1,-1)) for _ in x_syms[j:j+seq_len]]
            x_window = cas.vertcat(*x_window)
            ux_prev = cas.horzcat(u_window,x_window)   #stitching together u_prev, x_prev into single list
            x_next = forecaster_sym(ux_prev).reshape((-1, 1))
            g.append(x_syms[seq_len + j] - x_next)
            lbg += [0]*out_dim# equality: 0 <= g <= 0
            ubg += [0]*out_dim# equality: 0 <= g <= 0

        w = cas.vertcat(*w)
        g = cas.vertcat(*g)

        nlp = {
            'f': cost_sym(cas.vertcat(w,params[(seq_len*out_dim) + (seq_len-1)*u_dim:])),#function   (cost_sym)
            'x': w,          #input      (u_next, x_next)
            'p':params,      #parameters (u_prev, x_prev)
            'g': g}          #constraints(dynamics)


        solver = cas.nlpsol('solver', 'ipopt', nlp, opts)
        return solver 
    
    def single_shooting(self, solver, n_steps, p_plan0, p_cool0, T_pyro0, u_prev, x_prev, plotting=False, Vf_toggle = 1, Emax=1):  
        cost = self.cost_single_shooting
        forecaster = self.forecaster
        manager = self.manager
        f = self.f
        u_tilde = self.u_tilde
        x_tilde = self.x_tilde
        K_f = self.K_f
        Vf_toggle = torch.tensor(Vf_toggle, dtype=torch.float32).reshape(1,1)
  
        power_plan_opt = []
        latent_pred = []
        traj_pred = []
        cst_traj = []
        u_tilde = torch.tensor(u_tilde, dtype=torch.float32)
        x_tilde = torch.tensor(x_tilde, dtype=torch.float32)

        if p_plan0 is None:
            raise
            # u0_s,_ = self.create_u0x0(u_prev,x_prev,None,None)
        elif (p_plan0.max() > 1.0) or (p_plan0.min() < 0.0):
            print('Warm start is unscaled, scaling...')
            p0_s = manager.Pscaler.transform(p_plan0)
        else:
            p0_s = p_plan0
            p_plan0 = manager.Pscaler.inverse_transform(p_plan0)

        p_cool0_s = p_cool0
        T_pyro0_s = T_pyro0
        for k in range(n_steps):
            print(f'----------{k+1}----------')
            #UNSCALED
            # if u_prev[-1,-1] >= Emax:
            #     print(f'MAX ENERGY REACHED AT STEP {k}')
            #     break

            #SCALED
            p0_s = p0_s.flatten().detach().numpy()
            if T_pyro0 is None:
                u_other = p_cool0_s.flatten().reshape(-1,1).detach()
            elif p_cool0 is None:
                u_other = T_pyro0_s.flatten().reshape(-1,1).detach()
            else:
                u_other = torch.cat([p_cool0_s.flatten().reshape(-1,1),(), T_pyro0_s.flatten().reshape(-1,1)]).flatten().reshape(-1,1).detach()

            param_s = torch.cat([u_other, u_prev.flatten().reshape(-1,1),x_prev.flatten().reshape(-1,1), Vf_toggle]).detach().numpy() #scaled params
            # print(p0_s.shape, param_s.shape)
            sol = solver(
                lbx=np.zeros_like(p0_s),
                ubx=np.ones_like(p0_s),
                x0=p0_s,
                p=param_s
                    )
            p_opt_s = torch.tensor(np.array(sol['x']).reshape(f, 1), dtype=torch.float32) #Scaled optimal power plan
            cst = np.array(sol['f']).reshape(1,)
            cst_traj.append(cst)
            # print(p_opt_s.shape, p_cool0.shape)

            #UNSCALED
            p_opt = manager.Pscaler.inverse_transform(p_opt_s) #unscaling
            if T_pyro0 is None:
                Z_next, x_next, _ = cost.pred_loop(p_opt_s, p_cool0_s, u_prev, x_prev)#use the model with the first input to get next state
                p_cool0 = self.cost_single_shooting.cooling_power(self.outer_wall_picker@Z_next[0]).reshape(1,)
                u_next = torch.cat([p_opt[0].reshape(1,1),p_cool0.reshape(1,1),torch.zeros((1,1),dtype=torch.float32)], dim=1)
            elif p_cool0 is None:
                Z_next, x_next, _ = cost.pred_loop(p_opt_s,T_pyro0_s, u_prev, x_prev)#use the model with the first input to get next state
                T_pyro0 = self.pyro_picker@Z_next[0]
                u_next = torch.cat([p_opt[0].reshape(1,1), torch.zeros((1,1),dtype=torch.float32),T_pyro0.reshape(1,1)], dim=1)
            else:
                Z_next, x_next, _ = cost.pred_loop(p_opt_s, p_cool0_s,T_pyro0_s, u_prev, x_prev)#use the model with the first input to get next state
                p_cool0 = self.cost_single_shooting.cooling_power(self.outer_wall_picker@Z_next[0]).reshape(1,)
                T_pyro0 = self.pyro_picker@Z_next[0]
                u_next = torch.cat([p_opt[0].reshape(1,1),p_cool0.reshape(1,1),T_pyro0.reshape(1,1)], dim=1)

            #SCALED
            x_next = x_next.reshape(f,-1)
            Z_next = Z_next.reshape(f,-1)
            u_next_s = manager.Xscaler.transform(u_next)
            p_cool0_s = u_next_s[:,1].reshape(1,)
            T_pyro0_s = u_next_s[:,2].reshape(1,)
            if T_pyro0 is None:
                u_next_s = u_next_s[:,:2]
            elif p_cool0 is None:
                u_next_s = torch.cat([u_next_s[:,0], u_next_s[:,-1]]).reshape(1,-1)
            print(u_next_s)
            # if k <= 15:
            #     p_cool0_s = torch.ones_like(p_cool0_s)
            x_prev = self.update_window(x_prev, x_next[0])#update x_prev with xkp1
            u_prev = self.update_window(u_prev, u_next_s, des_len=self.seq_len-1)#update u_prev with u_opt0

            power_plan_opt.append(u_next_s[0].reshape(1,-1))
            latent_pred.append(x_next[0].reshape(1,-1))
            traj_pred.append(Z_next[0].reshape(1,-1))

            uN_s = K_f @ (x_next[-1] - x_tilde) + u_tilde 
            pNs = uN_s[0].reshape(1,1)#calculating the terminal control for the next warm start
            p0_s = self.update_window(p_opt_s, pNs, des_len=f) #SCALED next warm start


        Uopt = torch.cat(power_plan_opt).detach()
        Xpred = torch.cat(latent_pred).detach()
        Zpred = torch.cat(traj_pred).detach()
        if plotting:
           self.make_plots(Uopt, Xpred, Zpred)
        plt.plot(cst_traj)
        return Uopt, Xpred, Zpred
    
    def multiple_shooting(self, solver, n_steps, u_prev, x_prev, cooling_plan = None, plotting = None, u0 = None, x0 = None, Vf_toggle=1, Emax=1, lbg=None, ubg=None):         
        cost = self.cost_multiple_shooting
        shred=self.shred
        forecaster = self.forecaster
        manager = self.manager
        f = self.f
        u_tilde = self.u_tilde
        x_tilde = self.x_tilde
        K_f = self.K_f
        out_dim=self.out_dim
        seq_len=self.seq_len
        cooling_plan = self.cooling_plan if cooling_plan is None else cooling_plan
        Vf_toggle = torch.tensor(1,dtype=torch.float32).reshape(1,1)
        lbg = [0]*(out_dim*f) if lbg is None else lbg# equality: 0 <= g <= 0
        ubg = [0]*(out_dim*f) if ubg is None else ubg# equality: 0 <= g <= 0

        if u0 is None:
            raise NotImplementedError('Need to implement making u0 with the terminal controller')
        elif (u0.max() > 1.0) or (u0.min() < 0.0):
            print('Warm start is unscaled, scaling...')
            u0_s = manager.Uscaler.transform(u0)
        else:
            u0_s = u0

        u0_s,x0 = self.create_u0x0(u_prev, x_prev, u0_s, x0)
            
        power_plan_opt = []
        latent_pred = []
        traj_pred = []

        u_tilde = torch.tensor(u_tilde, dtype=torch.float32)
        x_tilde = torch.tensor(x_tilde, dtype=torch.float32)
        lbx = np.vstack([np.zeros((f,1)),-1*np.ones_like(x0.detach()).reshape(-1,1)])#,dim=1)
        ubx = np.vstack([np.ones((f,1)),1*np.ones_like(x0.detach()).reshape(-1,1)])#,dim=1)

        
        if (x0.max() > 1.0) or (x0.min() < -1.0):
            raise ValueError(f'{x0.min()}, {x0.max()}')

        for k in range(n_steps):
            print(f'----------{k+1}----------')
            #UNSCALED
            if u_prev[-1,1] >= Emax:
                print(f'MAX ENERGY REACHED AT STEP {k}')
                break
            u_unscaled = manager.Uscaler.inverse_transform(u_prev)
            E_now = u_unscaled[-1,1]

            #SCALED
            p0_s = u0_s[:,0].reshape(-1,1).detach() #scaled power plan guess
            E0_s = u0_s[:,1].reshape(-1,1).detach() #scaled E_plan guess
            C0_s = u0_s[:,2].reshape(-1,1).detach()#scaled C_plan guess
            param_s = torch.cat([
                u_prev.flatten().reshape(-1,1),
                x_prev.flatten().reshape(-1,1), 
                E0_s, 
                C0_s,
                Vf_toggle
                ]).detach().numpy() #scaled params

            w0_s = torch.cat([p0_s,x0.reshape(-1,1).detach()]).numpy()

            sol = solver(
                x0=w0_s, 
                p=param_s,
                lbg=lbg, 
                ubg=ubg, 
                lbx=lbx, 
                ubx=ubx)

            w_opt_s = np.array(sol['x']).flatten() #Scaled optimal power plan
            p_opt_s = w_opt_s[:f].reshape(f,1)
            x_opt = w_opt_s[f:].reshape(f,out_dim)

            #UNSCALED
            p_opt = manager.Pscaler.inverse_transform(torch.tensor(p_opt_s, dtype=torch.float32)) #unscaling
            p_opt_MW = p_opt/1000
            E0N = E_now + 0.5*torch.cumsum(p_opt_MW,dim=0).reshape(-1,1)
            C0N = cooling_plan[k:k+f].reshape(-1,1)
            u_next = torch.cat([p_opt,E0N,C0N],dim=1)

            #SCALED
            u_next_s = manager.Uscaler.transform(u_next)
            print(u_next[0,:])
            x_next = torch.tensor(x_opt,dtype=torch.float32)
            Z_next = shred.post(shred.sdn(x_next))#use the model with the first input to get next state
    
            x_next = x_next.reshape(f,-1)
            Z_next = Z_next.reshape(f,-1)
            x_prev = self.update_window(x_prev, x_next[0])#update x_prev with xkp1
            u_prev = self.update_window(u_prev, u_next_s[0], des_len=seq_len-1)#update u_prev with u_opt0

            # self.make_plots(u_prev.detach(), x_prev.detach(), self.shred.post(self.shred.sdn(x_prev)).detach())


            power_plan_opt.append(u_next_s[0].reshape(1,-1))
            latent_pred.append(x_next[0].reshape(1,-1))
            traj_pred.append(Z_next[0].reshape(1,-1))

            uN_s = K_f @ (x_next[-1] - x_tilde) + u_tilde #calculating the terminal control for the next warm start
            x_temp = torch.cat([x_prev[-f:], x_next])
            u_temp = torch.cat([u_prev[-(f-1):], u_next_s, uN_s.reshape(1,-1)])
            ux_temp = torch.cat([u_temp, x_temp],dim=1)
            xN = self.forecaster(ux_temp)
            u0 = self.update_window(u_next_s, uN_s, des_len = f) 
            x0 = self.update_window(x_next, xN, des_len = f) 

        Uopt = torch.cat(power_plan_opt).detach()
        Xpred = torch.cat(latent_pred).detach()
        Zpred = torch.cat(traj_pred).detach()
        if plotting:
           self.make_plots(Uopt, Xpred, Zpred)
        return Uopt, Xpred, Zpred
    
    def make_plots(self, Uopt, Xpred, Zpred, U0 = None, Z0 = None, extra_plots = False):
        if not U0 is None:
            Uopt = torch.cat([U0, Uopt])
        plt.title('MPC optimal input')
        plt.step(np.arange(len(Uopt)),Uopt[:,0],label='Heating power')
        for j in range(1,Uopt.shape[1]):
            plt.plot(np.arange(len(Uopt)),Uopt[:,j],label=f'Sensor {j}')
        # plt.plot(np.arange(len(Uopt)),Uopt[:,2],label='Net energy')
        plt.xlabel('Timestep')
        plt.ylabel('Normalized input')
        plt.legend()
        plt.show()

        if not Z0 is None:
            print(Z0.shape, Zpred.shape)
            Zpred = torch.cat([Z0, Zpred]).detach()

        c1  = Zpred @ self.c1_picker.T
        c2  = Zpred @ self.c2_picker.T
        sus = Zpred @ self.sus_picker.T
        outer = Zpred @ self.outer_wall_picker.T

        plt.plot(c1.mean(dim=1),color='steelblue', label='C1')
        plt.plot(c2.mean(dim=1),color='tomato', label='C2')
        plt.plot(sus.mean(dim=1),color='magenta', label='Sus')
        plt.hlines(self.c1_target,0,len(Uopt),ls='dashed',color='black')
        plt.hlines(-self.sus_target,0,len(Uopt),ls='dashed',color='black')
        plt.xlabel('Timestep')
        plt.ylabel('Temperature')
        plt.title('MPC output')
        plt.legend()
        plt.show()

        if not extra_plots:
            return
        zone_temps = {'C1': c1, 'C2': c2, 'Sus': sus, 'Shell': outer}
        fig_list = []
        for name, zone in zone_temps.items():
            mean = zone.mean(dim=1)
            lower = zone.min(dim=1)[0]
            upper = zone.max(dim=1)[0]
            t = np.arange(len(mean))
            fig_list = [
                go.Scatter(x=t, y=upper, mode='lines', line=dict(width=0), showlegend=False),
                go.Scatter(x=t, y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.2)', name='Min–Max range'),
                go.Scatter(x=t, y=mean, mode='lines', line=dict(color='blue'), name=name),
            ]
            fig = go.Figure(fig_list)
            fig.update_layout(xaxis_title='Time', yaxis_title='Temperature')
            fig.show()

"""
Worst-case linearization error bound using CasADi + l4casadi.

Given:
  - l4c_model: your l4casadi-wrapped torch model, callable as l4c_model(x_sym)
               where x_sym is an MX/SX column vector of shape (in_dim, 1)
  - A, b, x0:  the linearization  f(x) ~= A @ (x - x0) + b
               (A: (out_dim, in_dim), b: (out_dim,1), x0: (in_dim,1), all numeric)

For each output dimension i, solves two NLPs with IPOPT:
    max   (f(x)_i - f_lin(x)_i)     s.t.  x in [lb, ub]^in_dim
    max  -(f(x)_i - f_lin(x)_i)     s.t.  x in [lb, ub]^in_dim

Each NLP is solved from multiple random restarts (still needed: the problem
is non-convex, so IPOPT only finds LOCAL optima -- but each one found is
*exact* to first-order KKT conditions, unlike PGD with a soft-max surrogate).

The overall bound reported is the max over all dimensions, both directions,
and all restarts. Still a lower bound on the true global worst case, but a
tighter and more reliable one than sampling/PGD, since every restart
converges to an exact local maximum rather than an approximate one.
"""
import torch.nn as nn
import numpy as np
import casadi as cas
import l4casadi as l4c
import torch


def linearization_error_bound_casadi(
    l4c_model,          # callable: MX (in_dim,1) -> MX (out_dim,1)
    A, b, x0,            # numpy arrays: A (out_dim,in_dim), b (out_dim,1), x0 (in_dim,1)
    in_dim=68,
    out_dim=None,
    lb=-1.0,
    ub=1.0,
    n_restarts=20,
    exact_hessian=True,  # set False if jacjac trace is unavailable / unreliable
    seed=0,
):
    rng = np.random.default_rng(seed)
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1, 1)
    x0 = np.asarray(x0).reshape(-1, 1)
    if out_dim is None:
        out_dim = A.shape[0]

    x = cas.MX.sym('x', in_dim, 1)
    f_true = l4c_model(x)                       # (out_dim, 1), MX
    f_lin = cas.DM(A) @ (x - cas.DM(x0)) + cas.DM(b)   # (out_dim, 1)
    diff = f_true - f_lin                        # (out_dim, 1)

    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.hessian_approximation': 'exact' if exact_hessian else 'limited-memory',
    }

    lbx = np.full(in_dim, lb)
    ubx = np.full(in_dim, ub)

    overall_bound = -np.inf
    overall_x = None
    overall_dim = None
    overall_sign = None

    for i in range(out_dim):
        for sign in (+1, -1):
            # maximize sign * diff[i]  <=>  minimize -sign * diff[i]
            objective = -sign * diff[i]
            nlp = {'x': x, 'f': objective}
            solver = cas.nlpsol('solver', 'ipopt', nlp, opts)

            best_for_this = -np.inf
            best_x_for_this = None

            for r in range(n_restarts):
                x_guess = rng.uniform(lb, ub, size=(in_dim, 1))
                try:
                    sol = solver(x0=x_guess, lbx=lbx, ubx=ubx)
                    stats = solver.stats()
                    if not stats['success']:
                        continue
                    val = sign * float(-sol['f'])  # convert back to "sign*diff[i]" value
                    if val > best_for_this:
                        best_for_this = val
                        best_x_for_this = np.array(sol['x']).flatten()
                except Exception as e:
                    # occasional restart failures (bad initial guess, etc.) are normal
                    continue

            print(f"[dim {i}, sign {sign:+d}] best error found: {best_for_this:.6f} "
                  f"(over {n_restarts} restarts)")

            if best_for_this > overall_bound:
                overall_bound = best_for_this
                overall_x = best_x_for_this
                overall_dim = i
                overall_sign = sign

    print(f"\n==> Empirical worst-case linearization error bound: {overall_bound:.6f}")
    print(f"    achieved at output dim {overall_dim}, sign {overall_sign:+d}")
    print("    This is exact-gradient/exact-Hessian local optimization with multi-start;")
    print("    still not a certified global bound, but tighter/more reliable than")
    print("    sampling or soft-max PGD. Increase n_restarts to gain confidence.")

    return overall_bound, overall_x, overall_dim, overall_sign

class closed_loop_wrapper(nn.Module):
    def __init__(self, forecaster, controller, x_tilde, u_tilde):
        super(closed_loop_wrapper, self).__init__()
        self.forecaster = forecaster
        self.controller = torch.tensor(controller,dtype=torch.float32)
        self.x_tilde = torch.tensor(x_tilde,dtype=torch.float32).reshape(-1,1)
        self.u_tilde = torch.tensor(u_tilde,dtype=torch.float32).reshape(-1,1)
    def forward(self, x):
        print(x.shape)
        u = self.controller @ (x - self.x_tilde) + self.u_tilde
        print(u.shape)
        ux = torch.cat([u,x])
        return self.forecaster(ux)



if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example usage with a dummy l4casadi model -- replace with your own
    # ------------------------------------------------------------------
    import torch
    from forecaster_config_tester import make_sym
    from models import SHREDForecaster, UnivForecasterWrapper
    j=3
    forecaster_folder = 'forecaster_configs/'
    u_picker = np.loadtxt(forecaster_folder+'u_picker{j}.txt')
    u_dim, out_dim = 4, 64
    seq_len = 10
    forecaster = SHREDForecaster(
        u_dim=u_dim+out_dim, 
        out_dim=out_dim, 
        lstm_params = {
                    'seq_len': seq_len, 
                    'n_layers': 1,
                    'hidden_dim': 64,
                    },
        batch_first=True,
        data_manager=None,
    )
    forecaster_weights = torch.load(forecaster_folder+f'config{j}.json')
    forecaster.load_state_dict(forecaster_weights)
    forecaster = UnivForecasterWrapper(forecaster, u_picker)

    mpc_folder = 'mpc_configs/'
    u_tilde_prev = np.loadtxt(mpc_folder+'/u_tilde_prev.txt')
    x_tilde_prev = np.loadtxt(mpc_folder+'x_tilde_prev.txt')
    Kf = np.loadtxt(mpc_folder+'Kmatrix.txt')
    Ak = np.loadtxt(mpc_folder+'AKmatrix.txt')
    closed_loop = closed_loop_wrapper(forecaster, Kf, x_tilde=x_tilde_prev[-1], u_tilde=u_tilde_prev[-1])


    
    build_dir = f'_l4c_closedloop{j}_'
    input_shape = (10,70)
    input_shape = (input_shape[0]*input_shape[1],1)
    output_shape = (64,1)
    sym_model,forecaster,dforecaster = make_sym(None, f'_l4c_config{j}_', input_shape, output_shape, f'forecaster_config{j}', wrapper=None,jacjac=False,folder=forecaster_folder)

    

    x0_t = torch.zeros(in_dim, dtype=torch.float64, requires_grad=True)
    y0 = torch_model(x0_t.unsqueeze(0))
    A_t = torch.autograd.functional.jacobian(
        lambda xx: torch_model(xx.unsqueeze(0)).squeeze(0), x0_t.detach()
    )
    b_np = y0.detach().numpy().reshape(-1, 1)
    A_np = A_t.detach().numpy()
    x0_np = x0_t.detach().numpy().reshape(-1, 1)

    l4c_model = l4c.L4CasADi(torch_model, model_expects_batch_dim=True, device='cpu')

    bound, worst_x, dim, sign = linearization_error_bound_casadi(
        l4c_model, A_np, b_np, x0_np,
        in_dim=in_dim, out_dim=out_dim,
        n_restarts=10,
    )
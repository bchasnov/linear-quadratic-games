import numpy as np
from scipy.linalg import solve, solve_discrete_are, LinAlgError
from scipy.linalg import solve
from numpy.linalg import inv, norm, eigvals
from sys import stdout

def random_zero_sum(seed=0):
    """ Random zero-sum two-player linear quadratic game """
    np.random.seed(seed)
    N = 5
    Q = 3 * np.diagflat([-1, -1, 1, 1, 1]) + np.random.randn(N, N) * 0.01
    Q = (Q + Q.T) / 2

    A = 0.5 * np.identity(N)
    B = np.identity(N)
    B2, B1 = B[:, :2], B[:, 2:]

    R = np.diagflat([-1, -1, 1, 1, 1])
    R2, R1 = R[:2, :2], R[2:, 2:]

    # optimal feedback gains
    P = solve_discrete_are(A, B, Q, R)
    K_opt = solve(R + B.T @ P @ B , -B.T @ P @ A)
    L, K = K_opt[:2, :], K_opt[2:, :]
    
    cost_opt = np.trace( P )
    
    R = [[R1, R2], [-R1, -R2]]
    Q = [Q, -Q]
    B = [B1, B2]
    
    return dict(K=K, L=L),\
           dict(A=A, B=B, Q=Q, R=R),\
           dict(cost=(cost_opt, -cost_opt))
    
def random_non_zero_sum(seed=0, N=5):
    """ Random non-zero-sum two-player linear quadratic game """
    np.random.seed(seed)

    while True:
        A = np.random.randn(N, N)
        if np.all(np.abs(eigvals(A))<1):
            break
    B = np.identity(N)
    B1, B2 = B[:, :N//2], B[:, N//2:]

    def psd(n):
        M = np.random.randn(n)
        return M@M.T + np.eye(n)

    Q1 = psd(N)
    Q2 = psd(N)
    R11 = psd(N//2)
    R12 = psd(N-N//2)
    R21 = psd(N//2)
    R22 = psd(N-N//2)

    B = [B1, B2]
    Q = [Q1, Q2]
    R = [[R11, R12], [R21, R22]]

    K = np.zeros(B1.T.shape)
    L = np.zeros(B2.T.shape)
    # optimal feedback gains
    for _ in range(100):
        P, W = costtogo(K, L, A, B, Q, R)
        K, L = bestresponses(K, L, P, W, A, B, Q, R)

    cost1 = np.trace( P )
    cost2 = np.trace( W )
    
    return dict(K=K, L=L),\
           dict(A=A, B=B, Q=Q, R=R),\
           dict(cost=(cost1, cost2))
    

def costtogo(K, L, A, B, Q, R):
    """ Computes the cost-to-go matrix from the perspective of each player
        using the discrete algebraic riccati equation 
    """
    Q1, Q2 = Q
    B1, B2 = B
    (R11, R12), (R21, R22) = R
    try:
        P = solve_discrete_are(A + B2 @ L, B1, Q1 + L.T @ R12 @ L, R11)
        W = solve_discrete_are(A + B1 @ K, B2, Q2 + K.T @ R21 @ K, R22)
    except LinAlgError:
        print('Failed')
        raise LinAlgError
    return P, W

def gradients(K, L, P, W, A, B, Q, R):
    """ Computes the natural gradients for each player in a two-player LQR game """
    B1, B2 = B
    (R11, R12), (R21, R22) = R

    dK = 2 * ( R11 @ K + B1.T @ P @ (A + B1@K + B2@L) )
    dL = 2 * ( R22 @ L + B2.T @ W @ (A + B1@K + B2@L) )
    return dK, dL


def bestresponses(K, L, P, W, A, B, Q, R):
    """ Computes the best responses of a two-player LQR game """
    B1, B2 = B
    (R11, R12), (R21, R22) = R

    K = solve( R11 + B1.T @ P @ B1, -B1.T @ P @ (A + B2@L) )
    L = solve( R22 + B2.T @ W @ B2, -B2.T @ W @ (A + B1@K) )
    return K, L

def simulate(seed, K, L, A, B, Q, R, num_steps):
    """ Simulate a two-player linear dynamical system with quadratic costs"""
    n = A.shape[0]
    Q1, Q2 = Q
    B1, B2 = B
    (R11, R12), (R21, R22) = R

    np.random.seed(seed)
    x0 = np.random.randn(n)

    x = np.zeros((num_steps+1, n))
    u1 = np.zeros((num_steps, B1.shape[1]))
    u2 = np.zeros((num_steps, B2.shape[1]))
    c1 = np.zeros((num_steps, 2))
    c2 = np.zeros((num_steps, 2))

    x[0] = x0
    for t in range(num_steps):
        # Linear feedback (P1 and P2)
        u1[t] = K@x[t] 
        u2[t] = L@x[t] 

        # Costs (P1 and P2)
        c1[t] = x[t]@Q1@x[t] + u1[t]@R11@u1[t] + u2[t]@R12@u2[t]
        c2[t] = x[t]@Q2@x[t] + u1[t]@R21@u1[t] + u2[t]@R22@u2[t]

        # State transition
        x[t+1] = A@x[t] + B1@u1[t] + B2@u2[t]
    
    return x, (u1, u2), (c1, c2)
    
def train(seed: int, num_iter: int, lr1: float, lr2: float, K0, L0, **params):
    """ Trains a two-player linear quadratic game with simultaneous 
        or sequential methods.
        """

    print("\nTraining with learning rates {},{}".format(lr1, lr2))
    A = params['A']
    B1, B2 = params['B']
    Q1, Q2 = params['Q']
    
    # feedback policies
    K = np.zeros((num_iter+1, *B1.T.shape))*np.nan
    L = np.zeros((num_iter+1, *B2.T.shape))*np.nan
    # cost
    cK = np.zeros(num_iter)*np.nan
    cL = np.zeros(num_iter)*np.nan
    # gradients
    gK = np.zeros((num_iter, *B1.T.shape))*np.nan
    gL = np.zeros((num_iter, *B2.T.shape))*np.nan
    # best responses
    bK = np.zeros((num_iter, *B1.T.shape))*np.nan
    bL = np.zeros((num_iter, *B2.T.shape))*np.nan
    # cost-to-go
    P = np.zeros((num_iter, *Q1.shape))*np.nan
    W = np.zeros((num_iter, *Q2.shape))*np.nan
    # stability of closed loop system
    spec_radius = np.zeros(num_iter)*np.nan
    
    K[0] = K0
    L[0] = L0
    for t in range(num_iter):
        P[t], W[t] = costtogo(K[t], L[t], **params)
        gK[t], gL[t] = gradients(K[t], L[t], P[t], W[t], **params)
        bK[t], bL[t] = bestresponses(K[t], L[t], P[t], W[t], **params)

        if lr1 < float('inf'):
            K[t+1] = K[t] - lr1*gK[t]
        else:
            K[t+1] = bK[t]

        if lr2 < float('inf'):
            L[t+1] = L[t] - lr2*gL[t]
        else:
            L[t+1] = bL[t]

        spec_radius[t] = np.max(np.abs(eigvals(A+B1@K[t]+B2@L[t])))
        cK[t] = np.trace(P[t])
        cL[t] = np.trace(W[t])

        stdout.write('\r iter={} |gK|={:.2f} |gL|={:.2f}'.format(t+1, norm(gK[t]), norm(gL[t])))

    out = { 'K': K, 'L': L, 
            'cost (K)': cK,
            'cost (L)': cL,
            'gradient (K)': gK,
            'gradient (L)': gL,
            'best response (K)': bK,
            'best response (L)': bL,
            'cost to go (P)': P,
            'cost to go (W)': W,
            'spectral radius': spec_radius}

    return out
        

def plot(axs, out, Kopt, Lopt, title='', set_xlabel=False, set_ylabel=False):
    """ Plots the output of a training loop """
    axs[0].set_title(title)
    axs[0].plot(np.linalg.norm(out['gradient (K)'], axis=(1,2)))
    axs[0].plot(np.linalg.norm(out['gradient (L)'], axis=(1,2)))
    axs[0].set_yscale('log')
    axs[1].plot(out['spectral radius'], label='closed loop stability')
    [axs[2].axhline(y=_, color='k', ls=':') for _ in Kopt.flatten()]
    axs[2].plot(out['K'].reshape(-1, Kopt.shape[0]*Kopt.shape[1]))
    axs[2].plot(out['L'].reshape(-1, Lopt.shape[0]*Lopt.shape[1]))
    axs[3].plot(out['cost (K)'])
    axs[3].plot(out['cost (L)'])

    if set_ylabel: 
        axs[0].set_ylabel('Grad norm')
        axs[1].set_ylabel('Spectral radius')
        axs[2].set_ylabel('Feedback policy')
        axs[3].set_ylabel('Cost')
    if set_xlabel: 
        axs[3].set_xlabel('Iterations')

def test_and_plot():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_iter', type=int, default=int(1e2), help='number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='size of initial random perturbation')
    parser.add_argument('--type', type=str, default='zs', choices=['zs','nzs'], help='game class (zero-sum or non-zero-sum)')
    parser.add_argument('--save', type=str, default=None, help='output file (save as pickle)')
    parser.add_argument('--no-plot', default=False, action='store_false', help='plot results')
    args = parser.parse_args()

    if args.type == 'zs':
        opt, params, info = random_zero_sum()
        print("==Policy optimization for zero-sum linear quadratic game==")
    elif args.type == 'nzs':
        opt, params, info = random_non_zero_sum()
        print("==Policy optimization for non-zero-sum linear quadratic game==")
    else:
        raise Error

    Kopt, Lopt = opt['K'], opt['L']
    traj_opt = simulate(args.seed, K=Kopt, L=Lopt, num_steps=int(1e2), **params)

    Popt, Wopt = costtogo(Kopt, Lopt, **params)
    grad = gradients(Kopt, Lopt, Popt, Wopt, **params)
    br = bestresponses(Kopt, Lopt, Popt, Wopt, **params)
    
    assert np.isclose(np.linalg.norm(grad[0]), 0) and np.isclose(np.linalg.norm(grad[1]), 0), "Error: Gradients are not zero at Kopt, Lopt!"
    assert np.isclose(np.linalg.norm(br[0]-Kopt), 0) and np.isclose(np.linalg.norm(br[1]-Lopt), 0), "Error: Kopt,Lopt is not a Nash!"

    np.random.seed(args.seed)

    K0 = Kopt + args.eps*np.random.randn(*Kopt.shape)
    L0 = Lopt + args.eps*np.random.randn(*Lopt.shape)

    train_params = dict(seed=args.seed, K0=K0, L0=L0, num_iter=args.num_iter)

    out_sim  = train(lr1=args.lr, lr2=args.lr, **train_params, **params)
    out_seq1 = train(lr1=float('inf'), lr2=args.lr, **train_params, **params)
    out_seq2 = train(lr1=args.lr, lr2=float('inf'), **train_params,  **params)
    out_best = train(lr1=float('inf'), lr2=float('inf'), **train_params,  **params)
    print()

    if args.save:
        import pickle
        from os import path
        outs = dict(simultaneous=out_sim, sequential1=out_seq1, sequential2=out_seq2, bestresponse=out_best)
        with open(args.save+'.pkl', 'wb') as f:
            pickle.dump(outs, f)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=4, ncols=4)
        plot_params = dict(Kopt=Kopt, Lopt=Lopt)
        plot(axs[:,0], out_sim, title='Simultaneous', set_ylabel=True, **plot_params)
        plot(axs[:,1], out_seq1, title='Sequential (K)', set_xlabel=True, **plot_params)
        plot(axs[:,2], out_seq2, title='Sequential (L)', **plot_params)
        plot(axs[:,3], out_best, title='Coupled Riccati', **plot_params)
        plt.show()

        plt.plot(traj_opt[0])
        plt.show()


if __name__ == "__main__":
    test_and_plot()

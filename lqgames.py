""" Policy optimization for linear quadratic games.

    Author: Ben Chasnov (bchasnov@gmail.com)

    2021
"""
import numpy as np
from scipy.linalg import solve, solve_discrete_are, LinAlgError
from scipy.linalg import solve
from numpy.linalg import inv, norm, eigvals
from sys import stdout

def simulate(seed, K, L, A, B, Q, R, num_steps, noise=0):
    """ Simulate a two-player linear dynamical system with state feedback policies and quadratic costs.

          x(0) = x0
          x(t+1) = A*x(t) + B1*u1(t) + B2*u2(t), t=0,1,...

        where each player chooses action by state feedback

          u1(t) = K*x(t) + n1(t)
          u2(t) = L*x(t) + n2(t)

        with random gaussian noise n1, n2.

        The costs at each state x(t) and actions u1(t), u2(t) for each player are

          cost1(x,u1,u2) = x'*Q1*x + u1'*R11*u1 + u2'*R12*u2
          cost2(x,u1,u2) = x'*Q2*x + u1'*R21*u1 + u2'*R22*u2

        Returns the trajectories x, (u1, u2), and (cost1, cost2)
    """
    n = A.shape[0]
    Q1, Q2 = Q
    B1, B2 = B
    (R11, R12), (R21, R22) = R

    # Random initialization
    np.random.seed(seed)
    x0 = np.random.randn(n)

    # States, actions and costs
    x = np.zeros((num_steps+1, n))
    u1 = np.zeros((num_steps, B1.shape[1]))
    u2 = np.zeros((num_steps, B2.shape[1]))
    c1 = np.zeros((num_steps, 2))
    c2 = np.zeros((num_steps, 2))

    x[0] = x0
    for t in range(num_steps):
        # Linear feedback (P1 and P2)
        u1[t] = K@x[t] + noise*np.random.randn(K.shape[0])
        u2[t] = L@x[t] + noise*np.random.randn(L.shape[0])

        # Costs (P1 and P2)
        c1[t] = x[t]@Q1@x[t] + u1[t]@R11@u1[t] + u2[t]@R12@u2[t]
        c2[t] = x[t]@Q2@x[t] + u1[t]@R21@u1[t] + u2[t]@R22@u2[t]

        # State transition
        x[t+1] = A@x[t] + B1@u1[t] + B2@u2[t]
    
    return x, (u1, u2), (c1, c2)

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

    _K = solve( R11 + B1.T @ P @ B1, -B1.T @ P @ (A + B2@L) )
    _L = solve( R22 + B2.T @ W @ B2, -B2.T @ W @ (A + B1@K) )
    return _K, _L


def spectral_radius(A):
    """ Spectral radius of A. 

        If the spectral radius of A is less than 1, then
        x(t+1) = A*x(t) is exponentially stable.
    """
    return np.max(np.abs(eigvals(A)))

    
def train(seed: int, num_iter: int, lr1: float, lr2: float, K0, L0, **params):
    """ Policy optimization for a two-player linear quadratic game.

        Four training methods possible:

        1) Simultaneous policy gradient with learning rates (lr1, lr2):
            K(t+1) = K(t) - lr1*gradK(t)
            L(t+1) = L(t) - lr2*gradL(t)
          where
            gradK(t) = (d/dK) cost1(K(t), L(t))
            gradL(t) = (d/dL) cost2(K(t), L(t))

        2) Best response iteration:
            K(t+1) = bestK(t)
            L(t+1) = bestL(t)
          where
            bestK(t) = argmin_K costK(K, L(t))
            bestL(t) = argmin_L costL(K(t), L)

        3) Sequential update (K as leader) 
            K(t+1) = bestK(t)
            L(t+1) = L(t) - lr2*gradL(t)

        4) Sequential update (L as leader) 
            K(t+1) = K(t) - lr1*gradK(t)
            L(t+1) = bestL(t)

        returns a dict of results

        """
    print("\nTraining with learning rates {},{}".format(lr1, lr2))

    A = params['A']
    B1, B2 = params['B']
    Q1, Q2 = params['Q']
    
    K = np.empty((num_iter+1, *B1.T.shape))
    L = np.empty((num_iter+1, *B2.T.shape))
    P = np.empty((num_iter, *Q1.shape))
    W = np.empty((num_iter, *Q2.shape))

    costK = np.empty(num_iter)
    costL = np.empty(num_iter)
    gradK = np.empty((num_iter, *B1.T.shape))
    gradL = np.empty((num_iter, *B2.T.shape))
    bestK = np.empty((num_iter, *B1.T.shape))
    bestL = np.empty((num_iter, *B2.T.shape))

    spec_radius = np.empty(num_iter)
    
    K[0], L[0] = K0, L0
    for t in range(num_iter):
        P[t], W[t] = costtogo(K[t], L[t], **params)
        gradK[t], gradL[t] = gradients(K[t], L[t], P[t], W[t], **params)
        bestK[t], bestL[t] = bestresponses(K[t], L[t], P[t], W[t], **params)

        # Player 1
        if lr1 < float('inf'):
            K[t+1] = K[t] - lr1*gradK[t]
        else:
            K[t+1] = bestK[t]

        # Player 2
        if lr2 < float('inf'):
            L[t+1] = L[t] - lr2*gradL[t]
        else:
            L[t+1] = bestL[t]

        # spectral radius of closed loop system A+B1*K+B2*L
        spec_radius[t] = spectral_radius(A+B1@K[t]+B2@L[t])
        costK[t] = np.trace(P[t])
        costL[t] = np.trace(W[t])

        stdout.write('\r iter={} costK={:.1f} costL={:.1f} |gradK|={:.0e} |gradL|={:.0e}'.format(t+1, costK[t], costL[t], norm(gradK[t]), norm(gradL[t])))

    out = { 'K': K, 'L': L, 
            'cost (K)': costK,
            'cost (L)': costL,
            'gradient (K)': gradK,
            'gradient (L)': gradL,
            'best response (K)': bestK,
            'best response (L)': bestL,
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
        if spectral_radius(A) < 1: break
    B = np.identity(N)
    m = N//2
    B1, B2 = B[:, :m], B[:, m:]

    def psd(n):
        M = np.random.randn(n, n)
        return M@M.T + np.eye(n)
    def sym(n):
        M = np.random.randn(n, n)
        return M+M.T

    Q1 = psd(N)
    Q2 = psd(N)
    R11 = psd(m)
    R12 = 0.5*sym(N-m)
    R21 = 0.5*sym(m)
    R22 = psd(N-m)

    B = [B1, B2]
    Q = [Q1, Q2]
    R = [[R11, R12], [R21, R22]]

    K = np.zeros(B1.T.shape)
    L = np.zeros(B2.T.shape)
    # optimal feedback gains
    for _ in range(1000):
        P, W = costtogo(K, L, A, B, Q, R)
        K, L = bestresponses(K, L, P, W, A, B, Q, R)

    costK = np.trace( P )
    costL = np.trace( W )
    
    return dict(K=K, L=L),\
           dict(A=A, B=B, Q=Q, R=R),\
           dict(cost=(costK, costL))
    

def test_and_plot():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_iter', type=int, default=int(5e1), help='number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='size of initial random perturbation')
    parser.add_argument('--type', type=str, default='zs', choices=['zs','nzs'], help='game class (zero-sum or non-zero-sum)')
    parser.add_argument('--save', type=str, default=None, help='output file (save as pickle)')
    parser.add_argument('--no-plot', default=False, action='store_false', help='plot results')
    args = parser.parse_args()

    if args.type == 'zs':
        opt, params, info = random_zero_sum(args.seed)
        print("==Policy optimization for zero-sum linear quadratic game==")
    elif args.type == 'nzs':
        opt, params, info = random_non_zero_sum(args.seed)
        print("==Policy optimization for non-zero-sum linear quadratic game==")
    else:
        raise TypeError
    
    # Tests 
    Kopt, Lopt = opt['K'], opt['L']
    Popt, Wopt = costtogo(Kopt, Lopt, **params)
    grad = gradients(Kopt, Lopt, Popt, Wopt, **params)
    br = bestresponses(Kopt, Lopt, Popt, Wopt, **params)
    
    assert np.isclose(np.linalg.norm(grad[0]), 0) and np.isclose(np.linalg.norm(grad[1]), 0), "Error: Gradients at (Kopt, Lopt) are not zero!"
    assert np.isclose(np.linalg.norm(br[0]-Kopt), 0) and np.isclose(np.linalg.norm(br[1]-Lopt), 0), "Error: Joint policy (Kopt,Lopt) is not a Nash!"

    # Perturb initial condition eps away from optimal
    K0 = Kopt + args.eps*np.random.randn(*Kopt.shape)
    L0 = Lopt + args.eps*np.random.randn(*Lopt.shape)

    # Compare the training algorithms with different learning rates
    train_params = dict(seed=args.seed, K0=K0, L0=L0, num_iter=args.num_iter)
    out_sim  = train(lr1=args.lr, lr2=args.lr, **train_params, **params)
    out_seq1 = train(lr1=float('inf'), lr2=args.lr, **train_params, **params)
    out_seq2 = train(lr1=args.lr, lr2=float('inf'), **train_params,  **params)
    out_best = train(lr1=float('inf'), lr2=float('inf'), **train_params,  **params)
    print()

    # Save
    if args.save:
        import pickle
        from os import path
        outs = dict(simultaneous=out_sim, sequential1=out_seq1, sequential2=out_seq2, bestresponse=out_best)
        with open(args.save+'.pkl', 'wb') as f:
            pickle.dump(outs, f)

    # Plot
    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=4, ncols=4)
        plot_params = dict(Kopt=Kopt, Lopt=Lopt)
        plot(axs[:,0], out_sim, title='Simultaneous', set_ylabel=True, **plot_params)
        plot(axs[:,1], out_seq1, title='Sequential (K)', set_xlabel=True, **plot_params)
        plot(axs[:,2], out_seq2, title='Sequential (L)', **plot_params)
        plot(axs[:,3], out_best, title='Coupled Riccati', **plot_params)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_and_plot()

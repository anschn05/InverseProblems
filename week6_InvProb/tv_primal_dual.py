import numpy as np


def tv_primal_dual(A, L, b, lam):

    m, n = A.shape
    theta = 1.0

    # first derivative matrix*    D = np.diff(np.eye(n), axis=0)*
    # estimate ||K||
    K = np.vstack([A,L])

    lipc = np.linalg.norm(K, 2)

    tau = 0.99 / lipc

    sigma = 0.99 / lipc

    x = np.zeros(n)
    xbar = x.copy()

    pA = np.zeros(m)
    pL = np.zeros(n - 1)

    for k in range(5000):

        x_old = x.copy()

        # dual variable for data fidelity
        qA = pA + sigma * (A @ xbar)
        pA = (qA - sigma * b) / (1.0 + sigma)

        # dual variable for TV
        qL = pL + sigma * (L @ xbar)
        pL = np.clip(qL, -lam, lam)

        # primal update
        grad = A.T @ pA + L.T @ pL

        x = x - tau * grad

        # extrapolation
        xbar = x + theta * (x - x_old)

    return x
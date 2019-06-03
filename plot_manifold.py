import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


class ObjectiveFunctionClass():
    def __init__(self):
        self.neq = 2

        self.p0 = np.zeros(self.neq, float)
        self.p0[0] = -51
        self.p0[1] = -46

        self.p1 = np.zeros((self.neq, self.neq), float)
        self.p1[0, 0] = 2
        self.p1[0, 1] = 4
        self.p1[1, 0] = 3
        self.p1[1, 1] = 2

        self.p2 = np.zeros((self.neq, self.neq, self.neq), float)
        self.p2[0, 0, 0] = 2
        self.p2[0, 0, 1] = 3
        self.p2[0, 1, 0] = 0
        self.p2[0, 1, 1] = 1

        self.p2[1, 0, 0] = 1
        self.p2[1, 0, 1] = 2
        self.p2[1, 1, 0] = 0
        self.p2[1, 1, 1] = 2

        x = np.linspace(-2, 5, 7)
        y = np.linspace(-2, 5, 8)
        Xx, Xy = np.meshgrid(x, y)
        X = np.array([Xx, Xy])
        mask = np.ones_like(Xx)

    def objective_function(self, X):
        # D0
        D0 = np.einsum('i,i', self.p0, self.p0)

        # D1
        D1_b = np.einsum('i,ij,j', self.p0, self.p1, X)
        D1_g = np.einsum('j,ji,i', X.T, self.p1.T, self.p0)

        D1 = D1_b + D1_g

        # D2
        D2_b = np.einsum('i,ijk,j,k', self.p0.T, self.p2, X, X)
        D2_g = np.einsum('j,ji,ik,k', X.T, self.p1.T, self.p1, X)
        D2_y = np.einsum('k,j,kji,i', X.T, X.T, self.p2.T, self.p0)

        D2 = D2_b + D2_g + D2_y

        # D3
        D3_g = np.einsum('j,ji,ikl,k,l', X.T, self.p1.T, self.p2, X, X)
        D3_y = np.einsum('k,j,kji,il,l', X.T, X.T, self.p2.T, self.p1, X)

        D3 = D3_g + D3_y

        # D4
        D4 = np.einsum('k,j,kji,inm,n,m', X.T, X.T, self.p2.T, self.p2, X, X)

        D = D0 + D1 + D2 + D3 + D4
        return D


if __name__ == '__main__':
    pmin = -15
    pmax = 15
    x = np.linspace(pmin, pmax, 101)
    y = np.linspace(pmin, pmax, 101)
    Xx, Xy = np.meshgrid(x, y, indexing='xy')
    X = np.array([Xx, Xy])

    ObjectClass = ObjectiveFunctionClass()
    manifold = np.zeros_like(Xx)
    for y_idx in range(len(Xx)):
        for x_idx in range(len(Xx[0])):
            X = np.array([Xx[y_idx, x_idx], Xy[y_idx, x_idx]])
            manifold[y_idx, x_idx] = ObjectClass.objective_function(X)
    fig, ax = plt.subplots()
    cs = ax.contourf(Xx, Xy, manifold, 30, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    cbar = fig.colorbar(cs)
    plt.show()

import numpy as np

class BilinearInterpolator:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.nx = len(X)
        self.ny = len(Y)

    def eval(self, x, y):
        if x < self.X[0] or x > self.X[-1] or y < self.Y[0] or y > self.Y[-1]:
            return None, None, None

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)

        tx = x - x0
        ty = y - y0

        z00 = self.Z[y0, x0]
        z10 = self.Z[y0, x1]
        z01 = self.Z[y1, x0]
        z11 = self.Z[y1, x1]

        z0 = z00 * (1 - tx) + z10 * tx
        z1 = z01 * (1 - tx) + z11 * tx
        z = z0 * (1 - ty) + z1 * ty

        fx = (z10 - z00) * (1 - ty) + (z11 - z01) * ty
        fy = (z01 - z00) * (1 - tx) + (z11 - z10) * tx

        return z, fx, fy

import numpy as np

class PerlinNoise2D:
    def __init__(self, seed=None, grid_size=100, scale=10.0):
        self.seed = seed
        self.grid_size = grid_size
        self.scale = scale

        rng = np.random.default_rng(seed)
        self.grad = rng.normal(size=(grid_size, grid_size, 2))
        self.grad /= np.linalg.norm(self.grad, axis=2, keepdims=True)

    def fade(self, t):
        return 6*t**5 - 15*t**4 + 10*t**3

    def noise(self, x, y):
        x = x / self.scale
        y = y / self.scale

        x0 = int(np.floor(x)) % self.grid_size
        x1 = (x0 + 1) % self.grid_size
        y0 = int(np.floor(y)) % self.grid_size
        y1 = (y0 + 1) % self.grid_size

        tx = x - np.floor(x)
        ty = y - np.floor(y)

        g00 = self.grad[x0, y0]
        g10 = self.grad[x1, y0]
        g01 = self.grad[x0, y1]
        g11 = self.grad[x1, y1]

        v00 = np.array([tx, ty])
        v10 = np.array([tx-1, ty])
        v01 = np.array([tx, ty-1])
        v11 = np.array([tx-1, ty-1])

        d00 = np.dot(g00, v00)
        d10 = np.dot(g10, v10)
        d01 = np.dot(g01, v01)
        d11 = np.dot(g11, v11)

        u = self.fade(tx)
        v = self.fade(ty)

        x1_interp = d00 * (1-u) + d10 * u
        x2_interp = d01 * (1-u) + d11 * u
        result = x1_interp * (1-v) + x2_interp * v

        return result

    def generate_map(self, width, height):
        z = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                z[i, j] = self.noise(j, i)
        return z
    

class Landscape3D:
    def __init__(self) -> None:
        pass
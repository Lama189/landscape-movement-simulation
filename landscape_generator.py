import numpy as np
import plotly.graph_objects as go
from perlin import PerlinNoise2D
from plotly.io import to_html
from bilinear_interpolator import BilinearInterpolator
import heapq

class LandscapeGenerator:
    def __init__(self, width=200, height=100, scale=20.0, seed=None, grid_size=100, zscale=5):
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.grid_size = grid_size
        self.zscale = zscale

        self.perlin = PerlinNoise2D(seed=self.seed, grid_size=self.grid_size, scale=self.scale)


    def generate_heightmap(self):
        height_map = self.perlin.generate_map(self.width, self.height)
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        height_map *= self.zscale
        return height_map


    def simulate_motion(self, steps=200, dt=0.1, g=9.81, mu=0.05,
                    x0=10, y0=10, vx0=0.5, vy0=0.0):
        height_map = self.generate_heightmap()

        X = np.arange(0, self.width)
        Y = np.arange(0, self.height)
        X, Y = np.meshgrid(X, Y)
        interpolator = BilinearInterpolator(X, Y, height_map)

        def f(x, y):
            z, _, _ = interpolator.eval(x, y)
            return z if z is not None else 0

        def grad_f(x, y):
            _, fx, fy = interpolator.eval(x, y)
            if fx is None or fy is None:
                return 0.0, 0.0
            return fx, fy

        x, y = x0, y0
        vx, vy = vx0, vy0
        trajectory = [(x, y, f(x, y))]

        for _ in range(steps):
            fx, fy = grad_f(x, y)
            denom = np.sqrt(fx**2 + fy**2 + 1)

            ax = -g * fx / denom - mu * vx
            ay = -g * fy / denom - mu * vy

            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            z = f(x, y)

            z = f(x, y) + 0.5
            trajectory.append((x, y, z))

        return zip(*trajectory)


    def astar_continuous(self, interpolator, start, goal, step=2):
            def heuristic(a, b):
                return np.linalg.norm(np.array(a) - np.array(b))

            open_set = []
            heapq.heappush(open_set, (0, start))

            came_from = {}
            g_score = {start: 0}

            while open_set:
                _, current = heapq.heappop(open_set)
                if heuristic(current, goal) < step:  
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    return path[::-1]

                cx, cy = current
                cz, _, _ = interpolator.eval(cx, cy)

                for dx, dy in [(-step,0),(step,0),(0,-step),(0,step),
                            (-step,-step),(step,step),(-step,step),(step,-step)]:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue
                    nz, _, _ = interpolator.eval(nx, ny)
                    if nz is None: 
                        continue

                    dz = abs(nz - cz)
                    move_cost = np.sqrt(dx**2 + dy**2) + dz 
                    tentative_g = g_score[current] + move_cost

                    neighbor = (nx, ny)
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
            return None


    def find_path_continuous(self, start=(10,10), goal=(80,80), step=2):
        height_map = self.generate_heightmap()
        X = np.arange(0, self.width)
        Y = np.arange(0, self.height)
        interpolator = BilinearInterpolator(X, Y, height_map)

        path = self.astar_continuous(interpolator, start, goal, step=step)
        if not path:
            return [], [], []
        xs, ys = zip(*path)
        zs = [(interpolator.eval(x, y)[0] or 0) + 0.5 for x, y in path]
        return xs, ys, zs

    def plot(self, output_file=None, return_html=False, with_motion=False, with_path=False, animate=False):
        height_map = self.generate_heightmap()
        X = np.linspace(0, self.width, self.width)
        Y = np.linspace(0, self.height, self.height)
        X, Y = np.meshgrid(X, Y)

        fig = go.Figure()

        fig.add_trace(go.Surface(
            z=height_map,
            x=X,
            y=Y,
            colorscale="Earth",
            showscale=True
        ))

        if with_motion:
            xs, ys, zs = self.simulate_motion()
            xs, ys, zs = list(xs), list(ys), list(zs)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines", line=dict(color="red", width=4), name="Trajectory"
            ))

        if with_path:
            xs, ys, zs = self.find_path_continuous((10, 10), (80, 80))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color="yellow", width=4),
                name="A* Path"
            ))

            if xs and ys and zs:
                fig.add_trace(go.Scatter3d(
                    x=[xs[0]], y=[ys[0]], z=[zs[0]],
                    mode="markers+text",
                    text=["Start"],
                    textposition="top center",
                    marker=dict(size=6, color="green"),
                    name="Start"
                ))
                fig.add_trace(go.Scatter3d(
                    x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                    mode="markers+text",
                    text=["Goal"],
                    textposition="top center",
                    marker=dict(size=6, color="red"),
                    name="Goal"
                ))

            if animate and xs and ys and zs:
                frames = []
                for i in range(len(xs)):
                    frames.append(go.Frame(
                        data=[
                            go.Surface(
                                z=height_map,
                                x=X,
                                y=Y,
                                colorscale="Earth",
                                showscale=True
                            ),
                            go.Scatter3d(
                                x=xs, y=ys, z=zs,
                                mode="lines",
                                line=dict(color="yellow", width=4),
                                name="A* Path"
                            ),
                            go.Scatter3d(
                                x=[xs[i]], y=[ys[i]], z=[zs[i]],
                                mode="markers",
                                marker=dict(size=6, color="blue", symbol="circle"),
                                name="Object"
                            )
                        ],
                        name=f"frame{i}"
                    ))

                fig.frames = frames
                fig.update_layout(
                    updatemenus=[{
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 200, "redraw": True},
                                                "fromcurrent": True}]
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate"}]
                            }
                        ]
                    }]
                )

        fig.update_layout(
            title="3D Ландшафт (Шум Перлина)",
            scene=dict(
                xaxis=dict(range=[0, self.width]),
                yaxis=dict(range=[0, self.height]),
                zaxis=dict(range=[0, self.zscale]),
                aspectmode="manual",
                aspectratio=dict(
                    x=self.width / max(self.width, self.height),
                    y=self.height / max(self.width, self.height),
                    z=0.5
                )
            )
        )

        if return_html:
            return to_html(fig, include_plotlyjs="cdn", full_html=False)
        elif output_file:
            fig.write_html(output_file)
        else:
            fig.show()


import numpy as np
import plotly.graph_objects as go
from perlin import PerlinNoise2D
from plotly.io import to_html
import pandas as pd
import heapq

def plot_topview(height_map, return_html=True):
        fig = go.Figure(data=go.Heatmap(
            z=height_map,
            colorscale="Earth",
            showscale=True
        ))

        fig.update_layout(
            title="Выберите старт и финиш (кликните на 2 точки)",
            xaxis=dict(scaleanchor="y"),
            yaxis=dict(autorange="reversed"),
            dragmode="pan"
        )

        if return_html:
            return to_html(fig, include_plotlyjs="cdn", full_html=False)
        
        
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

    def heuristic(self, state, goal):
        x, y, vx, vy = state
        gx, gy = goal
        return abs(x - gx) + abs(y - gy) + abs(vx) + abs(vy)

    import pandas as pd

    def astar_discrete(self, height_map, start, goal, max_steps=5000):
        moves = [-1, 0, 1]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        logs = [] 
        while open_set:
            _, current = heapq.heappop(open_set)
            x, y, vx, vy = current

            if (x, y) == goal and vx == 0 and vy == 0:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path = path[::-1]

                df = pd.DataFrame(logs)
                return path, df 

            if g_score[current] > max_steps:
                continue

            for ax in moves:
                for ay in moves:
                    new_vx = vx + ax
                    new_vy = vy + ay
                    new_x = x + new_vx
                    new_y = y + new_vy

                    if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                        continue

                    dz = height_map[new_y, new_x] - height_map[y, x]
                    step_cost = 1 + max(0, dz * 10)
                    tentative_g = g_score[current] + step_cost

                    neighbor = (new_x, new_y, new_vx, new_vy)

                    if tentative_g < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))

                        logs.append({
                            "X": new_x, "Y": new_y,
                            "Vx": new_vx, "Vy": new_vy,
                            "Высота": round(height_map[new_y, new_x], 2),
                            "Δz": round(dz, 2),
                            "Стоимость шага": round(step_cost, 2),
                            "Накопленная стоимость": round(tentative_g, 2),
                            "Эвристика": round(self.heuristic(neighbor, goal), 2),
                            "F = g+h": round(f_score, 2)
                        })

        return None, pd.DataFrame()


    def find_path_discrete(self, start=(10, 10, 0, 0), goal=(80, 80)):
        height_map = self.generate_heightmap()
        path, df = self.astar_discrete(height_map, start, goal)

        if path is None:
            return [], [], [], pd.DataFrame()

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [height_map[y, x] + 0.5 for x, y, _, _ in path]

        return xs, ys, zs, df
    
    
    def plot(self, output_file=None, return_html=False, with_motion=False, with_path=False, animate=False, start=(10, 10, 0, 0), goal=(80, 80)):
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

        if with_path:
            xs, ys, zs, df = self.find_path_discrete(start, goal)
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

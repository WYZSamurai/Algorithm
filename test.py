import plotly.graph_objects as go
import torch


def func(x1: torch.Tensor, x2: torch.Tensor):
    y = x1**2+x2.t()**2
    return y


if __name__ == "__main__":
    n = 1000
    x1 = torch.linspace(-10, 10, n)
    x2 = torch.linspace(-10, 10, n)
    x = x1.reshape(n, 1)@torch.ones(1, n)
    y = x2.reshape(n, 1)@torch.ones(1, n)
    z = func(x, y)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=x1, y=x2, z=z, cauto=True, colorscale="rainbow"),
    )
    fig.update_layout(
        template="simple_white",
    )
    fig.show()

import plotly.graph_objects as go
import torch


def func(x1: torch.Tensor, x2: torch.Tensor):
    y = x1.pow(2)+x2.pow(2)
    return y


if __name__ == "__main__":
    n = 5
    x1 = torch.linspace(-2, 2, n).reshape(n, 1)@torch.ones(1, n)
    x2 = torch.linspace(-2, 2, n).reshape(n, 1)@torch.ones(1, n)
    z = x1**2+x2**2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=x1, y=x2, z=z)
    )
    fig.update_layout(
        template="simple_white",
    )
    fig.show()

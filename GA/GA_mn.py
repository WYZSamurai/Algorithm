import plotly.graph_objects as go
import torch


def func(x: torch.Tensor):
    y = x.t()@x
    return y


def plot():
    n = 1000
    x1 = torch.linspace(-10, 10, n)
    x2 = torch.linspace(-10, 10, n)
    x = x1.reshape(n, 1)@torch.ones(1, n)
    y = x2.reshape(n, 1)@torch.ones(1, n)
    z = x**2+y.t()**2

    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=x1, y=x2, z=z, cauto=True, colorscale="rainbow"),
    )
    fig.update_layout(
        template="simple_white",
    )
    fig.show()


def decode(dna: torch.Tensor, mx: float, md: float):
    pass


def selection(dna: torch.Tensor, fit: torch.Tensor):
    pass


def crossover(dna: torch.Tensor, Pc: float):
    pass


def mutation(dna: torch.Tensor, Pm: float):
    pass


def GA(NP=50, L=20, G=100, Pc=0.8, Pm=0.05, mx=0, md=10):
    pass


if __name__ == "__main__":
    plot()

    GA(NP=50, L=20, G=100, Pc=0.8, Pm=0.05, mx=0, md=10)

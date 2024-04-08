import plotly.graph_objects as go
import torch


# 线阵，等间距，等幅同相
def patt1(M: int, l: float, d: float, delta: int, theta_0: float) -> None:
    k = 2*torch.pi/l
    theta_0 = (torch.tensor(theta_0)*torch.pi)/180

    theta = torch.linspace(-torch.pi/2, torch.pi/2, delta)
    F = torch.zeros((delta,))
    for i in range(delta):
        fai = (k*d * torch.arange(
            0, M))*(torch.sin(theta[i])-torch.sin(theta_0)).to(dtype=torch.float)
        temp = torch.exp(torch.complex(
            torch.zeros(M,).to(dtype=torch.float), fai))
        F[i] = torch.abs(torch.sum(temp))

    theta = theta*180/torch.pi
    Fdb = 20*torch.log10(F/F.max())
    Fdb = torch.where(Fdb < -50, input=-50+torch.randint(-1,
                      1, (1,))*torch.rand((1,)), other=Fdb)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(x=theta, y=Fdb)
    )
    fig.update_layout(
        template="simple_white",
    )
    fig.show()
    return F


if __name__ == "__main__":
    (M, l, delta, theta_0) = (20, 1, 360, 0)
    d = l/2
    patt1(M, l, d, delta, theta_0)

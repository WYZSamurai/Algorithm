import plotly.graph_objects as go
import torch
import time


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
cpu = torch.device("cpu")


def func(x: torch.Tensor):
    y = torch.matmul(x.t(), x)
    return y


def pl():
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
    NP = dna.shape[0]
    L = dna.shape[3]

    scale = float((md-mx)/(pow(2, L)-1))
    a = (2*torch.ones(L,)).pow(torch.arange(0, L)
                               ).to(dtype=torch.float, device=device)
    a, _ = a.sort(descending=True)
    x = scale*torch.matmul(dna, a)+mx

    fit = torch.zeros(NP,).to(dtype=torch.float, device=device)
    for i in range(NP):
        fit[i] = func(x[i])

    maxindex = torch.argmax(fit)
    minindex = torch.argmin(fit)

    dnabest = dna[maxindex]
    xbest = x[maxindex]
    ybest = func(xbest)

    if fit[maxindex]-fit[minindex] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minindex])/(fit[maxindex]-fit[minindex])

    # print("dna值：\n", dna)
    # print("X值：\n", x)
    # print("fit值：\n", fit)
    print("最佳x值：\n", xbest)
    print("最佳y值：\n", ybest)
    return fit, xbest, ybest, dnabest


def selection(dna: torch.Tensor, fit: torch.Tensor):
    NP = dna.shape[0]
    P = (fit/fit.sum()).reshape(NP,)
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    dna = dna[index]
    return dna


def crossover(dna: torch.Tensor, Pc: float):
    NP = dna.shape[0]
    L = dna.shape[3]
    for i in range(NP, 2):
        P = torch.rand(size=(1,)).item()
        if P < Pc:
            randcut = torch.randint(int(L/2), L, (1,)).item()
            temp = dna[i, :, :, randcut:].clone()
            dna[i, :, :, randcut:] = dna[i+1, :, :, randcut:]
            dna[i+1, :, :, randcut:] = temp
    return dna


def mutation(dna: torch.Tensor, Pm: float):
    P = (torch.rand(size=dna.shape)-Pm * torch.ones(size=dna.shape))
    dna = torch.where(P > 0, input=dna, other=1-dna)
    return dna


def GA(NP=50, L=20, m=2, n=1, G=100, Pc=0.8, Pm=0.05, mx=0, md=10):
    xbest = torch.zeros(size=(G, m, n))
    ybest = torch.zeros(size=(G,))

    dna = torch.randint(0, 2, (NP, m, n, L)).to(
        dtype=torch.float, device=device)

    for i in range(G):
        print("第", i+1, "代")
        fit, xbest[i], ybest[i], dnabest = decode(dna, mx, md)
        dna = selection(dna, fit)
        dna = crossover(dna, Pc)
        dna = mutation(dna, Pm)
        dna[0] = dnabest

    bestindex = torch.argmax(ybest)
    print("算法结束")
    print("最佳x值为：\n", xbest[bestindex])
    print("最佳函数值为：\n", ybest[bestindex])
    return ybest


if __name__ == "__main__":
    G = 1000

    start_time = time.time()
    ybest = GA(G=G, NP=200, Pc=0.80, Pm=0.100, L=40)
    end_time = time.time()

    x = torch.arange(1, G+1).to(device=cpu)
    print("算法耗时：", end_time-start_time)

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(x=x, y=ybest)
    )
    fig.update_layout(
        # ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
        template="simple_white",
    )
    fig.show()

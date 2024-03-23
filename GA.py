import torch
import matplotlib.pyplot as plt
import time


# 选择设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
cpu = torch.device("cpu")


# 目标函数
def func(x: torch.Tensor):
    y = x+10*torch.cos(5*x)+7*torch.sin(4*x)
    return y


# 绘制图像
def prin(min: float, max: float, n: int):
    x = torch.linspace(min, max, n, device=device, dtype=torch.float)
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x.to(device=cpu), y.to(device=cpu))
    plt.show()


def prin2(x: torch.Tensor, y: torch.Tensor):
    fig, ax = plt.subplots()
    ax.plot(x.to(device=cpu), y.to(device=cpu))
    plt.show()


# 解码过程
def decode(dna: torch.Tensor, mx: float, md: float):
    L = dna.shape[1]
    # 设置大小比例
    scale = float((md-mx)/(pow(2, L)-1))
    # 二进制向量转为十进制向量
    temp = (2*torch.ones(size=(L,))).pow(torch.arange(0, L)
                                         ).to(device=device, dtype=torch.float)
    temp, _ = torch.sort(temp, descending=True)
    temp = torch.diag(temp)
    t2 = torch.mm(dna, temp)
    temp = torch.ones(size=(L, 1)).to(device=device, dtype=torch.float)
    # 定义个体十进制值
    x = scale*torch.mm(t2, temp)+mx
    # print("X值为：", x)
    # 定义个体适应度
    fit = -func(x=x)
    # print("初始适应值为：", fit)
    # print("平均适应值为：", (fit.sum()/NP).item())
    # 筛出适应度最值
    maxindex = torch.argmax(fit)
    minindex = torch.argmin(fit)
    # 保存最佳个体的dna、十进制值、适应度
    # dnabest = dna[maxindex]
    xbest = x[maxindex]
    ybest = func(xbest)
    print("此代最小值为：", ybest.item())
    # 适应值归一化操作
    # 注意dna全部一致情况
    if fit[maxindex]-fit[minindex] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minindex])/(fit[maxindex]-fit[minindex])
    # print("归一化适应值为：", fit)
    return fit, xbest, ybest


# 选择阶段
def selection(dna: torch.Tensor, fit: torch.Tensor):
    NP = dna.shape[0]
    # 赌轮盘选择
    P = (fit/fit.sum()).reshape(NP,)
    # print("群体概率为：", P)

    # 根据概率选择个体
    # replacement是否放回
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    dna = dna[index, :]
    return dna


# 交叉操作
def crossover(dna: torch.Tensor, Pc: float):
    L = dna.shape[1]
    P = torch.rand(size=(1,)).to(device=cpu).item()
    if P < Pc:
        cut = torch.randint(int(L/2), L, size=(1,)
                            ).to(device=cpu, dtype=torch.int).item()
        temp = dna[0::2, cut:].clone()
        dna[0::2, cut:] = dna[1::2, cut:]
        dna[1::2, cut:] = temp
    return dna


def bianyi(a: int):
    if a == 1:
        a = 0
    elif a == 0:
        a = 1
    return a


# 变异操作
def mutation(dna: torch.Tensor, Pm: float):
    (NP, L) = dna.shape
    for i in range(NP):
        P = torch.rand(size=(1,)).item()
        if P < Pm:
            tubian = torch.randint(0, L, size=(
                1,), device=device, dtype=torch.int)
            dna[i, tubian.item()] = bianyi(dna[i, tubian.item()])
    return dna


# 变异操作
def mutation(dna: torch.Tensor, Pm: float):
    (NP, L) = dna.shape
    for i in range(NP):
        P = torch.rand(size=(1,)).item()
        if P < Pm:
            tubian = torch.randint(0, L, size=(
                1,), device=device, dtype=torch.int)
            dna[i, tubian.item()] = bianyi(dna[i, tubian.item()])
    return dna


# 遗传算法
# NP->种群数 L->编码长度 G->迭代次数 Pc->交叉概率 Pm->变异概率 mx->x最小值 md->x最大值
def GA(NP=50, L=20, G=100, Pc=0.8, Pm=0.05, mx=0, md=10):
    # 保存最佳x值
    xbest = torch.zeros(size=(G,)).to(device=device)
    # 保存最佳y值
    ybest = xbest.clone()

    # 初始化种群编码（NP，L）
    dna = torch.randint(0, 2, size=(NP, L)).to(
        device=device, dtype=torch.float)

    # GA循环
    for t in range(G):
        print("第", t+1, "轮迭代")
        # print("群体dna为：", dna)
        # 解码及计算适应值
        fit, xbest[t], ybest[t] = decode(dna=dna, mx=mx, md=md)
        # 选择阶段
        dna = selection(dna=dna, fit=fit)
        # 交叉操作
        dna = crossover(dna=dna, Pc=Pc)
        # 变异操作
        dna = mutation(dna=dna, Pm=Pm)

    minindex = torch.argmin(ybest)
    print("最佳x值为：", xbest[minindex].item())
    print("函数最小值为：", ybest[minindex].item())

    return xbest, ybest


if __name__ == "__main__":

    start_time = time.time()
    G = 1000
    _, ybest = GA(G=G, NP=150, Pc=0.80, Pm=0.050)
    x = torch.arange(1, G+1).to(device=cpu)
    end_time = time.time()
    print("算法耗时：", end_time-start_time)
    prin2(x, ybest)

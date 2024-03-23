import torch
import matplotlib.pyplot as plt
import time


# 选择设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
cpu = torch.device("cpu")


# 目标函数
def func(x: torch.Tensor):
    y = x+10*torch.cos(5*x)+7*torch.sin(4*x)
    return y


# 绘制图像
def prin(min: float, max: float, n: int):
    x = torch.linspace(min, max, n, device=device, dtype=torch.float64)
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x.to(device=cpu), y.to(device=cpu))
    plt.show()


def prin2(x: torch.Tensor, y: torch.Tensor):
    fig, ax = plt.subplots()
    ax.plot(x.to(device=cpu), y.to(device=cpu))
    plt.show()


# 解码过程
def decode(dna: torch.Tensor, L: int, mx: float, md: float):
    # 设置大小比例
    scale = float((md-mx)/(pow(2, L)-1))
    # 二进制向量转为十进制向量
    temp = (2*torch.ones(size=(L,))).pow(torch.arange(0, L))
    temp, _ = torch.sort(temp, descending=True)
    temp = torch.diag(temp).to(device=device, dtype=torch.int64)
    t2 = torch.mm(dna, temp)
    temp = torch.ones(size=(L, 1)).to(device=device, dtype=torch.int64)
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
    # print("此代最小值为：", ybest.item())
    # 适应值归一化操作
    # 注意dna全部一致情况
    if fit[maxindex]-fit[minindex] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minindex])/(fit[maxindex]-fit[minindex])
    # print("归一化适应值为：", fit)
    return fit, xbest, ybest


# 选择阶段
def selection(dna: torch.Tensor, fit: torch.Tensor, NP: int):
    # 设置个体选择概率
    P = torch.zeros(size=(NP,), device=device, dtype=torch.float64)
    # 赌轮盘选择
    sumfit = fit.sum()
    for i in range(NP):
        P[i] = fit[i]/sumfit
    # 打印群体概率
    # print(P)
    # 根据概率选择个体
    # replacement是否放回
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    dna = dna[index]
    return dna


# 选择阶段
def selection1(dna: torch.Tensor, fit: torch.Tensor, NP: int):
    # 设置个体选择概率
    P = torch.zeros(size=(NP,), device=device, dtype=torch.float64)
    # 赌轮盘选择
    sumfit = fit.sum()
    for i in range(NP):
        P[i] = fit[i]/sumfit
    # 打印群体概率
    # print(P)
    # 根据概率选择个体
    # replacement是否放回
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    dna = dna[index]
    return dna


# 交叉操作
def crossover(dna: torch.Tensor, Pc: float, L: int, NP: int):
    P = torch.rand(size=(1,)).item()
    if P < Pc:
        for i in range(0, NP, 2):
            begin = torch.randint(0, L, size=(
                1,), device=device, dtype=torch.int).item()
            temp = dna[i, begin:]
            dna[i, begin:] = dna[i+1, begin:]
            dna[i+1, begin:] = temp
    return dna


def bianyi(a: int):
    if a == 1:
        a = 0
    elif a == 0:
        a = 1
    return a


# 变异操作
def mutation(dna: torch.Tensor, L: int, Pm: float, NP: int):
    for i in range(NP):
        P = torch.rand(size=(1,)).item()
        if P < Pm:
            tubian = torch.randint(0, L, size=(
                1,), device=device, dtype=torch.int)
            dna[i, tubian.item()] = bianyi(dna[i, tubian.item()])
            # tubian = torch.randint(0, 2, size=(
            #     L,), device=device, dtype=torch.int)
            # for j in range(L):
            #     if tubian[j].item() == 1:
            #         dna[i, j] = bianyi(dna[i, j])
    return dna


# 遗传算法
# NP->种群数 L->编码长度 G->迭代次数 Pc->交叉概率 Pm->变异概率 mx->x最小值 md->x最大值
def GA(NP=50, L=20, G=100, Pc=0.8, Pm=0.05, mx=0, md=10):
    # 保存最佳x值
    xbest = torch.zeros(size=(G,), device=device)
    # 保存最佳y值
    ybest = torch.zeros(size=(G,), device=device)
    epochs = torch.zeros(size=(G,), device=device)

    # 初始化种群编码（NP，L）
    dna = torch.randint(0, 2, size=(NP, L)).to(
        device=device, dtype=torch.int64)

    # GA循环
    for t in range(G):
        # print("第", t+1, "轮迭代")
        epochs[t] = t+1
        # print("群体dna为：", dna)
        # 解码及计算适应值
        fit, xbest[t], ybest[t] = decode(dna=dna, L=L, mx=mx, md=md)
        # 选择阶段
        dna = selection(dna=dna, fit=fit, NP=NP)
        # 交叉操作
        dna = crossover(dna=dna, Pc=Pc, L=L, NP=NP)
        # 变异操作
        dna = mutation(dna=dna, L=L, Pm=Pm, NP=NP)

    minindex = torch.argmin(ybest)
    print("最佳x值为：", xbest[minindex].item())
    print("函数最小值为：", ybest[minindex].item())

    return xbest, ybest, epochs


def main():
    start_time = time.time()
    xbest, y, x = GA(G=1000, NP=150, Pc=0.80, Pm=0.050)
    end_time = time.time()
    print("算法耗时：", end_time-start_time)
    prin2(x, y)


if __name__ == "__main__":
    main()

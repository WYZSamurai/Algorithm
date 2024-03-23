import torch

device = torch.device("cpu")
# device = torch.device("cuda")


if __name__ == "__main__":
    # 生成01数组
    dna = torch.randint(0, 2, size=(3, 5)).to(dtype=torch.int64)
    # print(dna)
    t2 = (2*torch.ones(size=(5,))).pow(torch.arange(0, 5))
    print(t2)
    t2, b = torch.sort(t2, descending=True)
    t2 = torch.diag(t2).to(dtype=torch.int64)
    dna = torch.mm(dna, t2)
    # print(dna)
    t2 = torch.ones(size=(5, 1)).to(dtype=torch.int64)
    # print(t2)
    # print(torch.mm(dna, t2))

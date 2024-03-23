import torch

device = torch.device("cpu")
# device = torch.device("cuda")


if __name__ == "__main__":
    dna = torch.randint(0, 2, size=(3, 5), device=device, dtype=torch.int64)
    print(dna)
    t2 = (2*torch.ones(size=(5,), dtype=torch.int64)).pow(torch.arange(0,
                                                                       5, dtype=torch.int64))
    t2, b = torch.sort(t2, descending=True)
    t2 = torch.diag(t2)
    print(t2)
    t1 = torch.mm(dna, t2)
    t2 = torch.ones()
    print(t1)

import torch

device = torch.device("cpu")
# device = torch.device("cuda")


if __name__ == "__main__":
    (NP, L) = (3, 5)
    # P = torch.rand(0, 2, (NP, 1)).to(dtype=torch.float)@torch.ones(1, L)
    P = torch.rand(NP, L)
    print(P)

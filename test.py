import torch

# device = torch.device("cpu")
device = torch.device("cuda")


# 定义一个函数来交换n阶矩阵的每两行
def swap_rows(matrix: torch.Tensor):
    # 确保矩阵有偶数行
    if matrix.shape[0] % 2 != 0:
        raise ValueError("矩阵必须有偶数行才能交换每两行。")

    # 为行创建一个新的索引顺序
    new_order = torch.arange(matrix.shape[0])
    new_order[::2], new_order[1::2] = new_order[1::2], new_order[::2]

    # 返回交换行后的矩阵
    return matrix[new_order]


if __name__ == "__main__":
    new_order = torch.arange(100).reshape(10, 10)
    print(new_order)
    temp = new_order[0::2, 4:].clone()
    new_order[0::2, 4:] = new_order[1::2, 4:]
    new_order[1::2, 4:] = temp
    print(temp)
    print(new_order)
    print(int(2.999))

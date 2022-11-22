import torch

def get_one_hot(label, N):
    assert label.dim() == 1
    print(label)
    ones = torch.sparse.torch.eye(N)
    print(ones)
    ones = label.unsqueeze(-1)*ones
    print(ones)
    mask = ones.sum(dim=1) > 0
    print(mask)
    return ones[mask]

gt = torch.tensor([0.6, 0, 0.7, 0])
gt_one_hot = get_one_hot(gt, 4)
print(gt_one_hot)
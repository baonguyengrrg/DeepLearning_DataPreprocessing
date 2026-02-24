import torch
import torch.nn as nn
def demo():
    sample = torch.tensor([
        [20.4100, 2.0657, -16.7879],
        [10.6843, -5.8452, -8.9860],
        [9.0335, 13.3803, -2.1926],
        [0.9666, -0.9664, 6.8204],
        [-3.5667, 16.0060, -5.7119],
    ], dtype=torch.float32)
    print(sample)
    print("Before Group Norm")
    mean_before = sample.mean(dim=1)
    var_before = sample.var(dim=1, unbiased=False)
    print("Mean:\n", mean_before)
    print("Variance:\n", var_before)
    gn = nn.GroupNorm(num_groups=1, num_channels=3)
    output = gn(sample)
    print(output.detach())
    print("After Group Norm")
    print("Mean:\n", output.mean(dim=1).detach())
    print("Variance:\n", output.var(dim=1, unbiased=False).detach())
if __name__ == '__main__':
    demo()

import torch
import torch.nn as nn
def demo():
    torch.manual_seed(0)
    dummy_input = torch.randn(5, 3) * 10 + 5
    print(dummy_input)
    print("Before Batch Norm:")
    print("Mean:\n", dummy_input.mean(dim=0))
    print("Variance:\n", dummy_input.var(dim=0, unbiased=False))
    batch_norm = nn.BatchNorm1d(num_features=3)
    output = batch_norm(dummy_input)
    print("After Batch Norm:")
    print("Mean:\n", output.mean(dim=0).detach())
    print("Variance:\n", output.var(dim=0, unbiased=False).detach())
if __name__ == '__main__':
    demo()


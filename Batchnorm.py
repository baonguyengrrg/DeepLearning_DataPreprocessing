try:
    import torch
    import torch.nn as nn
except Exception:
    print("Error: PyTorch (torch) is not installed or failed to import.")
    print("To install PyTorch (CPU-only) run:\n  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("For GPU/CUDA builds, follow instructions at https://pytorch.org/get-started/locally/")
    raise SystemExit(1)
def demo():
    torch.manual_seed(0)

    dummy_input = torch.randn(3, 3) * 10 + 5

    print("Before normalizing")
    print("Average mean:\n", dummy_input.mean(dim=0))
    print("Variance:\n", dummy_input.var(dim=0, unbiased=False))

    batch_norm = nn.BatchNorm1d(num_features=3)
    output = batch_norm(dummy_input)

    print("After normalizing")
    print("New mean:\n", output.mean(dim=0).detach())
    print("New variance:\n", output.var(dim=0, unbiased=False).detach())
if __name__ == '__main__':
    demo()

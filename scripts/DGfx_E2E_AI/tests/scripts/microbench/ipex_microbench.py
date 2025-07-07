import intel_extension_for_pytorch as ipex
import torch
import torch.nn as nn


def tensor_creation():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    x = x.to("xpu")
    print("Tensor:\n", x)
    print("Shape:", x.shape)
    print("Data type:", x.dtype)
    print(f"Device: {x.device}")
    return x

def tensor_operations(x):
    y = x + 2
    print("Addition:\n", y)
    y_check = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32)
    assert y.cpu().detach().numpy().all() == y_check.numpy().all(), "Failed at addition step"

    z = x * y
    print("Multiplication:\n", z)
    z_check = torch.tensor([[3, 8], [15, 24]], dtype=torch.float32)
    assert z.cpu().detach().numpy().all() == z_check.numpy().all(), "Failed at multiplication step"

    w = torch.matmul(x, y)
    print("Matrix multiplication:\n", w)
    w_check = torch.tensor([[13, 16], [29, 36]], dtype=torch.float32)
    assert w.cpu().detach().numpy().all() == w_check.numpy().all(), "Failed at matmul step"
    return x

def simple_model_forward():
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3)

        def forward(self, x):
            x = self.fc1(x)
            x = self.conv1(x)
            return x

    model = SimpleNet()
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters in the model:", total_params)
    assert total_params == 503, "Failed at simple_model_forward"

def check_pytorch_installation():
    print(torch.__version__)
    print(ipex.__version__)

def check_ipex():
    [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]


if __name__ == "__main__":
    print("\n")
    print("=== PyTorch Validation Check ===")
    check_pytorch_installation()
    check_ipex()
    tensor = tensor_creation()
    tensor = tensor_operations(tensor)
    simple_model_forward()
    print("=== PyTorch Validation Check Complete ===")
    print("\n")
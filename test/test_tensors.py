import torch


def test_float_tensor(torchd):
    class Model(torch.nn.Module):
        def forward(self, a):
            return a

    torchd.start(Model())
    res = torchd.forward([3.14])
    assert res == [3.14]


def test_float_scalar(torchd):
    class Model(torch.nn.Module):
        def forward(self, a):
            return a

    torchd.start(Model())
    res = torchd.forward(3.14)
    assert res == 3.14


def test_no_input_output(torchd):
    class Model(torch.nn.Module):
        def forward(self):
            return None

    torchd.start(Model())
    res = torchd.forward()
    assert res == None

import torch

from minselfflow.flow import add_noise, euler_sample, flow_loss


def test_add_noise_endpoints_and_per_token():
    x0, x1 = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    tau = torch.tensor([[0.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.0, 0.25]])
    x = add_noise(x0, x1, tau)
    assert torch.allclose(x[0, 0], x0[0, 0])  # t=0 -> clean
    assert torch.allclose(x[0, 1], x1[0, 1])  # t=1 -> noise
    assert torch.allclose(x[0, 2], 0.5 * (x0[0, 2] + x1[0, 2]))
    assert torch.allclose(x[1, 3], 0.75 * x0[1, 3] + 0.25 * x1[1, 3])


def test_euler_sample_hits_point_mass():
    # If all data equals c, the exact velocity field is v(x, t) = (x - c) / t.
    c = torch.tensor([2.0, -1.0])

    def oracle(x, tau, y):
        return (x - c) / tau[..., None]

    out = euler_sample(oracle, torch.randn(8, 1, 2), torch.zeros(8), steps=10)
    assert torch.allclose(out, c.expand_as(out), atol=1e-5)


def test_flow_loss_zero_for_oracle_on_point_mass():
    c = torch.tensor([2.0, -1.0])
    x0 = c.expand(16, 1, 2)

    def oracle(x, tau, y):
        return (x - c) / tau[..., None]

    assert flow_loss(oracle, x0, torch.zeros(16)) < 1e-8

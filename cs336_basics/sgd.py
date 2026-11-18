
from math import sqrt

import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get state associated with p
                state = self.state[p]
                # get iteration number from the state
                t = state.get("t", 0)
                # get the gradient of loss with respect to p
                grad = p.grad.data
                # update weight tensor in-place
                p.data -= lr/sqrt(t+1) * grad
                # incr iteration number
                state["t"] = t + 1
        return loss

def train_loop(lr:float):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)

    for t in range(10):
        # reset the gradients for all learnable parameters
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"Step {t:2d}, loss:{loss.cpu().item():.6f}")
        # run backward pass, which compute gradients
        loss.backward()
        # run optimizer step
        opt.step()

# python -m cs336_basics.sgd
for lr in [1e1,1e2,1e3]:
    print(f"Train with lr:{lr}")
    train_loop(lr)
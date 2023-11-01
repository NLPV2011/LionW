import torch
from torch.optim.optimizer import Optimizer

class LionW(Optimizer):
    # A new optimizer that combines Lion and AdamW
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        # Initialize the optimizer with the given parameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LionW, self).__init__(params, defaults)

    def step(self, closure=None):
        # Perform a single optimization step
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Get the parameters and hyperparameters for each group
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                # Skip if the parameter has no gradient
                if p.grad is None:
                    continue

                # Get the gradient and apply weight decay
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Get the state of the parameter
                state = self.state[p]

                # Initialize the state with zeros
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_max'] = torch.zeros_like(p.data)

                # Get the state variables
                exp_avg, exp_avg_sq, exp_avg_sq_max = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_sq_max']
                state['step'] += 1

                # Update the exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                # Update the maximum of exp_avg_sq_hat
                torch.max(exp_avg_sq_max, exp_avg_sq_hat, out=exp_avg_sq_max)

                # Update the parameter with Lion rule
                denom = exp_avg_sq_max.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss

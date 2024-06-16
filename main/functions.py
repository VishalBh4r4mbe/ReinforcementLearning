from typing import List
import numpy as np
import torch

def n_step_bellman_target(
    r_t: torch.Tensor,
    done: torch.Tensor,
    q_t: torch.Tensor,
    gamma: float,
    n_steps: int,
) -> torch.Tensor:
    r"""Computes n-step Bellman targets.

    See section 2.3 of R2D2 paper (which does not mention the logic around end of
    episode).

    Args:
      rewards: This is r_t in the equations below. Should be non-discounted, non-summed,
        shape [T, B] tensor.
      done: This is done_t in the equations below. done_t should be true
        if the episode is done just after
        experimenting reward r_t, shape [T, B] tensor.
      q_t: This is Q_target(s_{t+1}, a*) (where a* is an action chosen by the caller),
        shape [T, B] tensor.
      gamma: Exponential RL discounting.
      n_steps: The number of steps to look ahead for computing the Bellman targets.

    Returns:
      y_t targets as <float32>[time, batch_size] tensor.
      When n_steps=1, this is just:

      $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$

      In the general case, this is:

      $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
        \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$

      where notdone_{t,i} is defined as:

      $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$

      The last n_step-1 targets cannot be computed with n_step returns, since we
      run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
      returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
      multiple times.
    """

    # We append n_steps - 1 times the last q_target. They are divided by gamma **
    # k to correct for the fact that they are at a 'fake' indices, and will
    # therefore end up being multiplied back by gamma ** k in the loop below.
    # We prepend 0s that will be discarded at the first iteration below.
    bellman_target = torch.concat(
        [torch.zeros_like(q_t[0:1]), q_t] + [q_t[-1:] / gamma**k for k in range(1, n_steps)], dim=0
    )
    # Pad with n_steps 0s. They will be used to compute the last n_steps-1
    # targets (having 0 values is important).
    done = torch.concat([done] + [torch.zeros_like(done[0:1])] * n_steps, dim=0)
    rewards = torch.concat([r_t] + [torch.zeros_like(r_t[0:1])] * n_steps, dim=0)
    # Iteratively build the n_steps targets. After the i-th iteration (1-based),
    # bellman_target is effectively the i-step returns.
    for _ in range(n_steps):
        rewards = rewards[:-1]
        done = done[:-1]
        bellman_target = rewards + gamma * (1.0 - done.float()) * bellman_target[1:]

    return bellman_target

def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)
  
def calculate_distributed_priorities_from_td_error(td_error: torch.Tensor, eta: float) -> np.ndarray:
    td_errors = torch.clone(td_error).detach()
    absolute_td_errors = torch.abs(td_errors)
    priorities = eta * torch.max(absolute_td_errors,dim=0)[0] + (1-eta) * torch.mean(absolute_td_errors,dim=0)
    priorities = torch.clamp(priorities, min = 0.00001, max = 1000)
    priorities = priorities.cpu().numpy()
    return priorities

def get_actors_exploration_rate(n: int) -> List[float]:
    assert 1 <= n
    return np.power(0.4, np.linspace(1.0, 8.0, num=n)).flatten().tolist()
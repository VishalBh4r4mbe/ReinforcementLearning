from argparse import Action
from typing import Mapping, Text
import numpy as np
import torch
from main.abstractClasses.agent import Agent
from main.abstractClasses.timestep import TimeStepPair

def apply_egreedy_policy(
    q_values: torch.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> Action:
    """Apply e-greedy policy."""
    action_dim = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = q_values.argmax(-1).cpu().item()
    return a_t


class EpsilonGreedyActor(Agent):
    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 exploration_epsilon: float,
                 random_state:np.random.RandomState,
                 name:str):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=self._device)
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state
    
    def step(self,timestep:TimeStepPair) -> Action:
        return self._select_action(timestep)
        
    @torch.no_grad()
    def _select_action(self,timestep:TimeStepPair) -> Action:
        state_t = torch.tensor(timestep.observation[None,...]).to(device=self._device,dtype=torch.float32)
        q_values = self._network(state_t).q_values
        return apply_egreedy_policy(q_values,self._exploration_epsilon,self._random_state)
    def reset(self) -> None:
        """reset"""
        return
    def statistics(self) -> Mapping[str, float]:
        return {}

class DRQNEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 exploration_epsilon: float,
                 random_state:np.random.RandomState,
                 name: str):
        super().__init__(network,device,exploration_epsilon,random_state,name)
        self._lstm_state=None
    @torch.no_grad()
    def _select_action(self,timestep:TimeStepPair) -> Action:
        if self._lstm_state is None:
            raise ValueError("Reset Agent")
        state_t = torch.as_tensor(timestep.observation[None,None,...]).to(device=self._device,dtype=torch.float32)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        network_output = self._network(state_t,hidden_state)
        q_values = network_output.q_values
        self._lstm_state = network_output.hidden_state
        return apply_egreedy_policy(q_values,self._exploration_epsilon,self._random_state)
    def reset(self):
        self._lstm_state = self._network.get_initial_hidden_state(1)
        
class R2D2EpsilonGreedyActor(EpsilonGreedyActor):

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        super().__init__(
            network=network,
            exploration_epsilon=exploration_epsilon,
            random_state=random_state,
            device=device,
            name='R2D2-greedy',
        )
        self._last_action = None
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: TimeStepPair) -> Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        network_output = self._network(
                state_t=s_t[None, ...],
                action_t_minus_1=a_tm1[None, ...],
                reward_t=r_t[None, ...],
                hidden_state=hidden_s,
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_state

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def reset(self) -> None:
        self._last_action = 0 
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
import torch


class EpisodicBonusModule:
    def __init__(
        self,
        embedding_network : torch.nn.Module,
        device : torch.device,
        size : int,
        num_neighbors : int,
        kernel_epsilon : float,
        cluster_distance : float,
        maximum_similarity : float,
        c_constant : float
    ) -> None:
        self._embedding_network = embedding_network.to(device=device)
        self._device = device
        self._memory = torch.zeros(size, self._embedding_network.embedding_size,device=self._device)
        self._mask = torch.zeros(size, dtype=torch.bool,device=self._device)
        self._size = size
        self._counter = 0
        self._cluster_distance_normalizer = PyTorchRunningMeanStd(shape=(1,),device=self._device)
        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = maximum_similarity
        self._c_constant = c_constant
    def _add_to_memory(self,embedding:torch.Tensor):
        index = self._counter %  self._size
        self._memory[index] = embedding
        self._mask[index] = True
        self._counter += 1
    @torch.no_grad()
    def get_bonus(self,state_t:torch.Tensor)->torch.Tensor:
        embedding = self._embedding_network(state_t).squeeze(0)
        
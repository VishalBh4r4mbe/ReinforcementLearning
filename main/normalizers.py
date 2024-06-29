
import logging
import numpy as np
import torch

from models.common import numpy_to_tensor


class PytorchRunningMeanStd:
    def __init__(self, shape, device):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32,device=self.device)
        self.variance = torch.ones(shape, dtype=torch.float32,device=self.device)
        self.count = 0
        self.deltas = []
        self.minimun_size = 10
    @torch.no_grad()
    def update(self,x):
        x = x.to(self.device)
        batch_mean = torch.mean(x,dim=0)
        batch_variance = torch.var(x,dim=0)
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta *n/self.count
        m_a = self.variance * (self.count - n)
        m_b = batch_variance * n
        M2 = m_a + m_b + torch.square(delta)*n
        self.variance = M2/self.count
    @torch.no_grad()
    def update_single(self,x):
        self.deltas.append(x)
        if len(self.deltas) >= self.minimun_size:
            batched_x = torch.concat(self.deltas,dim=0)
            self.update(batched_x)
            del self.deltas[:]
    @torch.no_grad()
    def normalize(self,x):
        return (x.to(self.device) - self.mean)/torch.sqrt(self.variance + 1e-8)
    
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape=shape,dtype=np.float32)
        self.variance = np.ones(shape,dtype=np.float32)
        self.count = 0
        self.deltas = []
        self.minimun_size = 10
        
    def update(self,x):
        batched_mean = np.mean(x,axis=0)
        batched_variance = np.mean(x,axis=0)
        n = x.shape[0]
        self.count += n
        delta = batched_mean - self.mean
        self.mean += delta *n/self.count
        m_a = self.variance * (self.count - n)
        m_b = batched_variance * n
        M2 = m_a + m_b + np.square(delta)*n
        self.variance = M2/self.count

    def update_single(self,x):
        self.deltas.append(x)
        if len(self.deltas) >= self.minimun_size:
            # logging.info(self.deltas)
            batched_x = np.stack(self.deltas,axis=0)
            self.update(batched_x)
            del self.deltas[:]
    def normalize(self,x):
        return (x - self.mean)/np.sqrt(self.variance + 1e-8)
        
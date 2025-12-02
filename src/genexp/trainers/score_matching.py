from genexp.models import DiffusionModel
from genexp.sampling import SDE
import torch

class ScoreMatchingTrainer(object):
    def __init__(self, model: DiffusionModel, sde: SDE):
        self.model = model
        self.sde = sde
    


    def train_step(batch):
        x0 = batch
        t = torch.rand(x.shape[0]).to(x0.device)
        eps = torch.randn(*x0.shape).to(x0.device)
        
        

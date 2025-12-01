import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)

        # add metrics here

    def forward(self, batch):
        output = self.renderer.render(batch)
        rgb_gt = batch['rgb']
        rgb_coarse = output['rgb_coarse']
        loss_coarse = torch.mean((rgb_coarse - rgb_gt) ** 2)
        loss = loss_coarse
        loss_stats = {
            'loss_coarse': loss_coarse,
        }
        
        if 'rgb_fine' in output:
            rgb_fine = output['rgb_fine']
            loss_fine = torch.mean((rgb_fine - rgb_gt) ** 2)
            loss = loss + loss_fine
            loss_stats['loss_fine'] = loss_fine
        
        loss_stats['loss'] = loss
        
        return output, loss, loss_stats

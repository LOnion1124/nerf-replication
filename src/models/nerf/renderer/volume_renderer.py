import numpy as np
import torch
from src.config import cfg

class Renderer:
    def __init__(self, net):
        self.net = net
        self.N_samples = cfg.task_arg.N_samples
        self.N_importance = cfg.task_arg.N_importance
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.perturb = cfg.task_arg.perturb
        self.raw_noise_std = cfg.task_arg.raw_noise_std
        self.lindisp = cfg.task_arg.lindisp
        
        self.near = 2.0
        self.far = 6.0

    def sample_pts_along_ray(self, rays_o, rays_d, near, far, N_samples, perturb=False, lindisp=False):
        N_rays = rays_o.shape[0]
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
        
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        
        z_vals = z_vals.expand([N_rays, N_samples])
        
        # Perturb sampling positions
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (N_rays, N_samples, 3)
        
        return pts, z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False):
        weights = weights + 1e-5  # Prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (N_rays, N_samples + 1)
        
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_importance, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=bins.device)
        
        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (N_rays, N_importance, 2)
        
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_exp = cdf.unsqueeze(-2).expand(matched_shape)
        bins_exp = bins.unsqueeze(-2).expand(matched_shape)
        cdf_g = torch.gather(cdf_exp, -1, inds_g)
        bins_g = torch.gather(bins_exp, -1, inds_g)
        
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0):
        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)  # (N_rays, N_samples)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        rgb = torch.sigmoid(raw[..., :3])  # (N_rays, N_samples, 3)
        
        # Add noise to density for regularization during training
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
        
        # Calculate alpha values
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # (N_rays, N_samples)
        
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1
        )[..., :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        
        # Apply white background if needed
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map, disp_map, acc_map, weights, depth_map

    def render(self, batch):
        rays = batch['rays']
        if rays.dim() != 2:
            rays = rays.reshape(-1, rays.shape[-1])
        
        # Split rays into origins and directions
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]  # (N_rays, 3)
        
        device = next(self.net.parameters()).device
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        
        # Normalize ray directions
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.flatten(viewdirs, 0, -2) if viewdirs.dim() > 2 else viewdirs
        
        # ========== Coarse Network ========== 
        pts_coarse, z_vals_coarse = self.sample_pts_along_ray(
            rays_o, rays_d, self.near, self.far, self.N_samples, 
            perturb=self.perturb > 0, lindisp=self.lindisp
        )

        raw_coarse = self.net.forward(pts_coarse, viewdirs, model='coarse')

        rgb_coarse, disp_coarse, acc_coarse, weights_coarse, depth_coarse = self.raw2outputs(
            raw_coarse, z_vals_coarse, rays_d, self.raw_noise_std
        )

        ret = {
            'rgb_coarse': rgb_coarse,
            'disp_coarse': disp_coarse,
            'acc_coarse': acc_coarse,
            'depth_coarse': depth_coarse,
        }

        # ========== Fine Network ========== 
        if self.N_importance > 0:
            z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            z_samples = self.sample_pdf(
                z_vals_mid, weights_coarse[..., 1:-1], self.N_importance, 
                det=(self.perturb == 0)
            )
            z_samples = z_samples.detach()

            z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
            pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]

            raw_fine = self.net.forward(pts_fine, viewdirs, model='fine')

            rgb_fine, disp_fine, acc_fine, weights_fine, depth_fine = self.raw2outputs(
                raw_fine, z_vals_fine, rays_d, self.raw_noise_std
            )

            ret['rgb_fine'] = rgb_fine
            ret['disp_fine'] = disp_fine
            ret['acc_fine'] = acc_fine
            ret['depth_fine'] = depth_fine

        return ret

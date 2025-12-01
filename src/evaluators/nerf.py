import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255),
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255),
        )
        img_pred = (img_pred * 255).astype(np.uint8)

        ssim = compare_ssim(img_pred, img_gt, win_size=51, full=True, channel_axis=2)
        return ssim

    def evaluate(self, output, batch):
        H = batch['meta']['H'].item()
        W = batch['meta']['W'].item()
        
        if 'rgb_fine' in output:
            rgb_pred = output['rgb_fine']
        else:
            rgb_pred = output['rgb_coarse']
        
        rgb_gt = batch['rgb']
        
        img_pred = rgb_pred.reshape(H, W, 3).detach().cpu().numpy()
        img_gt = rgb_gt.reshape(H, W, 3).detach().cpu().numpy()
        
        # Clip to valid range [0, 1]
        img_pred = np.clip(img_pred, 0, 1)
        img_gt = np.clip(img_gt, 0, 1)
        
        # Compute MSE
        mse = np.mean((img_pred - img_gt) ** 2)
        self.mse.append(mse)
        
        # Compute PSNR
        psnr = self.psnr_metric(img_pred, img_gt)
        self.psnr.append(psnr)
        
        # Compute SSIM and save images
        num_imgs = len(self.imgs)
        img_gt_uint8 = (img_gt * 255).astype(np.uint8)
        ssim_val, _ = self.ssim_metric(img_pred, img_gt_uint8, batch, num_imgs, num_imgs + 1)
        self.ssim.append(ssim_val)
        
        self.imgs.append(img_pred)
        
        return None

    def summarize(self):
        ret = {}
        
        if len(self.mse) > 0:
            ret['mse'] = np.mean(self.mse)
        
        if len(self.psnr) > 0:
            ret['psnr'] = np.mean(self.psnr)
            print(f"Average PSNR: {ret['psnr']:.4f}")
        
        if len(self.ssim) > 0:
            ret['ssim'] = np.mean(self.ssim)
            print(f"Average SSIM: {ret['ssim']:.4f}")
        
        # Save metrics to JSON
        result_dir = os.path.join(cfg.result_dir, "images")
        os.makedirs(cfg.result_dir, exist_ok=True)
        
        import json
        with open(os.path.join(cfg.result_dir, "metrics.json"), "w") as f:
            json.dump(ret, f, indent=2, default=float)
        
        print(f"Metrics saved to {cfg.result_dir}/metrics.json")
        print(f"Images saved to {result_dir}")
        
        # Reset for next evaluation
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []
        
        return ret

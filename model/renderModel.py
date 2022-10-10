import torch
import torch.nn as nn
from .depthModel import DepthModel
import torchvision.transforms as T
from .utils import plane_sweep_torch, stable_softmax
from .gan import Generator
from PIL import Image


class RenderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_model = DepthModel()
        self.generator = Generator()

    def forward(self, 
                src_images_BN3HW,
                src_cams_BN244,
                tgt_cam_B244,
                depths_BD):
        B, N, C, H, W = src_images_BN3HW.shape
        D = depths_BD.shape[1]

        # depth_model output
        src_weights_BNDhw, depth_probs_BD1hw = self.depth_model(src_images_BN3HW, src_cams_BN244, tgt_cam_B244, depths_BD)

        # src_weights processing
        src_weights_full_size_BNDHW = T.Resize((H, W))(src_weights_BNDhw.reshape(B * N, D, H // 2, W // 2)).reshape(B, N, D, H, W)
        src_weights_softmax_BNDHW = stable_softmax(src_weights_full_size_BNDHW, dim=1)

        # depth_probs processing
        depth_probs_full_size_BD1HW = T.Resize((H, W))(depth_probs_BD1hw.reshape(B, D, H // 2, W // 2)).reshape(B, D, 1, H, W)
        depth_probs_softmax_BD1HW = stable_softmax(depth_probs_full_size_BD1HW, dim=1)

        # warp images
        warped_imgs_BND3HW = []
        masks_BNDHW = []
        src_intrins_BN33 = src_cams_BN244[:, :, 0, :3, :3]
        tgt_intrins_B133 = tgt_cam_B244[:, 0:1, :3, :3]
        src_w2cs_BN44 = src_cams_BN244[:, :, 1, :4, :4]
        tgt_w2cs_B144 = tgt_cam_B244[:, 1:2, :4, :4]
        for i in range(B):
            src2tgt_poses_N44 = torch.matmul(src_w2cs_BN44[i], torch.inverse(tgt_w2cs_B144[i]))
            warped_imgs_DN3HW, masks_DNHW = plane_sweep_torch(src_images_BN3HW[i], depths_BD[i], src2tgt_poses_N44, src_intrins_BN33[i], tgt_intrins_B133[i], H, W)
            warped_imgs_BND3HW.append(warped_imgs_DN3HW)
            masks_BNDHW.append(masks_DNHW)
        warped_imgs_BND3HW = torch.stack(warped_imgs_BND3HW, dim=0).transpose(1, 2)
        masks_BNDHW = torch.stack(masks_BNDHW, dim=0).transpose(1, 2)

        # mask src_weights
        src_weights_softmax_BNDHW = src_weights_softmax_BNDHW * masks_BNDHW
        src_weights_softmax_BNDHW = src_weights_softmax_BNDHW / (src_weights_softmax_BNDHW.sum(dim=1, keepdim=True) + 1e-8)

        # compute aggregated images
        weighted_warped_imgs_BD3HW = (warped_imgs_BND3HW * src_weights_softmax_BNDHW.unsqueeze(3)).sum(dim=1)
        aggregated_imgs_B3HW = (weighted_warped_imgs_BD3HW * depth_probs_softmax_BD1HW).sum(dim=1)

        warped_imgs_BN3HW = (warped_imgs_BND3HW * depth_probs_softmax_BD1HW.unsqueeze(1)).sum(dim=2)

        # generator render final image
        output_imgs_BN3HW = []
        confidences_BN3HW = []
        for i in range(N):
            output_img_b3HW, confidence_b1HW = self.generator(aggregated_imgs_B3HW, warped_imgs_BN3HW[:, i])
            output_imgs_BN3HW.append(output_img_b3HW)
            confidences_BN3HW.append(confidence_b1HW)
        output_imgs_BN3HW = torch.stack(output_imgs_BN3HW, dim=1)
        confidences_BN3HW = torch.stack(confidences_BN3HW, dim=1)

        confidences_norm_BN3HW = confidences_BN3HW / (confidences_BN3HW.sum(dim=1, keepdim=True) + 1e-8)
        final_out_img_B3HW = (output_imgs_BN3HW * confidences_norm_BN3HW).sum(dim=1)

        return aggregated_imgs_B3HW, final_out_img_B3HW, warped_imgs_BN3HW

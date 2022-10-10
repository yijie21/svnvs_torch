import torch
import torch.nn as nn
import torchvision.transforms as T
from .featureExtractor import FeatureExtractor2D
from .utils import plane_sweep_torch
from .lstm import DepthConvLSTM


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        self.feature_extractor = FeatureExtractor2D()
        self.depth_model = DepthConvLSTM()

    def forward(self, 
                src_images_BN3HW,
                src_cams_BN244,
                tgt_cam_B244,
                depths_BD):
        B, N, C, H, W = src_images_BN3HW.shape
        D = depths_BD.shape[1]

        # operates on half image size for memory efficiency
        src_images_BN3hw = T.Resize((H // 2, W // 2))(src_images_BN3HW.reshape(B * N, C, H, W)).reshape(B, N, C, H // 2, W // 2)

        # extract src_view 2D features
        src_features_BNChw = self.feature_extractor(src_images_BN3hw.reshape(B * N, C, H // 2, W // 2)).reshape(B, N, -1, H // 2, W // 2)
        _, _, C, h, w = src_features_BNChw.shape

        # warp src_feature to tgt_view
        src_intrins_BN33 = src_cams_BN244[:, :, 0, :3, :3]
        tgt_intrins_B133 = tgt_cam_B244[:, 0:1, :3, :3]
        src_w2cs_BN44 = src_cams_BN244[:, :, 1, :4, :4]
        tgt_w2cs_B144 = tgt_cam_B244[:, 1:2, :4, :4]
        warped_features_BNChwD = []
        for i in range(B):
            src2tgt_poses_N44 = torch.matmul(src_w2cs_BN44[i], torch.inverse(tgt_w2cs_B144[i]))
            warped_feature_DNChw, _ = plane_sweep_torch(src_features_BNChw[i], depths_BD[i], src2tgt_poses_N44, src_intrins_BN33[i], tgt_intrins_B133[i], h, w)
            warped_features_BNChwD.append(warped_feature_DNChw)
        warped_features_BNChwD = torch.stack(warped_features_BNChwD, dim=0).permute(0, 2, 3, 4, 5, 1)

        # compute cost volume
        view_cost_BN1hwD = []
        for i in range(B):
            cost_NNhwD = torch.einsum('nchwd, cmhwd -> nmhwd', warped_features_BNChwD[i], warped_features_BNChwD[i].transpose(0, 1))
            view_cost_NhwD = torch.mean(cost_NNhwD, dim=0)
            view_cost_BN1hwD.append(view_cost_NhwD)
        view_cost_BN1hwD = torch.stack(view_cost_BN1hwD, dim=0)[:, :, None, ...]
        view_cost_mean_BN1hwD = torch.mean(view_cost_BN1hwD, dim=1, keepdim=True).expand(-1, N, -1, -1, -1, -1)
        view_cost_BNChwD = torch.cat([warped_features_BNChwD, view_cost_BN1hwD, view_cost_mean_BN1hwD], dim=2)

        # rnn compute src_weights and depth_probs
        view_cost_BND_C_h_w = view_cost_BNChwD.permute(0, 1, 5, 2, 3, 4).reshape(-1, C + 2, h, w)
        src_weight_BNDHW, depth_probs_BD1HW = self.depth_model(view_cost_BND_C_h_w, N, D)

        return src_weight_BNDHW, depth_probs_BD1HW

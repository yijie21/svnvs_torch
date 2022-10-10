from torch.utils.data import Dataset
import os
import numpy as np
from .data_io import gen_tnt_list
from PIL import Image
from .utils import inv_depths, scale_img_cam


class NVSDataset(Dataset):
    def __init__(self,
                 data_folder,
                 mode,
                 n_views,
                 n_depths,
                 max_h=256,
                 max_w=448,
                 depth_min=0.5,
                 depth_max=100):
        super().__init__()

        self.sample_list = gen_tnt_list(data_folder, mode)
        self.n_views = n_views
        self.n_depths = n_depths
        self.max_h = max_h
        self.max_w = max_w
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.n_samples = len(self.sample_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # index to a sample
        tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, ts = self.sample_list[index]

        index_key = list(tgt_img_dict.keys())[0]
        index_key_int = int(index_key)

        tgt_count = np.load(tgt_count_dict[index_key])
        Ks_N33 = np.load(Ks)
        Rs_N33 = np.load(Rs)
        ts_N3 = np.load(ts)

        # load tgt image and cam
        tgt_img_HW3 = Image.open(tgt_img_dict[index_key])
        tgt_cam_244 = np.zeros((2, 4, 4))
        tgt_cam_244[0, :3, :3] = Ks_N33[index_key_int]
        tgt_cam_244[1, :3, :3] = Rs_N33[index_key_int]
        tgt_cam_244[1, :3, 3] = ts_N3[index_key_int]
        tgt_cam_244[1, 3, 3] = 1
        tgt_cam_244 = tgt_cam_244.astype(np.float32)

        tgt_img_HW3, tgt_cam_244 = scale_img_cam(tgt_img_HW3, tgt_cam_244, self.max_h, self.max_w)
        # TODO: normalize to [-1, 1]?
        # TODO: 这个scale的原理是什么
        tgt_img_3HW = np.array(tgt_img_HW3).transpose(2, 0, 1) / 255.0 * 2 - 1
        tgt_img_3HW = tgt_img_3HW.astype(np.float32)

        # load src images and cams
        count = np.argsort(tgt_count)[::-1][:self.n_views]
        count = count[np.random.permutation(self.n_views)]

        src_imgs_N3HW = []
        src_cams_N244 = []
        for i in range(self.n_views):
            src_index = count[i]

            if src_index == index_key_int:
                continue
            
            src_index_key = '%08d' % src_index

            src_K_33 = Ks_N33[src_index]
            src_R_33 = Rs_N33[src_index]
            src_t_3 = ts_N3[src_index]

            src_cam_244 = np.zeros((2, 4, 4))
            src_cam_244[0, :3, :3] = src_K_33
            src_cam_244[1, :3, :3] = src_R_33
            src_cam_244[1, :3, 3] = src_t_3
            src_cam_244[1, 3, 3] = 1
            src_cam_244 = src_cam_244.astype(np.float32)

            src_img_HW3 = Image.open(src_img_dict[src_index_key])
            src_img_HW3, src_cam_244 = scale_img_cam(src_img_HW3, src_cam_244, self.max_h, self.max_w)
            src_img_3HW = np.array(src_img_HW3).transpose(2, 0, 1) / 255.0 * 2 - 1
            src_img_3HW = src_img_3HW.astype(np.float32)

            src_imgs_N3HW.append(src_img_3HW)
            src_cams_N244.append(src_cam_244)
        
        src_imgs_N3HW = np.stack(src_imgs_N3HW, axis=0)
        src_cams_N244 = np.stack(src_cams_N244, axis=0)

        depths_D = inv_depths(self.depth_min, self.depth_max, self.n_depths)

        sample = {
            'src_imgs': src_imgs_N3HW,
            'src_cams': src_cams_N244,
            'tgt_img': tgt_img_3HW,
            'tgt_cam': tgt_cam_244,
            'depths': depths_D,
        }

        return sample



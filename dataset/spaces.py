import random
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from .spaces_dataset.code.utils import ReadScene
from PIL import Image
from glob import glob
import torch.nn.functional as F
from .utils import inv_depths


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

        self.data_folder = data_folder
        self.mode = mode
        self.n_views = n_views
        self.n_depths = n_depths
        self.max_h = max_h
        self.max_w = max_w
        self.depth_min = depth_min
        self.depth_max = depth_max

        # load metadata
        scenes_root_folder = os.path.join(data_folder, 'data/800')
        scenes = [ReadScene(path) for path in sorted(glob(scenes_root_folder + '/*'))]

        if self.mode == 'training':
            scenes = scenes[10:]
        # TODO: else后面是什么条件
        else:
            scenes = scenes[:10]
        
        self.scenes = scenes

        # create the table with scene id and camera id
        # i = scene index, j = camera index in a scene
        self.scene_cam_table = []
        n_scenes = len(self.scenes)
        for i in range(n_scenes):
            for j in range(len(self.scenes[i])):
                self.scene_cam_table.append((i, j))
        
        self.depths = inv_depths(self.depth_min, self.depth_max, self.n_depths)

    def __len__(self):
        return len(self.scene_cam_table)

    def __getitem__(self, index):
        idx_scene, idx_cam = self.scene_cam_table[index]
        if self.mode == 'training':
            indices_choices = np.random.choice(len(self.scenes[0][0]), 5, replace=False)
            self.indices_src = indices_choices[:-1]
            self.indices_tgt = indices_choices[-1:]
        else:
            pass

        indices_src_full = [(idx_scene, idx_cam, idx_img) for idx_img in self.indices_src]
        indices_tgt_full = [(idx_scene, idx_cam, idx_img) for idx_img in self.indices_tgt]

        imgs_info_src = [self.load_image_info(idx) for idx in indices_src_full]
        imgs_info_tgt = [self.load_image_info(idx) for idx in indices_tgt_full]

        sample = {
            'src_imgs': np.stack([info['image'] for info in imgs_info_src], axis=0),
            'src_cams': np.stack([info['cam'] for info in imgs_info_src], axis=0),
            'tgt_img': imgs_info_tgt[0]['image'],
            'tgt_cam': imgs_info_tgt[0]['cam'],
            'depths': self.depths
        }

        return sample


    def load_image_info(self, idx):
        '''
        :param idx: (scene_id, camera_id, image_id)
        '''
        data = self.scenes[idx[0]][idx[1]][idx[2]]
        intrin = data.camera.intrinsics
        w2c = data.camera.c_f_w
        image_path = data.image_path
        image = Image.open(image_path).convert('RGB')
        image, intrin = resize_totensor_intrinsics(image, intrin, self.max_w, self.max_h)
        cam = np.zeros((2, 4, 4))
        cam[0, :3, :3] = intrin
        cam[1, :4, :4] = w2c
        return {
            'image': image,
            'cam': cam
        }


def scale_intrinsics(intrin, sy, sx):
    '''
    :param intrin: tensor (3, 3)
    :param sy: scale factor for y
    :param sx: scale factor for x
    :return:
        intrin: scaled intrinsics
    '''
    intrin[0, 0] *= sx
    intrin[1, 1] *= sy
    intrin[0, 2] *= sx
    intrin[1, 2] *= sy
    return intrin


def resize_totensor_intrinsics(img, intrin, img_tgt_w, img_tgt_h):
    '''
    :param img: PIL image format
    :param intrin: intrinsics
    :param img_tgt_w: target image width
    :param img_tgt_h: target image height
    :return:
        img: tensor (3, H, W) in range [0, 1]
        intrin: tensor (3, 3)
    '''
    intrin_s = scale_intrinsics(intrin, img_tgt_h / img.height, img_tgt_w / img.width)
    img_s = img.resize((img_tgt_w, img_tgt_h))
    img_t = np.array(img_s) / 255
    return img_t.transpose(2, 0, 1), intrin_s
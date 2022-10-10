import os
import torch


def gen_tnt_list(data_folder, mode='training'):
    validation = {'training/Truck':
                      [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196],
                  'intermediate/M60':
                      [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
                  'intermediate/Playground':
                      [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252],
                  'intermediate/Train':
                      [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]}

    sample_list = []
    if mode == 'training':
        sub_dirs = sorted(os.listdir(data_folder))      # ['training', 'intermediate', 'advanced']
        for sub_dir in sub_dirs:
            sub_dir_full = os.path.join(data_folder, sub_dir)
            scenes = sorted(os.listdir(sub_dir_full))
            for scene in scenes:
                scene_key = os.path.join(sub_dir, scene).replace('\\', '/')
                if scene_key not in validation.keys():
                    scene_file_dir = os.path.join(sub_dir_full, scene, 'dense', 'ibr3d_pw_0.25')

                    Ks = os.path.join(scene_file_dir, 'Ks.npy')
                    Rs = os.path.join(scene_file_dir, 'Rs.npy')
                    ts = os.path.join(scene_file_dir, 'ts.npy')

                    count_dict = {}
                    im_dict = {}

                    scene_files = os.listdir(scene_file_dir)
                    for scene_file in scene_files:
                        if scene_file not in ['Ks.npy', 'Rs.npy', 'ts.npy', 'counts.npy']:
                            name, index = (scene_file.split('.')[0]).split('_')
                            if name == 'count':
                                count_dict[index] = os.path.join(scene_file_dir, scene_file)
                            elif name == 'im':
                                im_dict[index] = os.path.join(scene_file_dir, scene_file)
                    
                    assert im_dict.keys() == count_dict.keys()

                    indices = im_dict.keys()

                    for index in indices:
                        tgt_img_dict = {index: im_dict[index]}
                        tgt_count_dict = {index: count_dict[index]}
                        src_img_dict = {key: value for key, value in im_dict.items() if key != index}
                        sample_list.append([tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, ts])
    else:
        if mode == 'Truck':
            data_name = 'training/Truck'
        elif mode == 'M60':
            data_name = 'intermediate/M60'
        elif mode == 'Playground':
            data_name = 'intermediate/Playground'
        elif mode == 'Train':
            data_name = 'intermediate/Train'
        scene_file_dir = os.path.join(data_folder, data_name, 'dense', 'ibr3d_pw_0.25')

        Ks = os.path.join(scene_file_dir, 'Ks.npy')
        Rs = os.path.join(scene_file_dir, 'Rs.npy')
        ts = os.path.join(scene_file_dir, 'ts.npy')

        count_dict = {}
        im_dict = {}

        scene_files = os.listdir(scene_file_dir)
        for scene_file in scene_files:
            if scene_file not in ['Ks.npy', 'Rs.npy', 'ts.npy', 'counts.npy']:
                name, index = (scene_file.split('.')[0]).split('_')
            if name == 'count':
                count_dict[index] = os.path.join(scene_file_dir, scene_file)
            elif name == 'im':
                im_dict[index] = os.path.join(scene_file_dir, scene_file)

        for item in validation[data_name]:
            index = '%08d' % item
            tgt_img_dict = {index: im_dict[index]}
            tgt_count_dict = {index: count_dict[index]}
            src_img_dict = {key: value for key, value in im_dict.items() if key != index}
            sample_list.append([tgt_img_dict, tgt_count_dict, src_img_dict, Ks, Rs, ts])

    return sample_list

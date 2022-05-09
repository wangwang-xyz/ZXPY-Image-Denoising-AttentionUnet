import os.path
import struct
import numpy as np
import rawpy
import torch

from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, rootdir, dataname, gtname, data_idx, black_level=1024, white_level=16383):
        self.data_idx = data_idx
        self.data_path = os.path.join(rootdir, dataname)
        self.gt_path = os.path.join(rootdir, gtname)
        self.black_level = black_level
        self.white_level = white_level

    def __getitem__(self, idx):
        #print(idx)
        raw_data, height, width = read_image(os.path.join(self.data_path, r"{}_noise.dng".format(self.data_idx[idx])))
        raw_data_normal = normalization(raw_data, self.black_level, self.white_level)
        raw_data_normal = torch.from_numpy(np.transpose(
            raw_data_normal.reshape(height // 2, width // 2, 4), (2, 0, 1))).float()

        gt, height, width = read_image(os.path.join(self.gt_path, r"{}_gt.dng".format(self.data_idx[idx])))
        gt = normalization(gt, self.black_level, self.white_level)
        gt = torch.from_numpy(np.transpose(
            gt.reshape(height // 2, width // 2, 4), (2, 0, 1))).float()
        return raw_data_normal, gt

    def __len__(self):
        return len(self.data_idx)

def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width

def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data

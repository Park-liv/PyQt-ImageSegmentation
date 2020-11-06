import voxel
import numpy as np
import natsort
import os

# Voxel = voxel.PyVoxel()

# img_dir = 'LV ground truth/20/RAW'
# fname = os.listdir(img_dir)
# Voxel.ReadFromRaw(img_dir + '/' + fname[0])
# img_20 = Voxel.m_Voxel.copy()

# img_dir = 'LV ground truth/21/RAW'
# fname = os.listdir(img_dir)
# Voxel.ReadFromRaw(img_dir + '/' + fname[0])
# img_21 = Voxel.m_Voxel.copy()

# img_dir = 'LV ground truth/22/RAW'
# fname = os.listdir(img_dir)
# Voxel.ReadFromRaw(img_dir + '/' + fname[0])
# img_22 = Voxel.m_Voxel.copy()

# img_dir = 'LV ground truth/23/RAW'
# fname = os.listdir(img_dir)
# Voxel.ReadFromRaw(img_dir + '/' + fname[0])
# img_23 = Voxel.m_Voxel.copy()

# img = np.concatenate((img_20, img_21, img_22, img_23),axis=0)
# np.save('./20_23_img.npy', img)
# print(img.shape)

f = np.load('masks.npy')
print(f.dtype, f.shape)
# s = np.load('masks_onehot_2.npy')
# print('done')
# masks = np.concatenate((np.load('masks_onehot_1.npy'), np.load('masks_onehot_2.npy')), axis=0)
# print(masks.shape)
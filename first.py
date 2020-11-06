import voxel
import numpy as np
import natsort
import os

Voxel = voxel.PyVoxel()

Voxel.ReadFromBin('label/20_LV_ALL_1.bin')
lv_all = Voxel.m_Voxel.copy()
Voxel.ReadFromBin('label/20_LV_Only_1.bin')
lv_only = Voxel.m_Voxel.copy()
lv_20 = lv_all+lv_only
np.save('./21_masks.npy', lv_20)
print('done')

Voxel = voxel.PyVoxel()
Voxel.ReadFromBin('label/21_LV_ALL_1.bin')
lv_all = Voxel.m_Voxel.copy()
Voxel.ReadFromBin('label/21_LV_Only_1.bin')
lv_only = Voxel.m_Voxel.copy()
lv_21 = lv_all+lv_only
np.save('./21_masks.npy', lv_21)
print('done')

Voxel.ReadFromBin('label/22_LV_ALL_1.bin')
lv_all = Voxel.m_Voxel.copy()
Voxel.ReadFromBin('label/22_LV_Only_1.bin')
lv_only = Voxel.m_Voxel.copy()
lv_22 = lv_all+lv_only
np.save('./22_masks.npy', lv_22)
print('done')

Voxel.ReadFromBin('label/23_LV_ALL_1.bin')
lv_all = Voxel.m_Voxel.copy()
Voxel.ReadFromBin('label/23_LV_Only_1.bin')
lv_only = Voxel.m_Voxel.copy()
lv_23 = lv_all+lv_only
np.save('./23_masks.npy', lv_23)
print('done')

lv = np.concatenate((lv_20, lv_21, lv_22, lv_23),axis=0)
print(lv.shape)
np.save('./20_23_masks.npy', lv)
print('done')
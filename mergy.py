import voxel
import numpy as np
import natsort
import os
from tensorflow.keras.utils import to_categorical

Voxel = voxel.PyVoxel()

lv_list = natsort.natsorted(os.listdir('label/LV'))
only_list = natsort.natsorted(os.listdir('label/LV_only'))
# del lv_list[4]
# del only_list[4]
print(lv_list)
print(only_list)

mask = []
for i in range(len(lv_list[:-6])):
    Voxel.ReadFromBin('label/LV/'+lv_list[i])
    lv_all = Voxel.m_Voxel.copy()
    Voxel.ReadFromBin('label/LV_only/'+only_list[i])
    lv_only = Voxel.m_Voxel.copy()
    merge = lv_all + lv_only
    # print(np.unique(merge))
    # merge = to_categorical(merge)
    print(merge.shape)
    mask.append(merge)


print(len(mask))

mask_merge = np.concatenate(mask, axis=0)
mask_merge = to_categorical(mask_merge)
np.save('./masks_2_14.npy', mask_merge)
print(mask_merge.shape, 'done')

img = []
Voxel.ReadFromRaw('LV ground truth/02/RAW/SYO_7385371.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/12/RAW/19.KKO.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/13/RAW/20.CDS.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/14/RAW/YCS_2408322.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/19/RAW/KKS_7856792.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/20/RAW/POI_3219818.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/21/RAW/KKS_7831491.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/22/RAW/KKH_7826160.raw')
img.append(Voxel.m_Voxel.copy())
Voxel.ReadFromRaw('LV ground truth/23/RAW/YYC_7800157.raw')
img.append(Voxel.m_Voxel.copy())
print(len(img))

img_merge = np.concatenate(img[:-6], axis=0)
np.save('./imgs_2_14.npy', img_merge)
print(img_merge.shape)
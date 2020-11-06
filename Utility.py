import numpy as np
import SimpleITK as itk
import numpy as np
import natsort
import sys, time, os
import time
import pydicom

def load_filenames(path):
    filenames = os.listdir(path)
    return natsort.natsorted(filenames)      

def extention_validation(files):
    temp = []
    for file in files:
        if file.endswith('.dcm'):
            temp.append(file)
        elif file.endswith('.IMA'):
            temp.append(file)
        else:
            pass
    
    return temp

def ordering_check(origin_dir, files):
    temp = []
    for file in files:
        ds = pydicom.dcmread(os.path.join(origin_dir, file))
        temp.append(ds.InstanceNumber)
    
    filesorted = [x for _,x in sorted(zip(temp,files))]
    
    return filesorted


def search_folders(path):
    filenames = os.listdir(path)
    filenames = natsort.natsorted(filenames)
    temp = []
    for file in filenames:
        if file[-4:] == "fake":
            pass
        else:
            temp.append(file)

    return temp

def AdjustPixelRange(images, level, width):
    Lower = level - (width/2.0)
    Upper = level + (width/2.0)
 
    range_ratio = (Upper - Lower) / 256.0

    for i in range(images.shape[0]):
        img_adjusted = (images[i] - Lower)/range_ratio
        images[i] = img_adjusted.clip(0, 255)

    return images

def load_dicomseries(filenames, folder_path):
    Images = []
    for i in range(len(filenames)):
        ImgData = itk.ReadImage(os.path.join(folder_path, filenames[i]))
        ImgArray = itk.GetArrayFromImage(ImgData)   
        source = np.asarray(ImgArray, dtype=np.float32)
        Images.append(source)

    temp = np.array(Images)
    return np.squeeze(temp)

def write_dicom(origin_dir, dir_path, images, fakeimage):
    fakeimage = np.asarray(fakeimage, dtype='int16')
    ds = pydicom.dcmread(os.path.join(origin_dir, images[0]))
    
    if not ds.Manufacturer == 'TOSHIBA':
        fakeimage = np.add(fakeimage, 1024)

    for i in range(len(images)):
        ds = pydicom.dcmread(os.path.join(origin_dir, images[i]))
        ds.PixelData = fakeimage[i].tostring()
        saving_path = dir_path + "/fake_{}".format(i)
        ds.save_as(saving_path +".dcm")


def ct_tanh_norm(X, max_=2048, min_=-1024):
    
    X = np.clip(X, -1024, 2048)

    max_value = np.max(X, axis=(1, 2))
    min_value = np.full(max_value.shape, -1024)
    # min_value = np.min(X, axis=(1, 2, 3))
    for i in range(X.shape[0]):
        X[i] = ((X[i] - min_value[i]) / (max_value[i] - min_value[i]))*2 - 1.0

    return X, min_value, max_value
     
# def reverse_ct_tanh_norm(X, min_value, max_value):

#     for i in range(X.shape[0]):
#         X = (((X + 1.0)*0.5) * (max_value - min_value)) + min_value
        
#     X = np.clip(X, -1024, 2048)

    
#     return X

def reverse_ct_tanh_norm(X, min_value, max_value):
    
    for i in range(X.shape[0]):
        X[i] = (((X[i] + 1.0)*0.5) * (max_value[i] - min_value[i])) + min_value[i]
        
    X = np.clip(X, -1024, 2048)

    return X

def HU_intensity_norm(X, max_=2048, min_=-1024):
    X = np.clip(X, min_, max_)
    
    img = X.astype(np.float32)
    img = (img - min_) / (max_ - min_)

    return img


def revers_HU_intensity_norm(X, max_=2048, min_=-1024):
    img = X.astype(np.float32)
    img = (img * (max_ - min_)) + min_
    
    img = np.clip(img, min_, max_)

    return img
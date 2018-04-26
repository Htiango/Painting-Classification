from PIL import Image
import os
import numpy as np
from config import *
import torch
import torchvision.models as models
from torch.autograd import Variable
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_fnames():
    im_paths = []
    for sub_dir in os.listdir(dname):
        path = dname+sub_dir + '/'
        print(path)
        try:
            print(len(os.listdir(path)))
            for fname in os.listdir(path):
                if fname.endswith(".jpg") or fname.endswith(".JPG") or fname.endswith("Jpg"):
                    tmp = path + fname
                    im_paths.append(tmp)
                else:
                	print(fname)
                    # print(tmp)
        except:
            print("Exception!")
            continue
    return np.array(im_paths)

def crop_and_scale_image(im):
    """ Crops and scales a given image. 
        Args: 
            im (PIL Image) : image object to be cropped and scaled
        Returns: 
            (PIL Image) : cropped and scaled image object
    """
    x, y = im.size
#     print(im.size)
    if x > y:
        gap = x - y
        start = int(gap / 2)
#         print(start)
        im_cropped = im.crop((start, 0, start + y, y))
    else:
        gap = y - x
        start = int(gap / 2)
#         print(start)
#         print(y - start - x)
        im_cropped = im.crop((0, start, x, start + x))
    im_cropped = im_cropped.resize((img_size_cnn, img_size_cnn), Image.ANTIALIAS)
    return im_cropped

def fname_to_vgg_input(fname):
    """ Creates the input for a VGG network from the filename 
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (numpy ndarray) : the array to be passed into the 
                              VGG network as a single example
    """
    im = Image.open(fname)
    im = crop_and_scale_image(im)
    im = im.convert('RGB')
    # im.save("thumbnail.JPEG")

    res = np.array(im)

#     print(np.array(im).shape)
#     res = np.rollaxis(np.array(im), 2) # change to bgr

    return res / 255

def fname_to_type(fname):
    """ Extracts the painting type from a filename
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (string) : the type extracted from the filename
    """
    painting_type = os.path.split(os.path.split(fname)[0])[-1]
    return painting_type

def fnames_to_labels(fnames, mode="style"):
    """ Given a list of filenames, generate western or eastern labels
        Args: 
            fnames (list) : A list of filenames
        Returns: 
            (numpy ndarray) : a 1D array of numerical labels
    """
    label_all = [fname_to_type(fname) for fname in fnames]
    dic = painting_label_dic

    labels = np.array([dic[label] 
                       for label in label_all])
    return labels

def get_XY(fnames):
    try:
        sample = fname_to_vgg_input(fnames[0])
        size = sample.shape
        sample_num = len(fnames)
        print("Total number of input are: " + str(sample_num))
    except:
        print("Exception!")
        return None
    X = np.zeros((sample_num, size[0], size[1], size[2]))
    Y = fnames_to_labels(fnames)
    for i, fname in enumerate(fnames):
        if i%10 == 0:
            print("processing: " + str(i))
        X[i] = fname_to_vgg_input(fname)
    return X, Y

def main():
    im_paths = get_fnames()
    print("file number: " + str(len(im_paths)))
    P = np.random.permutation(len(im_paths))
    split = int(len(im_paths) * training_ratio)

    fnames_tr = im_paths[P[:split]]
    np.savetxt(tr_path_cnn, fnames_tr, fmt='%s')
    print("training set size: " + str(len(fnames_tr)))

    fnames_te = im_paths[P[split:]]
    np.savetxt(te_path_cnn, fnames_te, fmt='%s')
    print("testing set size: " + str(len(fnames_te)))

    X_tr, Y_tr = get_XY(fnames_tr)
    X_tr_flatten = np.reshape(X_tr, (X_tr.shape[0], -1))
    print(X_tr_flatten.shape)
    np.savetxt(X_tr_path_cnn, X_tr_flatten)
    print("Saving X_tr flatten features")
    np.savetxt(Y_tr_path_cnn, Y_tr, fmt='%i')
    print("Saving Y_tr features")
    print()

    X_te, Y_te = get_XY(fnames_te)
    X_te_flatten = np.reshape(X_te, (X_te.shape[0], -1))
    np.savetxt(X_te_path_cnn, X_te_flatten)
    print("Saving X_te flatten features")
    np.savetxt(Y_te_path_cnn, Y_te, fmt='%i')
    print("Saving Y_te features")
    print()

if __name__ == "__main__":
    main()
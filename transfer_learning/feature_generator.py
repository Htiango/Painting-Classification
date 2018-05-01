"""
This script:
1. Canonicalizing Images
2. Generate the vgg_16 features using pytorch. 
3. Use pre-train model to accelerate. 
"""

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
    im_cropped = im_cropped.resize((img_size, img_size), Image.ANTIALIAS)
    return im_cropped


def fname_to_type(fname):
    """ Extracts the painting type from a filename
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (string) : the type extracted from the filename
    """
    painting_type = os.path.split(os.path.split(fname)[0])[-1]
    return painting_type
    
    
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

#     print(np.array(im).shape)
    res = np.rollaxis(np.array(im), 2) # change to bgr
    return res

def VGG_16():
    """ Loads a pretrained VGG network. 
        Returns: 
            (pytorch VGG model) : the VGG-16 model
    """
    vgg = models.vgg16(pretrained=True)
    return vgg



def fnames_to_labels(fnames, mode="stype"):
    """ Given a list of filenames, generate western or eastern labels
        Args: 
            fnames (list) : A list of filenames
        Returns: 
            (numpy ndarray) : a 1D array of numerical labels
    """
    label_all = [fname_to_type(fname) for fname in fnames]
    
    if mode=="binary":
        dic = painting_type_dic
    else:
        dic = painting_label_dic

    labels = np.array([dic[label] 
                       for label in label_all])
    return labels

def fnames_to_features(fnames, vgg, mode="style"):
    """ Given a list of filenames and a VGG16 model, generate the corresponding array of VGG features
        Args: 
            fnames (list) : A list of filenames
            vgg (pytorch model) : a pretrained VGG16 model
        Returns: 
            X (np array) : a (m x 4608)-dimensional Variable of features generated from the VGG model,
                                 where m is the number of filenames
            Y (np array) : labels
    """
    try:
        sample = fname_to_vgg_input(fnames[0])
        size = sample.shape
        sample_num = len(fnames)
        batch_num = sample_num // batch_size
        print("Total number of input are: " + str(sample_num))
        print("Total batch number are: " + str(batch_num))
        print("Total number of actual input are: " + str(batch_num * batch_size))
    except:
        print("Exception!")
        return None

    feature_ls = []
    for batch_idx in range(sample_num // batch_size):
        vgg_input = np.zeros((batch_size, size[0], size[1], size[2]))
        start = batch_idx * batch_size
        for i in range(batch_size):
            fname = fnames[i + start]
            vgg_input[i] = fname_to_vgg_input(fname)
        vgg_input = Variable(torch.Tensor(vgg_input))
        feature = vgg.features(vgg_input).view(batch_size, -1).data.numpy()
        feature_ls.extend(feature)
        print("[" + str(batch_idx) + "]: finish partial painting")
    
    X = np.concatenate([[feat] for feat in feature_ls])
    print(X.shape)
    Y = fnames_to_labels(fnames[: batch_num * batch_size], mode)
    return X, Y


def get_fnames():
    im_paths = []
    for sub_dir in os.listdir(dname):
        path = dname+sub_dir + '/'
        # print(path)
        try:
            for fname in os.listdir(path):
                if fname.endswith(".jpg"):
                    tmp = path + fname
                    im_paths.append(tmp)
                    # print(tmp)
        except:
            continue
    return np.array(im_paths)


def generator(args):
    mode = args.mode
    print("choose mode: " + mode)
    im_paths = get_fnames()

    print("file number: " + str(len(im_paths)))
    P = np.random.permutation(len(im_paths))
    
    split1 = int(len(im_paths) * 0.7)
    split2 = split1 + int(len(im_paths) * 0.15)
    
    fnames_tr = im_paths[P[:split1]]
    np.savetxt(tr_path, fnames_tr, fmt='%s')
    print("training set size: " + str(len(fnames_tr)))
    fnames_va = im_paths[P[split1:split2]]
    np.savetxt(va_path, fnames_va, fmt='%s')
    print("validation set size: " + str(len(fnames_va)))
    fnames_te = im_paths[P[split2:]]
    np.savetxt(te_path, fnames_te, fmt='%s')
    print("testing set size: " + str(len(fnames_te)))

    vgg_model = VGG_16()
    print("Loaded VGG model")
    print()

    X_tr, Y_tr = fnames_to_features(fnames_tr, vgg_model, mode)
    np.savetxt(X_tr_path, X_tr)
    print("Saving X_tr features")
    np.savetxt(Y_tr_path, Y_tr, fmt='%i')
    print("Saving Y_tr features")
    print()


    X_va, Y_va = fnames_to_features(fnames_va, vgg_model, mode)
    np.savetxt(X_va_path, X_va)
    print("Saving X_va features")
    np.savetxt(Y_va_path, Y_va, fmt='%i')
    print("Saving Y_va features")
    print()

    X_te, Y_te = fnames_to_features(fnames_te, vgg_model, mode)
    np.savetxt(X_te_path, X_te)
    print("Saving X_te features")
    np.savetxt(Y_te_path, Y_te, fmt='%i')
    print("Saving Y_te features")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help = "select mode by 'style' or binary",
        choices = ["binary", "style"], default = "style")

    args = parser.parse_args()
    generator(args)

if __name__ == "__main__":
    main()




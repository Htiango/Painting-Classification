import numpy as np
import os

dname = "../data/paintings/"

training_ratio = 0.9

img_size = 100

img_size_cnn = 100

# western style is 1, eastern style is 0
painting_type_dic={ "gongbi": 0, 
                    "shin-hanga": 0,
                    "ink-and-wash-painting": 0,
                    "gothic": 1, 
                    "northern-renaissance": 1, 
                    "realism": 1, 
                    "abstract-art": 1, 
                    "international-gothic": 1, 
                    "impressionism": 1 }


painting_label_dic={ "gongbi": 0, 
                    "shin-hanga": 1,
                    "ink-and-wash-painting": 2,
                    "gothic": 3, 
                    "northern-renaissance": 4, 
                    "realism": 5, 
                    "abstract-art": 6, 
                    "international-gothic": 7, 
                    "impressionism": 8 }

tr_path = "../data/features/tr.txt"
X_tr_path = "../data/features/VGG_X_tr.txt"
Y_tr_path = "../data/features/VGG_Y_tr.txt"

va_path = "../data/features/va.txt"
X_va_path = "../data/features/VGG_X_va.txt"
Y_va_path = "../data/features/VGG_Y_va.txt"

te_path = "../data/features/te.txt"
X_te_path = "../data/features/VGG_X_te.txt"
Y_te_path = "../data/features/VGG_Y_te.txt"


# cnn preprocessing path
tr_path_cnn = "../data/features_cnn/tr.text"
X_tr_path_cnn = "../data/features_cnn/X_tr.txt"
Y_tr_path_cnn = "../data/features_cnn/Y_tr.txt"

te_path_cnn = "../data/features_cnn/te.text"
X_te_path_cnn = "../data/features_cnn/X_te.txt"
Y_te_path_cnn = "../data/features_cnn/Y_te.txt"


batch_size = 32
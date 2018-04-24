import numpy as np
import os

dname = "../data/paintings/"


img_size = 100

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

tr_path = "../data/features/tr.txt"
X_tr_path = "../data/features/VGG_X_tr.txt"
Y_tr_path = "../data/features/VGG_Y_tr.txt"

va_path = "../data/features/va.txt"
X_va_path = "../data/features/VGG_X_va.txt"
Y_va_path = "../data/features/VGG_Y_va.txt"

te_path = "../data/features/te.txt"
X_te_path = "../data/features/VGG_X_te.txt"
Y_te_path = "../data/features/VGG_Y_te.txt"


batch_size = 32
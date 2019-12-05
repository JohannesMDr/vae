from keras.models import Model
from keras.layers import *
from keras.layers.normalization import BatchNormalization

'''
ResNet18: nb_blocks = [2,2,2,2], wide = 2, nottleneck = False
ResNet34: nb_blocks = [3,4,6,3], wide = 2, nottleneck = False
ResNet50: nb_blocks = [3,4,6,3], wide = 2, nottleneck = True
ResNet101: nb_blocks = [3,4,23,3], wide = 2, nottleneck = True
ResNet152: nb_blocks = [3,8,36,3], wide = 2, nottleneck = True
WideResNet: nb_blocks = [3,3,3], wide >= 3, nottleneck = False  
'''

def ResBlock(X, n_filter, nb_blocks, resblock_num):
    for j in range(nb_blocks):
        shortcut = X
        X = BatchNormalization()(X)

        if resblock_num>0 and j == 0:
            shortcut =  Conv3D(n_filter, (1, 1, 1), strides=(2, 2, 2),
                            kernel_initializer='he_normal')(shortcut)
            X = Activation("relu")(X)
            X = Conv3D(n_filter, (3, 3, 3), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(X)
            X = BatchNormalization()(X)
    
        else:      
            X = Activation("relu")(X)
            X = Conv3D(n_filter, (3, 3, 3), padding="same", kernel_initializer='he_normal')(X)
            X = BatchNormalization()(X)

        X = Activation("relu")(X)
        X = Conv3D(n_filter, (3, 3, 3), padding="same",kernel_initializer='he_normal')(X)

        # ショートカットとマージ
        X = Add()([X, shortcut])

    return X

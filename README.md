# ANODEF
This repository is a modified clone of https://github.com/rtqichen/ffjord

1. The experiments on 2D data can be found inside the 'examples' folder in the following folders: 'hash_gaussian' and 'checker_board'.
        The default command to run the original cnfs is:  python cnf_normal.py
        The default command to run the augmented cnfs is: python cnf_augmented.py
2. The experiments on MNIST and CIFAR10 can be found inside the folder named 'affjord_joint'.
        The default command in the CIFAR10 case is: python train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 1 --layer_type hyper --multiscale True
        The default command in the MNIST case is:   python train_cnf.py --data mnist --dims 64,64,64 --strides 1,1,1,1 --num_blocks 1 --layer_type hyper --multiscale True

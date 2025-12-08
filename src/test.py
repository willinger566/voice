import numpy as np
for s in ['train','val','test']:
    L=np.load(f'data/features/CASIA/{s}/labels.npy')
    print(s, np.bincount(L))
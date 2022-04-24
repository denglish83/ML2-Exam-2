import numpy as np
import glob
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed=101
directory = '/home/ubuntu/data'

filelist = glob.glob('/home/ubuntu/data/train/*.txt')
filelist2 = sorted(filelist)
#print(filelist2)

y = np.array([])
for fname in filelist2:
    fle = open(fname,'r')
    lines = fle.readlines()
    cats = np.empty([7], dtype=int)
    for x in range(7):
        cats[x] = 0
    for line in lines:
        if line.strip() == 'red blood cell':
            cats[0] = 1
        elif line.strip() == 'difficult':
            cats[1] = 1
        elif line.strip() == 'gametocyte':
            cats[2] = 1
        elif line.strip() == 'trophozoite':
            cats[3] = 1
        elif line.strip() == 'ring':
            cats[4] = 1
        elif line.strip() == 'schizont':
            cats[5] = 1
        elif line.strip() == 'leukocyte':
            cats[6] = 1
    y = np.vstack([y, cats]) if y.size else cats
    fle.close()
#print(y)
yy = torch.from_numpy(y)


transform = transforms.Compose([transforms.Resize((800,600)), # full size is 1600x1200, but that causes memory errors
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

dataset = datasets.ImageFolder(directory, transform=transform)

x_train, x_valid, y_train, y_valid = train_test_split(dataset, yy, test_size=validation_split)

torch.save(x_train, 'x_train.pt')
torch.save(y_train, 'y_train.pt')
torch.save(x_valid, 'x_valid.pt')
torch.save(y_valid, 'y_valid.pt')
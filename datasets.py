# python libraries
import pandas as pd
from PIL import Image

# pytorch libraries
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils import calculate_normalized_stats

class HAM10KDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.df['path'][index])
        label = torch.tensor(int(self.df['class'][index]))
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(image_size):
    # norm_mean,norm_std = calculate_normalized_stats(glob(os.path.join(os.getcwd(), 'data/skin-cancer-mnist-ham10000/*/*.jpg')), image_size)

    # Saved values for HAM10K dataset
    norm_mean = [0.76304483, 0.54564637, 0.5700451]
    norm_std = [0.14092779, 0.15261324, 0.16997057]

    # ImageNet mean and std
    # norm_mean = [0.485, 0.456, 0.406]
    # norm_std = [0.229, 0.224, 0.225]

    # Transforms
    train_transforms = transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(20),
    #                                     transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                          transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(norm_mean, norm_std)])

    val_transforms = transforms.Compose([transforms.Resize((image_size,image_size)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    test_transforms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])
    return train_transforms, val_transforms, test_transforms

def upsample(df_train):
    max_size = df_train['dx'].value_counts().max()
    lst = [df_train]
    for _, group in df_train.groupby('dx'):
        lst.append(group.sample(max_size-len(group), replace=True))
    upsampled_df = pd.concat(lst)
    upsampled_df.reset_index(inplace=True)
    return upsampled_df

def balanced_batch_sample(df_train):
    class_counts = [(df_train['class']==i).sum() for i in range(7)]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[i] for i in df_train['class'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df_train), replacement=True)
    return sampler

def get_train_loader(df_train, batch_size, image_size, sampling_method='NONE'):
    train_transforms, _, _ = get_transforms(image_size)

    if sampling_method == 'UPSAMPLE':
        # Have to return df_train since upsample does not modify in place
        df_train = upsample(df_train)

    train_set = HAM10KDataset(df_train, transform=train_transforms)
    
    if sampling_method == 'BALANCED':
        sampler = balanced_batch_sample(df_train)
        return df_train, DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        return df_train, DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

def get_val_loader(df_val, batch_size, image_size):
    _, val_transforms, _ = get_transforms(image_size)
    val_set = HAM10KDataset(df_val, transform=val_transforms)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def get_test_loader(df_test, batch_size, image_size):
    _, _, test_transforms = get_transforms(image_size)
    train_set = HAM10KDataset(df_test, transform=test_transforms)
    return DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as lr_scheduler

def calculate_normalized_stats(image_paths, image_size):
    num_images = len(image_paths)
    channels = 3  # Assuming RGB images

    sum_means = np.zeros(channels)
    sum_squared_means = np.zeros(channels)

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.0

        # Calculate per-channel means
        means = np.mean(image, axis=(0, 1))

        # Calculate per-channel squared means
        squared_means = np.mean(image**2, axis=(0, 1))

        sum_means += means
        sum_squared_means += squared_means

    normalized_means = sum_means / num_images
    std_dev = np.sqrt(sum_squared_means / num_images - normalized_means**2)

    return normalized_means[::-1], std_dev[::-1] # BRG -> RGB

def get_scheduler(scheduler_name, optimizer, num_epochs):
    if scheduler_name == 'STEP':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    elif scheduler_name == 'EXPONENTIAL':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_name == 'COSINE':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'SGDR':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    elif scheduler_name == 'NONE':
        scheduler = None
    return scheduler

def calculate_class_weights(weight_method, data, labels):
    class_counts = data['dx'].value_counts()
    if weight_method == 'MEDIAN':
        class_weights = torch.FloatTensor(class_counts.median() / class_counts[labels].values)
    elif weight_method == 'INVFREQ':
        class_weights = torch.FloatTensor(1 / class_counts[labels].values)
    elif weight_method == 'INVPRO':
        class_weights = torch.FloatTensor(data.shape[0] / class_counts[labels].values)
    elif weight_method == 'NONE':
        class_weights = None
    return class_weights
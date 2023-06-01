import torch.nn as nn
import torchvision.models as models

import data

def initialise_model(model_name):
    # Setup this way so more models can be supported in future
    n_classes = len(data.classes)
    if model_name == "B0":
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        image_size = 224
        
    elif model_name == "B1":
        model = models.efficientnet_b1(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        image_size = 224
    
    elif model_name == "B2":
        model = models.efficientnet_b2(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        image_size = 288
        
    elif model_name == "B3":
        model = models.efficientnet_b3(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        image_size = 300
        
    elif model_name == "B4":
        model = models.efficientnet_b4(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        image_size = 380
            
    return model, image_size
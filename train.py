import os, argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import data
import utils
import models
import datasets
import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--train-folder', type=str, default='skin-cancer-mnist-ham10000')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--init-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--scheduler-type', type=str, default='NONE')
    parser.add_argument('--weight-method', type=str, default='NONE')
    parser.add_argument('--sampling-method', type=str, default='NONE')
    parser.add_argument('--snapshots', type=int, nargs='+')

    args = parser.parse_args()
    return args

def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Have to add this because it is called from validate.py
    model.train()
    train_loss = 0
    train_acc = 0

    for i, data in enumerate(tqdm(train_loader)):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()

        if (i + 1) % 100 == 0:
            avg_loss = train_loss / (i + 1)
            avg_acc = train_acc / ((i + 1) * images.size(0))
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), avg_loss, avg_acc))

    if scheduler:
        scheduler.step()

    avg_loss = train_loss / len(train_loader)
    avg_acc = train_acc / (len(train_loader) * train_loader.batch_size)
    return avg_loss, avg_acc

def main():
    filename = '_'.join([args.model_type, 'TRAIN', str(args.init_lr), str(args.batch_size), str(args.n_epochs), 
                        args.scheduler_type, args.weight_method, args.sampling_method])
    model, image_size = models.initialise_model(args.model_type)
    model = model.to(device)
    df_train = data.get_df_train(os.path.join(args.data_dir, args.train_folder))
    df_train, train_loader = datasets.get_train_loader(df_train, args.batch_size, image_size, args.sampling_method)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = utils.get_scheduler(args.scheduler_type, optimizer, args.n_epochs)
    class_weights = utils.calculate_class_weights(args.weight_method, df_train, data.classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    total_loss_train, total_acc_train = [],[]
    for epoch in range(args.n_epochs):
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler)
        total_loss_train.append(train_loss)
        total_acc_train.append(train_acc)
        if args.snapshots:
            if epoch in args.snapshots:
                filepath = os.path.join(args.model_dir, f'{filename}_{epoch}.pth')
                torch.save(model.state_dict(), filepath)
    filepath = os.path.join(args.model_dir, f'{filename}.pth')
    torch.save(model.state_dict(), filepath)
    evaluate.plot_accuracy_loss(total_loss_train, total_acc_train)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_dir, exist_ok=True)
    main()



    






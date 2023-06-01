import os, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import data
import utils
import models
import datasets
import evaluate
import train

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
    parser.add_argument('--n-splits', type=int, default=5)

    args = parser.parse_args()
    return args

def validate_epoch(val_loader, model, criterion, epoch):
    model.eval()
    val_loss_sum = 0
    val_acc_sum = 0
    total_samples = 0
    y_label = []
    y_predict = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            y_label.extend(labels.cpu().numpy())
            y_predict.extend(prediction.cpu().numpy())

            val_acc_sum += prediction.eq(labels.view_as(prediction)).sum().item()
            val_loss_sum += criterion(outputs, labels).item()
            total_samples += labels.size(0)

    balanced_acc = balanced_accuracy_score(y_label, y_predict)
    val_loss_avg = val_loss_sum / total_samples
    val_acc_avg = val_acc_sum / total_samples
    print('------------------------------------------------------------')
    print('[epoch %d], [val BMCA %.5f], [val loss %.5f], [val acc %.5f]' % (epoch, balanced_acc, val_loss_avg, val_acc_avg))
    print('------------------------------------------------------------')
    return balanced_acc, val_loss_avg, val_acc_avg

def train_split(num_epochs, model, criterion, optimizer, scheduler, train_loader, val_loader):
    best_BMCA = 0
    best_ACC = 0
    total_loss_train, total_acc_train, total_loss_val, total_acc_val = [],[],[],[]
    for epoch in range(1, num_epochs +1):
        train_loss, train_acc = train.train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler)
        BMCA, val_loss, val_acc = validate_epoch(val_loader, model, criterion, epoch)
        total_loss_val.append(val_loss)
        total_acc_val.append(val_acc)
        total_loss_train.append(train_loss)
        total_acc_train.append(train_acc)
        if BMCA > best_BMCA:
            best_BMCA = BMCA
            # torch.save(model.state_dict(), os.path.join(args.model_dir,'train_BMCA_best.pth'))
            print('*****************************************************')
            print('best BMCA record: [epoch %d], [val BMCA %.5f], [val loss %.5f], [val acc %.5f]' % (epoch, BMCA, val_loss, val_acc))
            print('*****************************************************')
        if val_acc > best_ACC:
            best_ACC = val_acc
            # torch.save(model.state_dict(), os.path.join(args.model_dir, 'train_val_best.pth'))
            print('*****************************************************')
            print('best VAL record: [epoch %d], [val BMCA %.5f], [val loss %.5f], [val acc %.5f]' % (epoch, BMCA, val_loss, val_acc))
            print('*****************************************************')
    # torch.save(model.state_dict(), os.path.join(args.model_dir,'train_last.pth'))
    # evaluate.plot_accuracy_loss(total_loss_train, total_acc_train)
    # evaluate.plot_accuracy_loss(total_loss_val, total_acc_val)
    return best_BMCA, best_ACC

def main():
    filename = '_'.join([args.model_type, 'CV', str(args.init_lr), str(args.batch_size), str(args.n_epochs), 
                        args.scheduler_type, args.weight_method, args.sampling_method, str(args.n_splits)])

    # k-fold stratified cross validation
    # Remove duplicated lesions in split to avoid information leak between test and val set
    df_train = data.get_df_train(os.path.join(args.data_dir, args.train_folder))
    df_unique = df_train[~df_train['lesion_id'].duplicated(keep=False)]
    targets = df_unique['dx']
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # Loop over the splits
    best_BMCA = []
    best_ACC = []
    for i, (_, val_index) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'=================================Split {i+1} =================================')
        # Reset all
        model, image_size = models.initialise_model(args.model_type)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
        scheduler = utils.get_scheduler(args.scheduler_type, optimizer, args.n_epochs)

        # Get the training and validation data for this split
        df_train_unique, df_val = data.get_train_val_split(df_train, df_unique.iloc[val_index])
        df_train_unique, train_loader = datasets.get_train_loader(df_train_unique, args.batch_size, image_size, args.sampling_method)
        val_loader = datasets.get_val_loader(df_val, args.batch_size, image_size)

        print(len(df_val))
        print(len(df_train_unique))

        class_weights = utils.calculate_class_weights(args.weight_method, df_train_unique, data.classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

        # CV
        BMCA, ACC = train_split(args.n_epochs, model, criterion, optimizer, scheduler, train_loader, val_loader)
        best_BMCA.append(BMCA)
        best_ACC.append(ACC)
        
    print(f"CV BMCA: {sum(best_BMCA) / args.n_splits:.3f} CV Accuracy: {sum(best_ACC) / args.n_splits:.3f}")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_dir, exist_ok=True)
    main()
import os, argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision.models as models


# sklearn libraries
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score

import data
import models
import datasets
import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--test-folder', type=str, default='ham10000-test')
    parser.add_argument('--ensemble-dir', type=str, default='./ensemble')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--input-size', type=int, default=224)

    args = parser.parse_args()
    return args

def make_ensemble(model_paths):
    ensemble = []
    for model_path in model_paths:
        model_type = model_path.split('_')[0]
        model, _ = models.initialise_model(model_type)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(args.ensemble_dir, model_path)))
        ensemble.append(model)
    return ensemble

# TTAx5
# https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/2c4a5428c6d410e97d2a74aacd5f86b3750d32cf/train.py#L97
def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

def test(ensemble):
    y_label = []
    y_predict = []

    df_test = data.get_df_test(os.path.join(args.data_dir, args.test_folder))
    
    test_loader = datasets.get_test_loader(df_test, args.batch_size, args.input_size)
    # For recording inference times
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = len(test_loader)
    timings=np.zeros((repetitions,1))
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            probs = torch.zeros((images.shape[0], len(data.classes))).to(device)

            starter.record()

            for model in ensemble:
                model.eval()
                for I in range(5):
                    l = model(get_trans(images, I))
                    probs += l.softmax(1)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time / len(images)

            probs /= 5
            probs /= len(ensemble)
            pred = probs.argmax(dim=1)
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(pred.cpu().numpy())
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    return mean_syn, std_syn, y_label, y_predict



def main():
    ensemble = make_ensemble(os.listdir(args.ensemble_dir))

    mean_syn, std_syn, y_label, y_predict = test(ensemble)
    print(f"Inference time: {mean_syn:.3f}\u00B1{std_syn:.3f}ms")
    
    # Evaluation metrics and graphs
    evaluate.plot_confusion_matrix(y_label, y_predict, data.classes, title='Confusion matrix')
    evaluate.plot_class_accuracy(y_label, y_predict, data.classes)
    report = classification_report(y_label, y_predict, target_names=data.classes, digits=3)
    print(report)
    BMCA = balanced_accuracy_score(y_label, y_predict)
    ACC = accuracy_score(y_label, y_predict)
    print(f"BMCA: {BMCA:.3f}, ACC: {ACC:.3f}")

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
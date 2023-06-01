import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_accuracy_loss(total_loss, total_acc):
    plt.figure()
    plt.plot(total_loss, label='loss')
    plt.plot(total_acc, label='accuracy')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.0])
    plt.show()

def plot_confusion_matrix(y_label, y_predict, classes, title='Confusion matrix'):
    cm = confusion_matrix(y_label, y_predict, normalize='true')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_class_accuracy(y_label, y_predict, classes):
    num_classes = len(classes)
    label_counts = np.zeros(num_classes)
    correct_counts = np.zeros(num_classes)

    # Calculate label counts and correct counts
    for i in range(len(y_label)):
        label = y_label[i]
        prediction = y_predict[i]
        label_counts[label] += 1
        if label == prediction:
            correct_counts[label] += 1

    # Calculate fraction classified incorrectly
    label_frac_error = 1 - correct_counts / label_counts

    # Create bar plot
    plt.bar(classes, label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')
    plt.show()

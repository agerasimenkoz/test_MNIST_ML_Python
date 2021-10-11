import itertools
import os
import matplotlib.pyplot as plt
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import seaborn as sns
import numpy as np


class ShowFunction:
    def __init__(self, history):
        self.history = history
        pass

    def show_history(self, save_dir="tmp/"):
        print(self.history.keys())
        accuracy = self.history['accuracy']
        val_accuracy = self.history['val_accuracy']
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy.png'))
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.show()
        pass

    def show_history_2(self):
        # Plot the loss and accuracy curves for training and validation
        fig, ax = plt.subplots(2, 1, figsize=(18, 10))
        ax[0].plot(self.history['loss'], color='b', label="Training loss")
        ax[0].plot(self.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        plt.legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(self.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(self.history['val_acc'], color='r', label="Validation accuracy")
        plt.legend = ax[1].legend(loc='best', shadow=True)
        plt.show()

    @staticmethod
    def confusion_matrix(predict_label, test_label):
        mat = confusion_matrix(test_label, predict_label)  # Confusion matrix

        # Plot Confusion matrix
        sns.heatmap(mat.T, square=True, annot=True, cbar=False, annot_kws={"size": 16})
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()

    # Look at confusion matrix
    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

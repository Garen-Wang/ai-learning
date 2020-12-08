import numpy as np
import matplotlib.pyplot as plt
import pickle


class TrainingHistory(object):
    def __init__(self):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []
        self.loss_val = []
        self.accuracy_val = []

    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.figure(figsize=(12, 5))
        axes = plt.subplot(1, 2, 1)
        p1, = axes.plot(self.epoch_seq, self.loss_val)
        p2, = axes.plot(self.epoch_seq, self.loss_train)
        axes.legend([p1, p2], ['validation', 'train'])
        axes.set_title('Loss')
        axes.set_xlabel('epoch')
        axes.set_ylabel('loss')
        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            axes.axis([xmin, xmax, ymin, ymax])

        axes = plt.subplot(1, 2, 2)
        p1, = axes.plot(self.epoch_seq, self.accuracy_val)
        p2, = axes.plot(self.epoch_seq, self.accuracy_train)
        axes.legend([p1, p2], ['validation', 'train'])
        axes.set_title('Accuracy')
        axes.set_xlabel('epoch')
        axes.set_ylabel('accuracy')

        title = params.toString()
        plt.suptitle(title)
        plt.show()

    def Load(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def Dump(self, file_name):
        with open(file_name, 'wb') as f:
            return pickle.dump(f)

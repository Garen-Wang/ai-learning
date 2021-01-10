import matplotlib.pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.epoch_seq = []
        self.iteration_seq = []
        self.loss_train = []
        self.accuracy_train = []
        self.loss_valid = []
        self.accuracy_valid = []

    def add(self, epoch, iteration, loss_train, accuracy_train, loss_valid, accuracy_valid):
        self.epoch_seq.append(epoch)
        self.iteration_seq.append(iteration)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        self.loss_valid.append(loss_valid)
        self.accuracy_valid.append(accuracy_valid)

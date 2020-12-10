import numpy as np
import matplotlib.pyplot as plt

class TrainingHistory(object):
    def __init__(self):
        self.loss_history = []
        self.iteration = []

    def append(self, loss, iteration):
        self.loss_history.append(loss)
        self.iteration.append(iteration)

    def show(self, params):
        plt.plot(self.iteration, self.loss_history)
        plt.title(params.getTitle())
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()

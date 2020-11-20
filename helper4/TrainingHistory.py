from helper4.HyperParameters import *
import matplotlib.pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        self.iteration = []

    def addLossHistory(self, iteration, loss, w, b):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.w_history.append(w)
        self.b_history.append(b)

    def showLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        title = params.toString()
        plt.title(title)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        if xmin is not None and ymin is not None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()

    def getLast(self):
        return self.loss_history[-1], self.w_history[-1], self.b_history[-1]

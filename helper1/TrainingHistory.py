import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from HyperParameters import *

class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []
        self.w_history = []
        self.b_history = []
    
    def addHistory(self, iteration, loss, w, b):
        self.loss_history.append(loss)
        self.iteration.append(iteration)
        self.w_history.append(w)
        self.b_history.append(b)
    
    def showHistory(self, params, xmin = None, xmax = None, ymin = None, ymax = None):
        plt.plot(self.iteration, self.loss_history)
        title = params.toString()
        plt.title(title)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()

    def getLast(self):
        return self.loss_history[-1], self.w_history[-1], self.b_history[-1]
    

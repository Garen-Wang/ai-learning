import numpy as np
import struct
from ch12.helperClass.DataReader import DataReader

train_image_filename  = '../../ai-data/Data/train-images-10'
train_label_filename  = '../../ai-data/Data/train-labels-10'
test_image_filename  = '../../ai-data/Data/test-images-10'
test_label_filename  = '../../ai-data/Data/test-labels-10'


class ImageDataReader(DataReader):
    def __init__(self, mode='image'):
        self.train_image_filename = train_image_filename
        self.train_label_filename = train_label_filename
        self.test_image_filename = test_image_filename
        self.test_label_filename = test_label_filename
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0
        self.mode = mode

    @staticmethod
    def __read_image(filename):
        with open(filename, 'rb') as f:
            a = f.read(4)
            b = f.read(4)
            c = f.read(4)
            d = f.read(4)
            num_image = int.from_bytes(b, byteorder='big')
            num_row = int.from_bytes(c, byteorder='big')
            num_col = int.from_bytes(d, byteorder='big')
            image_size = num_row * num_col
            image_data = np.empty((num_image, 1, num_row, num_col))
            for i in range(num_image):
                bin_data = f.read(image_size)
                unpacked_data = struct.unpack('>{}B'.format(image_size), bin_data)
                array_data = np.array(unpacked_data)
                array_data = array_data.reshape((1, num_row, num_col))
                image_data[i] = array_data

        return image_data

    @staticmethod
    def __read_label(filename):
        with open(filename, 'rb') as f:
            f.read(4)
            a = f.read(4)
            num_label = int.from_bytes(a, byteorder='big')

            label_data = np.empty((num_label, 1))
            for i in range(num_label):
                bin_data = f.read(1)
                unpacked_data = struct.unpack('>B', bin_data)
                label_data[i] = unpacked_data

        return label_data

    def read_data(self):
        self.XTrainRaw = self.__read_image(self.train_image_filename)
        self.XTestRaw = self.__read_image(self.test_image_filename)
        self.YTrainRaw = self.__read_label(self.train_label_filename)
        self.YTestRaw = self.__read_label(self.test_label_filename)

        # self.XTrainRaw = self.XTrainRaw[:count, :]
        # self.YTrainRaw = self.YTrainRaw[:count, :]

        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = len(np.unique(self.YTrainRaw))
        self.num_train = self.XTrainRaw.shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_valid = 0
        if self.mode == 'vector':
            self.num_feature = 784

    def __normalize(self, raw):
        maxv = np.max(raw)
        minv = np.min(raw)
        norm = (raw - minv) / (maxv - minv)
        return norm

    def normalize_x(self):
        self.XTrain = self.__normalize(self.XTrainRaw)
        self.XTest = self.__normalize(self.XTestRaw)

    def get_batch_train_samples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        batch_x = self.XTrain[start:end]
        batch_y = self.YTrain[start:end]

        if self.mode == 'vector':
            return batch_x.reshape(-1, 784), batch_y
        elif self.mode == 'image':
            return batch_x, batch_y

    def get_batch_test_samples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        batch_x = self.XTest[start:end]
        batch_y = self.YTest[start:end]

        if self.mode == 'vector':
            return batch_x.reshape(batch_size, -1), batch_y
        elif self.mode == 'image':
            return batch_x, batch_y

    def get_validation_set(self):
        if self.mode == 'vector':
            return self.XValid.reshape((self.num_valid, -1)), self.YValid
        elif self.mode == 'image':
            return self.XValid, self.YValid

    def get_test_set(self):
        if self.mode == 'vector':
            return self.XTest.reshape((self.num_test, -1)), self.YTest
        elif self.mode == 'image':
            return self.XTest, self.YTest


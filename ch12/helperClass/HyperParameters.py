from ch12.helperClass.EnumDef import NetType, InitMethod


class HyperParameters(object):
    def __init__(self, num_input, num_hidden1, num_hidden2, num_output, batch_size, max_epoch, eta, eps, net_type, init_method):
        self.num_input = num_input
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_output = num_output

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.eta = eta
        self.eps = eps

        self.net_type = net_type
        self.init_method = init_method

    def get_string(self):
        return "bz={1}, eta={2}, hidden={3}*{4}".format(self.batch_size, self.eta, self.num_hidden1, self.num_hidden2)


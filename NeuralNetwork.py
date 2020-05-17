import argparse
import csv
import numpy as np
import sys


class linear_layer:
    def __init__(self, input_d, output_d):
        self.params = dict()
        self.gradient = dict()
        my_sigma = 0.1
        self.params['W'] = np.random.normal(0, my_sigma, size=(input_d, output_d))
        self.params['b'] = np.random.normal(0, my_sigma, size=(1, output_d))
        self.gradient['W'] = np.zeros((input_d, output_d))
        self.gradient['b'] = np.zeros((1, output_d))

    def forward(self, X):
        output_d = self.params['b'].shape[1]
        forward_output = np.dot(X, self.params['W']) + self.params['b'].reshape(output_d, )
        return forward_output

    def backward(self, X, grad):
        backward_output = np.dot(grad, self.params['W'].transpose())
        self.gradient['W'] = np.dot(X.transpose(), grad)
        output_d = self.gradient['b'].shape[1]
        self.gradient['b'] = np.sum(grad, axis=0).reshape(1, output_d)
        return backward_output


class relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        forward_output = X * self.mask
        return forward_output

    def backward(self, X, grad):
        backward_output = self.mask * grad
        return backward_output


class softmax_cross_entropy:
    def __init__(self):
        self.Y = None
        self.e = None
        self.sum_e = None
        self.prob = None

    def forward(self, X, Y):
        n, d = X.shape
        self.Y = np.zeros(X.shape)
        self.Y = self.Y.reshape(-1)
        y_k = Y.astype(int).reshape(-1)
        y_k += np.arange(n) * d
        self.Y[y_k] = 1.0
        self.Y = self.Y.reshape(X.shape)
        x_max = np.amax(X, axis=1, keepdims=True)
        self.e = np.exp(X - x_max)
        self.sum_e = np.sum(self.e, axis=1, keepdims=True)
        self.prob = self.e / self.sum_e
        log = np.multiply(self.Y, X - x_max - np.log(self.sum_e))
        forward_output = - np.sum(log) / n
        return forward_output

    def backward(self, X, Y):
        n, d = X.shape
        return - (self.Y - self.prob) / n


class dropput:

    def __init__(self, rate):
        self.mask = None
        self.rate = rate

    def forward(self, X, train):
        if train:
            self.mask = np.random.uniform(0.0, 1.0, X.shape) >= self.rate
            self.mask = self.mask.astype(float) * (1.0 / (1.0 - self.rate))
        else:
            self.mask = np.ones(X.shape)

        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        return np.multiply(grad, self.mask)


def sgd_update(model, learning_rate):
    for module in model.values():
        if hasattr(module, 'params'):
            for key, value in module.params.items():
                grad = module.gradient[key]
                module.params[key] -= learning_rate * grad
    return model


def main(main_params):
    train_path = main_params[1]
    train_label_path = main_params[2]
    test_path = main_params[3]

    train_set = np.array(parseCSV(train_path)).astype(float) / 255
    train_label = np.array(parseCSV(train_label_path)).astype(float)
    test_set = np.array(parseCSV(test_path)).astype(float).astype(float) / 255
    # test_label = np.array(parseCSV(test_label_path))

    # print(train_set.shape)
    # print(train_label.shape)
    # print(test_set.shape)
    # print(test_label.shape)

    np.random.seed(7)
    n, d = train_set.shape
    num_epoch = 15
    learning_rate = 0.01
    dropput_rate = 0.1
    batch_size = 10
    l1_size = 1000
    l2_size = 10
    step = 25

    model = dict()
    model['l1'] = linear_layer(d, l1_size)
    model['l2'] = linear_layer(l1_size, l2_size)
    model['act1'] = relu()
    model['dropout'] = dropput(dropput_rate)
    model['softmax_loss'] = softmax_cross_entropy()

    for epoch in range(num_epoch):
        if epoch != 0 and epoch % step == 0:
            learning_rate *= 0.2

        order = np.random.permutation(n)
        for i in range(int(np.floor(n / batch_size))):
            index = order[batch_size * i: batch_size * (i + 1)]
            X = np.zeros((batch_size, d))
            y = np.zeros((batch_size, 1))

            for j in range(batch_size):
                X[j] = train_set[index[j]]
                y[j] = train_label[index[j]]

            l1 = model['l1'].forward(X)
            a1 = model['act1'].forward(l1)
            d1 = model['dropout'].forward(a1, True)
            l2 = model['l2'].forward(d1)
            model['softmax_loss'].forward(l2, y)

            grad_l2 = model['softmax_loss'].backward(l2, y)
            grad_d1 = model['l2'].backward(d1, grad_l2)
            grad_a1 = model['dropout'].backward(a1, grad_d1)
            grad_l1 = model['act1'].backward(l1, grad_a1)
            model['l1'].backward(X, grad_l1)

            model = sgd_update(model, learning_rate)

    X = test_set
    l1 = model['l1'].forward(X)
    a1 = model['act1'].forward(l1)
    d1 = model['dropout'].forward(a1, False)
    l2 = model['l2'].forward(d1)
    my_predict = np.argmax(l2, axis=1)
    with open('test_predictions.csv', mode='w') as csv_file:
        for i in range(len(my_predict)):
            csv_file.write(str(my_predict[i]) + '\n')


def parseCSV(path):
    data_set = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_set.append(row)
        data_set = np.array(data_set)
    return data_set


if __name__ == "__main__":
    main(sys.argv)

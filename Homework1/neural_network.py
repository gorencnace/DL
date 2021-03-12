import pandas as pd
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]

    def train(self, training_data,training_class, epochs, mini_batch_size, eta):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch"+str(j))
            loss_avg = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = net.backward_pass(output, mini_batch[1], Zs, As)

                self.update_network(gw, gb, eta)

                loss = cross_entropy(mini_batch[1], output)
                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))

    def eval_network(self, validation_data,validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)


            loss = cross_entropy(example_class, output)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))

    def update_network(self, gw, gb, eta):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        ########### Implement the update function
        for l in range(len(self.weights)):
            self.weights[l] = self.weights[l] - eta * gw[l]
            self.biases[l] = self.biases[l] - eta * gb[l]

    def forward_pass(self, input):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        ########## Implement the forward pass
        output, Zs, As = None, list(), [input]
        n_max = len(self.biases)
        n = 1
        for b, w in zip(self.biases, self.weights):
            Zs.append(np.dot(w, input) + b)
            if n < n_max:
                input = sigmoid(Zs[-1])
                As.append(input)
            else:
                output = softmax(Zs[-1])
                As.append(output)
            n += 1
        return output, Zs, As


    def backward_pass(self, output, target, Zs, activations):
        ########## Implement the backward pass
        n = activations[0].shape[0]
        gw, gb = list(), list()
        dz = softmax_dLdZ(output, target) #mogoce treba se kej mnozit z odvodom sigme
        dw = np.dot(dz, activations[-2].transpose()) / n
        db = np.sum(dz, axis=1, keepdims=True) / n
        da_prev = np.dot(self.weights[-1].transpose(), dz)

        gw.append(dw)
        gb.append(db)

        for l in range(len(self.weights) - 2, -1, -1):
            dz = da_prev * sigmoid_prime(Zs[l])
            dw = np.dot(dz, activations[l].transpose()) / n
            db = np.sum(dz, axis=1, keepdims=True) / n
            if l > 0:
                da_prev = np.dot(self.weights[l].transpose(), dz)
            gw.append(dw)
            gb.append(db)
        gw.reverse()
        gb.reverse()
        return gw, gb



def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def cross_entropy(y,x, epsilon=1e-12):
    targets = y.transpose()
    predictions = x.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train_np = train.to_numpy()
    test_np = test.to_numpy()
    train_data = train_np[:,1:] / 255.0
    train_class = train_np[:,:1]
    train_class_one_hot = np.zeros((train_data.shape[0],10))
    train_class_one_hot[np.arange(train_class.shape[0]),train_class[:,0]] = 1.0
    test_data = test_np[:,1:] / 255.0
    test_class = test_np[:,:1]
    test_class_one_hot = np.zeros((test_class.shape[0],10))
    test_class_one_hot[np.arange(test_class.shape[0]),test_class[:,0]] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()


if __name__ == "__main__":
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    train_data, train_class, test_data, test_class = load_data(train_file, test_file)
    # The network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the
    net = Network([train_data.shape[0],100, 100,10])
    net.train(train_data,train_class, 20, 128, 0.1)
    net.eval_network(test_data, test_class)

from random import seed, randrange, random
from csv import reader
from math import exp

def load_csv(filename):
    dataset = list()
    check = 0
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for X in csv_reader:
            if not X:
                continue
            if check != 0:
                dataset.append(X)
            check += 1
    change_dataset(dataset)
    return dataset

def change_dataset(dataset):
    for element in dataset:
        for i in range(len(element) - 1):
            element[i] = float(element[i].strip())
        element[-1] = 1.0 if element[-1] == 'Yes' else 0.0

def class_label(dataset, column):
    class_values = [X[column] for X in dataset]
    unique = set(class_values)
    hash = dict()
    for i, value in enumerate(unique):
        hash[value] = i
    for X in dataset:
        X[column] = hash[X[column]]
    return hash

def split_into_k_folds(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def compute_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def cross_validate(dataset, algorithm, n_folds, *args):
    folds = split_into_k_folds(dataset, n_folds)
    accuracy_values = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for X in fold:
            row_copy = list(X)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [X[-1] for X in fold]
        accuracy = compute_accuracy(actual, predicted)
        accuracy_values.append(accuracy)
    return accuracy_values

def make_predictions(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'w':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'w':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, X):
    inputs = X
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = make_predictions(neuron['w'], inputs)
            neuron['o'] = sigmoid(activation)
            new_inputs.append(neuron['o'])
        inputs = new_inputs
    return inputs

def derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['w'][j] * neuron['d'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['o'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['d'] = errors[j] * derivative(neuron['o'])

def update_weights(network, X, l_rate):
    for i in range(len(network)):
        inputs = X[:-1]
        if i != 0:
            inputs = [neuron['o'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['w'][j] += l_rate * neuron['d'] * inputs[j]
            neuron['w'][-1] += l_rate * neuron['d']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for X in train:
            expected = [0 for i in range(n_outputs)]
            expected[X[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, X, l_rate)

def predict(network, X):
    outputs = forward_propagate(network, X)
    return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([X[-1] for X in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for X in test:
        prediction = predict(network, X)
        predictions.append(prediction)
    return(predictions)

seed(1)

filename = raw_input("Enter the file name: ")
learning_rate = raw_input("Enter the learning rate: ")
epochs = raw_input("Enter the number of epochs: ")
k_folds = raw_input("Enter the number of folds: ")
k_hidden = raw_input("Enter number of hidden neurons: ")

dataset = load_csv(filename)
class_label(dataset, len(dataset[0])-1)

accuracy_values = cross_validate(dataset, back_propagation, int(k_folds), float(learning_rate), int(epochs), int(k_hidden))
print('Accuracy: %s' % accuracy_values)
print('Average accuracy: %f' % (sum(accuracy_values)/float(len(accuracy_values))))
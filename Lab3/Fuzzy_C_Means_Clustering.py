from csv import reader
from math import sqrt
from random import randrange

def load_csv(filename):
    dataset = list()
    check = 0
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if check != 0:
                dataset.append(row)
            check += 1
    change_dataset(dataset)
    return dataset

def change_dataset(dataset):
    for element in dataset:
        for i in range(len(element) - 1):
            element[i] = float(element[i].strip())
        element[-1] = 1.0 if element[-1] == 'Yes' else 0.0

def pick_centroids(dataset, c):
    dataset_copy = list(dataset)
    clusters = []
    for i in range(c):
        index = randrange(len(dataset_copy))
        clusters.append(dataset_copy.pop(index))
    return clusters

def euclidian_distance(datapoint_1, datapoint_2):
    value = 0
    for i in range(len(datapoint_1) - 1):
        value += (datapoint_1[i] - datapoint_2[i])**2
    return sqrt(value)

def compute_membership(dataset, clusters, c, m):
    membership_values = {}
    for i in range(len(dataset)):
        center1_dist = euclidian_distance(dataset[i], clusters[0])
        center2_dist = euclidian_distance(dataset[i], clusters[1])
        exp = float(2)/ (m - 1)
        if(center2_dist == 0):
            value = float(0)
        else:
            value = 1 + (float(center1_dist)/ center2_dist)**exp
            value = 1/ value
        if(0 not in membership_values):
            membership_values[0] = []
        membership_values[0].append(value)
        if(1 not in membership_values):
            membership_values[1] = []
        membership_values[1].append(1 - value)
    return membership_values

def compute_new_clusters(membership_values):
    clusters = []
    for cluster_num, membership_value in membership_values.iteritems():
        membership_squares = [i**2 for i in membership_value]
        cluster_weights = list(list())
        for i in range(len(dataset)):
            cluster_weights.append([x*membership_squares[i] for x in dataset[i]])
        clusters.append([float(sum(i)) / sum(membership_squares) for i in zip(*cluster_weights)])
    return clusters

def dataset_segregation(dataset, membership_values):
    segregation = {}
    segregation[0] = []
    segregation[1] = []
    membership_value = membership_values[0]
    for i in range(len(membership_value)):
        if(membership_value[i] > 0.5):
            segregation[0].append(dataset[i])
        else:
            segregation[1].append(dataset[i])
    return segregation

def fuzzy_c_means_clustering(dataset, c, m, num_iterations):
    clusters = pick_centroids(dataset, c)
    prev_clusters = []
    for i in range(num_iterations):
        if (prev_clusters == clusters):
            break
        membership_values = compute_membership(dataset, clusters, c, m)
        new_clusters = compute_new_clusters(membership_values)
        prev_clusters = clusters
        clusters = new_clusters
    segregation = dataset_segregation(dataset, membership_values)
    print("Total iterations used: %s" %(i + 1))
    return segregation

def accuracy(segregation):
    correct = 0
    for cluster_num, data_points in segregation.iteritems():
        count_0 = count_1 = 0
        for data_point in data_points:
            if(data_point[-1] == 0):
                count_0 = count_0 + 1
            else:
                count_1 = count_1 + 1

        if(count_0 > count_1):
            class_assigned = 0
        else:
            class_assigned = 1
        print("Cluster: %s; Class Assigned: %s; Number of elements: %s" %(cluster_num, class_assigned, len(data_points)))

        for data_point in data_points:
            if(data_point[-1] == class_assigned):
                correct = correct + 1
    return correct
    
filename = raw_input("Enter file name: ")
c = 2
m = int(raw_input("Enter fuzziness parameter: "))
num_iterations = int(raw_input("Enter the maximum number of iterations: "))

dataset = load_csv(filename)
segregation = fuzzy_c_means_clustering(dataset, c, m, num_iterations)
print("Clusters: %s" % segregation)
correct = accuracy(segregation)
print("Accuracy: %s" %(correct/float(len(dataset)) * 100))

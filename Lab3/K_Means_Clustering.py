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

def pick_centers(dataset, k):
    dataset_copy = list(dataset)
    clusters = []
    for i in range(k):
        index = randrange(len(dataset_copy))
        clusters.append(dataset_copy.pop(index))
    return clusters

def euclidian_distance(datapoint_1, datapoint_2):
    value = 0
    for i in range(len(datapoint_1) - 1):
        value += (datapoint_1[i] - datapoint_2[i])**2
    return sqrt(value)

def dataset_segregation(dataset, assigned_clusters):
    segregated_dataset = {}
    for i in range(len(dataset)):
        if (assigned_clusters[i] not in segregated_dataset):
            segregated_dataset[assigned_clusters[i]] = []
        segregated_dataset[assigned_clusters[i]].append(dataset[i])
    return segregated_dataset

def compute_new_clusters(dataset, assigned_clusters):
    clusters = []
    segregated_dataset = dataset_segregation(dataset, assigned_clusters)
    for cluster_num, data_points in segregated_dataset.iteritems():
        clusters.append([float(sum(i)) / len(i) for i in zip(*data_points)])
    return clusters

def assign_clusters(clusters, dataset, k):
    assigned_clusters = []
    for data_point in dataset:
        distances = []
        for i in range(k):
            distances.append(euclidian_distance(data_point, clusters[i]))
        cluster = distances.index(min(distances))
        assigned_clusters.append(cluster)
    return assigned_clusters

def k_means_clustering(dataset, k, num_iterations):
    clusters = pick_centers(dataset, k)
    prev_clusters = []
    for i in range(num_iterations):
        if (prev_clusters == clusters):
            break
        assigned_clusters = assign_clusters(clusters, dataset, k)
        new_clusters = compute_new_clusters(dataset, assigned_clusters)
        prev_clusters = clusters
        clusters = new_clusters
    final_segregation = dataset_segregation(dataset, assigned_clusters)
    print("Total iterations used: %s" %(i + 1))
    return final_segregation

def accuracy(final_segregation):
    correct = 0
    for cluster_num, data_points in final_segregation.iteritems():
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
k = 2
num_iterations = int(raw_input("Enter the maximum number of iterations: "))

dataset = load_csv(filename)
final_segregation = k_means_clustering(dataset, k, num_iterations)
print("Clusters: %s" % final_segregation)
correct = accuracy(final_segregation)
print("Accuracy: %s" %(correct/float(len(dataset)) * 100))
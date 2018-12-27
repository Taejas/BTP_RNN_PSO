import csv
import numpy as np

# Delta_t
DELTA_t = 0.1

# Sigmoid activation function.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def generate_dataset(N, w, beta, tao, time_points):
    dataset = []
    dataset.append(np.random.uniform(0, 1, N))
    for i in range(time_points - 1):
        e_curr = dataset[i]
        e_next = np.zeros(N)
        for n in range(N):
            e_next[n] = (DELTA_t / tao[n]) * (sigmoid(np.dot(w[n], e_curr)) + beta[n]) + (1 - DELTA_t / tao[n]) * e_curr[n]
        dataset.append(e_next)
    return dataset
    

if __name__ == '__main__':
    
    N = 4
    w = np.array([
            [20.0, -20.0, 0.0, 0.0],
            [15.0, -10.0, 0.0, 0.0],
            [0.0, -8.0, 12.0, 0.0],
            [0.0, 0.0, 8.0, -12.0]
            ])
    beta = np.array([0.0, -5.0, 0.0, 0.0])
    tao = np.array([10.0, 5.0, 5.0, 5.0])
    time_points = 50
    
    dataset = generate_dataset(N, w, beta, tao, time_points)
    
    headings = ['Gene_1', 'Gene_2', 'Gene_3', 'Gene_4']
    with open('artificial.csv', 'w', newline = '') as csv_file:
        csvWriter = csv.writer(csv_file, delimiter = ',')
        csvWriter.writerow(headings)
        csvWriter.writerows(dataset)
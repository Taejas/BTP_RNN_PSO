import copy
import numpy as np
import pandas as pd



# File name
FILE_NAME = 'artificial.csv'
# Number of particles to be generated
NUM_PARTICLES = 30
# Number of iterations
NUM_ITERATIONS = 500

# Results
predicted_weights = None
fit_pw = None

# Dataset (initialised in main)
data = None
# Number of genes in the system (initialised in main)
N = None
# Number of time points (initialised in main)
T = None

# Range limits
W_MIN = -50
W_MAX = 50
BETA_MIN = -10
BETA_MAX = 10
TAO_MIN = 0
TAO_MAX = 20
V_MAX = 6

# Delta_t
DELTA_t = 0.1
# Inertia weight
W_I = 0.7
# Acceleration constants
C_1 = 2
C_2 = 2
# Del
DEL = 0.05


# D = N(N + 2) dimensional vector of 0s and 1s.
class Position(object):
    
    def __init__(self):
        # Weights; w[i][j]: Effect of jth gene on the ith gene
        self.w = np.random.randint(2, size = (N, N))
        # Biases
        self.beta = np.random.randint(2, size = N)
        # Time constants
        self.tao = np.random.randint(2, size = N)

# D = N(N + 2) dimensional vector; identical in structure to Position.
class Velocity(object):
    
    def __init__(self):
        # Weights; w[i][j]: Effect of jth gene on the ith gene
        self.w = np.random.uniform(-V_MAX, V_MAX, (N, N))
        # Biases
        self.beta = np.random.uniform(-V_MAX, V_MAX, N)
        # Time constants
        self.tao = np.random.uniform(-V_MAX, V_MAX, N)

        
class Particle(object):
    
    def __init__(self):
        # Position
        self.x = Position()
        # Velocity
        self.v = Velocity()
        # Own previous best position
        self.p = copy.deepcopy(self.x)
        # Fitness for best position
        self.fit_p = float('inf')


# Sigmoid activation function.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Model for estimating expression levels at next time point.
def model_estimate(x, t, n):
    estimate = (sigmoid(np.dot(x.w[n], data[t])) + x.beta[n]) - data[t][n]
    return estimate

# Fitness function to measure deviation of network output from target.
def fitness(x):
    fitness_x = 0
    for t in range(T - 1):
        for n in range(N):
            fitness_x += (model_estimate(x, t, n) - data[t + 1][n]) ** 2
    fitness_x /= (T - 1) * N
    return fitness_x
        

def PSO_based_RNN_training(num_particles, num_iterations):
    
    # 1. Initialising population of particles.
    particles = [Particle() for _ in range(num_particles)]
    
    # Global best position.
    p_g = Position()    # Arbitrary position initialization; will be overwritten.
    fit_p_g = float('inf')
    
    # 6. Repetition until stopping criterion is met.
    for iter in range(num_iterations):
        
        # 2. Evaluating optimization fitness function for each particle.
        fitness_values = np.zeros(len(particles))
        for i in range(len(particles)):
            fitness_values[i] = fitness(particles[i].x)
        
        # 3. Updating best position for each particle.
        for i in range(len(particles)):
            if(fitness_values[i] < particles[i].fit_p):
                particles[i].fit_p = fitness_values[i]
                particles[i].p = copy.deepcopy(particles[i].x)
                # 4. Updating global best position.
                if(fitness_values[i] < fit_p_g):
                    fit_p_g = particles[i].fit_p
                    p_g = copy.deepcopy(particles[i].p)
        
        # 5. Updating velocity and position of particles.
        for i in range(len(particles)):
            v_next_w = W_I * particles[i].v.w + C_1 * np.random.uniform(0, 1) * (particles[i].p.w - particles[i].x.w) + C_2 * np.random.uniform(0, 1) * (p_g.w - particles[i].x.w)
            v_next_beta = W_I * particles[i].v.beta + C_1 * np.random.uniform(0, 1) * (particles[i].p.beta - particles[i].x.beta) + C_2 * np.random.uniform(0, 1) * (p_g.beta - particles[i].x.beta)
            v_next_tao = W_I * particles[i].v.tao + C_1 * np.random.uniform(0, 1) * (particles[i].p.tao - particles[i].x.tao) + C_2 * np.random.uniform(0, 1) * (p_g.tao - particles[i].x.tao)
            v_next_w = np.clip(v_next_w, -V_MAX, V_MAX)
            v_next_beta = np.clip(v_next_beta, -V_MAX, V_MAX)
            v_next_tao = np.clip(v_next_tao, -V_MAX, V_MAX)
            particles[i].v.w = v_next_w
            particles[i].v.beta = v_next_beta
            particles[i].v.tao = v_next_tao
            v_next_w_prob = sigmoid(v_next_w)
            v_next_beta_prob = sigmoid(v_next_beta)
            v_next_tao_prob = sigmoid(v_next_tao)
            particles[i].x.w = (np.random.uniform(0, 1) + DEL < v_next_w_prob).astype(np.int)
            particles[i].x.beta = (np.random.uniform(0, 1) + DEL < v_next_beta_prob).astype(np.int)
            particles[i].x.tao = (np.random.uniform(0, 1) + DEL < v_next_tao_prob).astype(np.int)

    return p_g.w, fit_p_g
        


if __name__ == '__main__':

    # Reading the data; rows represent time points and columns represent genes.
    df = pd.read_csv(FILE_NAME)
    data = df.values
    N = len(data[0])
    T = len(data)
    
    predicted_weights, fit_pw = PSO_based_RNN_training(NUM_PARTICLES, NUM_ITERATIONS)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import random

#make the environment for problem
env = gym.make('CartPole-v0')

#model parameters
alpha = 0.01   #step size parameter for each update of the network's weights
gamma = 0.9    #discount factor, if close to 0, model weights nearer timesteps

class GradDescModel:

    """Neural network for a gradient descent algorithm"""
    
    def __init__(self, env, alpha, gamma):
    
        self.action_space = env.action_space     #possible actions
        self.observation_space = env.observation_space     #possible states
        self.alpha = alpha
        self.gamma = gamma
        self.S = []         #stochastic observations/states for a single episode
        self.A = []         #stochastic actions for a single episode
        self.R = []         #stochastic rewards for a single episode
        
        self.lifespan = []    #records how long each episode lasts for
        
        self.model = self.build_model()
        
        
    def build_model(self):
        
        """Creates the neural network"""
        
        model = keras.Sequential([
            keras.Input(self.observation_space.shape, name="observations", dtype=np.float64),
            layers.Dense(32, activation='relu', dtype=np.float64),
            layers.Dense(32, activation='relu', dtype=np.float64),
            layers.Dense(self.action_space.n, activation="softmax", name="actions", dtype=np.float64),
        ])
        
        return model
        
    def save(self, obs, action, reward):
    
        """Records the state, action and reward at each time step"""
    
        self.S.append(obs)
        self.A.append(action)
        self.R.append(reward)
        
    def sumG(self, t):
    
        """Calculates the return G from time t"""
        
        g = 0
        N = len(self.R)
        
        for k in range(t+1, N):
            g += self.gamma**(k-t-1) * self.R[k]
            
        return g
   
#function for return the rolling average of an array
def rollingAvg(x):

    result = np.zeros(len(x))
    result[0] = x[0]
    
    for i in range(1, len(x)):
    
        result[i] = ((float(i)-1.0)*result[i-1] + x[i])/float(i)
    
    return result
        
#create model class instance
agent = GradDescModel(env, alpha, gamma)
print(agent.model.summary())

#NOTES :
#Input needs to be 2D array, of form (n_batches, 2) as model.predict takes in batches of inputs
#need to cast output of model to numpy and then ravel it for 1d array

for i_episode in range(50):

    #get initial state of system
    obs = env.reset()
    
    #generate an episode
    for t in range(1000):
    
        env.render()
        #reshape
        obs = np.reshape(obs, (1,4))
        #return action with highest probability
        action = np.argmax(agent.model(obs).numpy().ravel())
        #take this action
        obs, reward, done, info = env.step(action)
        #record next state, action and reward
        agent.save(obs, action, reward)
        #check for termination
        if done:
            print("{} finished after {} timesteps".format(i_episode, t+1))
            agent.lifespan.append(t+1)
            break
           
    #now loop for every step of the episode
    N = len(agent.R)
    
    for t in range(N):
        
        #compute return from time t
        g = agent.sumG(t)
        
        #compute the gradient of the logarithm
        with tf.GradientTape() as tape:
            tape.watch(agent.model.trainable_variables)
            result = tf.math.log(agent.model(np.reshape(agent.S[t], (1,4))))

        nabla = tape.gradient(result, agent.model.trainable_variables)
        
        #update the model weights
        for i in range(len(nabla)):
            agent.model.trainable_variables[i].assign( tf.math.add(agent.model.trainable_variables[i], tf.math.scalar_mul( agent.alpha * agent.gamma**t * g, nabla[i]  )  )  )
    
    #unload the previous episode and start again
    agent.S = []
    agent.A = []
    agent.R = []

env.close()

plt.plot(rollingAvg(agent.lifespan))
plt.show()


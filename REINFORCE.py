import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import random
import h5py

#make the environment for problem
env = gym.make('CartPole-v0')

#model parameters
alpha = 1e-10   #step size parameter for each update of the network's weights
gamma = 0.1    #discount factor, if close to 0, model weights nearer timesteps

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
            #layers.Dense(64, activation='relu', dtype=np.float64),
            #layers.Dense(4, activation='relu', dtype=np.float64),
            #layers.Dense(64, activation='relu', dtype=np.float64),
            layers.Dense(self.action_space.n, activation="softmax", name="actions", dtype=np.float64),
        ])
        
        #for i in range(len(model.trainable_variables)):
        #    model.trainable_variables[i].assign( tf.ones_like (model.trainable_variables[i]) * 0.5 )
        
        return model
        
    def save(self, obs, action, reward):
    
        """Records the state, action and reward at each time step"""
    
        self.S.append(obs)
        self.A.append(action)
        self.R.append(reward)

    def initialiseWeights(self, N):

        """Keeps randomly initialising the weights until they are decent"""

        initialR = 0
        initialcount = 0
        while initialR < N:

            obs = env.reset()
            
            #generate an episode
            for t in range(1000):
            
                #env.render()
                #reshape
                obs = np.reshape(obs, (1,4))
                #return action with highest probability
                action = np.argmax(self.model(obs).numpy().ravel())
                #take this action
                newobs, reward, done, info = env.step(action)
                #record next state, action and reward
                self.save(obs, action, reward)
                obs = newobs
                #check for termination
                if done:
                    #print("Initialising {} finished after {} timesteps".format(initialcount, t+1))
                    #self.lifespan.append(t+1)
                    break
            initialcount += 1
            initialR = t+1
            if initialR < N:
                self.model = self.build_model()
                #print('new model')

        
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
    
        result[i] = ((float(i))*result[i-1] + x[i])/float(i+1.0)
    
    return result
    

#NOTES :
#Input needs to be 2D array, of form (n_batches, 2) as model.predict takes in batches of inputs
#need to cast output of model to numpy and then ravel it for 1d array

alphaList = [1*10**(-exp) for exp in range(5, 15)]
print(alphaList)
gammaList = np.round([0.1+i*0.2 for i in range(5)],2)
print(gammaList)

episodes = 300
averageN = 50

alphaGrid, gammaGrid = np.meshgrid(alphaList, gammaList)
rGrid = np.zeros((len(alphaList), len(gammaList)))
initialRGrid = np.zeros((len(alphaList), len(gammaList)))

for alpha_i, alpha in enumerate(alphaList):
    for gamma_i, gamma in enumerate(gammaList):
        print(alpha, gamma)
        #create model class instance
        agent = GradDescModel(env, alpha, gamma)
        #print(agent.model.summary())
        agent.initialiseWeights(50)
        #print(agent.model.trainable_variables)

        for i_episode in range(episodes):

            #get initial state of system
            obs = env.reset()
            
            #generate an episode
            for t in range(1000):
            
                #env.render()
                #reshape
                obs = np.reshape(obs, (1,4))
                #return action with highest probability
                action = np.argmax(agent.model(obs).numpy().ravel())
                #take this action
                newobs, reward, done, info = env.step(action)
                #record next state, action and reward
                agent.save(obs, action, reward)
                obs = newobs
                #check for termination
                if done:
                    #print("{} finished after {} timesteps".format(i_episode, t+1))
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
                    actions = agent.model(tf.reshape(agent.S[t], (1,4)))
                    #print(actions)
                    b = tf.sort(actions,axis=-1,direction='DESCENDING',name=None)
                    #print(b)
                    action = b[0][0]
                    #print(action)
                    result = tf.math.log(action)
                    #print(result)
                    
                    #result = tf.math.log(agent.model(np.reshape(agent.S[t], (1,4))))

                nabla = tape.gradient(result, agent.model.trainable_variables)
                #print(nabla)
                #update the model weights
                for i in range(len(nabla)):
                    agent.model.trainable_variables[i].assign( tf.math.add(agent.model.trainable_variables[i], tf.math.scalar_mul( agent.alpha * agent.gamma**t * g, nabla[i]  )  )  )
                    
                #print("variables: ", agent.model.trainable_variables)
                #print("-----------")
                #print("nabla: ", nabla)
                #print("-----------")
            
            #unload the previous episode and start again
            agent.S = []
            agent.A = []
            agent.R = []

        rGrid[alpha_i,gamma_i] = np.average(agent.lifespan[-averageN:])
        initialRGrid[alpha_i,gamma_i] = np.average(agent.lifespan[:averageN])
        print(initialRGrid[alpha_i,gamma_i])
        print(rGrid[alpha_i,gamma_i])

env.close()

f = h5py.File('cartPole.hdf5', 'w')

grp = f.create_group('noLayersModel')
grp2 = grp.create_group('initialiseCondition')
grp3 = grp2.create_group('50plus')
dset = grp3.create_dataset('alphaGrid', data=alphaGrid)
dset2 = grp3.create_dataset('gammaGrid', data=gammaGrid)
dset3 = grp3.create_dataset('rGrid_0', data=rGrid)
dset3.attrs['averageOver'] = averageN
dset3.attrs['averageAfter'] = episodes-averageN
dset4 = grp3.create_dataset('initialRGrid_0', data=initialRGrid)
dset4.attrs['averageOver'] = averageN

f.close()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_ylabel('alpha')
ax.set_xlabel('gamma')
ax.set_xticklabels(alphaList)
ax.set_yticklabels(gammaList)

plt.imshow(rGrid, origin='lower')
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_xticklabels(alphaList)
ax.set_yticklabels(gammaList)

plt.imshow(initialRGrid, origin='lower')
plt.colorbar()
plt.show()

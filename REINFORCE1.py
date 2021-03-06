#https://github.com/clearya1/Reinforcement-Learning.git
import numpy as np
import tensorflow as tf
#import keras as ke
#def get_policy():
#    a=ke.


import numpy as np
import tensorflow as tf
from tensorflow import keras as ke
#import keras as ke
from tensorflow.keras import layers as la
import gym,random
import matplotlib.pyplot as plt
 
#where to save file
locc="rl2_model1.h5"

env = gym.make('CartPole-v0')

#tries to load an existing model from file for more training.
#otherwise, produces a model that takes in an observation and action and predicts reward


def loss_f(y_true, y_pred):
    return -tf.tensordot(y_true,tf.math.log(y_pred),axes=[1,1],name="val")
try:
    model=ke.models.load_model(locc)
except OSError:
    model = ke.Sequential()
    model.add( la.Input(shape=env.observation_space.shape,name="obs"))
    #p2 = la.Input(shape=(1,),name="act")
    model.add(la.Dense(60,activation="relu"))
    #p4=la.Dense(20,activation="relu")(p2)
    #p5=la.Concatenate(axis=-1)([p3,p4])
    model.add(la.Dense(60,activation="relu"))
    #p7=la.Dense(40,activation="relu")(p6)
    model.add(la.Dense(env.action_space.n,activation="softmax"))
    #mask=la.Input(shape=[env.action_space.n],name="obs")
    #model2=ke.Model(inputs=model.inputs+[mask],outputs=[la.Dot(axes=1)([model.output,mask])])
    #model = ke.Model(inputs=[p1,p2], outputs=p8, name="mnist_model")
    model.summary()
    model.compile( optimizer=ke.optimizers.Adam(learning_rate=0.003),loss=loss_f)#ke.losses.MeanSquaredError())#,metrics=[ke.losses.MeanSquaredError()])
    
#assert 0
gamma=0.02
rav=10
def ExpDecay(x):
    global rav
    rwd=np.cumsum(x[::-1])[::-1]
    av=np.mean(rwd)
    ff=0.1
    oav=rav
    rav*=1-ff
    rav+=av*ff
    
    
    return rwd-30#-oav#- 10
    #converts x[i] into sum(x[j]*gamma**j; for j>=i)*(1-gamma)
    #this gives a reward to train the agent to predict.

    #first note that this function is linear.
    #for any i and j>i, note that j-i has a binary expansion, with (j-i).bit_length() at most n.bit_length()
    #values are stepped backwards in powers of 2, and multiplied by y**(2**s) as they are moved.
    #there is only one sequence of stepping back and staying in place that lets x[j] influence x[i]
    #and all the y terms multiply together to make y**(j-i)

    #I wrote it like this because numpy allows quick batch processing. 
    
    n=x.size
    y=1-gamma
    x=x*(1-y)
    for i in range(n.bit_length()):
        j=1<<i
        x[:-j]+=x[j:]*y
        y=y*y
    return x
#acts=[]

#rwds=[]
#these act as piles of data to train on
def s():
    model.save(locc)

for itr in range(1000):
    observation=env.reset()

    rewards_n=[]
    obss=[]
    acts=[]
    for i in range(1000):
        
        env.render()
        obss.append(observation)

        
        

        #take the action with the higher predicted reward
        prr=model.predict({"obs":observation[None,:]})
        act=random.choices(range(env.action_space.n),weights=prr.ravel(),k=1)[0]
        #print(prr,act,type(act))
        f=env.action_space.sample()
        #print(f,type(f))

        observation, reward, done, info=env.step(act) # take an action
        
        acts.append(act)
        rewards_n.append(reward)
        
        if done:
            break
    l=len(acts)
    print(l)
    obss=np.array(obss)
    mask=np.zeros([l,env.action_space.n],float)
    rewards_n=ExpDecay(np.array(rewards_n))
    mask[np.arange(l),acts]=rewards_n
    model.train_on_batch(x=obss,y=mask)
##    
##    rwds.extend(list(ExpDecay(np.array(rewards_n))))#apply ExpDecay 
##    mlen=5000
##    rwds=rwds[-mlen:]
##    obss=obss[-mlen:]
##    acts=acts[-mlen:]
##    #get rid of any data thats too old.
##    
##    rwdsN=np.array(rwds)[:,None]
##    obssN=np.stack(obss,axis=0)
##    actsN=np.array(acts)[:,None]
##    #turn into np arrays for training on
##    
##    
##    print(itr,i,model.train_on_batch({"obs":obssN,"act":actsN},rwdsN))
env.close()

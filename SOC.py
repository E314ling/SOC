import numpy as np
import tensorflow as tf
import keras
from keras import layers,losses

import matplotlib.pyplot as plt
'''
h - Finite time horizon
N - Number of discrete points in [0,h]
num_sim - Number of simulations
'''
class SOC_DQN():
    def __init__(self):
        self.num_epi  = 2001
        self.gam = 0.95

        self.d_a = 2
        self.d_x = 1

        self.h = 1
        self.N = 10

        self.sig = np.sqrt(2)

        self.dt = self.h/self.N

        self.time_axis = np.linspace(0,self.h,self.N)
        self.actions = list(np.linspace(-1,1,self.d_a))
        print(self.actions)
        self.X = np.zeros((self.num_epi, self.N,self.d_x))
       
        self.model = self.build_model()

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.reward_history = []
        self.action_index_history = []
        self.in_S_history = []

        self.batch_size = 64
        self.max_memory_length = 1000

    def check_in_S(self,x):
        '''
        x = [x_0,...,x_dim]
        '''
        if (self.d_x == 1):
            Sy = [3, 6]
            if (x<= Sy[1] and x >= Sy[0]):
                return True
            else:
                return False

        else:
            print('Einstellungen für Dimension nicht gefunden')

            return np.NAN
    # functions for the process and the cost
    def b(self,x, a):
        return x-a

    # f and g are for the accumulated costs
    def f(self,n,x,a, finished):
        if (finished):
            return 0
        else:
            return a**2

    def g(self, n,x,a):
        return 0

    def build_model(self):
        input = keras.Input((self.d_x,))

        x = layers.Dense(units= 128, activation= 'relu', name = "Dense_1")(input)
        x = layers.Dense(units= 128, activation= 'relu', name = "Dense_2")(x)

        output = layers.Dense(units= self.d_a, name = 'output_layer')(x)

        model = keras.Model(inputs = input, outputs = output, name = "SOC_Model")

        return model


    def TD_update_Replay(self,i_epi, n,in_S):

        x = self.X[i_epi][n]
        weights = self.model.get_weights()
        
        a = 100
        eps = 0.7* (1/ (1 + i_epi))
        alp = 0.01* (a / (a + i_epi))

        self.model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=alp),
        )

        Q_vals = self.model(x)
       
        # perform action and choose epsilon greedy
        if (np.random.rand() <= eps):
            rand_ind = np.random.choice(self.d_a)
            a = self.actions[rand_ind]
            action_val = Q_vals[0][rand_ind]

        else:
            
            #index of the action with highest prop
            action_index = np.argmin(Q_vals[0])
            a = self.actions[action_index]
            action_val = Q_vals[0][action_index]
        
        new_x = self.X[i_epi][n] + (self.X[i_epi][n] + a)* self.dt + self.sig * np.sqrt(self.dt)* np.random.normal()
        self.X[i_epi][n+1] = new_x

        self.action_history.append(a)
        self.state_history.append(x)
        self.state_next_history.append(new_x)
        self.reward_history.append(self.f(n,x,a, in_S))
        self.in_S_history.append(in_S)
        
        if (len(self.in_S_history) >= self.batch_size):
            indices = np.random.choice(range(len(self.in_S_history)), size=self.batch_size)
            # sample from replay buffer
            state_sample = np.asarray([self.state_history[i] for i in indices])
            state_next_sample = np.asarray([self.state_next_history[i] for i in indices])
            reward_sample = np.asarray([self.reward_history[i] for i in indices])

            action_sample = np.asarray([self.action_history[i] for i in indices])
            in_S_sample = np.asarray([self.in_S_history[i] for i in indices])

            # get the new best action
            new_Q_vals = self.model(state_next_sample)
            
            label = self.model.predict(state_sample)

            for i in range(self.batch_size):
                if (in_S_sample[i]):
                    y = reward_sample[i]
                else:
                    y = reward_sample[i] + self.gam * np.min(new_Q_vals[i])
                
                a_ind = self.actions.index(action_sample[i])
                label[i][a_ind] = y
                
            loss = self.model.fit(
                x = state_sample,
                y = label,
                verbose = '0',
                batch_size= self.batch_size,
                shuffle= False,
            )

            if len(self.in_S_history) > self.max_memory_length:
                        del self.reward_history[0]
                        np.delete(self.state_history,0)
                        np.delete(self.state_next_history,0)
                        del self.action_history[0]
                        del self.in_S_history[0]  
        return new_x

    def TD_update(self,i_epi, n,in_S):
        
        with tf.GradientTape() as tape:
            
            x = self.X[i_epi][n]

            weights = self.model.get_weights()
          
            a = 100
            eps = 0.3* (a/ (a + i_epi))

            alp = 0.01* (a / (a + i_epi))
            Q_vals = self.model(x)
        
            # perform action and choose epsilon greedy
            if (np.random.rand() <= eps):
                rand_ind = np.random.choice(self.d_a)
                a = self.actions[rand_ind]
                action_val = Q_vals[0][rand_ind]

            else:
                
                #index of the action with highest prop
                action_index = np.argmin(Q_vals[0])
                a = self.actions[action_index]
                action_val = Q_vals[0][action_index]
            
            new_x = self.X[i_epi][n] + (self.X[i_epi][n] - a)* self.dt + self.sig * np.sqrt(self.dt)* np.random.normal()
            self.X[i_epi][n+1] = new_x

            # get the new best action
            new_Q_vals = tf.stop_gradient(self.model(new_x))
            new_action_val = np.min(new_Q_vals[0])

            delta_t = self.f(n,x,a, in_S) + self.gam * new_action_val - action_val
            
            loss = delta_t**2 *np.ones(self.d_a)
    
        grad =  tape.gradient(loss, self.model.trainable_variables)
       
        opt = tf.keras.optimizers.Adam(learning_rate=alp)
       
        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        #new_weights = weights + alp*grad

        #self.model.set_weights(new_weights)

        return new_x

if __name__ == "__main__":
    DQN = SOC_DQN()

    stopping_time = np.zeros(DQN.num_epi)

    for i_epi in range(DQN.num_epi):
        print(i_epi)
        in_S = False
        for n in range(DQN.N-1):
            
            new_x = DQN.TD_update_Replay(i_epi,n, in_S)

            in_S = DQN.check_in_S(new_x)

            if (in_S == True):  
                stopping_time[i_epi] = DQN.dt * n
                
                break
                
        if (i_epi % 20 == 0 and i_epi != 0):

            fig, ax = plt.subplots(2)
            err = np.zeros((DQN.N,1))
            cost = np.zeros((DQN.N,1))
            for i in range(50):
                X = np.zeros((DQN.N,1))
                X_true = np.zeros((DQN.N,1))
                

                for n in range(DQN.N -1):
                    q_vals = DQN.model.predict(X[n])
                    value = np.min(q_vals[0])
                    best_action_index = np.argmin(q_vals[0])
                    a = DQN.actions[best_action_index]
                    rand = np.random.normal()

                    X[n+1] = X[n] + (X[n] + a)* DQN.dt + DQN.sig * np.sqrt(DQN.dt) * rand
                    X_true[n+1] = X_true[n] + (X[n] + 1)* DQN.dt + DQN.sig * np.sqrt(DQN.dt) * rand
                    
                if (i <= 1):
                    ax[0].set_xlim(0,DQN.h)
                    ax[0].plot(DQN.time_axis, X, color = 'blue', alpha = 0.4, linewidth = 0.8)
                    ax[0].plot(DQN.time_axis, X_true, color = 'red', alpha = 0.4, linewidth = 0.8)
                err +=  np.abs(X - X_true)
            ax[0].fill_between(DQN.time_axis,3,6 , color = 'green', alpha = 0.2)
            ax[0].set_title('Episode {}'.format(i_epi))
            
            err = (1/50)* err
            ax[1].plot(DQN.time_axis, err)

            fig.savefig('./Bilder_Episoden/Episode_{}.png'.format(i_epi))
            print('Bild für Episode {} fertig'.format(i_epi))
        
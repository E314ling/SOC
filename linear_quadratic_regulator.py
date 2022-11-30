import numpy as np
import tensorflow as tf
import keras
from keras import losses, layers
import matplotlib.pyplot as plt

class set_S():

    def __init__(self,d):
        self.dim = d
        self.r = 2

    def has_exit(self,x):
        if (np.linalg.norm(x) < self.r):
            return False
        else:
            return True

    def plot_S_and_X(self,X):
        '''
        X is a Trajektorie of the Process X.shape() = (N,dim)
        N- is the number of time steps
        '''

        t_axis = np.linspace(0, len(X), len(X))
        plt.plot(t_axis, X)
        plt.fill_between(t_axis, -self.r,self.r , color = 'green', alpha = 0.2)

class LQR_DQN():
    
    def __init__(self,load):
        
        self.num_epi  = 2001
        self.gam = 1

        self.d_a = 5
        self.d_x = 1

        self.h = 2
        self.N = 40

        self.sig = 0

        self.dt = self.h/self.N

        self.time_axis = np.linspace(0,self.h,self.N)
        self.actions = list(np.linspace(-2,2,self.d_a))
        print(self.actions)

        self.X = np.zeros((self.N,self.d_x))
       
        self.model = self.build_model()
        self.target_model = self.build_model()

        #stopping time model
        self.stopping_model = self.build_stopping_model()

        self.label = []

        self.stopping_label = []
        #stopping time history
        self.done_history = []


        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.reward_history = []
        self.action_index_history = []

        self.exit_history_label = []
        self.exit_state_history = []

        self.batch_size = 32
        self.exit_batch_size = 1
        self.max_memory_length = 5000
        
        self.loss_history = []
        self.av_loss_history = []
        self.temp_loss = []
        self.cost_history = []
        # setup set S
        self.set_S = set_S(self.d_x)

        if (load == True):
            self.model = tf.keras.models.load_model('LQR_Model.h5')
            self.target_model = tf.keras.models.load_model('LQR_Model.h5')
            self.stopping_model = tf.keras.models.load_model('Stopping_Time_Model.h5')

    def b(self,x, a):
        return x+a

    # f and g are for the accumulated costs
    def f(self,n,x,a):
       
        return a**2

    def g(self, n,x,a):
        return 0

    def build_model(self):
        input = keras.Input((self.d_x,))

        x = layers.BatchNormalization()(input)
        x = layers.Dense(units= 256, activation= 'relu', name = "Dense_1")(x)
        x = layers.Dense(units= 256, activation= 'relu', name = "Dense_2")(x)

        output = layers.Dense(units= self.d_a, name = 'output_layer')(x)

        model = keras.Model(inputs = input, outputs = output, name = "SOC_Model")
        model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025),
        )

        
        return model

    def build_stopping_model(self):
        input = keras.Input((self.d_x,))

        x = layers.Dense(units= 128, activation= 'relu', name = "ST_Dense_1")(input)
        x = layers.Dense(units= 128, activation= 'relu', name = "ST_Dense_2")(x)

        output = layers.Dense(units= 2, name = 'ST_output_layer', activation = 'softmax')(x)

        model = keras.Model(inputs = input, outputs = output, name = "Stopping_Model")
        model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025),
        )
        return model

    
    def TD_update_Replay(self,i_epi, n):

        x = self.X[n]
          
        a = 100
        eps = np.max([0.05, 0.7* (a/ (a + i_epi))])
        alp = 0.01#* (a / (a + i_epi))

        Q_vals = self.model(x)
       
        # perform action and choose epsilon greedy
        if (np.random.rand() <= eps):
            rand_ind = np.random.choice(self.d_a)
            a = self.actions[rand_ind]
            action_val = Q_vals[0][rand_ind]

        else:
            
            #index of the action with highest prop
            action_index = np.argmax(Q_vals[0])
            a = self.actions[action_index]
            action_val = Q_vals[0][action_index]
        
        new_x = self.X[n] + (self.X[n] + a)* self.dt + self.sig * np.sqrt(self.dt)* np.random.normal()
        self.X[n+1] = new_x

        new_Q_vals = self.target_model(new_x)

    
        self.action_history.append(a)
        self.state_history.append(x)
        self.state_next_history.append(new_x)
        self.reward_history.append(self.f(n,x,a))
        # calc if we have stopped the process
        done = self.set_S.has_exit(new_x)

        if (done ):
            self.label.append(self.f(n,x,a))
            self.stopping_label.append(np.array([0,1]))

            self.exit_history_label.append(np.array([0,1]))
            self.exit_state_history.append(new_x)
            self.exit_batch_size = np.min(np.array([self.batch_size, len(self.exit_history_label)]))
        else:
            self.label.append(self.f(n,x,a) + self.gam * np.max(new_Q_vals[0]))
            self.stopping_label.append(np.array([1,0]))

        

        self.done_history.append(done)
         
        if (i_epi % 5 == 0 and i_epi !=0):
            self.target_model.set_weights(self.model.get_weights())
        
        if (len(self.done_history) >= self.batch_size):
            indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
            # sample from replay buffer
            state_sample = np.asarray([self.state_history[i] for i in indices])
            state_next_sample = np.asarray([self.state_next_history[i] for i in indices])
            reward_sample = np.asarray([self.reward_history[i] for i in indices])

            action_sample = np.asarray([self.action_history[i] for i in indices])
            done_sample = np.asarray([self.action_history[i] for i in indices])

            label_sample =  np.asarray([self.label[i] for i in indices])
            stopping_label_sample =  np.asarray([self.stopping_label[i] for i in indices])

            loss = self.model.fit(
                x = state_sample,
                y = label_sample,
                verbose = '0',
                batch_size= self.batch_size,
                shuffle= False,
            )
            
            self.loss_history.append(loss.history['loss'])
            self.temp_loss.append(loss.history['loss'])

        
            if (len(self.loss_history) >= 20000):
                del self.loss_history[0]

            if (len(self.exit_history_label) >= self.exit_batch_size):
                indices = np.random.choice(range(len(self.exit_history_label)), size=self.exit_batch_size)
                # sample from replay buffer
                exit_sample = np.asarray([self.exit_state_history[i] for i in indices])
                exit_sample_label = np.asarray([self.exit_history_label[i] for i in indices])

                loss = self.stopping_model.fit(
                x = exit_sample,
                y = exit_sample_label,
                verbose = '0',
                batch_size= self.batch_size,
                shuffle= False,
                )
            

            if(len(self.exit_history_label) > self.max_memory_length ):
                del self.exit_history_label[0]
                del self.exit_state_history[0]
            if len(self.done_history) > self.max_memory_length:
                del self.reward_history[0]
                np.delete(self.state_history,0)
                np.delete(self.state_next_history,0)
                del self.action_history[0]
                del self.done_history[0]

        cost = self.f(n, x, a)
       
        return done, cost


if __name__ == "__main__":
    loading = True
    LQR = LQR_DQN(loading)

    num_epi = LQR.num_epi
    N = LQR.N
    r = LQR.set_S.r
    for i_epi in range(num_epi):
        cost = 0
        for n in range(N-1):
            done, c = LQR.TD_update_Replay(i_epi, n)
            cost += c

            if (done):
                
                LQR.av_loss_history.append(np.mean(np.array(LQR.temp_loss).reshape(len(LQR.temp_loss))))

                LQR.temp_loss = []
                break
       
        LQR.av_loss_history.append(np.mean(np.array(LQR.temp_loss).reshape(len(LQR.temp_loss))))

        LQR.temp_loss = []

        LQR.cost_history.append(c)
        if (i_epi % 1 == 0):
            print('episode', i_epi)
        if (i_epi % 100 == 0):
            LQR.model.save("LQR_Model.h5")
            LQR.stopping_model.save("Stopping_Time_Model.h5")

            num_sim = 1
            fig,ax = plt.subplots(2,3)
            for i in range(num_sim):
                X = np.zeros((LQR.N,1))
                Tau = np.zeros(LQR.N)
                True_Tau = np.zeros(LQR.N)
                Tau_done = []
                Tau_True = []

                for i_n in range(N-1):
                    
                    tau = LQR.stopping_model.predict(X[i_n])
                    done = np.argmax(tau)
                    if (done ==1):
                        if (len(Tau_done) == 0):
                            Tau_done.append(i_n)
                            Tau[i_n] = 1
                    
                    done_true = LQR.set_S.has_exit(X[i_n])
                    if (done_true):
                        if (len(Tau_True) == 0):
                            Tau_True.append(i_n)
                            True_Tau[i_n] = 1

                    Q_vals = LQR.model.predict(X[i_n])
                    action_index = np.argmax(Q_vals[0])
                    a = LQR.actions[action_index]
                    action_val = Q_vals[0][action_index]

                    X[i_n+1] = X[i_n] + (X[i_n] + a)* LQR.dt + LQR.sig * np.sqrt(LQR.dt)* np.random.normal()
                ax[0][0].plot(LQR.time_axis, X)
                ax[0][0].fill_between(LQR.time_axis,-r,r , color = 'green', alpha = 0.3)
                print(Tau_done)
                ax[0][0].set_ylim([-2*r, 2*r])
                ax[0][0].set_title('trajectories')
                
                ax[0][0].vlines(LQR.time_axis[Tau_True], ymin = -2*r, ymax = 2*r, color ='black',   alpha = 0.3, label = 'real exit')
                ax[0][0].vlines(LQR.time_axis[Tau_done], ymin = -2*r, ymax = 2*r,color = 'r',  alpha = 0.3, label = 'predicted exit')

                ax[0][0].legend()
                
                ax[0][1].plot(np.linspace(0,len(LQR.cost_history), len(LQR.cost_history)), LQR.cost_history)
                ax[0][1].set_title('cost')
                ax[0][2].plot(np.linspace(0,len(LQR.loss_history),len(LQR.loss_history)), LQR.loss_history)
                ax[0][2].set_title('loss')
               
                ax[1][2].plot(np.linspace(0,len(LQR.av_loss_history), len(LQR.av_loss_history)), LQR.av_loss_history)
                ax[1][2].set_title('Average_Loss per episode')
            num_x = 20
            x_axis = np.linspace(-2,2, num_x)
            V = np.zeros(num_x)
            A = np.zeros(num_x)

            for j in range(num_x):
                Q_vals = LQR.model.predict(np.array([x_axis[j]]))
                V[j] = np.max(Q_vals)
                a_ind = np.argmax(Q_vals)
                A[j] = LQR.actions[a_ind]
            
            ax[1][0].plot(x_axis, V)
            ax[1][0].set_title('value function')

            ax[1][1].plot(x_axis,A, label = 'control')
            ax[1][1].set_title('control')
            plt.show()
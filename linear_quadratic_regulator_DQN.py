import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt

import linear_quadratic_regulator_DP as LQR

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
       
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
       
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class experience_memory():

    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim+1), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.int32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim+1), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]


        self.buffer_counter += 1

class ActorCritic():

    def __init__(self, state_dim, action_dim,load_model):

        self.batch_size = 32
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.tau = 0.005
        self.lower_action_bound = -2
        self.upper_action_bound = 2

        self.action_space = np.linspace(self.lower_action_bound,self.upper_action_bound,40)
        self.num_a = len(self.action_space)

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic = tf.keras.models.load_model('./Models/LQR_DQM_model.h5')
            self.target_critic = tf.keras.models.load_model('./Models/LQR_DQM_model.h5')

        else:
            self.critic = self.get_critic_NN()
            self.target_critic = self.get_critic_NN()
            self.target_critic.set_weights(self.critic.get_weights())


        self.lr = 0.0001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr)

        self.eps = 1
        self.eps_decay = 1
        self.lr_decay = 0.9999
        
    def save_model(self):
        self.critic.save('./Models/LQR_DQM_model.h5')
    def update_lr(self):
        self.lr = self.lr * self.lr_decay
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr)

    def update_eps(self):
        self.eps = np.max([0.1,self.eps * self.eps_decay])

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # next_q_vals =  self.target_critic(next_state_batch, training=True)
        
        # target_vals = tf.reshape(tf.reduce_max(next_q_vals, axis =1),[self.batch_size, 1])
        
        # y = reward_batch + (1-done_batch)* self.gamma*target_vals
        
        # self.critic.fit(state_batch,
        # y = y,
        # verbose = 0,
        # batch_size =  self.batch_size)
        
        next_q_vals =  self.target_critic(next_state_batch, training=True)
            
        target_vals = tf.reshape(tf.reduce_min(next_q_vals, axis =1),[self.batch_size,1])
        y = reward_batch + (1-done_batch)* self.gamma*target_vals
        y = tf.reshape(y,[self.batch_size,])
        mask = tf.reshape(tf.one_hot(action_batch, self.num_a, dtype=tf.float32),[self.batch_size, self.num_a])

        with tf.GradientTape() as tape:
            
            critic_value = self.critic(state_batch)
     
            critic_pred = tf.reduce_sum(tf.multiply(critic_value,mask), axis=1)
            
            #critic_loss = tf.reduce_mean(tf.math.square(y-critic_value))
            critic_loss = losses.huber_loss(y,critic_pred)
        
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        #critic_grad, _ = tf.clip_by_global_norm(critic_grad, 5.0)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
    
    @tf.function
    def update_without_replay(self, state, action,reward, next_state, done):

        with tf.GradientTape() as tape:
            
            next_q_vals =  self.target_critic(next_state)
            
            target_vals = tf.reduce_min(next_q_vals, axis =1)
            
            y = tf.convert_to_tensor(reward) + tf.convert_to_tensor(self.gamma*(1-done))*target_vals
           
            critic_value = tf.reduce_min(self.critic(state), axis =1)
            critic_loss = tf.math.reduce_mean(tf.math.square(y- critic_value))
            
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))


    def learn_without_replay(self, state, action,reward, next_state, done):
        if done:
            done_num = 1
        else:
            done_num = 0
        self.update_without_replay(state, action,reward, next_state, done_num)

    def learn(self):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)
        
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        self.update(state_batch, action_batch, reward_batch, next_state_batch,done_batch)

    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim+1,))

        out = layers.BatchNormalization()(state_input)
    
        out = layers.Dense(512, activation = 'relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation = 'relu')(out)
        out = layers.Dense(self.num_a, kernel_initializer= last_init)(out)

        model = keras.Model(inputs = state_input, outputs = out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )
        return model

    def epsilon_greedy(self, state, eps):

        q_vals = self.critic(state)
        
        if (eps > np.random.rand()):
           
            rand_ind = np.random.choice(self.num_a)
            
            return self.action_space[rand_ind], rand_ind
        
        else:
            
            a_ind = tf.argmin(q_vals,axis = 1)
           
            return self.action_space[a_ind],a_ind
       
class CaseOne():

    def __init__(self):
        
       # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = 1
        self.B = 1
        self.sig = 0

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = 1
        self.f_B = 1

        # g(x) = D * ||x||^2
        self.D = 0

        self.num_episodes = 1000
        self.state_dim = 1
        self.action_dim = 1
        self.AC = ActorCritic(self.state_dim, self.action_dim,False)

        self.T = 1
        self.N = 20
        self.dt = self.T/self.N

        self.r = 0
        
    
    def f(self, n,x,a):
        
        return np.float32(self.f_B *np.linalg.norm(a)**2 + self.f_A*np.linalg.norm(x)**2)
        

    def g(self, n,x):
        
        return self.D * np.linalg.norm(x)**2
    
    def has_exit(self,x):
        if self.r == 0:
            return False
        else:
            if (np.linalg.norm(x) < self.r):
                return False
            else:
                return True

    def check_if_done(self,n,x):
        if n == self.N-2:
            return True
        else:
            if self.r == 0:
                return False
            else:
                if (np.linalg.norm(x) < self.r):
                    return False
                else:
                    return True

    def run_episodes(self,n_x,V_t,A_t, base):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        for ep in range(self.num_episodes):
            X = np.zeros((self.N), dtype= np.float32)
            X[0] = 2.5*np.random.rand() - 2.5
            n=0
            episodic_reward = 0
            while(True):
                state = np.array([n,X[n]], np.float32)
                state = tf.expand_dims(tf.convert_to_tensor(state),0)

                eps = self.AC.eps
                self.AC.update_eps()

                action,action_ind = self.AC.epsilon_greedy(state,eps)
                
                reward = self.f(n,X[n], action)

                X[n+1] = (self.A * X[n] + self.B * action) + self.sig * np.random.normal()
                
                new_state = np.array([n+1,X[n+1]], np.float32)
                new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)

                done = self.check_if_done(n,new_state)
                
                episodic_reward += reward

                # warm up
                if (ep >= 100):
                    self.AC.eps_decay = 0.999
                    self.AC.buffer.record((state,action_ind,reward, new_state, done))
                    self.AC.learn()
                    self.AC.update_target(self.AC.target_critic.variables, self.AC.critic.variables)
                self.AC.update_lr()
                
                n += 1
                if (done):
                    break
                
                
                
            
            if (ep % 100 == 0 and ep != 0):
                plt.plot(avg_reward_list)
                plt.hlines(base,xmin = 0, xmax = len(avg_reward_list), color = 'black', label = 'baseline: {}'.format(base))
                plt.xlabel("Episode")
                plt.ylabel("Avg. Epsiodic Reward")
                plt.legend()
                plt.show()
                self.plots(n_x,V_t,A_t)

            if (ep >= 100):
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                print("Episode * {} * Avg Reward is ==> {}, eps ==> {}, lr ==> {}".format(ep, avg_reward,self.AC.eps, self.AC.lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")

        self.AC.save_model()
        plt.show()
        self.plots(n_x,V_t,A_t)
    
    def plots(self,n_x,V_t,A_t):
        
        n_a = 20

        state_space = np.linspace(-2,2, n_x)
        
        fig, ax = plt.subplots(3)
 
        V = np.zeros(n_x)

        P = np.zeros(n_x)

        for ix in range(n_x):
            
            state = np.array([0,state_space[ix]])
            q_vals = self.AC.critic(tf.expand_dims(tf.convert_to_tensor(state),0))
            
            a_ind = tf.argmin(q_vals[0])
            
            P[ix] = self.AC.action_space[a_ind]

            v = tf.reduce_min(q_vals)

            V[ix] = v
        
        ax[0].plot(state_space, V)
        ax[0].set_title('value function')
        ax[0].plot(state_space, V_t[0], label = 'true_solution')
        ax[0].legend()

        ax[1].plot(state_space, P)
        ax[1].set_title('policy function')
        ax[1].plot(state_space, A_t[0],label = 'true_solution')
        ax[1].legend()

        X = np.zeros(self.N)

        for i in range(self.N-1):
            state = np.array([i,X[i]])
            q_vals = self.AC.critic(tf.expand_dims(tf.convert_to_tensor([i,X[i]]),0))
            a_ind = tf.argmin(q_vals[0])
            a = self.AC.action_space[a_ind]

            X[i+1] = X[i] + (X[i] + a)*self.dt + self.sig * np.sqrt(self.dt) * np.random.normal()
        ax[2].plot(np.linspace(0, self.T, self.N), X)
        ax[2].set_ylim([-5,5])
        ax[2].fill_between(np.linspace(0, self.T, self.N),-self.r,self.r , color = 'green', alpha = 0.3)
        plt.show()

if __name__ == "__main__":

   
    
    lqr = CaseOne()
    n_x = 40
    
    V_t,A_t,base = LQR.Solution(lqr).create_solution(n_x)
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x,V_t,A_t, base)
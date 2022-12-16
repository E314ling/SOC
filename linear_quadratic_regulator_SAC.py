import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import linear_quadratic_regulator_DP as LQR

import tensorflow_probability as tfp

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
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim+1), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

        # saves the parameters that are estimated by the actor
        #self.param_buffer = np.zeros((self.buffer_capacity, 2), dtype= np.float32)

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
        #self.param_buffer[index] = obs_tuple[5]

        self.buffer_counter += 1


class ActorCritic():

    def __init__(self, state_dim, action_dim, load_model):

        self.batch_size = 512
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.005

        # for the entropy regulization
        self.temp = 0.4

        self.lower_action_bound = -4
        self.upper_action_bound = 4

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic_1 = tf.keras.models.load_model('./Models/LQR_SAC_critic_1.h5')
            self.target_critic_1 = tf.keras.models.load_model('./Models/LQR_SAC_critic_1.h5')
            self.critic_2 = tf.keras.models.load_model('./Models/LQR_SAC_critic_1.h5')
            self.target_critic_2 = tf.keras.models.load_model('./Models/LQR_SAC_critic_1.h5')

            self.actor = tf.keras.models.load_model('./Models/LQR_SAC_actor.h5')
            
        else:
            self.critic_1 = self.get_critic()
            self.target_critic_1 = self.get_critic()
            #self.target_critic_1.set_weights(self.critic_1.get_weights())

            self.critic_2 = self.get_critic()
            self.target_critic_2 = self.get_critic()
            #self.target_critic_2.set_weights(self.critic_2.get_weights())
            
            self.actor = self.get_actor()
            
        self.critic_lr = 0.001
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)

        self.actor_lr = 0.001
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
   
        self.lr_decay = 0.9999

        self.update = 1
       
    def save_model(self):
        self.critic_1.save('./Models/LQR_SAC_critic_1.h5')
        self.critic_2.save('./Models/LQR_SAC_critic_2.h5')
        self.actor.save('./Models/LQR_SAC_actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def get_critic(self):
      input_state = keras.Input(shape =(self.state_dim+1,))
      input_action = keras.Input(shape =(self.action_dim,))
      input = tf.concat([input_state, input_action], axis = 1 )
      d1 = layers.Dense(64, activation = 'relu')(input)
      d2 = layers.Dense(64, activation = 'relu')(d1)
      out = layers.Dense(1)(d2)
      model = keras.Model(inputs =  [input_state, input_action], outputs = out)
      return model

    
    def get_actor(self):
      input = keras.Input(shape = (self.state_dim+1,))

      d1 = layers.Dense(64, activation = 'relu')(input)
      d2 = layers.Dense(64, activation = 'relu')(d1)

      mu = layers.Dense(self.action_dim)(d2)
      log_std = layers.Dense(self.action_dim)(d2)

      model = keras.Model(inputs = input, outputs = [mu, log_std])

      return model

    def transform_actor(self, mu, log_std):
      clip_log_std = tf.clip_by_value(log_std, -20,2)

      std =  tf.exp(clip_log_std)
      
      dist = tfp.distributions.Normal(mu, std,allow_nan_stats=False)

      action_ = dist.sample()
      action = tf.tanh(action_)
    
      log_pi = dist.log_prob(action_)
      log_pi_a = log_pi - tf.reduce_sum(tf.math.log(self.upper_action_bound*(1-action**2) + 1e-6), axis = 1, keepdims = True)
      action = self.upper_action_bound*action
      return action, log_pi_a
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        mu, std = self.actor(next_state_batch)
        pi_a ,log_pi_a = self.transform_actor(mu, std)
                                              
        target_1 = self.target_critic_1([next_state_batch, pi_a])
        target_2 = self.target_critic_1([next_state_batch, pi_a])
        
        target_vals =  tf.minimum(target_1, target_2)

        # soft target
      
        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*(target_vals - self.temp*log_pi_a))

        with tf.GradientTape() as tape1:
            
            critic_value_1 = self.critic_1([state_batch, action_batch])
            
            critic_1_loss = losses.MSE(y,critic_value_1 )

        with tf.GradientTape() as tape2:
            
            critic_value_2 = self.critic_2([state_batch, action_batch])
            
            critic_2_loss = losses.MSE(y,critic_value_2)

        critic_1_grad = tape1.gradient(critic_1_loss, self.critic_1.trainable_variables)   
        self.critic_optimizer_1.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))

        critic_2_grad = tape2.gradient(critic_2_loss, self.critic_2.trainable_variables)   
        self.critic_optimizer_2.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))

       

    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            mu,std = self.actor(state_batch)
            pi_a, log_pi_a = self.transform_actor(mu, std)
            critic_value_1 = self.critic_1([state_batch, pi_a])
            critic_value_2 = self.critic_2([state_batch, pi_a])
            min_critic = tf.minimum(critic_value_1, critic_value_2)

            soft_q = min_critic - self.temp * log_pi_a

            # for maximize add a minus '-tf.math.reduce_mean(soft_q)'
            actor_loss = -tf.math.reduce_mean(soft_q)
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
     
     
    def learn(self,episode):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        #param_batch = tf.convert_to_tensor(self.buffer.param_buffer[batch_indices])

        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (episode % self.update == 0 and episode != 0):
            self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)
        
class CaseOne():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = 1
        self.B = 1
        self.sig = 1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = 1
        self.f_B = 1

        # g(x) = D * ||x||^2
        self.D = 1

        self.num_episodes = 2200
        self.state_dim = 1
        self.action_dim = 1
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 20
        self.dt = self.T/self.N

        self.r = 0
        self.dashboard_num = 500
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P = []

    def f(self, n,x,a):
        return np.float32(self.B *np.linalg.norm(a)**2 + self.A*np.linalg.norm(x)**2)
        

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
        if n == self.N-1:
            return True
        else:
            if self.r == 0:
                return False
            else:
                if (np.linalg.norm(x) < self.r):
                    return False
                else:
                    return True
    
    def run_episodes(self, n_x,V_t,A_t, base):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        X = np.zeros((self.N), dtype= np.float32)
        X[0] = 2.5*np.random.rand() - 2.5
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                state = np.array([n,X[n]], np.float32)
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)

                if (done):
                    reward = -self.g(n,X[n])
                    mu, std = self.AC.actor(state)
                    

                    action_,_ = self.AC.transform_actor(mu,std)
                    action = action_.numpy()[0]
                    X = np.zeros((self.N), dtype= np.float32)
                    X[0] = 2.5*np.random.rand() - 2.5
                    new_state = np.array([0,X[0]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:
                    mu, std = self.AC.actor(state)
                    

                    action_,_ = self.AC.transform_actor(mu,std)
                    action = action_.numpy()[0]
                    reward = -self.f(n,X[n], action)
                    #X[n+1] =  X[n] + (X[n] + action)*self.dt + np.sqrt(self.sig*self.dt)  * np.random.normal()
                    X[n+1] =  (X[n] + action) + self.sig  * np.random.normal()

                    new_state = np.array([n+1,X[n+1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)

                self.AC.buffer.record((state,action,reward, new_state, done))
                 
                episodic_reward += reward
                # warm up
                if (ep >= 200):
                    self.AC.learn(n)
                    
                    if (n % self.AC.update == 0 and n != 0):
                        self.AC.update_target(self.AC.target_critic_1.trainable_variables, self.AC.critic_1.trainable_variables)
                        self.AC.update_target(self.AC.target_critic_2.trainable_variables, self.AC.critic_2.trainable_variables)
                        
                        self.AC.update_lr()
                        
                if(done):
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0 and ep > 200):
                self.dashboard(n_x,V_t,A_t,avg_reward_list)
            
            if (ep >= 200):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-100:])
                print("Episode * {} * Avg Reward is ==> {},  actor_lr ==> {}".format(ep, avg_reward,  self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,V_t,A_t,avg_reward_list)
        
    def dashboard(self,n_x,V_t,A_t,avg_reward_list):

        state_space = np.linspace(-2,2, n_x)
        
        fig, ax = plt.subplots(2,3)
        
        V1 = np.zeros(n_x)
        P = np.zeros(n_x)
        stddev = np.zeros(n_x)
        V2 = np.zeros(n_x)
        t0 = 1

        ax[0][0].plot(avg_reward_list)
        ax[0][0].set_xlabel('Episode')
        ax[0][0].set_ylabel('Avg. Epsiodic Reward')
        ax[0][0].hlines(base,xmin = 0, xmax = len(avg_reward_list), color = 'black', label = 'baseline: {}'.format(np.round(base,)))
        ax[0][0].set_title('baseline value: {} \n Avg 100 Episodes: {}'.format(np.round(base,2), np.round(avg_reward_list[-1],2)))

        for ix in range(n_x):
           state = np.array([t0,state_space[ix]])
           
           mu, std = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
           
           action_arr = np.zeros(20)
           for ia in range(20):
            action_,_ = self.AC.transform_actor(mu,std)
           
            action = action_[0].numpy()
            action_arr[ia] = action
           
           print('action_arr', action_arr)
           action = np.mean(action_arr)
           print(action)
           std = np.std(action_arr)
           print(std)
           action_ = tf.expand_dims(tf.convert_to_tensor(action),0)

           P[ix] = action
           stddev[ix] = 2*std

           v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action_]), self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action_])
           
           V1[ix] = v1
          
           V2[ix] = v2

        error_v_1 = np.abs(V_t[t0] - V1)
        error_v_2 = np.abs(V_t[t0] - V2)
        self.mean_abs_error_v1.append(np.mean(np.abs(V_t[t0] - V1)))
        self.mean_abs_error_v2.append(np.mean(np.abs(V_t[t0] - V2)))

        error_P = np.abs(A_t[t0] - P)
        self.mean_abs_error_P.append(np.mean(np.abs(A_t[t0] - P)))
        
        ax[0][1].plot(state_space, V1, label = 'approx value function 1')
        ax[0][1].plot(state_space, V2,label = 'approx value function 1')
        ax[0][1].plot(state_space, -V_t[t0], label = 'true value function', color = 'black')
        
        ax[0][1].set_title('value function t = {}'.format(t0))

        ax[0][2].plot(state_space, P, label = 'policy function approximation', color='blue')
        ax[0][2].fill_between(state_space, P+stddev, P -stddev, facecolor='blue', alpha = 0.3)
        ax[0][2].plot(state_space, A_t[t0],label = 'true policy function', color = 'black')
        
        ax[0][2].set_title('policy function t = {}'.format(t0))

        ###########################################################
        episode_ax = self.dashboard_num * np.linspace(1,len(self.mean_abs_error_P), len(self.mean_abs_error_P))

        ax[1][1].scatter(episode_ax,  self.mean_abs_error_v1, label = 'Mean Abs Error 1: {}'.format(np.round(np.mean(error_v_1),2)))
        ax[1][1].scatter(episode_ax,  self.mean_abs_error_v2, label = 'Mean Abs Error 1: {}'.format(np.round(np.mean(error_v_2),2)))
        ax[1][1].plot(episode_ax,  self.mean_abs_error_v1)
        ax[1][1].plot(episode_ax,  self.mean_abs_error_v2)

        ax[1][1].set_title('abs error value function t = {} '.format(t0))
        ax[1][1].legend()
        ax[1][1].set_xlim([np.min(episode_ax)-1, np.max(episode_ax)+1])

        ax[1][2].scatter(episode_ax, self.mean_abs_error_P, label = 'Mean Abs Error: {}'.format(np.round(np.mean(error_P),2)))
        ax[1][2].plot(episode_ax, self.mean_abs_error_P)

        ax[1][2].set_title('abs error policy function t = {} '.format(t0))
        ax[1][2].legend()
        ax[1][2].set_xlim([np.min(episode_ax)-1, np.max(episode_ax)+1])
        # terminal value function
        for ix in range(n_x):
           state = np.array([self.N-1,state_space[ix]])
           mu,std = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
           action_arr = np.zeros(20)
           for ia in range(20):
               action_,_ = self.AC.transform_actor(mu,std)
           
               action = action_[0].numpy()
               action_arr[ia] = action
           
           action = np.mean(action_arr)
           std = np.sqrt(np.var(action_arr))
           action_ = tf.expand_dims(tf.convert_to_tensor(action),0)
           P[ix] = action

           v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action_]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action_])
           
           V1[ix] = v1
          
           V2[ix] = v2
        ax[1][0].plot(state_space, V1, label = 'approx value function 1')
        ax[1][0].plot(state_space, V2,label = 'approx value function 1')
        ax[1][0].plot(state_space, -V_t[self.N-1], label = 'true value function', color = 'black')
        
        ax[1][0].set_title('terminal value function t = {}'.format(self.N-1))
        fig.set_size_inches(w = 15, h= 7.5)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        #fig.savefig('.\Bilder_Episoden\TD3_Dashboard_Episode_{}'.format(len(avg_reward_list)))
        plt.show()
    
   

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 40
    
    V_t,A_t,base = LQR.Solution(lqr).create_solution(n_x)
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x,V_t,A_t, -base)
    
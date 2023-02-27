import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import linear_quadratic_regulator_DP as LQR
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.1, dt=1e-2, x_initial=None):
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
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
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

    def __init__(self, state_dim, action_dim, load_model):

        self.batch_size = 512
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.lower_action_bound = -2
        self.upper_action_bound = 2

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic = tf.keras.models.load_model('./Models/LQR_TD3_critic.h5')
            self.target_critic = tf.keras.models.load_model('./Models/LQR_TD3_critic.h5')


            self.actor = tf.keras.models.load_model('./Models/LQR_DDPG_Actor.h5')
            self.target_actor = tf.keras.models.load_model('./Models/LQR_DDPG_Actor.h5')
        else:
            self.critic = self.get_critic_NN()
            self.target_critic = self.get_critic_NN()
            self.target_critic.set_weights(self.critic.get_weights())

           
            
            self.actor = self.get_actor_NN()
            self.target_actor = self.get_actor_NN()
            self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
       
        self.actor_lr = 0.0001
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 0.1
        self.var_decay = 0
        self.lr_decay = 1

       
    def save_model(self):
        self.critic.save('./Models/LQR_TD3_critic.h5')
        
        self.actor.save('./Models/LQR_TD3_actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def update_var(self):
        self.var = np.max([0.1,self.var * self.var_decay])
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        target_actions = self.target_policy(next_state_batch) 
        target_1, target_2 = self.target_critic([next_state_batch, target_actions])
        
        target_vals =  tf.minimum(target_1, target_2)

        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*target_vals)

        with tf.GradientTape() as tape:
            tape.watch(self.critic.trainable_variables)
            
            critic_value_1, critic_value_2 = self.critic([state_batch, action_batch])
            
            critic_loss = losses.MSE(y, critic_value_1) + losses.MSE(y, critic_value_2)

        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))


       

    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions = self.actor(state_batch)
            critic_value_1,critic_value_2 = self.critic([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value_1)
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

         
    def learn(self,episode):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)
        
        batch_indices = np.append(batch_indices, record_range)
        
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (episode % 2 == 0 and episode != 0):
            self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

        
    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim+1,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(128, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(128, activation = 'relu')(out_1)
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        out_2 = layers.Dense(128, activation = 'relu')(input)
        out_2 = layers.BatchNormalization()(out_1)
        out_2 = layers.Dense(128, activation = 'relu')(out_2)
        out_2 = layers.Dense(1, kernel_initializer= last_init)(out_2)



        model = keras.Model(inputs = [state_input, action_input], outputs = [out_1,out_2])

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim+1,))
       
        out = layers.Dense(128, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(128, activation="relu")(out)
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)
      
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = self.actor(state)
       
        #noice = self.noice_Obj()
      
        sampled_actions = sampled_actions + tf.random.normal(shape = sampled_actions.get_shape(),mean= 0, stddev=self.var, dtype= tf.float32)

        legal_action = tf.clip_by_value(sampled_actions, clip_value_min= -1, clip_value_max=1)
        return legal_action
        #return [np.squeeze(sampled_actions)]
    
    @tf.function
    def target_policy(self, state):
        sampled_actions = self.target_actor(state)
        
        #noice = self.noice_Obj()
        noice = tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = 0.2, dtype = tf.float32)
        noice = tf.clip_by_value(noice, -0.5, 0.5)
        sampled_actions = sampled_actions + noice
        
        legal_action = tf.clip_by_value(sampled_actions, clip_value_min = -1, clip_value_max =1)

        return legal_action
        
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

        self.num_episodes = 5100
        self.state_dim = 1
        self.action_dim = 2
        
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 50
        self.dt = self.T/self.N

        self.dashboard_num = 100
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P = []

        
        self.alpha = 0.5
        self.lam = 0.2
        self.beta = 0.9
        self.r = 0.1
        self.mu = 0
        
        self.k = ((self.alpha/(1-self.alpha))* (0.5*((self.r-self.mu)**2) / (self.sig**2 * (self.alpha -1))) - self.r + (self.beta + self.lam)/self.alpha )*(self.alpha -1)

    def f(self, n,x,a):
        c = a[0]
        return self.dt*(np.exp(-(self.beta + self.lam)*n*self.dt) * (c**self.alpha)/self.alpha)
        

    def g(self, n,x):
        return (x**self.alpha)/self.alpha
    
   
    def check_if_done(self,n,x):
        if n == self.N-1:
            return True
        else:
            return False
    
    def run_episodes(self, n_x):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        X = np.zeros((self.N), dtype= np.float32)
        X[0] = 2*np.random.rand() +0.5
        for ep in range(self.num_episodes):
            
            
            episodic_reward = 0
           
            state = np.array([X[n]], np.float32)
            
            state = tf.expand_dims(tf.convert_to_tensor(state),0)
            
            
            
            if (ep <= 100):
                action = 2* np.random.rand(2) -1
                action[0] = 0.5*(action[0] +1) + 10e-1

                action_env = self.AC.upper_action_bound * action
            else:
                action = self.AC.policy(state).numpy()[0]
                action[0] = 0.5*(action[0] +1) + 10e-1
                action_env = self.AC.upper_action_bound*action

            
                        
            

            reward = self.f(n,X[n], action_env)
            c = action_env[0]
            pi = action_env[1]
            X[n+1] =  X[n] + (X[n]*(self.r +pi*(self.mu -self.r)) - c)*self.dt + self.sig*np.sqrt(self.dt)*pi*X[n]  * np.random.normal()
            if (X[n+1] <= 0):
                X[n+1] = 0.1
            new_state = np.array([n+1,X[n+1]], np.float32)
            new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
               
                
                
            self.AC.buffer.record((state.numpy()[0],action,reward, new_state.numpy()[0]))
                
            episodic_reward += reward
            
            if (ep >= 100):
                self.AC.learn(ep)
                
                if (ep % 2 == 0 and ep != 0):
                    self.AC.update_target(self.AC.target_critic.variables, self.AC.critic.variables)
                    self.AC.update_target(self.AC.target_actor.variables, self.AC.actor.variables)
                    self.AC.update_lr()
                    self.AC.update_var()

            if (ep % self.dashboard_num == 0):
                self.dashboard(n_x,avg_reward_list,self.AC)
            
            if (ep >= 100):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-100:])
                print("Episode * {} * Avg Reward is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
                if (avg_reward == np.nan):
                    print('done,reward,X[n], action, action_env', done,reward,X[n],action, action_env)
                # warm up
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,avg_reward_list,self.AC)
    
    def H_f(self,x,t):
        if (self.k == 0):
            return 1 + self.T -t
        else:
            return np.exp(self.k *(self.T -t))
    
    def H_f_prime(self,x,t):
        if (self.k == 0):
            return -1
        else:
            return -self.k*np.exp(self.k *(self.T -t))
    
    def H_x(self,x,t):

        return (self.H_f(x,t)**(1-self.alpha))*(x**(self.alpha-1))

    def H_xx(self,x,t):
        return (self.alpha -1) * (self.H_f(x,t)**(1-self.alpha))*(x**(self.alpha-2))
    def H_t(self,x,t):
        return (1- self.alpha)* self.H_f(x,t) *self.H_f_prime(x,t) * (x**(self.alpha)/ self.alpha)

    def value_fct(self,x,t):
        
        return (self.H_f(x,t)**(1-self.alpha))* (x**(self.alpha)/ self.alpha)

    def opt_cosumption(self, x,t):
        return self.H_x(x,t)**(1/(self.alpha))
    
    def opt_invest(self, x,t):
        return  -((self.mu -self.r)* (self.H_x(x,t) / self.H_xx(x,t)) )/self.sig**2

    def dashboard(self,n_x,avg_reward_list,AC:ActorCritic):

        state_space = np.linspace(0.1,4, n_x)
        
        fig, ax = plt.subplots(2,3)
        
        V1 = np.zeros(n_x)
        P_consume = np.zeros(n_x)
        P_invest = np.zeros(n_x)
        V2 = np.zeros(n_x)
        V_true = np.zeros(n_x)

        optimal_consume = np.zeros(n_x)
        optimal_invest= np.zeros(n_x)
        
        t0 = 1

        ax[0][0].plot(avg_reward_list)
        ax[0][0].set_xlabel('Episode')
        ax[0][0].set_ylabel('Avg. Epsiodic Reward')
        ax[0][0].set_xlim([0,self.num_episodes])
       
        if (len(avg_reward_list)> 0):
            ax[0][0].set_title('Avg 100 Episodes: {}'.format( np.round(avg_reward_list[-1],2)))

        for ix in range(n_x):
           state = np.array([t0,state_space[ix]])
           action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
          
           v1,v2 = self.AC.critic([tf.expand_dims(tf.convert_to_tensor(state),0),action])
           action = AC.upper_action_bound*action.numpy()[0]
           V1[ix] = v1
          
           V2[ix] = v2

           V_true[ix] = self.value_fct(state_space[ix], self.dt*t0)
           P_consume[ix] = action[0]
           P_invest[ix] = action[1]
           optimal_consume[ix] = self.opt_cosumption(state_space[ix], self.dt*t0)
           optimal_invest[ix] = self.opt_invest(state_space[ix], self.dt*t0)

        error_v_1 = (V_true - V1)**2
        error_v_2 = (V_true - V2)**2
        self.mean_abs_error_v1.append(np.mean((V_true - V1)**2))
        self.mean_abs_error_v2.append(np.mean((V_true - V2)**2))

        error_P = np.abs(optimal_consume - P_consume)
        self.mean_abs_error_P.append(np.mean(np.abs(optimal_consume - P_consume)))
        
        ax[0][1].plot(state_space, V1, label = 'approx value function 1')
        ax[0][1].plot(state_space, V2,label = 'approx value function 1')
        ax[0][1].plot(state_space,V_true, label = 'true value function', color = 'black')
        
        ax[0][1].set_title('value function t = {}'.format(t0))
       

        ax[0][2].plot(state_space, P_consume, label = 'policy function approximation')
        ax[0][2].plot(state_space, optimal_consume,label = 'true policy function', color = 'black')
        
        ax[0][2].set_title('consume_function function x = 0')
        

        ###########################################################
        episode_ax = self.dashboard_num * np.linspace(1,len(self.mean_abs_error_P), len(self.mean_abs_error_P))

        #ax[1][1].scatter(episode_ax,  self.mean_abs_error_v1, label = 'MSE 1: {}'.format(np.round(np.mean(error_v_1),2)))
        #ax[1][1].scatter(episode_ax,  self.mean_abs_error_v2, label = 'MSE 2: {}'.format(np.round(np.mean(error_v_2),2)))
        ax[1][1].plot(episode_ax,  self.mean_abs_error_v1, label = 'MSE 1: {}'.format(np.round(np.mean(error_v_1),4)))
        ax[1][1].plot(episode_ax,  self.mean_abs_error_v2, label = 'MSE 2: {}'.format(np.round(np.mean(error_v_2),4)))

        ax[1][1].set_title('MSE value function t = {} '.format(t0))
        ax[1][1].legend()
        ax[1][1].set_xlim([0,self.num_episodes])

        #ax[1][2].scatter(episode_ax, self.mean_abs_error_P, label = 'MSE: {}'.format(np.round(np.mean(error_P),2)))
        ax[1][2].plot(episode_ax, self.mean_abs_error_P, label = 'MSE: {}'.format(np.round(np.mean(error_P),4)))

        ax[1][2].set_title('MSE policy function t = {} '.format(t0))
        ax[1][2].legend()
        ax[1][2].set_xlim([0,self.num_episodes])
        # terminal value function
        

        ax[1][0].plot(state_space, P_invest, label = 'policy function approximation')
        ax[1][0].plot(state_space, optimal_invest,label = 'true policy function', color = 'black')
        
        ax[1][0].set_title('invest function function t = {} '.format(t0))
       

        fig.set_size_inches(w = 15, h= 7.5)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig('.\Bilder_Episoden\OCI_TD3_Dashboard_Episode_{}'.format(len(avg_reward_list)))
        
    

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 40
    
    lqr.run_episodes(n_x)
    
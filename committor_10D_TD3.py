import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
#import linear_quadratic_regulator_DP as LQR



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
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
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

        self.batch_size = 128
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.tau_actor = 0.001
        self.lower_action_bound = -5
        self.upper_action_bound = 5

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic_1 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_1.h5')
            self.target_critic_1 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_1.h5')

            self.critic_2 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_2.h5')
            self.target_critic_2 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_2.h5')


            self.actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_Actor.h5')
            self.target_actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_Actor.h5')
        else:
            self.critic_1 = self.get_critic_NN()
            self.target_critic_1 = self.get_critic_NN()
            #self.target_critic_1.set_weights(self.critic_1.get_weights())

            self.critic_2 = self.get_critic_NN()
            self.target_critic_2 = self.get_critic_NN()
            #self.target_critic_2.set_weights(self.critic_2.get_weights())

           
            
            self.actor = self.get_actor_NN()
            self.target_actor = self.get_actor_NN()
            self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0001
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
       
        self.actor_lr = 0.0001
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 1
        self.var_target = 0.2
        self.var_min = 0.1
        self.var_decay = 0.999
        self.lr_decay = 1
        

        self.update_frames = 2
        self.actor_loss = []
        self.critic_1_loss = []
        self.critic_2_loss = []
       
    def save_model(self):
        self.critic_1.save('./Models/LQR_2D_TD3_critic_1.h5')
        self.critic_2.save('./Models/LQR_2D_TD3_critic_2.h5')
        
        self.actor.save('./Models/LQR_2D_TD3_actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def update_var(self):
        self.var = np.max([self.var_min,self.var * self.var_decay])
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        target_actions = self.target_policy(next_state_batch) 
        target_1, target_2 = self.target_critic_1([next_state_batch, target_actions]),self.target_critic_2([next_state_batch, target_actions])
        
        target_vals =  tf.minimum(target_1, target_2)

        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*target_vals)

        with tf.GradientTape() as tape1:
            tape1.watch(self.critic_1.trainable_variables)
            
            critic_value_1 = self.critic_1([state_batch, action_batch])
            
            critic_loss_1 = tf.reduce_mean((y-critic_value_1)**2)

        with tf.GradientTape() as tape2:
            tape2.watch(self.critic_2.trainable_variables)
            
            critic_value_2 = self.critic_2([state_batch, action_batch])
            
            critic_loss_2 = tf.reduce_mean((y-critic_value_2)**2)

        
        critic_grad_1 = tape1.gradient(critic_loss_1, self.critic_1.trainable_variables) 
        self.critic_optimizer_1.apply_gradients(zip(critic_grad_1, self.critic_1.trainable_variables))

        critic_grad_2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(zip(critic_grad_2, self.critic_2.trainable_variables))

        return critic_loss_1,critic_loss_2

    @tf.function
    def update_actor(self, state_batch):
        
        with tf.GradientTape() as tape:
            
            actions = self.actor(state_batch)
            critic_value_1,critic_value_2 = self.critic_1([state_batch, actions]), self.critic_2([state_batch, actions])
            actor_loss = tf.math.reduce_mean(tf.minimum(critic_value_1,critic_value_2))
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        return actor_loss

   
    def learn(self,frame_num):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        critic_loss_1,critic_loss_2 = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        
        self.critic_1_loss.append(critic_loss_1)
        self.critic_2_loss.append(critic_loss_2)

        if (frame_num % self.update_frames == 0 and frame_num != 0):
            actor_loss = self.update_actor(state_batch)
            self.actor_loss.append(actor_loss)
           

    @tf.function
    def update_target_critic(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)
    @tf.function
    def update_target_actor(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau_actor *b + (1-self.tau_actor) *a)
        
    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(256, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(256, activation = 'relu')(out_1)
        
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
       
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(256, activation="relu")(out)
       
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        #outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = self.actor(state)
      
        sampled_actions = sampled_actions + tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = self.var, dtype = tf.float32)

        legal_action = tf.clip_by_value(sampled_actions, clip_value_min= -1, clip_value_max =1)
        

        return legal_action
        
    
    @tf.function
    def target_policy(self, state):
        sampled_actions = self.target_actor(state)
        
        #noice = self.noice_Obj()
        noice = tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = self.var_target, dtype = tf.float32)
        noice = tf.clip_by_value(noice, -0.5, 0.5)
        sampled_actions = sampled_actions + noice
        
        legal_action = tf.clip_by_value(sampled_actions, clip_value_min = -1, clip_value_max =1)

        return legal_action
        
class CaseOne():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = 0*np.identity(10)
        self.B = np.identity(10)
        self.sig = 1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.discrete_problem = False
        self.f_A = 0*np.identity(10)
        self.f_B = np.identity(10)

        # g(x) = D * ||x||^2
        self.D = 0*np.identity(10)

        self.num_episodes = 5100
        self.state_dim = 10
        self.action_dim = 10
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 1000
        self.dt = self.T / self.N

        self.r1 = 1

        self.r2 = 3

        self.dashboard_num = 100

        self.change_V1 = []
        self.change_V2 = []
        
    def f(self, n,x,a):

        y1= np.dot(np.transpose(x), self.f_A)
        
        y2 = np.dot(np.transpose(a), self.f_B)

        if (self.discrete_problem):
            return np.float32(np.dot(y1,x) + np.dot(y2,a))
        else:
            return 0.5*self.dt*np.linalg.norm(a)**2

    def g(self, n,x, exit_C):
        eps = 10e-4
        if (exit_C):
            const = 1
        else:
            const =0
        
        return -np.log(const + eps)
    
    def free_energy(self,x):

        r = np.linalg.norm(x)
        h = (self.r1**2  - r**(2-self.state_dim)*self.r1**self.state_dim) /(self.r1**2 - self.r2**(2-self.state_dim)*self.r1**self.state_dim)
        return -np.log( h + 10e-4)

    def start_state(self):
        r1 = self.r1 + 0.1
        r2 = self.r2 - 0.1
        rand_ind = np.random.choice(range(10))
        x = (r2-r1)*np.random.rand() +r1
       
        X = np.zeros(self.state_dim)
        X[rand_ind] = x
           
        return X
        

    def check_if_done(self,n,x):
      
        if n == self.N-1:
            return True, None
        else:
            norm = np.linalg.norm(x)
          
            if ( norm <= self.r1) or (self.r2 <= norm):
                if ( norm <= self.r1):
                    return True, None
                if ( self.r2 <= norm):
                    return True, 'exit_C'
            
            else:
                return False, None
    
    def run_episodes(self, n_x):
        ep_reward_list = []
        stopping_time_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        avg_stopping_list = []
        X = np.zeros((self.N,self.state_dim), dtype= np.float32)
        
        X[0] = self.start_state()
        
        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                
                state = X[n]
                
                done, exits = self.check_if_done(n,state)
         
                if (exits == 'exit_C'):
                    exit_C = True
                else:
                    exit_C = False

                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                action = self.AC.policy(state).numpy()[0]
                action_env = self.AC.upper_action_bound*action

                if (done):
                    reward = self.g(n,X[n], exit_C)
                    
                    X = np.zeros((self.N,self.state_dim), dtype= np.float32)
                    X[0] = self.start_state()
                    new_state = X[0]
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:
                    
                    reward = self.f(n,X[n], action_env)
                    #print('X[n], action,reward ', X[n], action,reward )
                    
                    if (self.discrete_problem):
                    
                        X[n+1] =  (X[n] + action_env) + self.sig*np.random.normal(size = self.state_dim)
                    else:
                        X[n+1] =  X[n] + action_env*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size = self.state_dim)
                   
                 
                    new_state = tf.expand_dims(tf.convert_to_tensor(X[n+1]),0)
                
                
                self.AC.buffer.record((state.numpy()[0],action,reward, new_state.numpy()[0], done))
                
                episodic_reward += reward
                # warm up
                if (ep >= 100):
                    self.AC.learn(n)
                    
                    if (n % self.AC.update_frames == 0 and n != 0):
                        self.AC.update_target_critic(self.AC.target_critic_1.variables, self.AC.critic_1.variables)
                        self.AC.update_target_critic(self.AC.target_critic_2.variables, self.AC.critic_2.variables)
                        self.AC.update_target_actor(self.AC.target_actor.variables, self.AC.actor.variables)
                        self.AC.update_lr()
                        self.AC.update_var()
                frame_num += 1

                if(done):
                    stopping_time_list.append(self.dt*(n+ 0.5))
                    avg_stopping_time = np.mean(stopping_time_list[-500:])
                    avg_stopping_list.append(avg_stopping_time)
                    break
                else:
                    n += 1
            if (ep == 0):
                self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC)
            if (ep % self.dashboard_num == 0 and ep >100):
                self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC)
            
            if (ep >= 100):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-500:])
                print("Episode * {} * Avg Reward is ==> {} * Avg Stopping Time is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward,avg_stopping_time, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC)
    
   
            
    def dashboard(self,n_x,avg_reward_list, avg_stopping_list,AC: ActorCritic):
        
        fig = plt.figure()
        
        ax = fig.add_subplot()

        free_x = np.linspace(self.r1, self.r2, 30)
        y = 0
        free_energy =  np.zeros(30)
        free_energy_approx_1 = np.zeros(30)
        free_energy_approx_2 = np.zeros(30)
        for i in range(30):
            state = np.zeros(self.state_dim)
            state[0] = free_x[i]

            free_energy[i] = self.free_energy(state)
            
            state = tf.expand_dims(tf.convert_to_tensor(state),0)
            action = self.AC.actor(state)
            free_energy_approx_1[i] = AC.critic_1([state,action]).numpy()[0]
            free_energy_approx_2[i] = AC.critic_2([state,action]).numpy()[0]

        ax.plot(free_x,free_energy, label = 'free energy', color = 'black')
        ax.plot(free_x,free_energy_approx_1, label = 'free energy approx 1')
        ax.plot(free_x,free_energy_approx_2, label = 'free energy approx 2')
        fig.savefig('.\Bilder_SOC\D10_TD3_Committor_Episode_{}'.format(len(avg_reward_list)))
        #plt.show()
    

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 40
    
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x)
    
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

        self.batch_size = 512
        self.max_memory_size = 100000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.tau_actor = 0.001
        self.lower_action_bound = -4
        self.upper_action_bound = 4

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
            #self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0003
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
       
        self.actor_lr = 0.0003
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 1
        self.var_target = 0.2
        self.var_min = 0.1
        self.var_decay = 0
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
       
        out_1 = layers.Dense(128, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(128, activation = 'relu')(out_1)
        
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
       
        out = layers.Dense(128, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(128, activation="relu")(out)
       
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
        self.A = 0*np.identity(2)
        self.B = np.identity(2)
        self.sig = 1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.discrete_problem = False
        self.f_A = 0*np.identity(2)
        self.f_B = np.identity(2)

        # g(x) = D * ||x||^2
        self.D = 0*np.identity(2)

        self.num_episodes = 5100
        self.warmup = 100
        self.state_dim = 2
        self.action_dim = 2
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 50
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
    
    def free_energy(self,x,y):

        r = np.linalg.norm(np.array([x,y]))
        return -np.log( ((np.log(self.r1) -np.log(r)) / (np.log(self.r1)-np.log(self.r2))) + 10e-4)

    def opt_control(self, x,y):
        r = np.linalg.norm(np.array([x,y]))

        # Ax = x/(( ((np.log(self.r1) - 0.5*np.log(x**2 + y**2)) / (np.log(self.r1)-np.log(self.r2))) + 10e-4)*((x**2 +y**2)*(np.log(self.r1)-np.log(self.r2))))
        # Ay = y/(( ((np.log(self.r1) - 0.5*np.log(x**2 + y**2)) / (np.log(self.r1)-np.log(self.r2))) + 10e-4)*((x**2 +y**2)*(np.log(self.r1)-np.log(self.r2))))
        # return -self.sig*np.array([Ax,Ay])

        A = ((np.log(self.r1) -np.log(self.r2))* r*(((np.log(self.r1) -np.log(r)) / (np.log(self.r1)-np.log(self.r2))) + 10e-4))
        u = x/np.sqrt(x**2 + y**2)
        v = y/np.sqrt(x**2 + y**2)
        
        return -self.sig*A*np.array([u,v])
    def start_state(self):
        r1 = self.r1 + self.dt
        r2 = self.r2 - self.dt
        start_r = (r2 -r1)* np.random.rand() + r1
        random_pi = 2*np.pi *np.random.rand()
        
        X = np.array([start_r*np.cos(random_pi),start_r*np.sin(random_pi)])

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
    def get_baseline(self):

        num_sim = 500
        reward_arr = np.zeros(num_sim)
        stopping_arr = np.zeros(num_sim)

        for i_sim in range(num_sim):
            X = np.zeros((self.N,self.state_dim), dtype= np.float32)
            X[0] = self.start_state()
            episodic_reward = 0
            n = 0
            while(True):
                state = np.array([X[n][0],X[n][1]], np.float32)
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                done, exits = self.check_if_done(n,state)
                if (exits == 'exit_C'):
                    exit_C = True
                else:
                    exit_C = False
                
                
                action_env = self.opt_control(X[n][0], X[n][1])

                if (done):
                    reward = self.g(n,X[n], exit_C)
                    
                    X = np.zeros((self.N,self.state_dim), dtype= np.float32)
                    X[0] = self.start_state()
                    new_state = np.array([X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    reward = self.f(n,X[n], action_env)
                    
                    X[n+1] =  X[n] + action_env*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size = 2)
                    
                     
                episodic_reward += reward

                if(done):
                    reward_arr[i_sim] = episodic_reward
                    stopping_arr[i_sim] = self.dt*(n+ 0.5)
                    break
                else:
                    n += 1
        return np.mean(reward_arr),  np.mean(stopping_arr)

    def run_episodes(self, n_x):
        base, base_st = self.get_baseline()
        ep_reward_list = []
        stopping_time_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        avg_stopping_list = []
        X = np.zeros((self.N,2), dtype= np.float32)
        
        X[0] = self.start_state()
        
        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                
                state = np.array([X[n][0],X[n][1]], np.float32)
                
                done, exits = self.check_if_done(n,state)

                if (exits == 'exit_C'):
                    exit_C = True
                else:
                    exit_C = False
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                if (ep <= self.warmup):
                    action = tf.convert_to_tensor(2*np.random.rand(self.state_dim) -1)
                    action_env = self.AC.upper_action_bound * action
                else:
                    action = self.AC.policy(state).numpy()[0]
                    action_env = self.AC.upper_action_bound*action  
                if (done):
                    reward = self.g(n,X[n], exit_C)
                    
                    X = np.zeros((self.N,2), dtype= np.float32)
                    X[0] = self.start_state()
                    
                    new_state = np.array([X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:
                    
                    reward = self.f(n,X[n], action_env)
                    #print('X[n], action,reward ', X[n], action,reward )
                    
                    if (self.discrete_problem):
                    
                        X[n+1] =  (X[n] + action_env) + self.sig*np.random.normal(2)
                    else:
                        X[n+1] =  X[n] + action_env*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size = 2)
                    new_state = np.array([X[n+1][0],X[n+1][1]], np.float32)
                   
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                
                
                self.AC.buffer.record((state.numpy()[0],action,reward, new_state.numpy()[0], done))
                
                episodic_reward += reward
                # warm up
                if (ep >= self.warmup):
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
                self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC,base, base_st)
            if (ep % self.dashboard_num == 0 and ep >100):
                self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC,base, base_st)
            
            if (ep >= self.warmup):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-500:])
                print("Episode * {} * Avg Reward is ==> {} * Avg Stopping Time is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward,avg_stopping_time, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC,base, base_st)
    
    def run_simulation(self, num_sim,avg_reward_list,AC:ActorCritic):
        fig = plt.figure(figsize= (6,6))
        for sim in range(num_sim):
            X = np.zeros((self.N,2), dtype= np.float32)
            x,y = np.zeros(self.N, dtype= np.float32), np.zeros(self.N, dtype= np.float32)
            X[0] = self.start_state()
            n = 0
            while(True):
                x[n] = X[n][0]
                y[n] = X[n][1]
                state = np.array([X[n][0],X[n][1]], np.float32)
                done, exits = self.check_if_done(n,state)

             
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                action = self.AC.actor(state).numpy()[0]
                action_env = AC.upper_action_bound*action
                if (done):
                    print('done')      
                else:

                    if (self.discrete_problem):
                    
                        X[n+1] =  (X[n] + action) + self.sig*np.random.normal(size=2)
                    else:
                        X[n+1] =  X[n] + (np.dot(self.A,X[n]) + np.dot(self.B,action_env))*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size = 2)
                    
                if(done):
                    x[n:] = x[n]
                    y[n:] = y[n]
                    break
                else:
                    n += 1
               
            plt.plot(x, y)
        time = np.linspace(0,2*np.pi,100)
        circ_x_1 = self.r1*np.cos(time)
        circ_y_1 = self.r1*np.sin(time)
        plt.plot(circ_x_1, circ_y_1, color = 'black')
        plt.title('simulation episode {}'.format(len(avg_reward_list)))

        circ_x_2 = self.r2*np.cos(time)
        circ_y_2 = self.r2*np.sin(time)
        plt.plot(circ_x_2, circ_y_2, color = 'black')
        fig.savefig('.\Bilder_SOC\Sim_Balls_Committor_Episode_{}'.format(len(avg_reward_list)))
        #plt.show()
            
    def dashboard(self,n_x,avg_reward_list, avg_stopping_list,AC: ActorCritic,base, base_st):
        if (len(self.change_V1) == 0):
            self.old_V1 = np.zeros((n_x,n_x))
            self.old_V2 = np.zeros((n_x,n_x))

        self.run_simulation(10, avg_reward_list,AC)
        x_space = np.linspace(-self.r2,self.r2, n_x)
        one_ind = int(np.where(x_space == 1)[0][0])
        
        minus_one_ind = int(np.where(x_space == -1)[0][0]) +1 
        y_space = np.linspace(-self.r2,self.r2, n_x)

        fig = plt.figure()
        
        V1 = np.zeros((n_x,n_x))
        P = np.zeros((n_x,n_x,2))
    
        V2 = np.zeros((n_x,n_x))
        t0 = 1
        V_true = np.zeros((n_x,n_x))

        ax = fig.add_subplot(2, 3, 1)
        if (len(avg_reward_list)> 0):
            ax.plot(avg_reward_list, label = 'Avg Reward: {}'.format(np.round(avg_reward_list[-1],2)))
        else:
            ax.plot(avg_reward_list)
        ax.set_xlim([0,self.num_episodes-100])
        ax.set_xlabel('Episode')
        ax.set_title('Avg. Epsiodic Reward')
        ax.hlines(base,xmin = 0, xmax = self.num_episodes, color = 'black', label = 'base: {}'.format(np.round(base,2)))
       
        ax.legend()

        ax = fig.add_subplot(2, 3, 2)
        if (len(avg_stopping_list) > 0):
            ax.plot(avg_stopping_list, label = 'Avg stopping time: {}'.format(np.round(avg_stopping_list[-1],2)))
        else:
            ax.plot(avg_stopping_list)
        ax.set_xlabel('Episode')
        ax.set_xlim([0,self.num_episodes-100])
        ax.set_title('Avg. Stopping Time')
        ax.hlines(base_st,xmin = 0, xmax = self.num_episodes, color = 'black', label = 'base: {}'.format(np.round(base_st,2)))
        ax.legend()
        
        

        # for y axis poilcy
        policy_x = np.zeros((n_x,n_x))
        opt_policy_x = np.zeros((n_x, n_x))
        # for x axis poilcy
        policy_y = np.zeros((n_x,n_x))
        
    

        for ix in range(n_x):
            for iy in range(n_x):
                state = np.array([x_space[ix],y_space[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix][iy] = AC.upper_action_bound*action[0]

                v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix][iy] = v1
                
                V2[ix][iy] = v2

                if (np.linalg.norm(state)< 1):
                    V_true[ix][iy] = np.nan
                else:
                    V_true[ix][iy] = self.free_energy(x_space[ix],y_space[iy])
              
                policy_x[ix][iy] = AC.upper_action_bound*action[0][0]
                opt_policy_x[ix][iy] = self.opt_control(x_space[ix], y_space[iy])[0]
                policy_y[ix][iy] = AC.upper_action_bound*action[0][1]

        change_V1 = (self.old_V1 - V1)**2
        change_V2 = (self.old_V2 - V2)**2

        self.change_V1.append(np.mean(change_V1))
        
        self.change_V2.append(np.mean(change_V2))
        self.old_V1 = V1
        self.old_V2 = V2
       
        

        X,Y = np.meshgrid(x_space, y_space)

        # ax = fig.add_subplot(2, 3, 3)
        # ep_axis = np.linspace(0,len(avg_reward_list), len(self.change_V1))
        # if (len(AC.critic_1_loss) != 0):
        #     ax.plot(np.array(AC.critic_1_loss), label = 'critic loss 1: {}'.format(AC.critic_1_loss[-1]))
        #     ax.plot(AC.critic_2_loss,  label = 'critic loss 2: {}'.format(AC.critic_2_loss[-1]))

        # ax.set_xlabel('Training steps')
      
        # #ax.set_xlim([0,self.num_episodes])
        # ax.set_title('critic losses')
        # ax.legend()
        
        ax = fig.add_subplot(2,3,3)
         # for y axis poilcy
        policy_x2 = np.zeros((20,20))
        policy_y2 = np.zeros((20,20))
        x_space2 = np.linspace(-self.r2,self.r2, 20)
        y_space2 = np.linspace(-self.r2,self.r2, 20)
        X2,Y2 = np.meshgrid(x_space2, y_space2)
        # for x axis poilcy
        
        opt_policy_x2 = np.zeros((20,20))
        opt_policy_y2 = np.zeros((20,20))
        for ix in range(20):
            for iy in range(20):
                state = np.array([x_space2[ix],y_space2[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                opt_action = self.opt_control(x_space2[ix],y_space2[iy])
                policy_x2[iy][ix] = AC.upper_action_bound*action[0][0]
                    
                policy_y2[iy][ix] = AC.upper_action_bound*action[0][1]
                opt_policy_x2[iy][ix] = opt_action[0]
                opt_policy_y2[iy][ix] = opt_action[1]
       
        ax.quiver(X2,Y2, policy_x2, policy_y2, color = 'blue')
        ax.quiver(X2,Y2, opt_policy_x2, opt_policy_y2, color = 'black', alpha = 0.4)
        
        time = np.linspace(0,2*np.pi,100)
        circ_x_1 = self.r1*np.cos(time)
        circ_y_1 = self.r1*np.sin(time)
        ax.plot(circ_x_1, circ_y_1, color = 'black')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        circ_x_2 = self.r2*np.cos(time)
        circ_y_2 = self.r2*np.sin(time)
        ax.plot(circ_x_2, circ_y_2, color = 'black')
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
       
        ax.set_title('policy function vector field')
        
        ax = fig.add_subplot(2, 3, 4, projection = '3d')

        #ax.set_zlim(AC.lower_action_bound, AC.upper_action_bound)
        ax.plot_surface(X,Y, policy_x, label = 'policy function approximation x-direction')# vmin = AC.lower_action_bound, vmax = AC.upper_action_bound
        ax.plot_surface(X,Y, opt_policy_x, label = 'policy function approximation x-direction',color = 'black',lw=0.5, rstride=1, cstride=1, alpha=0.4)
        #ax.contour(X,Y,policy_x, levels = 9,lw=2, cmap = 'viridis', offset = AC.lower_action_bound, linestyles="solid", vmin = AC.lower_action_bound, vmax = AC.upper_action_bound)
        #ax.contour(X, Y, policy_x, 10, lw=0.5, colors="k", linestyles="solid")
        ax.set_title('policy function x-direction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        free_x = np.linspace(self.r1, self.r2, 30)
        y = 0
        free_energy =  np.zeros(30)
        free_energy_approx_1 = np.zeros(30)
        free_energy_approx_2 = np.zeros(30)
        for i in range(30):
            free_energy[i] = self.free_energy(free_x[i], y)
            state = np.array([free_x[i],y])
            state = tf.expand_dims(tf.convert_to_tensor(state),0)
            action = self.AC.actor(state)
            free_energy_approx_1[i] = AC.critic_1([state,action]).numpy()[0]
            free_energy_approx_2[i] = AC.critic_2([state,action]).numpy()[0]
        
        ax = fig.add_subplot(2, 3, 5)
        
        ax.plot(free_x,free_energy, label = 'free energy y = 0', color = 'black')
        ax.plot(free_x,free_energy_approx_1, label = 'free energy approx 1')
        ax.plot(free_x,free_energy_approx_2, label = 'free energy approx 2')
        ax.set_xlabel('x')
        
        ax.set_title('free energy')
       

        ax = fig.add_subplot(2, 3, 6, projection = '3d')

        ax.plot_surface(X,Y, V_true, label = 'value function', color = 'black', alpha = 0.4)
        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        ax.plot_surface(X,Y, V2, label = 'approx value function 2')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
     

        
        ax.set_title('value function')
  
        fig.set_size_inches(w = 18, h= 8)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.25)
        fig.savefig('.\Bilder_SOC\TD3_Committor_Episode_{}'.format(len(avg_reward_list)))
        #plt.show()
    

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 40
    
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x)
    
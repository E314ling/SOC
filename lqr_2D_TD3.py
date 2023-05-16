import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import linear_quadratic_regulator_DP as LQR


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
        self.terminal_condition_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
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
        self.terminal_condition_buffer[index] = obs_tuple[5]

        self.buffer_counter += 1

class ActorCritic():

    def __init__(self, state_dim, action_dim, load_model,run):

        self.batch_size = 255
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.005
        self.tau_actor = 0.005
        self.lower_action_bound = -10
        self.upper_action_bound = 10

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        self.polynom = False
        
        # init the neural nets
        if (load_model):
            if self.polynom:
                self.critic_1 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))
                self.target_critic_1 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))

                self.critic_2 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))
                self.target_critic_2 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))


                self.actor = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_actor_run_{}.h5'.format(run))
                self.target_actor = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_actor_run_{}.h5'.format(run))
            else:
                self.critic_1 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))
                self.target_critic_1 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))

                self.critic_2 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))
                self.target_critic_2 = tf.keras.models.load_model('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))


                self.actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_actor_run_{}.h5'.format(run))
                self.target_actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_actor_run_{}.h5'.format(run))
        else:
            if self.polynom:
                print('Polynomial Ansatz')
                self.critic_1 = self.get_critic_NN_poly()
                self.target_critic_1 = self.get_critic_NN_poly()
                #self.target_critic_1.set_weights(self.critic_1.get_weights())

                self.critic_2 = self.get_critic_NN_poly()
                self.target_critic_2 = self.get_critic_NN_poly()
                #self.target_critic_2.set_weights(self.critic_2.get_weights())

                self.actor = self.get_actor_NN_poly()
                self.target_actor = self.get_actor_NN_poly()
                #self.target_actor.set_weights(self.actor.get_weights())
            else:
                self.critic_1 = self.get_critic_NN()
                self.target_critic_1 = self.get_critic_NN()
                self.target_critic_1.set_weights(self.critic_1.get_weights())

                self.critic_2 = self.get_critic_NN()
                self.target_critic_2 = self.get_critic_NN()
                self.target_critic_2.set_weights(self.critic_2.get_weights())

                self.actor = self.get_actor_NN()
                self.target_actor = self.get_actor_NN()
                self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0003
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr, clipvalue=5)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr, clipvalue=5)
       
        self.actor_lr = 0.0003
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 0.5
        self.var_decay = 0
        self.lr_decay = 1
        self.var_min = 0.1 / self.upper_action_bound
        self.var_target = 0.2 / self.upper_action_bound
        self.update_frames = 2

       
    def save_model(self, run):
        if self.polynom:
            self.critic_1.save('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))
            self.critic_2.save('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))
            
            self.actor.save('./Models/poly_LQR_2D_TD3_actor_run_{}.h5'.format(run))
        else:
            self.critic_1.save('./Models/poly_LQR_2D_TD3_critic_1_run_{}.h5'.format(run))
            self.critic_2.save('./Models/poly_LQR_2D_TD3_critic_2_run_{}.h5'.format(run))
            
            self.actor.save('./Models/poly_LQR_2D_TD3_actor_run_{}.h5'.format(run))
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def update_var(self):
        self.var = np.max([self.var_min,self.var * self.var_decay])
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch,terminal_condition_batch):
        target_actions = self.target_policy(next_state_batch) 
        
        target_1, target_2 = self.target_critic_1([next_state_batch, target_actions]),self.target_critic_2([next_state_batch, target_actions])
        
        target_vals =  tf.minimum(target_1, target_2)

        y = (1-done_batch)*tf.stop_gradient(reward_batch + self.gamma*target_vals) + done_batch*terminal_condition_batch

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


    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions = self.actor(state_batch)
            critic_value_1,critic_value_2 = self.critic_1([state_batch, actions]), self.critic_2([state_batch, actions])
            actor_loss = tf.math.reduce_mean(tf.minimum(critic_value_1,critic_value_2))
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
      
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

   
    def learn(self,frame_num):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)
        
        batch_indices = np.append(batch_indices, record_range-1)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])
        
        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        terminal_condition_batch = tf.convert_to_tensor(self.buffer.terminal_condition_buffer[batch_indices])
        
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch, terminal_condition_batch)

        if (frame_num % self.update_frames == 0 and frame_num != 0):
            for _ in range(self.update_frames):
                self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    
    
    @tf.function
    def update_target_critic(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

    @tf.function
    def update_target_actor(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau_actor *b + (1-self.tau_actor) *a)

    def get_critic_NN_poly(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        state_input = layers.Input(shape =(self.state_dim+1,))
        action_input = layers.Input(shape =(self.action_dim,))

        a = action_input
        x = state_input
        
        x_a = tf.einsum("nu,nv->nuv", state_input, action_input)
        z = tf.ones_like(x_a)
        z = tf.linalg.band_part(z,0,-1)
        x_a = x_a*z
        x_a = layers.Flatten()(x_a)

        x_x = tf.einsum("nu,nv->nuv", state_input, state_input)
        z = tf.ones_like(x_x)
        z = tf.linalg.band_part(z,0,-1)
        x_x = x_x*z
        x_x = layers.Flatten()(x_x)

        a_a = tf.einsum("nu,nv->nuv", action_input, action_input)
        a_a = layers.Flatten()(a_a)

        input = tf.concat([x,a,x_a,x_x,a_a],1)

        # out = theta_1 x_1**2 + theta_2 x_2**2 + theta_3 a_1**2 + theta_4 a_2**2 + theta_5
        out_1 = layers.Dense(1, kernel_initializer= last_init)(input)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model


    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        state_input = layers.Input(shape =(self.state_dim+1,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
        #out = layers.BatchNormalization()(input)
        out_1 = layers.Dense(128, activation = 'relu')(input)
        
        out_1 = layers.Dense(128, activation = 'relu')(out_1)
       
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model
    
    def get_actor_NN_poly(self):
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        inputs = layers.Input(shape=(self.state_dim+1,))
        
        # this is just a linear model y = theta_1 x_1 + theta_2 x_2 + theta_3
        x = inputs
        x_x = tf.einsum("nu,nv->nuv", x, x)
        z = tf.ones_like(x_x)
        z = tf.linalg.band_part(z,0,-1) - tf.linalg.band_part(z,0,0)
        x_x = x_x*z
        x_x = layers.Flatten()(x)
        input = tf.concat([x,x_x],1)
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(input)

        # Our upper bound is 2.0 .
        #outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-1, maxval=1)

        inputs = layers.Input(shape=(self.state_dim+1,))
        #inputs = layers.BatchNormalization()(inputs)
        out = layers.Dense(128, activation="relu")(inputs)
        out = layers.Dense(128, activation="relu")(out)
        
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        #outputs = outputs * self.upper_action_bound
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
        noice = tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = self.var_target, dtype = tf.float32)
        noice = tf.clip_by_value(noice, -0.5, 0.5)
        sampled_actions = sampled_actions + noice
        
        legal_action = tf.clip_by_value(sampled_actions, clip_value_min = -1, clip_value_max =1)

        return legal_action
        
class CaseOne():

    def __init__(self,run):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = np.identity(2)
        self.B = np.identity(2)
        self.sig = np.sqrt(2)

        self.discrete_problem = False
        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = np.identity(2)
        self.f_B = np.identity(2)

        # g(x) = D * ||x||^2
        self.D = np.identity(2)

        self.num_episodes = 5000
        self.warmup = 0
        self.state_dim = 2
        self.action_dim = 2
        self.load = False
        self.AC = ActorCritic(self.state_dim, self.action_dim, self.load, run)

        self.T = 1
        self.N = 100
        self.dt = self.T/self.N

        self.r = 0
        self.dashboard_num = 100
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P_x_0 = []
        self.mean_abs_error_P_y_0 = []

        self.run = run

        self.max_steps = 10e6


    def f(self, n,x,a):

        y1= np.dot(np.transpose(x), self.f_A)
        
        y2 = np.dot(np.transpose(a), self.f_B)
      
        if(self.discrete_problem):
            return np.float32(np.dot(y1,x) + np.dot(y2,a))
        else:
            return self.dt*(np.float32(np.dot(y1,x) + np.dot(y2,a)))
        
    def g(self, n,x):
        y = np.dot(np.transpose(x), self.D)
        return np.dot(y,x)
    
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
    

    def start_state(self):
        r1 = 0 + 0.1
        r2 = 3 - 0.1
        start_r = (r2 -r1)* np.random.rand() + r1
        random_pi = 2*np.pi *np.random.rand()
        
        X = np.array([start_r*np.cos(random_pi),start_r*np.sin(random_pi)])

        return X
    def run_episodes(self, n_x,V_t,A_t, base):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        X = np.zeros((self.N,2), dtype= np.float32)
        #X[0] = 1*np.random.rand(self.state_dim) - 1
        X[0] = self.start_state()
        frame_num = 0
        
        for ep in range(self.num_episodes):
            
            n=0
            done_training = False
            episodic_reward = 0
            while(True):
                X[n] = np.clip(X[n], a_min= -4,a_max = 4)
                state = np.array([n,X[n][0],X[n][1]], np.float32)
               
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)

                

                if (ep <= self.warmup):
                    action = tf.convert_to_tensor(2* np.random.rand(2) -1)
                    action_env = self.AC.upper_action_bound * action
                else:
                    action = self.AC.policy(state)[0]
                    if self.AC.polynom:
                        action_env = action
                    else:
                        action_env = self.AC.upper_action_bound*action

                if (done):
                    reward = self.g(n,X[n])
                    
                    X = np.zeros((self.N,2), dtype= np.float32)
                    #X[0] = 1*np.random.rand(self.state_dim) - 1

                    #X[0] = np.clip(X[0], a_min= -1,a_max = 1)
                    X[0] = self.start_state()
                    new_state = np.array([0,X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    reward = self.f(n,X[n], action_env)
                    if (self.discrete_problem):
                    
                        X[n+1] =  (X[n] + action_env) + self.sig*np.random.normal(size = 2)
                    else:
                        X[n+1] =  X[n] + (X[n] + action_env)*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size=2)
                    new_state = np.array([(n+1),X[n+1][0],X[n+1][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                
                if (n <= self.N-2):
                    terminal_cond = 0
                    if (n== self.N-2):
                        done_training = True

                        terminal_cond = self.g(n,X[n+1])

                    self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done_training, terminal_cond))
            
                self.AC.learn(n)
                
                if (n % self.AC.update_frames == 0 and n != 0):
                    self.AC.update_target_critic(self.AC.target_critic_1.variables, self.AC.critic_1.variables)
                    self.AC.update_target_critic(self.AC.target_critic_2.variables, self.AC.critic_2.variables)
                    self.AC.update_target_actor(self.AC.target_actor.variables, self.AC.actor.variables)
                    self.AC.update_lr()
                    self.AC.update_var()
                frame_num += 1

                episodic_reward += reward
                if(done):
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0):
                self.save_all(n_x,V_t,A_t,ep_reward_list,avg_reward_list,self.AC,base)
                #self.dashboard(n_x,V_t,A_t,avg_reward_list,self.AC,base)
            
           
                
            ep_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-500:])
            print("Episode * {} * Avg Reward is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward, self.AC.var, self.AC.actor_lr))
            avg_reward_list.append(avg_reward)

            
        # Plotting graph
        # Episodes versus Avg. Rewards
        
        #self.dashboard(n_x,V_t,A_t,avg_reward_list,self.AC,base)
        self.save_all(n_x,V_t,A_t,ep_reward_list,avg_reward_list,self.AC,base)
        self.AC.save_model(self.run)
    

    def save_all(self, n_x, V_t,A_t,ep_reward_list, avg_reward_list, AC:ActorCritic,base):
        steps = len(avg_reward_list)
        # save avg_reward and base
        np.save('./Saved_Runs/LQR/ep_reward_list_{}_run_{}'.format(steps, self.run), np.array(ep_reward_list))
        np.save('./Saved_Runs/LQR/avg_reward_step_{}_run_{}'.format(steps, self.run), np.array(avg_reward_list))
        np.save('./Saved_Runs/LQR/base_run_{}'.format(self.run), base)

        x_space = np.linspace(-2,2, n_x)
        y_space = np.linspace(-2,2, n_x)

        fig = plt.figure()
        
        V1 = np.zeros((n_x,n_x))
        V2 = np.zeros((n_x,n_x))

        P_x = np.zeros((n_x,n_x))
        P_y = np.zeros((n_x,n_x))

        True_P_x = np.zeros((n_x,n_x))
        True_P_y = np.zeros((n_x,n_x))

       
        for t0 in np.linspace(0,self.N,6):
            t0 = int(t0)
            if t0 == self.N:
                t0 -= 1

            for ix in range(n_x):
                for iy in range(n_x):
                    state = np.array([t0,x_space[ix],y_space[iy]])
                    
                    action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                    
                    if self.AC.polynom:
                        P_x[ix][iy] =  action[0][0]
                        P_y[ix][iy] =  action[0][1]
                    else:
                        P_x[ix][iy] = AC.upper_action_bound* action[0][0]
                        P_y[ix][iy] = AC.upper_action_bound* action[0][1]

                    v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                    
                    V1[ix][iy] = v1
                    
                    V2[ix][iy] = v2
                    True_P_x[ix][iy] = A_t[t0][ix][iy][0]
                    True_P_y[ix][iy] = A_t[t0][ix][iy][1]
            
            #save value function and policy
            np.save('./Saved_Runs/LQR/value_fct_1_n_{}_{}_run_{}'.format(t0, steps, self.run), V1)
            np.save('./Saved_Runs/LQR/value_fct_2_n_{}_{}_run_{}'.format(t0, steps, self.run), V2)
            np.save('./Saved_Runs/LQR/true_value_fct_{}'.format(t0), V_t[t0])

            np.save('./Saved_Runs/LQR/policy_X_n_{}_{}_run_{}'.format(t0, steps, self.run), P_x)
            np.save('./Saved_Runs/LQR/policy_Y_n_{}_{}_run_{}'.format(t0, steps, self.run), P_y)

            np.save('./Saved_Runs/LQR/true_policy_x_n_{}'.format(t0), True_P_x)
            np.save('./Saved_Runs/LQR/true_policy_y_n_{}'.format(t0), True_P_y)


        # save x and y space
        np.save('./Saved_Runs/LQR/horizon', self.N)
        np.save('./Saved_Runs/LQR/X_space_{}'.format(n_x), x_space)
        np.save('./Saved_Runs/LQR/Y_space_{}'.format(n_x), y_space)

if __name__ == "__main__":

   
    n_x = 40
    runs = 1

    for i in range(0,runs):
        lqr = CaseOne(i)
        V_t,A_t,base = LQR.Solution_2_D(lqr).create_solution(n_x)
        #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
        lqr.run_episodes(n_x,V_t,A_t, base)
    
import numpy as np
import matplotlib.pyplot as plt

import time

# file names

num_runs = 1
iteration_steps = 100

#\OUP\avg_reward_step_0_run_0.npy

max_P_mean_total_t0 = 0
max_P_stddev_total_t0 = 0
max_P_mean_total_t1 = 0
max_P_stddev_total_t1 = 0

max_V_mean_total_t0 = 0
max_V_mean_total_t1 = 0
max_V_stddev_total_t0 = 0
max_V_stddev_total_t1 = 0

min_P_mean_total_t0 = 0
min_P_stddev_total_t0 = 0
min_P_mean_total_t1 = 0
min_P_stddev_total_t1 = 0

min_V_mean_total_t0 = 0
min_V_mean_total_t1 = 0
min_V_stddev_total_t0 = 0
min_V_stddev_total_t1 = 0


for i in np.arange(4900,5000,100):

    x_space = np.load('./OUP/X_space_40.npy')
   

    
    n_x = len(x_space)

    rewards = np.zeros((num_runs,i))
    bases = np.zeros(num_runs)
    V1_t0 = np.zeros((num_runs,n_x))
    V2_t0 = np.zeros((num_runs,n_x))

    V1_t1 = np.zeros((num_runs,n_x))
    V2_t1 = np.zeros((num_runs,n_x))

    P_t0 = np.zeros((num_runs,n_x))
    
    P_t1 = np.zeros((num_runs,n_x))
  

    t0 = 0
    t1 = 19

    V_true_t0 = np.load('./OUP/true_value_fct_0.npy'.format(t0))
    V_true_t1 = np.load('./OUP/true_value_fct_{}.npy'.format(t1))

    P_true_t0 = np.load('./OUP/true_policy_n_{}.npy'.format(t0))
    P_true_t1 = np.load('./OUP/true_policy_n_{}.npy'.format(t1))
    P_true_time = np.load('./OUP/true_policy_X0_time.npy')
    
    N = len(P_true_time)
    print(N)
    t_axis = np.linspace(0,1,N)
    P_time = np.zeros((num_runs, N))

    fill_axis = np.linspace(0,i,i)
    for r in range(num_runs):

        rewards[r] = np.load('./OUP/avg_reward_step_{}_run_{}.npy'.format(i,r))
        bases[r] = np.load('./OUP/base_run_{}.npy'.format(r))

        V1_t0[r] = np.load('./OUP/value_fct_1_n_{}_{}_run_{}.npy'.format(t0,i,r))
        V2_t0[r] = np.load('./OUP/value_fct_2_n_{}_{}_run_{}.npy'.format(t0,i,r))

        V1_t1[r] = np.load('./OUP/value_fct_1_n_{}_{}_run_{}.npy'.format(t1,i,r))
        V2_t1[r] = np.load('./OUP/value_fct_2_n_{}_{}_run_{}.npy'.format(t1,i,r))

        P_t0[r] = np.load('./OUP/policy_n_{}_{}_run_{}.npy'.format(t0,i,r))
   

        P_t1[r] = np.load('./OUP/policy_n_{}_{}_run_{}.npy'.format(t1,i,r))

        P_time[r] = np.load('./OUP/policy_X0_time_{}_run_{}.npy'.format(i,r))


    reward_mean = np.mean(rewards, axis=0)
    reward_stddev = np.std(rewards, axis = 0)
    
    base_mean = np.mean(bases)
    base_stddev = np.std(bases)

    V1_t0_mean = np.mean(V1_t0,axis=0)
    V2_t0_mean = np.mean(V2_t0,axis=0)
    V1_t1_mean = np.mean(V1_t1,axis=0)
    V2_t1_mean = np.mean(V2_t1,axis=0)

    max_V_mean = np.max([V1_t0_mean,V2_t0_mean,V1_t1_mean,V2_t1_mean])

    V1_t0_stddev = np.std(V1_t0,axis=0)
    V2_t0_stddev = np.std(V2_t0,axis=0)
    V1_t1_stddev = np.std(V1_t1,axis=0)
    V2_t1_stddev = np.std(V2_t1,axis=0)

    max_V_stddev = np.max([V1_t0_stddev,V2_t0_stddev,V1_t1_stddev,V2_t1_stddev])


    P_t0_mean = np.mean(P_t0,axis=0)
    P_t1_mean = np.mean(P_t1,axis=0)

    P_time_mean = np.mean(P_time,axis=0)


    max_P_mean = np.max([P_t0_mean,P_t1_mean])

    P_t0_stddev = np.std(P_t0,axis=0)
    
    P_t1_stddev = np.std(P_t1,axis=0)
    P_time_stddev = np.std(P_time,axis=0)
   

    max_P_stddev = np.max([P_t0_stddev,P_t1_stddev])

    max_P_mean_total_t0 = np.max([max_P_mean_total_t0, np.max(P_t0_mean)])
    max_P_stddev_total_t0 = np.max([max_P_stddev_total_t0,np.max(P_t0_stddev)])
    max_P_mean_total_t1 = np.max([max_P_mean_total_t1, np.max(P_t1_mean)])
    max_P_stddev_total_t1 = np.max([max_P_stddev_total_t1, np.max(P_t1_stddev)])
   

    max_V_mean_total_t0 = np.max([max_V_mean_total_t0, np.max(V1_t0_mean), np.max(V2_t0_mean)])
    max_V_mean_total_t1 = np.max([max_V_mean_total_t1, np.max(V1_t1_mean), np.max(V2_t1_mean)])
    max_V_stddev_total_t0 = np.max([max_V_stddev_total_t0, np.max(V1_t0_stddev), np.max(V2_t0_stddev)])
    max_V_stddev_total_t1 = np.max([max_V_stddev_total_t1, np.max(V1_t1_stddev), np.max(V2_t1_stddev)])

    min_P_mean_total_t0 = np.min([min_P_mean_total_t0, np.min(P_t0_mean)])
    min_P_stddev_total_t0 = np.min([min_P_stddev_total_t0,np.min(P_t0_stddev)])
    min_P_mean_total_t1 = np.min([min_P_mean_total_t1, np.min(P_t1_mean)])
    min_P_stddev_total_t1 = np.min([min_P_stddev_total_t1, np.min(P_t1_stddev)])
   

    min_V_mean_total_t0 = np.min([min_V_mean_total_t0, np.min(V1_t0_mean), np.min(V2_t0_mean)])
    min_V_mean_total_t1 = np.min([min_V_mean_total_t1, np.min(V1_t1_mean), np.min(V2_t1_mean)])
    min_V_stddev_total_t0 = np.min([min_V_stddev_total_t0, np.min(V1_t0_stddev), np.min(V2_t0_stddev)])
    min_V_stddev_total_t1 = np.min([min_V_stddev_total_t1, np.min(V1_t1_stddev), np.min(V2_t1_stddev)])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('average cost over 500 episodes', fontsize = 9, fontweight='bold')
    
    ax1.set_xlim([0,5000])
    #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])

    ax1.set_ylim([-1, np.max([0, np.max(reward_mean)])+5])
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('average cost')

    ax1.plot(reward_mean, label = 'mean: {}'.format(np.round(reward_mean[-1],2)),color='blue')
    ax1.fill_between(fill_axis,reward_mean,reward_mean + reward_stddev, alpha = 0.3, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(reward_stddev[-1],2)))
    ax1.fill_between(fill_axis,reward_mean,reward_mean - reward_stddev, alpha = 0.3, color='dodgerblue')
    ax1.legend()
    #ax1.hlines(base_mean, xmin = 0, xmax = 5000, color = 'black')
    #ax1.fill_between([0,5000],[base_mean -base_stddev,base_mean-base_stddev], [base_mean +base_stddev,base_mean + base_stddev], alpha = 0.2, color='black')
    

    fig2 =plt.figure()
    fig2.set_tight_layout('tight')
    fig2.set_figheight(6)
    fig2.set_figwidth(7.5)
    
    ax2_1 = fig2.add_subplot(2,2,1)
 

    
    ax2_1.plot(x_space, V_true_t0, color= 'black', alpha = 0.2)
    ax2_1.plot(x_space, V1_t0_mean, color= 'dodgerblue')
    ax2_1.plot(x_space, V2_t0_mean, color= 'orange')
    ax2_1.set_title('value function n = {}'.format(t0), fontsize = 9, fontweight='bold')
    
    ax2_2 = fig2.add_subplot(2,2,2)
    

    
    ax2_2.plot(x_space, V_true_t1, color= 'black', alpha = 0.2)
    ax2_2.plot(x_space, V1_t1_mean, color= 'dodgerblue')
    ax2_2.plot(x_space, V2_t1_mean, color= 'orange')
    ax2_2.set_title('value function n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax2_1.set_ylim([-1,1])
    ax2_2.set_ylim([-1,1])
    
    
    ax2_3 = fig2.add_subplot(2,2,3 )
    #ax2_3.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_3.set_title('standard deviation n = {}'.format(t0), fontsize = 9, fontweight='bold')
    cont2_3 = ax2_3.plot(x_space, 0.5*(V1_t0_stddev + V2_t0_stddev))

    ax2_4 = fig2.add_subplot(2,2,4)
    #ax2_4.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_4.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')
    cont2_3 = ax2_4.plot(x_space, 0.5*(V1_t1_stddev + V2_t1_stddev))

    fig3 =plt.figure()
    fig3.set_figheight(6)
    fig3.set_figwidth(13.5)
    fig3.set_tight_layout('tight')

    ax3_1 = fig3.add_subplot(2,3,1)
    
    ax3_1.plot(x_space, P_true_t0, color= 'black', alpha = 0.2)
    ax3_1.plot(x_space, P_t0_mean, color= 'dodgerblue')
    ax3_1.set_ylim([-2,2])
    ax3_1.set_title('x component policy n = {}'.format(t0), fontsize = 9, fontweight='bold')

    ax3_3 = fig3.add_subplot(2,3,2)
    
    ax3_3.plot(x_space, P_true_t1, color= 'black', alpha = 0.2)
    ax3_3.plot(x_space, P_t1_mean, color= 'dodgerblue')
    ax3_3.set_ylim([-2,2])
    ax3_3.set_title('policy n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax3_3 = fig3.add_subplot(2,3,3)
    
    ax3_3.plot(t_axis, P_true_time, color= 'black', alpha = 0.2)
    ax3_3.plot(t_axis, P_time_mean, color= 'dodgerblue')
    ax3_3.set_ylim([-2,2])
    ax3_3.set_title('policy n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax3_5 = fig3.add_subplot(2,3,4)
    #ax3_5.set_aspect('equal')
    ax3_5.plot(x_space,  P_t0_stddev)
    ax3_5.set_title('standard deviation n = {}'.format(t0), fontsize = 9, fontweight='bold')
   
   
    ax3_6 = fig3.add_subplot(2,3,5)
    #ax3_6.set_aspect('equal')
    cont2 = ax3_6.plot(x_space,  P_t1_stddev)
    ax3_6.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax3_6 = fig3.add_subplot(2,3,6)
    #ax3_6.set_aspect('equal')
    cont2 = ax3_6.plot(t_axis,  P_time_stddev)
    ax3_6.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')
 
    plt.show()




# max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1
# 4.043304014205932, 2.935147692749849, 3.703972911834717, 1.4305743895184417, 24.30153121948242, 19.183624267578125, 1.5091561974096852, 1.24356255476597
print('max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1', max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1)

# min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1
#  -4.05456805229187 0.0 -4.011117887496948 0.0 0.0 0.0 0.0 0.0
print('min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1', min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1)

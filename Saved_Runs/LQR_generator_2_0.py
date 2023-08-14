import numpy as np
import matplotlib.pyplot as plt

import time

# file names

num_runs = 1
iteration_steps = 100

#\LQR\avg_reward_step_0_run_0.npy

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

approx_type = 'polynomial'
#approx_type = 'fourier'
#approx_type = 'rbf'

folder = './LQR/' + approx_type + '_'

folder = './LQR/'
for i in range(1400,1500,100):
    id = i
# for i in np.arange(100,200,100):
#     id = i +10000

    
    x_space = np.load(folder +'X_space_40.npy')
    y_space = np.load(folder +'Y_space_40.npy')
    N = np.load(folder +'horizon.npy')
    tn = 6
    time_points = np.linspace(0,N,tn)
    n_x = len(x_space)

    rewards = np.zeros((num_runs,i))
    avg_rewards = np.zeros((num_runs,i))
    bases = np.zeros(num_runs)
    V1 = np.zeros((tn,num_runs,n_x,n_x))
    V2 = np.zeros((tn,num_runs,n_x,n_x))

    V_true = np.zeros((tn,n_x,n_x))

    P_x = np.zeros((tn,num_runs,n_x,n_x))
    P_y = np.zeros((tn,num_runs,n_x,n_x))

    P_x_true = np.zeros((tn,n_x,n_x))
    P_y_true = np.zeros((tn,n_x,n_x))

    for j in range(tn):
        t0 = int(time_points[j])
        if t0 == N:
            t0 -= 1
        V_true[j] = np.load(folder +'true_value_fct_{}.npy'.format(t0))
        
        P_x_true[j] = np.load(folder +'true_policy_x_n_{}.npy'.format(t0))
        
        P_y_true[j] = np.load(folder +'true_policy_y_n_{}.npy'.format(t0))
        
        fill_axis = np.linspace(0,i,i)
        for r in range(num_runs):

            avg_rewards[r] = np.load(folder +'avg_reward_step_{}_run_{}.npy'.format(id,r))
            bases[r] = np.load(folder +'base_run_{}.npy'.format(r))
            
            rewards[r] = np.load(folder +'ep_reward_list_{}_run_{}.npy'.format(id,r))
            V1[j][r] = np.load(folder +'value_fct_1_n_{}_{}_run_{}.npy'.format(t0,id,r))
            V2[j][r] = np.load(folder +'value_fct_2_n_{}_{}_run_{}.npy'.format(t0,id,r))

            P_x[j][r] = np.load(folder +'policy_X_n_{}_{}_run_{}.npy'.format(t0,id,r))
            P_y[j][r] = np.load(folder +'policy_Y_n_{}_{}_run_{}.npy'.format(t0,id,r))

        
    
    reward_mean = np.mean(rewards, axis=0)
    reward_stddev = np.std(rewards, axis = 0)

    avg_reward_mean = np.mean(avg_rewards, axis=0)
    avg_reward_stddev = np.std(avg_rewards, axis = 0)
    
    base_mean = np.mean(bases)
    base_stddev = np.std(bases)


    V1_mean = np.mean(V1,axis=1)
 
    V2_mean = np.mean(V2,axis=1)
    

    max_V_mean = np.max([V1_mean,V2_mean])

    V1_stddev = np.std(V1,axis=1)
    V2_stddev = np.std(V2,axis=1)
    

    max_V_stddev = np.max([V1_stddev,V2_stddev])

   
    P_x_mean = np.mean(P_x,axis=1)
   
    P_y_mean = np.mean(P_y,axis=1)

    
    max_P_mean = np.max([P_x_mean,P_y_mean])

    P_x_stddev = np.std(P_x,axis=1)
    P_y_stddev = np.std(P_y,axis=1)
   

    max_P_stddev = np.max([P_x_stddev,P_y_stddev])

    max_P_mean_total = np.max([max_P_mean_total_t0, np.max(P_x_mean),np.max(P_y_mean)])
    max_P_stddev_total = np.max([max_P_stddev_total_t0,np.max(P_x_stddev), np.max(P_y_stddev)])
    
   

    max_V_mean_total = np.max([max_V_mean_total_t0, np.max(V1_mean), np.max(V2_mean)])
    max_V_mean_total = np.max([max_V_mean_total_t1, np.max(V1_mean), np.max(V2_mean)])
    

    min_P_mean_total= np.min([min_P_mean_total_t0, np.min(P_x_mean),np.min(P_y_mean)])
    min_P_stddev_total = np.min([min_P_stddev_total_t0,np.min(P_x_stddev), np.min(P_y_stddev)])
    

    min_V_mean_total = np.min([min_V_mean_total_t0, np.min(V1_mean), np.min(V2_mean)])
    min_V_mean_total = np.min([min_V_mean_total_t1, np.min(V1_mean), np.min(V2_mean)])
   

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('average cost over 500 episodes', fontsize = 9, fontweight='bold')
    
    ax1.set_xlim([0,6000])
    #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])
    
    ax1.set_ylim([10, 90])
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('average cost')

    if(len(reward_mean)== 0):
        ax1.plot(avg_reward_mean, label = 'mean:',color='blue')
        ax1.fill_between(fill_axis,avg_reward_mean,avg_reward_mean + avg_reward_stddev, alpha = 0.3, color='dodgerblue', label = 'standard deviation:')
        ax1.fill_between(fill_axis,avg_reward_mean,avg_reward_mean - avg_reward_stddev, alpha = 0.3, color='dodgerblue')
    else:
        ax1.plot(avg_reward_mean, label = 'mean: {}'.format(np.round(avg_reward_mean[-1],2)),color='blue')
        ax1.fill_between(fill_axis,avg_reward_mean,avg_reward_mean + avg_reward_stddev, alpha = 0.3, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(avg_reward_stddev[-1],2)))
        ax1.fill_between(fill_axis,avg_reward_mean,avg_reward_mean - avg_reward_stddev, alpha = 0.3, color='dodgerblue')
    ax1.legend()
    ax1.hlines(base_mean, xmin = 0, xmax = 6000, color = 'black')
    ax1.fill_between([0,6000],[base_mean -base_stddev,base_mean-base_stddev], [base_mean +base_stddev,base_mean + base_stddev], alpha = 0.2, color='black')
    

    fig2, ax2 = plt.subplots(2,3, subplot_kw=dict(projection='3d'))
    fig2.suptitle('Episode: {}'.format(i))
    X,Y = np.meshgrid(x_space, y_space)
    fig2.set_tight_layout('tight')
    fig2.set_figheight(6)
    fig2.set_figwidth(7.5)
    ax2 = ax2.ravel()
    for j in range(tn):
        t0 = int(time_points[j])
        if t0 == N:
            t0 -= 1
    
        ax2[j].view_init(elev=15, azim=-70, roll=0)
        ax2[j].plot_surface(X,Y, V_true[j], color= 'black', alpha = 0.2)

        ax2[j].plot_surface(X,Y, V1_mean[j], color= 'dodgerblue')
        ax2[j].plot_surface(X,Y, V2_mean[j], color= 'orange')
        #ax2[j].plot_surface(X,Y, 0.5*(V2_mean[j] +V1_mean[j]), color= 'dodgerblue')
        ax2[j].set_title('value function n = {}'.format(t0), fontsize = 9, fontweight='bold')
    
    

    fig3, ax3 = plt.subplots(2,3, subplot_kw=dict(projection='3d'))
   
    fig3.suptitle('Episode: {}'.format(i))
    fig3.set_figheight(6)
    fig3.set_figwidth(13.5)
    fig3.set_tight_layout('tight')

    ax3 = ax3.ravel()
    for j in range(tn):
        t0 = int(time_points[j])
        if t0 == N:
            t0 -= 1

    
        ax3[j].view_init(elev=15, azim=-70, roll=0)

   
        ax3[j].plot_surface(X,Y, P_x_true[j], color= 'black', alpha = 0.2)
        ax3[j].plot_surface(X,Y, P_x_mean[j], color= 'dodgerblue')
        ax3[j].set_zlim([-5,5])
        ax3[j].set_title('x component policy n = {}'.format(t0), fontsize = 9, fontweight='bold')

    fig4, ax4 = plt.subplots(2,3, subplot_kw=dict(projection='3d'))
   
    fig4.suptitle('Episode: {}'.format(i))
    fig4.set_figheight(6)
    fig4.set_figwidth(13.5)
    fig4.set_tight_layout('tight')

    ax4 = ax4.ravel()
    for j in range(tn):
        t0 = int(time_points[j])
        if t0 == N:
            t0 -= 1

    
        ax4[j].view_init(elev=15, azim=-70, roll=0)

   
        ax4[j].plot_surface(X,Y, P_y_true[j], color= 'black', alpha = 0.2)
        ax4[j].plot_surface(X,Y, P_y_mean[j], color= 'dodgerblue')
        ax4[j].set_zlim([-5,5])
        ax4[j].set_title('y component policy n = {}'.format(t0), fontsize = 9, fontweight='bold')
    plt.show()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    



# max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1
# 4.043304014205932, 2.935147692749849, 3.703972911834717, 1.4305743895184417, 24.30153121948242, 19.183624267578125, 1.5091561974096852, 1.24356255476597
print('max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1', max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1)

# min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1
#  -4.05456805229187 0.0 -4.011117887496948 0.0 0.0 0.0 0.0 0.0
print('min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1', min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1)

print('np.max([0, np.max(reward_mean)])+5', np.max([0, np.max(reward_mean)])+5)
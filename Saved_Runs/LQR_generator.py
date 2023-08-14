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

#folder = './LQR/'
for i in range(5000,5100,100):
    id = i
# for i in np.arange(100,200,100):
#     id = i +10000
    x_space = np.load(folder +'X_space_40.npy')
    y_space = np.load(folder +'Y_space_40.npy')
    
    
    n_x = len(x_space)

    rewards = np.zeros((num_runs,i))
    avg_rewards = np.zeros((num_runs,i))
    bases = np.zeros(num_runs)
    V1_t0 = np.zeros((num_runs,n_x,n_x))
    V2_t0 = np.zeros((num_runs,n_x,n_x))

    V1_t1 = np.zeros((num_runs,n_x,n_x))
    V2_t1 = np.zeros((num_runs,n_x,n_x))

    P_x_t0 = np.zeros((num_runs,n_x,n_x))
    P_y_t0 = np.zeros((num_runs,n_x,n_x))

    P_x_t1 = np.zeros((num_runs,n_x,n_x))
    P_y_t1 = np.zeros((num_runs,n_x,n_x))

    t0 = 80
    t1 = 99

    V_true_t0 = np.load(folder +'true_value_fct_0.npy'.format(t0))
    V_true_t1 = np.load(folder +'true_value_fct_{}.npy'.format(t1))

    P_x_true_t0 = np.load(folder +'true_policy_x_n_{}.npy'.format(t0))
    P_x_true_t1 = np.load(folder +'true_policy_x_n_{}.npy'.format(t1))
    P_y_true_t0 = np.load(folder +'true_policy_y_n_{}.npy'.format(t0))
    P_y_true_t1 = np.load(folder +'true_policy_y_n_{}.npy'.format(t1))

    fill_axis = np.linspace(0,i,i)
    for r in range(num_runs):

        avg_rewards[r] = np.load(folder +'avg_reward_step_{}_run_{}.npy'.format(id,r))
        bases[r] = np.load(folder +'base_run_{}.npy'.format(r))
        
        rewards[r] = np.load(folder +'ep_reward_list_{}_run_{}.npy'.format(id,r))
        V1_t0[r] = np.load(folder +'value_fct_1_n_{}_{}_run_{}.npy'.format(t0,id,r))
        V2_t0[r] = np.load(folder +'value_fct_2_n_{}_{}_run_{}.npy'.format(t0,id,r))

        V1_t1[r] = np.load(folder +'value_fct_1_n_{}_{}_run_{}.npy'.format(t1,id,r))
        V2_t1[r] = np.load(folder +'value_fct_2_n_{}_{}_run_{}.npy'.format(t1,id,r))

        P_x_t0[r] = np.load(folder +'policy_X_n_{}_{}_run_{}.npy'.format(t0,id,r))
        P_y_t0[r] = np.load(folder +'policy_Y_n_{}_{}_run_{}.npy'.format(t0,id,r))

        P_x_t1[r] = np.load(folder +'policy_X_n_{}_{}_run_{}.npy'.format(t1,id,r))
        P_y_t1[r] = np.load(folder +'policy_Y_n_{}_{}_run_{}.npy'.format(t1,id,r))
    
    reward_mean = np.mean(rewards, axis=0)
    reward_stddev = np.std(rewards, axis = 0)

    avg_reward_mean = np.mean(avg_rewards, axis=0)
    avg_reward_stddev = np.std(avg_rewards, axis = 0)
    
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

   
    P_x_t0_mean = np.mean(P_x_t0,axis=0)
   
    P_y_t0_mean = np.mean(P_y_t0,axis=0)
    P_x_t1_mean = np.mean(P_x_t1,axis=0)
    P_y_t1_mean = np.mean(P_y_t1,axis=0)
    
    max_P_mean = np.max([P_x_t0_mean,P_y_t0_mean,P_x_t1_mean,P_y_t1_mean])

    P_x_t0_stddev = np.std(P_x_t0,axis=0)
    P_y_t0_stddev = np.std(P_y_t0,axis=0)
    P_x_t1_stddev = np.std(P_x_t1,axis=0)
    P_y_t1_stddev = np.std(P_y_t1,axis=0)

    max_P_stddev = np.max([P_x_t0_stddev,P_y_t0_stddev,P_x_t1_stddev,P_y_t1_stddev])

    max_P_mean_total_t0 = np.max([max_P_mean_total_t0, np.max(P_x_t0_mean),np.max(P_y_t0_mean)])
    max_P_stddev_total_t0 = np.max([max_P_stddev_total_t0,np.max(P_x_t0_stddev), np.max(P_y_t0_stddev)])
    max_P_mean_total_t1 = np.max([max_P_mean_total_t1, np.max(P_x_t1_mean), np.max(P_y_t1_mean)])
    max_P_stddev_total_t1 = np.max([max_P_stddev_total_t1, np.max(P_x_t1_stddev), np.max(P_y_t1_stddev)])
   

    max_V_mean_total_t0 = np.max([max_V_mean_total_t0, np.max(V1_t0_mean), np.max(V2_t0_mean)])
    max_V_mean_total_t1 = np.max([max_V_mean_total_t1, np.max(V1_t1_mean), np.max(V2_t1_mean)])
    max_V_stddev_total_t0 = np.max([max_V_stddev_total_t0, np.max(V1_t0_stddev), np.max(V2_t0_stddev)])
    max_V_stddev_total_t1 = np.max([max_V_stddev_total_t1, np.max(V1_t1_stddev), np.max(V2_t1_stddev)])

    min_P_mean_total_t0 = np.min([min_P_mean_total_t0, np.min(P_x_t0_mean),np.min(P_y_t0_mean)])
    min_P_stddev_total_t0 = np.min([min_P_stddev_total_t0,np.min(P_x_t0_stddev), np.min(P_y_t0_stddev)])
    min_P_mean_total_t1 = np.min([min_P_mean_total_t1, np.min(P_x_t1_mean), np.min(P_y_t1_mean)])
    min_P_stddev_total_t1 = np.min([min_P_stddev_total_t1, np.min(P_x_t1_stddev), np.min(P_y_t1_stddev)])
   

    min_V_mean_total_t0 = np.min([min_V_mean_total_t0, np.min(V1_t0_mean), np.min(V2_t0_mean)])
    min_V_mean_total_t1 = np.min([min_V_mean_total_t1, np.min(V1_t1_mean), np.min(V2_t1_mean)])
    min_V_stddev_total_t0 = np.min([min_V_stddev_total_t0, np.min(V1_t0_stddev), np.min(V2_t0_stddev)])
    min_V_stddev_total_t1 = np.min([min_V_stddev_total_t1, np.min(V1_t1_stddev), np.min(V2_t1_stddev)])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('average cost over 500 episodes', fontsize = 9, fontweight='bold')
    
    ax1.set_xlim([0,5000])
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
    #ax1.hlines(base_mean, xmin = 0, xmax = 5000, color = 'black')
    #ax1.fill_between([0,5000],[base_mean -base_stddev,base_mean-base_stddev], [base_mean +base_stddev,base_mean + base_stddev], alpha = 0.2, color='black')
    

    fig2 =plt.figure()
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[1.5,1])
    fig2.suptitle('Episode: {}'.format(i))
    
    fig2.set_tight_layout('tight')
    fig2.set_figheight(6)
    fig2.set_figwidth(7.5)
    
    ax2_1 = fig2.add_subplot(gs2[0,0] ,projection='3d')
    ax2_1.view_init(elev=15, azim=-70, roll=0)

    X,Y = np.meshgrid(x_space, y_space)
    ax2_1.plot_surface(X,Y, V_true_t0, color= 'black', alpha = 0.2)
    # ax2_1.plot_surface(X,Y, V1_t0_mean+2, color= 'dodgerblue')
    # ax2_1.plot_surface(X,Y, V2_t0_mean+2, color= 'orange')
    ax2_1.plot_surface(X,Y, 0.5*(V2_t0_mean +V1_t0_mean), color= 'dodgerblue')
    ax2_1.set_title('value function n = {}'.format(t0), fontsize = 9, fontweight='bold')
    
    ax2_2 = fig2.add_subplot(gs2[0,1] ,projection='3d')
    ax2_2.view_init(elev=15, azim=-70, roll=0)

    
    ax2_2.plot_surface(X,Y, V_true_t1, color= 'black', alpha = 0.2)
    # ax2_2.plot_surface(X,Y, V1_t1_mean+2, color= 'dodgerblue')
    # ax2_2.plot_surface(X,Y, V2_t1_mean+2, color= 'orange')
    ax2_2.plot_surface(X,Y, 0.5*(V2_t1_mean +V1_t1_mean), color= 'dodgerblue')
    
    ax2_2.set_title('value function n = {}'.format(t1), fontsize = 9, fontweight='bold')

    #ax2_1.set_zlim([-6,25])
    #ax2_2.set_zlim([-6,25])
    
    
    ax2_3 = fig2.add_subplot(gs2[1,0] )
    ax2_3.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_3.set_title('standard deviation n = {}'.format(t0), fontsize = 9, fontweight='bold')
    cont2_3 = ax2_3.pcolormesh(X,Y, 0.5*(V1_t0_stddev + V2_t0_stddev),vmin=0, vmax=12)

    ax2_4 = fig2.add_subplot(gs2[1,1])
    ax2_4.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_4.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')
    cont2_4 = ax2_4.pcolormesh(X,Y, 0.5*(V1_t1_stddev + V2_t1_stddev),vmin=0, vmax=12)

    c_bar2 = fig2.colorbar(cont2_4, orientation='vertical')
    


    fig3 =plt.figure()
    gs3 = fig3.add_gridspec(2, 4, height_ratios=[1.5,1])
    fig3.suptitle('Episode: {}'.format(i))
    fig3.set_figheight(6)
    fig3.set_figwidth(13.5)
    fig3.set_tight_layout('tight')

    ax3_1 = fig3.add_subplot(gs3[0,0],projection='3d')
    ax3_1.view_init(elev=15, azim=-70, roll=0)

   
    ax3_1.plot_surface(X,Y, P_x_true_t0, color= 'black', alpha = 0.2)
    ax3_1.plot_surface(X,Y, P_x_t0_mean, color= 'dodgerblue')
    ax3_1.set_zlim([-5,5])
    ax3_1.set_title('x component policy n = {}'.format(t0), fontsize = 9, fontweight='bold')

    ax3_2 = fig3.add_subplot(gs3[0,1],projection='3d')
    ax3_2.view_init(elev=15, azim=-70, roll=0)
    ax3_2.plot_surface(X,Y, P_y_true_t0, color= 'black', alpha = 0.2)
    ax3_2.plot_surface(X,Y, P_y_t0_mean, color= 'dodgerblue')
    #ax3_2.set_zlim([-5,5])
    ax3_2.set_title('y component policy n = {}'.format(t0), fontsize = 9, fontweight='bold')


    ax3_3 = fig3.add_subplot(gs3[0,2],projection='3d')
    ax3_3.view_init(elev=15, azim=-70, roll=0)

   
    ax3_3.plot_surface(X,Y, P_x_true_t1, color= 'black', alpha = 0.2)
    ax3_3.plot_surface(X,Y, P_x_t1_mean, color= 'dodgerblue')
    #ax3_3.set_zlim([-5,5])
    ax3_3.set_title('x component policy n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax3_4 = fig3.add_subplot(gs3[0,3],projection='3d')
    ax3_4.view_init(elev=15, azim=-70, roll=0)
    ax3_4.plot_surface(X,Y, P_y_true_t1, color= 'black', alpha = 0.2)
    ax3_4.plot_surface(X,Y, P_y_t1_mean, color= 'dodgerblue')
    #ax3_4.set_zlim([-5,5])
    ax3_4.set_title('y component policy n = {}'.format(t1), fontsize = 9, fontweight='bold')

    ax3_5 = fig3.add_subplot(gs3[1,0])
    ax3_5.set_aspect('equal')
    ax3_5.pcolormesh(X,Y,  P_x_t0_stddev,vmin=0, vmax=5)
    ax3_5.set_title('standard deviation n = {}'.format(t0), fontsize = 9, fontweight='bold')
   
   
    ax3_6 = fig3.add_subplot(gs3[1,1])
    ax3_6.set_aspect('equal')
    cont2 = ax3_6.pcolormesh(X,Y,  P_y_t0_stddev,vmin=0, vmax=5)
    ax3_6.set_title('standard deviation n = {}'.format(t0), fontsize = 9, fontweight='bold')

    ax3_7 = fig3.add_subplot(gs3[1,2])
    ax3_7.set_aspect('equal')
    ax3_7.pcolormesh(X,Y,  P_x_t1_stddev,vmin=0, vmax=5)
    ax3_7.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')
   
   
    ax3_8 = fig3.add_subplot(gs3[1,3])
    ax3_8.set_aspect('equal')
    cont8 = ax3_8.pcolormesh(X,Y,  P_y_t1_stddev,vmin=0, vmax=5)
    ax3_8.set_title('standard deviation n = {}'.format(t1), fontsize = 9, fontweight='bold')




    fig3.colorbar(cont8, orientation='vertical')
   
    # fig1.savefig('./LQR/images/reward_epi_{}'.format(i))
    # fig2.savefig('./LQR/images/value_fct_epi_{}'.format(i))
    # fig3.savefig('./LQR/images/policy_epi_{}'.format(i))
    plt.show()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    



# max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1
# 4.043304014205932, 2.935147692749849, 3.703972911834717, 1.4305743895184417, 24.30153121948242, 19.183624267578125, 1.5091561974096852, 1.24356255476597
print('max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1', max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1)

# min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1
#  -4.05456805229187 0.0 -4.011117887496948 0.0 0.0 0.0 0.0 0.0
print('min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1', min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1)

print('np.max([0, np.max(reward_mean)])+5', np.max([0, np.max(reward_mean)])+5)
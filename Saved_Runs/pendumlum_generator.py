import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# file names

num_runs = 1
iteration_steps = 100

#\pendulum\avg_reward_step_0_run_0.npy

max_P_mean_total = 0
max_P_stddev_total = 0


max_V_mean_total = 0

max_V_stddev_total = 0


min_P_mean_total= 0
min_P_stddev_total = 0


min_V_mean_total = 0
min_V_mean_total = 0



for i in np.arange(2000,2100,100):
    X = np.load('./pendulum/X_space.npy')
   
    n_x = len(X)

    

    rewards = np.zeros((num_runs,i))


    #bases = np.zeros(num_runs)
    V1 = np.zeros((num_runs,n_x))
    V2 = np.zeros((num_runs,n_x))

    P = np.zeros((num_runs,n_x))

    V_true = np.load('./pendulum/true_value_fct.npy')
   

    P_true = np.load('./pendulum/true_policy.npy')
 
    fill_axis = np.linspace(0,i,i)
    for r in range(num_runs):
        print(r)
        rewards[r] = np.load('./pendulum/avg_reward_step_{}_run_{}.npy'.format(i,r))
        

        #bases[r] = np.load('./pendulum/base_run_{}.npy'.format(r))

        V1[r] = np.load('./pendulum/value_fct_1_{}_run_{}.npy'.format(i,r))
        V2[r] = np.load('./pendulum/value_fct_2_{}_run_{}.npy'.format(i,r))

        
        P[r] = np.load('./pendulum/policy_{}_run_{}.npy'.format(i,r))
       


    reward_mean = np.mean(rewards, axis=0)
    
    reward_stddev = np.std(rewards, axis = 0)

   
    
    # base_mean = np.mean(bases)
    # base_stddev = np.std(bases)

    V1_mean = np.mean(V1,axis=0)
    V2_mean = np.mean(V2,axis=0)
    
    max_V_mean = np.max([V1_mean,V2_mean])

    V1_stddev = np.std(V1,axis=0)
    V2_stddev = np.std(V2,axis=0)
  
    max_V_stddev = np.max([V1_stddev,V2_stddev])


    P_mean = np.mean(P, axis=0)
    

    P_stddev = np.std(P, axis=0)
  

    max_P_mean_total = np.max([max_P_mean_total, np.max(P_mean)])
    max_P_stddev_total = np.max([max_P_stddev_total,np.max(P_stddev)])
   
   
    max_V_mean_total = np.max([max_V_mean_total, np.max(V1_mean), np.max(V2_mean)])
    max_V_stddev_total = np.max([max_V_stddev_total, np.max(V1_stddev), np.max(V2_stddev)])
   

    min_P_mean_total = np.min([min_P_mean_total, np.min(P_mean)])
    min_P_stddev_total= np.min([min_P_stddev_total,np.min(P_stddev)])
    
   

    min_V_mean_total = np.min([min_V_mean_total, np.min(V1_mean), np.min(V2_mean)])
    min_V_mean_total = np.min([min_V_mean_total, np.min(V1_mean), np.min(V2_mean)])
    

    fig1 = plt.figure()
    #fig1.set_tight_layout('tight')
    #fig1.set_figheight(5)
    #fig1.set_figwidth(10)

    ax1_1 = fig1.add_subplot()
    
    
    ax1_1.set_title('average cost over 500 episodes', fontsize = 9, fontweight='bold')
    
    ax1_1.set_xlim([0,5000])
    #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])

   
    ax1_1.set_xlabel('episodes')
    ax1_1.set_ylabel('average cost')
    print('len(reward_mean)',len(reward_mean))
    if(len(reward_mean)== 0):
        ax1_1.plot(reward_mean, label = 'mean: ', color='dodgerblue')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean + reward_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: ')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean - reward_stddev, alpha = 0.2, color='dodgerblue')
    else:    
        ax1_1.plot(reward_mean, label = 'mean: {}'.format(np.round(reward_mean[-1],2)), color='dodgerblue')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean + reward_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(reward_stddev[-1],2)))
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean - reward_stddev, alpha = 0.2, color='dodgerblue')
    #ax1_1.set_ylim([0, 15])
    ax1_1.legend()
   
    # ax1_2 = fig1.add_subplot(1,2,2)
    
    # ax1_2.set_title('average stopping time over 500 episodes', fontsize = 9, fontweight='bold')
    
    # ax1_2.set_xlim([0,5000])
    # #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])

    # ax1_2.set_ylim([0, np.max([0, np.max(stopping_mean)])+0.2])
    # ax1_2.set_xlabel('episodes')
    # ax1_2.set_ylabel('average stopping time')

    # ax1_2.plot(stopping_mean, label = 'mean: {}'.format(np.round(stopping_mean[-1],2)),color='dodgerblue')
    # ax1_2.fill_between(fill_axis,stopping_mean,stopping_mean + stopping_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(stopping_stddev[-1],2)))
    # ax1_2.fill_between(fill_axis,stopping_mean,stopping_mean - stopping_stddev, alpha = 0.2, color='dodgerblue')
    # ax1_2.legend()
    #ax1.hlines(base_mean, xmin = 0, xmax = 5000, color = 'black')
    #ax1.fill_between([0,5000],[base_mean -base_stddev,base_mean-base_stddev], [base_mean +base_stddev,base_mean + base_stddev], alpha = 0.2, color='black')
    

    fig2 =plt.figure()
    fig2.suptitle('Epsiode: {}'.format(i))
    gs2 = fig2.add_gridspec(2,1 , height_ratios = [1,1])
    fig2.set_tight_layout('tight')
    fig2.set_figheight(7)
    fig2.set_figwidth(9)
   
    
    ax2_1 = fig2.add_subplot(gs2[0])

    
    ax2_1.plot(X, V_true, color= 'black', alpha = 0.2)
    # ax2_1.plot_surface(X,Y, V1_mean, color= 'dodgerblue')
    # ax2_1.plot_surface(X,Y, V2_mean, color= 'coral')
    ax2_1.plot(X, 0.5*(V1_mean + V2_mean), color= 'dodgerblue')
    ax2_1.set_title('value function', fontsize = 9, fontweight='bold')
     
    
    ax2_2 = fig2.add_subplot(gs2[1])
    ax2_2.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_2.set_title('standard deviation', fontsize = 9, fontweight='bold')
    ax2_2.plot(X, 0.5*(V1_stddev + V2_stddev))


    fig3 =plt.figure()
    fig3.suptitle('Epsiode: {}'.format(i))
    gs3 = fig2.add_gridspec(2,1 , height_ratios = [1,1])
    fig3.set_tight_layout('tight')
    fig3.set_figheight(6)
    fig3.set_figwidth(7)
   

    ax3_1 = fig3.add_subplot(gs3[0])
    
    ax3_1.plot(X, P_true, color= 'black', alpha = 0.2, label="$u^*(x) = -\sigma \nabla V^\epsilon(x)$")
    ax3_1.plot(X, P_mean, color= 'dodgerblue')
    #ax3_1.set_zlim([-4.1,4.1])
    ax3_1.set_title('policy', fontsize = 9, fontweight='bold')
    
    ax3_2 = fig3.add_subplot(gs3[1])

    ax3_2.set_aspect('equal')
    ax3_2.plot(X,  P_stddev)
    ax3_2.set_title('standard deviation', fontsize = 9, fontweight='bold')
    
    plt.show()
    # fig1.savefig('./pendulum/images/reward_epi_{}'.format(i))
    # fig2.savefig('./pendulum/images/value_fct_epi_{}'.format(i))
    # fig3.savefig('./pendulum/images/policy_epi_{}'.format(i))
   

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
  
    #plt.show()

# max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1
# 4.043304014205932, 2.935147692749849, 3.703972911834717, 1.4305743895184417, 24.30153121948242, 19.183624267578125, 1.5091561974096852, 1.24356255476597
print('max_P_mean_total,max_P_stddev_total,max_V_mean_total,max_V_stddev_total', max_P_mean_total,max_P_stddev_total,max_V_mean_total,max_V_stddev_total)

# min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1
#  -4.05456805229187 0.0 -4.011117887496948 0.0 0.0 0.0 0.0 0.0
print('min_P_mean_total,min_P_stddev_total,min_V_mean_total,min_V_mean_total', min_P_mean_total,min_P_stddev_total,min_V_mean_total,min_V_mean_total)

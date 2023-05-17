import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def visualization_one_epoch(
    ego_accel, 
    lead_accel, 
    ego_velocity, 
    lead_velocity, 
    deviation_opt_distance):
    
    t = range(len(ego_accel))
    
    fig, axs = plt.subplots(3, 1)
    
    axs[0].plot(lead_accel, label="Vorausfahrendes Auto",color='#ff7f0e')
    axs[0].plot(t, ego_accel, label="Ego Auto", color='#1f77b4')
    axs[0].set_xlim(0, len(t))
    axs[0].set_title('Beschleunigung (in m/s²)')
    axs[0].set_xlabel('Zeit (in 100ms)')
    fig.legend(bbox_to_anchor=(0.965, 1.015), prop={'size':8})

    axs[1].plot(t, lead_velocity, color='#ff7f0e')
    axs[1].plot(t, ego_velocity, color='#1f77b4')
    axs[1].set_xlim(0, len(t))
    axs[1].set_title('Geschwindigkeit (in m/s)')
    axs[1].set_xlabel('Zeit (in 100ms)')

    axs[2].plot(deviation_opt_distance, color="green")
    axs[2].set_xlim(0, len(t))
    axs[2].set_title('Abweichung vom Optimalabstand (in m)')
    axs[2].set_xlabel('Zeit (in 100ms)')

    fig.tight_layout()
    plt.show()
    fig.savefig("bild", dpi=1000)

def plot_return(reward_list, path):
    fig, ax = plt.subplots()
    ax.plot(reward_list)
    ax.set(xlabel='Episode', ylabel='Durchschittlicher Return',)
    fig.savefig(path + "return.png", dpi=1000)
    plt.show()


def save_data(path, ep_reward_list, avg_reward_list, num_steps_list):
    with open(path + 'ep_reward_list.pickle', 'wb') as handle:
        pickle.dump(ep_reward_list, handle)
    with open(path + 'avg_reward_list.pickle', 'wb') as handle:
        pickle.dump(avg_reward_list, handle)
    with open(path + 'num_steps_list.pickle', 'wb') as handle:
        pickle.dump(num_steps_list, handle)
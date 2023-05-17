from time import time


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


from model.noise import OUActionNoise
from model.buffer import Buffer
from utils import *


def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(3,))
    out = layers.Dense(32, activation="relu")(inputs)
    out = layers.Dense(32, activation="relu")(out)
    output = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # scale output
    output = output * 3
    model = tf.keras.Model(inputs, output)
    return model


def get_critic():
    inputs = layers.Input(shape=(4,))
    out = layers.Dense(32, activation="relu")(inputs)
    out = layers.Dense(32, activation="relu")(out)
    output = layers.Dense(1)(out)

    model = tf.keras.Model(inputs, output)
    return model


class DDPG:
    def __init__(self, path=""):
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # intialize models
        self.actor_model = get_actor()
        self.critic_model = get_critic()
        self.target_actor = get_actor()
        self.target_critic = get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # initialize optimizers
        critic_lr = 0.002
        actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.total_episodes = 1000
        self.num_steps = 300
        
        self.gamma = 0.99
        self.tau = 0.005

        self.buffer = Buffer(500000, 64)
        self.path = path


    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            state_action_tuple = tf.concat([next_state_batch, target_actions], axis=1) 
            y = reward_batch + self.gamma * self.target_critic(
                [state_action_tuple], training=True
            )
            state_action_tuple = tf.concat([state_batch, action_batch], axis=1)
            critic_value = self.critic_model([state_action_tuple], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        # update actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            state_action_tuple = tf.concat([state_batch, actions], axis=1)
            critic_value = self.critic_model([state_action_tuple], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )


    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        # add noise
        sampled_actions = sampled_actions.numpy() + noise
        action = np.clip(sampled_actions, -3, 3)

        return [np.squeeze(action)]


    def train(self, env):
        ep_reward_list = []
        avg_reward_list = []
        num_steps_list = []

        highest_reward = 0
        highest_avg_reward = 0
        
        start_time = time()
        for ep in range(self.total_episodes):
            
            prev_state = env.reset()
            episodic_reward = 0
            num_steps = 0
            start_ep_time = time()
            for i in range(self.num_steps):
                
                num_steps += 1

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state[0]), 0)

                action = self.policy(tf_prev_state)
                state, reward, done = env.step(action)

                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()
                self.update(state_batch, action_batch, reward_batch, next_state_batch)
                self.update_target(self.target_actor.variables, self.actor_model.variables)
                self.update_target(self.target_critic.variables, self.critic_model.variables)

                if done:
                    break

                prev_state = state

            end_ep_time = time()

            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Time: {:4.4f}| Episode {} Avg Reward is {:4.4f}".format((end_ep_time - start_ep_time)/60, ep, avg_reward))
            avg_reward_list.append(avg_reward)
            num_steps_list.append(num_steps)

            if highest_reward < episodic_reward:
                highest_reward = episodic_reward
                self.actor_model.save(self.path + "actor_highhest_reward_model.h5")
                self.critic_model.save(self.path + "critic_highhest_reward_model.h5")
                self.target_actor.save(self.path + "t_actor_highhest_reward_model.h5")
                self.target_critic.save(self.path + "t_critic_highhest_reward_model.h5")

            if highest_avg_reward < avg_reward:
                highest_avg_reward = avg_reward
                self.actor_model.save(self.path + "actor_highhest_avg_reward_model.h5")
                self.critic_model.save(self.path + "critic_highhest_avg_reward_model.h5")
                self.target_actor.save(self.path + "t_actor_highhest_avg_reward_model.h5")
                self.target_critic.save(self.path + "t_critic_highhest_avg_reward_model.h5")
        
        end_time = time()
        print("Training time: {:20.4f}".format((end_time - start_time)/60))
        self.actor_model.save(self.path + "last_episode_model.h5")
        self.critic_model.save(self.path + "critic_last_episode_model.h5")
        self.target_actor.save(self.path + "t_actor_last_episode_model.h5")
        self.target_critic.save(self.path + "t_critic_last_episode_model.h5")
        print("Model saved.")

        return ep_reward_list, avg_reward_list, num_steps_list
    

    def eval(self, env, seed=32, num_ep=1):
        set_random_seed(seed)
        state = env.reset()
        
        ego_velocities = [env.ego_velocity]
        lead_velocities = [env.lead_velocity]
        ego_accelerations = []
        lead_accelerations = []
        jerks = []
        opt_distances = [env.opt_distance]
        distances = [env.distance]
        deviations_opt_distance = [env.deviation_opt_distance]
        collision = []
        ttcs = []
        rewards = []

        for _ in range(num_ep):
            for i in range(600):

                sampled_actions = tf.squeeze(self.actor_model(state))
                sampled_actions = sampled_actions.numpy()
                legal_action = np.clip(sampled_actions, -3, 3)
                action = [np.squeeze(legal_action)]
                
                state, reward, done = env.step(action)

                # data 
                ego_velocities.append(env.ego_velocity)
                lead_velocities.append(env.lead_velocity)
                ego_accelerations.append(env.accel)
                lead_accelerations.append(env.lead_accel)
                jerks.append(env.accel - env.prev_accel)
                opt_distances.append(env.opt_distance)
                distances.append(env.distance)
                deviations_opt_distance.append(env.deviation_opt_distance)
                if np.clip(env.ego_velocity - env.lead_velocity, 0, 30) != 0:
                    ttcs.append(env.distance/np.clip(env.ego_velocity - env.lead_velocity, 0, 30))
                rewards.append(reward)
                

                # End this episode when `done` is True
                if done:
                    collision.append("step: " + str(i))
                    break
            
        return {
            "ego_velocities": ego_velocities,
            "ego_accelerations": ego_accelerations,
            "lead_accelerations": lead_accelerations, 
            "jerks": jerks, 
            "opt_distances": opt_distances, 
            "distances": distances, 
            "deviations_opt_distance": deviations_opt_distance, 
            "collision": collision,
            "ttcs": ttcs,
            "rewards": rewards,
            }
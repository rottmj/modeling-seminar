import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self, size=100000, batch_size=64, num_actions=1, num_states=3):
        self.size = size
        self.batch_size = batch_size

        self.counter = 0
        
        self.num_actions = num_actions
        self.num_states = num_states

        self.state_buffer = np.zeros((self.size, self.num_states))
        self.action_buffer = np.zeros((self.size, self.num_actions))
        self.reward_buffer = np.zeros((self.size, 1))
        self.next_state_buffer = np.zeros((self.size, self.num_states))


    def record(self, obs_tuple):
        index = self.counter % self.size
        
        self.state_buffer[index] = obs_tuple[0][0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.counter += 1


    def sample(self):
        record_range = min(self.counter, self.size)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch
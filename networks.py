import os


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=16, fc2_dims=16, 
            name='critic', dir=""):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.dir = dir
        self.file = os.path.join(self.dir, 
                    self.model_name + '.h5')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu', kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.q = Dense(1, activation=None, kernel_initializer='he_normal')

    @tf.function
    def call(self, state, action):
        action_value = self.bn1(tf.concat([state, action], axis=1))
        action_value = self.fc1(action_value)
        action_value = self.bn2(action_value)
        action_value = self.fc2(action_value)
        action_value = self.bn3(action_value)
        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=16, fc2_dims=16, n_actions=1, 
            name='actor', dir=""):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.dir = dir
        self.file = os.path.join(self.dir, 
                    self.model_name + '.h5')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu', kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.output_layer = Dense(1, activation='tanh', kernel_initializer='he_normal')

    @tf.function
    def call(self, state):
        state = self.bn1(state)
        activation = self.fc1(state)
        activation = self.bn2(activation)
        activation = self.fc2(activation)
        activation = self.bn3(activation)
        action = self.output_layer(activation)

        return action

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np


from DDPG.buffer import ExperienceReplayMemory
from DDPG.networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_shape, alpha=0.001, beta=0.0001,
            gamma=0.99, n_actions=1, max_size=500000, tau=0.005,
            batch_size=64, noise=0.02, dir=""):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ExperienceReplayMemory(max_size, input_shape, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise

        # Minimal and maximal action value [m/s²]
        self.max_action = 10
        self.min_action = -10

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', dir=dir)
        self.critic = CriticNetwork(name='critic', dir=dir)
        self.target_actor = ActorNetwork(n_actions=n_actions, 
                                        name='target_actor', dir=dir)
        self.target_critic = CriticNetwork(name='target_critic', dir=dir)

        self.actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm = 1.0))
        self.critic.compile(optimizer=Adam(learning_rate=beta, clipnorm = 1.0))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha, clipnorm = 1.0))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta, clipnorm = 1.0))

        self.update_network_parameters(tau=1)

    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)


    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)


    def save_models(self): 
        print('... saving models ...')
        self.actor.save_weights(self.actor.file)
        self.target_actor.save_weights(self.target_actor.file)
        self.critic.save_weights(self.critic.file)
        self.target_critic.save_weights(self.target_critic.file)
        print('... models saved ...')


    def load_actor_model(self, path):
        x = np.array([[0, 0]])
        self.actor(x)
        self.actor.load_weights(path)
        print('... actor model loaded ...')


    def load_models(self, paths):
        x = np.array([[0, 0]])
        y = np.array([[0]])
        self.actor(x)
        self.critic(x, y)
        self.target_actor(x)
        self.target_critic(x, y)
        self.actor.load_weights(paths[0])
        self.target_actor.load_weights(paths[1])
        self.critic.load_weights(paths[2])
        self.target_critic.load_weights(paths[3])
        print('... models loaded ...')


    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)

        if not evaluate:
            action += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
       
        # Clip or scale action
        # action = tf.clip_by_value(action, self.min_action, self.max_action)
        action = tf.math.multiply(3, action)
        action = tf.clip_by_value(action, -3, 3)
        #print(action)
        return action[0]


    @tf.function
    def update_critic(self, states, actions, rewards, next_state, done):
        target_actions = self.target_actor(next_state)
        critic_value_ = tf.squeeze(self.target_critic(next_state, target_actions), 1)
        critic_value = tf.squeeze(self.critic(states, actions), 1)
        target = rewards + self.gamma*critic_value_#*(1-done)
        return keras.losses.MSE(target, critic_value)

    @tf.function
    def update_actor(self, states):
        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions)
        actor_loss = tf.math.reduce_mean(actor_loss)
        return actor_loss
    
    
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        import time
        tic = time.perf_counter()
        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)
        done = tf.convert_to_tensor(done, dtype=tf.bool)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        
        




        
        tac = time.perf_counter()
        #print(f" loading {tac-tic:0.4f} seconds")

        tic = time.perf_counter()

        with tf.GradientTape() as tape:
            result = self.update_critic(states, actions, rewards, next_state, done)
        critic_network_gradient = tape.gradient(result, self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        tac = time.perf_counter()
        #print(f"critic {tac-tic:0.4f} seconds")
        tic = time.perf_counter()

        with tf.GradientTape() as tape:
            result = self.update_actor(states)

        actor_network_gradient = tape.gradient(result, self.actor.trainable_variables)
        a = self.actor.trainable_variables
        print(actor_network_gradient)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        tac = time.perf_counter()
        print(a == self.actor.trainable_variables)
        #print(f"actor {tac-tic:0.4f} seconds")


        tic = time.perf_counter()
        self.update_network_parameters()
        tac = time.perf_counter()
        #print(f"update{tac-tic:0.4f} seconds")

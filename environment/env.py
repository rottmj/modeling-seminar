import numpy as np 


class CarFollowingEnvironment:
    def __init__(
            self,
            config,
            alpha=1,    
            beta=1,
            gamma=1):
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def reset(self):
        # initialize ego vehicle 
        self.ego_velocity = np.random.uniform(
            low=self.config["ego_low_velocity"], 
            high=self.config["ego_high_velocity"])
        self.accel = 0
        self.prev_accel = 0
        
        # initialize lead vehicle
        self.lead_velocity = np.random.uniform(
            low=self.config["lead_low_velocity"], 
            high=self.config["lead_high_velocity"])
        self.lead_accel = 0
        
        # initialize distance values
        self.distance = np.random.uniform(
            low=self.config["low_distance"], 
            high=self.config["high_distance"])
        self.opt_distance = \
            self.config["time_gap"] * self.lead_velocity + self.config["minimum_gap"]
        self.deviation_opt_distance = self.distance - self.opt_distance
        
        return self._get_state()


    def step(self, action):
        # update ego velocity
        self.prev_accel = self.accel
        self.accel = action[0]
        self.prev_ego_velocity = self.ego_velocity
        self.ego_velocity = np.clip(
            self.ego_velocity + self.config["time_step"] * self.accel, 
            self.config["ego_low_velocity"], 
            self.config["ego_high_velocity"])
        
        # update lead velocity
        self.prev_lead_velocity = self.lead_velocity
        self._update_lead_velocity()
        
        # update distance values
        self._update_distance()
        self.opt_distance = \
            self.config["time_gap"] * self.lead_velocity + self.config["minimum_gap"]
        self.deviation_opt_distance = self.distance - self.opt_distance
        
        # calculate reward
        reward = self._get_reward()

        done = False
        
        # if collision, terminate episode
        if self.distance < 0:
            reward = self.config["collision_reward"]
            done = True
        
        return self._get_state(), reward, done
    
    
    def _get_state(self):
        # velocity difference
        diff_velocity = \
            (self.lead_velocity - self.ego_velocity)/self.config["state_normalizing_factors"][0]
        # deviation from optimal diatance
        deviation_opt_distance = \
            self.deviation_opt_distance/self.config["state_normalizing_factors"][1]
        # current ego acceleration
        ego_accel = self.accel/self.config["state_normalizing_factors"][2]
        
        return np.array([[diff_velocity, deviation_opt_distance, ego_accel]])

    
    def _update_lead_velocity(self):
        prev_velocity = self.lead_velocity
        self.lead_velocity += 0.132 * (7.5 - self.lead_velocity) \
            * 0.1 + 3.847 * np.random.normal(0, 0.1)
        self.lead_velocity = np.clip(
            self.lead_velocity, 
            self.config["lead_low_velocity"], 
            self.config["lead_high_velocity"])
        self.lead_accel = 10 * (prev_velocity - self.lead_velocity)


    def _update_distance(self):
        # velocities during the current time step
        current_lead_velocity = (self.lead_velocity + self.prev_lead_velocity)/2
        current_ego_velocity = (self.ego_velocity + self.prev_ego_velocity)/2
        self.distance += self.config["time_step"] * (current_lead_velocity - current_ego_velocity)
        

    def _get_reward(self):
        # comfort term
        jerk = np.abs(self.accel - self.prev_accel)
        comfort_reward = 1 - (jerk/6)**2
        
        # efficiency term
        effiency_reward = 1 - (np.clip(
            (self.deviation_opt_distance/80), -1, 1))**2
        
        # safety term
        velocity_diff = np.clip(self.ego_velocity - self.lead_velocity, 0, 30)
        safety_reward = 0
        if velocity_diff != 0:
            ttc = self.distance/velocity_diff
            ttc = np.clip(ttc/self.config["maximum_ttc"], 0.01, 1)
            safety_reward = np.clip(np.log(ttc), -10, 0)

        return self.alpha * comfort_reward  \
            + self.beta * effiency_reward  \
            + self.gamma * safety_reward
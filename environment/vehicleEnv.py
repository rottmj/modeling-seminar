import numpy as np


class Vehicle():
    def __init__(self, velocity, position, max_velocity, min_velocity=0):
        self.velocity = np.array([velocity])
        self.position = np.array([position])
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity


    def get_velocity(self):
        return self.velocity


    def set_velocity(self, v):
        if type(v) != float:
            float(v)
        if v < self.min_velocity:
            v = self.min_velocity
        if v > self.max_velocity:
            v = self.max_velocity
        self.velocity[[0]] = v


    def get_position(self):
        return self.position


    def set_position(self, pos):
        if type(pos) != float:
            float(pos)
        self.position[[0]] = pos


class VehicleEnv:
    def __init__(self, lead_velocity, lead_position, 
        ego_velocity=0.0, ego_position=0.0, safety_distance=50, 
        max_velocity=60, max_gap=250, reward_type='default',
        random_init_values=False, low_lead_velocity=25,
        high_lead_velocity=36, low_ego_velocity=15,
        high_ego_velocity=39, low_gap= 75, high_gap=125):
       
        if random_init_values:
            lead_velocity = np.random.default_rng().uniform(low_lead_velocity, high_lead_velocity)
            ego_velocity = np.random.default_rng().uniform(low_ego_velocity, high_ego_velocity)
            lead_position = np.random.default_rng().uniform(low_gap, high_gap)
            if type(lead_position) != float:
                float(lead_position)
            safety_distance = ego_velocity / 2
        
        self.ego = Vehicle(ego_velocity, ego_position, max_velocity)
        self.lead = Vehicle(lead_velocity, lead_position, max_velocity)
        self.safety_distance = safety_distance
        self.max_velocity = max_velocity
        self.max_gap = max_gap
        self.reward_type = reward_type


    def get_state(self):
        gap = (self.lead.get_position() - self.ego.get_position())[0] -self.safety_distance
        lead_velocity = self.lead.get_velocity()[0]
        # If the lead vehicle is too far away, 
        # the ego vehicle can't sense it anymore.
        # Then use maximal values.
        if gap > self.max_gap - self.safety_distance:
            gap = self.max_gap
            lead_velocity = self.max_velocity
        return [
            lead_velocity - self.ego.get_velocity()[0],
            gap
        ]


    def get_reward_LinMcPheeAzad(self, gap, acceleration, alpha=0.7, 
        beta=0.3, nominal_max_error=150.0, max_control_input=2.6):
        reward = -alpha * abs(gap)/nominal_max_error - beta * abs(acceleration)/ max_control_input
        if reward < -1.0:
            return -1.0
        return reward

    def get_reward_v0(self, gap, acceleration):
        distance = abs(gap - self.safety_distance) / 250
        distance = min(distance, 1)
        distance = 1 - distance
        acceleration = abs(acceleration)/3
        acceleration = min(acceleration, 1)
        acceleration = 1 -acceleration
        return 0.7 * distance + 0.3 * acceleration


    def step(self, acceleration):
        # Set velocity for current time step
        

        velocity = (self.ego.get_velocity() + acceleration)
        self.ego.set_velocity(velocity)
        # Calculate gap
        gap = self.lead.get_position() + 1/10 * self.lead.get_velocity() - self.ego.get_position() - 1/10 * velocity
        
        # Detect if a collision happens
        collision = False
        if gap <= 0:
            collision = True

        if collision:
            # Update positions
            self.lead.set_position(self.lead.get_position() + 1/10 * self.lead.get_velocity())
            self.ego.set_position(self.lead.get_position())

            # Get state, reward, done
            state = self.get_state()
            reward = -1
            done = True
            return state, reward, done

        # Update positions
        self.lead.set_position(self.lead.get_position() + 1/10 * self.lead.get_velocity())
        self.ego.set_position(self.ego.get_position() + 1/10 * self.ego.get_velocity())
        
        # Get state, reward, done
        state = self.get_state()
        reward = self.get_reward_v0(gap, acceleration)

        done = False
        
        return state, reward, done

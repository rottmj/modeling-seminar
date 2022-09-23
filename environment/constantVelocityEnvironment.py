import numpy as np


class ConstantVelocityEnvironment:
    def __init__(self, ego_velocity=25, lead_velocity=27.8, gap=55, safety_gap=50, lead_goal_velocity=23):
        self.ego_velocity = ego_velocity
        self.lead_velocity = lead_velocity
        self.gap = gap
        self.safety_gap = safety_gap
        self.lead_goal_velocity = lead_goal_velocity
        self.start_values = [
            ego_velocity,
            lead_velocity,
            gap,
            safety_gap
        ]

    def reset(self):
        self.ego_velocity = self.start_values[0]
        self.lead_velocity = self.start_values[1]
        self.gap = self.start_values[2]
        self.safety_gap = self.start_values[3]

    def get_state(self):
        velocity_diff = self.lead_velocity - self.ego_velocity
        gap_diff = self.gap - self.safety_gap
        return [
            velocity_diff,
            gap_diff
        ]

    def set_velocity(self, new_velocity):
        if new_velocity < 0.0:
            self.ego_velocity = 0.0
        elif new_velocity > 30.0:
            self.ego_velocity = 30.0
        else:
            self.ego_velocity = new_velocity

    def set_gap(self):
        gap_diff = 1/10 * (self.lead_velocity - self.ego_velocity)
        self.gap += gap_diff

    def get_reward(self, accel, alpha=1 / 2, beta=1 / 2):
        alpha_term = alpha * abs(self.gap - self.safety_gap) / 10
        beta_term = beta * abs(accel) / 3
        res = alpha_term + beta_term
        if res > 1:
            res = 1
        return 1 - res

    def step(self, accel):
        # update lead if lead velocity is not constant during one episode
        # self.lead_velocity = min(30, self.lead_velocity + self.lead_velocity * np.random.uniform(-0.005, 0.005))
        # self.safety_gap = self.lead_velocity * 3.6 / 2

        new_velocity = self.ego_velocity + 1/10 * accel
        self.set_velocity(new_velocity)
        self.set_gap()
        if self.gap <= 0:
            state = self.get_state()
            reward = -0.001
            done = True
            return state, reward, done
        state = self.get_state()
        reward = self.get_reward(accel)
        done = False
        return state, reward, done

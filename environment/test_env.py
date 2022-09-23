import numpy as np


class VehicleEnvironment:
    def __init__(self, ego_velocity, lead_velocity, gap, lead_goal_velocity):
        self.ego_velocity = ego_velocity
        self.lead_velocity = lead_velocity
        self.gap = gap
        self.safety_gap = self.lead_velocity * 3.6 / 2
        self.lead_goal_velocity = lead_goal_velocity
        self.mode = np.random.randint(3)
        self.old_gap = 0.0
        self.old_velocity = 0.0
        self.cut_in_time = np.random.randint(20, 140)
        self.cut_out_time = self.cut_in_time + np.random.randint(40, 80)
        self.bool_cut_in = False

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
        gap_diff = 1 / 10 * (self.lead_velocity - self.ego_velocity)
        self.gap += gap_diff

    def get_reward(self, accel, alpha=1 / 2, beta=1 / 2):
        alpha_term = alpha * abs(self.gap - self.safety_gap) / 10
        beta_term = beta * abs(accel) / 3
        res = alpha_term + beta_term
        if res > 1:
            res = 1
        return 1 - res

    def step_default(self, accel):
        self.lead_velocity = min(30, self.lead_velocity + self.lead_velocity * np.random.uniform(-0.005, 0.005))
        self.safety_gap = self.lead_velocity * 3.6 / 2

        new_velocity = self.ego_velocity + 1 / 10 * accel
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

    def step_cut_in(self, accel, t):
        if t == self.cut_in_time:
            if self.gap - 27 > 0:
                cut_in_position = np.random.uniform(0, self.gap - 27)
                self.gap = cut_in_position + 10
                self.lead_velocity = np.random.uniform(low=20.0, high=28.0)

        self.lead_velocity = min(30, self.lead_velocity + self.lead_velocity * np.random.uniform(-0.005, 0.005))
        self.safety_gap = self.lead_velocity * 3.6 / 2

        new_velocity = self.ego_velocity + 1 / 10 * accel
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

    def step_cut_in_out(self, accel, t):
        if t == self.cut_in_time:
            if self.gap - 27 > 0:
                self.bool_cut_in = True
                cut_in_position = np.random.uniform(0, self.gap - 27)
                self.old_gap = self.gap
                self.gap = cut_in_position + 10
                self.old_velocity = self.lead_velocity
                self.lead_velocity = np.random.uniform(low=20.0, high=28.0)

        if t == self. cut_out_time:
            if self.bool_cut_in:
                self.gap = self.old_gap
                self.lead_velocity = self.old_velocity

        self.lead_velocity = min(30, self.lead_velocity + self.lead_velocity * np.random.uniform(-0.005, 0.005))
        self.safety_gap = self.lead_velocity * 3.6 / 2

        new_velocity = self.ego_velocity + 1 / 10 * accel
        self.set_velocity(new_velocity)
        self.set_gap()

        if t > self.cut_in_time:
            self.old_velocity = min(30, self.old_velocity + self.old_velocity * np.random.uniform(-0.005, 0.005))
            gap_diff = 1 / 10 * (self.old_velocity - self.ego_velocity)
            self.old_gap += gap_diff

        if self.gap <= 0:
            state = self.get_state()
            reward = -0.001
            done = True
            return state, reward, done
        state = self.get_state()
        reward = self.get_reward(accel)
        done = False
        return state, reward, done

    def step(self, accel, t):
        if self.mode == 0:
            return self.step_default(accel)
        elif self.mode == 1:
            return self.step_cut_in(accel, t)
        else:
            return self.step_cut_in_out(accel, t)

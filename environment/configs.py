simulation_config = {
    # ego vehicle parameters
    "ego_low_velocity": 0.0,
    "ego_high_velocity": 30.0,
    # lead vehicle parameters
    "lead_low_velocity": 0.0,
    "lead_high_velocity": 32.0,
    # distance parameters
    "low_distance": 10,
    "high_distance": 100,
    "time_gap": 1.8,
    "minimum_gap": 2,
    # normalizing_factors
    "state_normalizing_factors": (32, 80, 3),
    # others
    "time_step": 0.1,
    "collision_reward": -10,
    "maximum_ttc": 7,
}
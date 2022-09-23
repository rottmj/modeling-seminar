import numpy as np


def reward_lin_mcphee_azad_19(gap, accel, nom_gap, alpha=1 / 2, beta=1 / 2):
    alpha_term = alpha * abs(gap) / nom_gap
    beta_term = beta * abs(accel) / 3
    return -1 * (alpha_term + beta_term)


def reward_lin_mcphee_azad_20(gap, accel, jerk, nom_gap, alpha=1 / 3, beta=1 / 3, gamma=1 / 3, eps=10 ** -8):
    alpha_term = alpha * np.sqrt((gap / nom_gap) ** 2 + eps)
    beta_term = beta * np.sqrt((accel / 3) ** 2 + eps)
    gamma_term = gamma * np.sqrt((jerk / 0.6) ** 2 + eps)
    return -1 * (alpha_term + beta_term + gamma_term)

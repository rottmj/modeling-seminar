import matplotlib.pyplot as plt

# TODO ts in 1/10 ts ändern
# TODO Refactoring


def plot_1d_graph(data, x_label, y_label, file_name, path, color='b'):
    """
    Plots graph with one dimensional data.
    """
    plt.plot(data, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path + "/" + file_name, dpi=1000)
    # plt.close()


def plot_undiscounted_episode_reward(episode_reward, path):
    plot_1d_graph(episode_reward, 't [1/10s]', 'undiscounted episode reward', 'undiscounted_episode_reward.jpg', path)


def plot_speed(speed, path):
    plot_1d_graph(speed, 't [1/10s]', 'velocity [m/s]', 'speed.jpg', path)


def plot_gap(gap, path):
    plot_1d_graph(gap, 't [1/10s]', 'gap [m]', 'gap.jpg', path)


def plot_acceleration(acceleration, path):
    plot_1d_graph(acceleration, 't [1/10s]', 'acceleration [m/s²]', 'acceleration.jpg', path)


def plot_jerk(jerk, path):
    plot_1d_graph(jerk, 't [1/10s]', 'jerk [m/s³]', 'jerk.jpg', path)


def plot_control_input(control_input, path):
    plot_1d_graph(control_input, 't [1/10s]', 'control input [m/s²]', 'control input.jpg', path)


def plot_reward(reward, path):
    plot_1d_graph(reward, 't [1/10s]', 'reward', 'reward.jpg', path)


def plot_return(reward, path):
    plot_1d_graph(reward, '#', 'return', 'return.jpg', path)


def kmh_to_ms(kmh):
    print(kmh / 60 / 60 * 1000)

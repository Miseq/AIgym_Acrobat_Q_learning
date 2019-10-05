import numpy as np


def get_discrete_state(state, DISCRETE_OS_SIZE, env):
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(
        np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


def print_episode(episode, reward):
    success = '\033[1;32;49m'
    failure = '\033[1;31;49m'
    if reward == 0:
        print(f'{success} Success at episode number: {episode}')
    else:
        print(f'{failure} {episode}')


def check_if_render(episode, env, SHOW_EVERY):
    if not episode % SHOW_EVERY:
        env.render()
    else:
        pass


def get_action(epsilon, q_table, discrete_state, env):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = np.random.randint(0, env.action_space.n)
    return action


def update_q_table(done, q_table, discrete_state, new_discrete_state, action, reward, LEARNING_RATE, DISCOUNT):
    if not done:
        try:
            max_future_q = np.max(q_table[new_discrete_state])
        except IndexError as ie:  # istnieje mala szansa na wytworzenie indeksu tuż poza granicą
            max_future_q = np.max(q_table[new_discrete_state - 1])

        current_q = q_table[discrete_state + (action,)]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state + (action,)] = new_q

    elif reward == 0:
        q_table[discrete_state + (action,)] = 0

    return q_table

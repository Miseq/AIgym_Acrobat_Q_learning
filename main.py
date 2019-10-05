import gym

from functions import *

env = gym.make("Acrobot-v1")

LEARNING_RATE = 0.12
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = int(EPISODES / 100)  # stała(20) oznacza ile razy będzie renderowane środowisko
successes = 0
stats_log = []

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

for episode in range(1, EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset(), DISCRETE_OS_SIZE, env)
    done = False
    show_stats(stats_log, SHOW_EVERY, episode, successes)

    while not done:
        action = get_action(epsilon, q_table, discrete_state, env)
        new_state, reward, done, info_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state, DISCRETE_OS_SIZE, env)
        check_if_render(episode, env, SHOW_EVERY)
        update_q_table(done, q_table, discrete_state, new_discrete_state, action, reward, successes, LEARNING_RATE,
                       DISCOUNT)
        successes = update_successes()
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    print_episode(episode, reward)

env.close()
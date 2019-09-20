import gym
import matplotlib.pyplot as plt
import numpy as np


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


env = gym.make("Acrobot-v1")

LEARNING_RATE = 0.12
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = int(EPISODES / 20)
STATS_EVERY = int(EPISODES / 100)

list_of_rewards = []
aggr_ep_rewards = {'episode': [], 'avg': [], 'max': [], 'min': []}

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


for episode in range(EPISODES):

    episode_reward = None
    discrete_state = get_discrete_state(env.reset())
    done = False
    render = False


    if episode % SHOW_EVERY == 0:
        render = True

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action5
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, info_ = env.step(action)
        episode_reward += int(reward)

        new_discrete_state = get_discrete_state(new_state)
        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table

        if not done:

            # Maximum possible Q value in next step (for new state)
            try:
                max_future_q = np.max(q_table[new_discrete_state])
            except IndexError as ie:
                print(ie)
                max_future_q = np.max(q_table[new_discrete_state] - 1)
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly

        elif reward == 0:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    list_of_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(list_of_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(list_of_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(list_of_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

#TODO kolory wyników zielony/czerwony
#TODO statystyki
#TODO wykres kołowo -udane/nieudane
#TODO wykres jak z czasem zmienia sie ilosc udanych/nieudanych
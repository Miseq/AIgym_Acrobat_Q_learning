import gym
import numpy as np


class Acrobot():

    def __init__(self, enviroment='MountainCarContinuous-v0', episodes=5000, learning_rate=0.12, discount=0.95,
                 epsilon=1,
                 times_to_render=10, ):

        self.env = gym.make(enviroment)
        self.discrete_os_size = [7] * len(self.env.observation_space.high)
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.successes = 0
        self.epsilon_decreasing_start = 1
        self.epsilon_decreasing_stop = self.episodes // 2
        self.epsilon_decreasing_velocity = float(self.epsilon / (self.epsilon_decreasing_stop))
        self.all_stats = []
        self.show_every = self.episodes // times_to_render

    def get_discrete_state(self, state):
        discrete_os_win_size = (
                                           self.env.observation_space.high - self.env.observation_space.low) / self.discrete_os_size
        discrete_state = (state - self.env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    def update_epsilon(self, episode):
        if self.epsilon_decreasing_stop > episode >= self.epsilon_decreasing_start:
            self.epsilon = self.epsilon - self.epsilon_decreasing_velocity

    def check_if_render(self, ep):
        if not ep % self.show_every:
            # self.env.render()
            pass
        else:
            pass

    def show_stats(self, episode):
        stats_prefix = '\033[0;35;49m'
        if not episode % self.show_every:
            success_rate = float(self.successes / self.show_every * 100)
            stats = f'For past {self.show_every}({episode - self.show_every}-{episode}) ' \
                    f'episodes, success_rate is {success_rate}%'
            print(f'{stats_prefix} {stats}')
            self.all_stats.append(stats)
            self.successes = 0

    def get_action(self, q_table, discrete_state):
        if np.random.random() > self.epsilon:  # random generuje watosci 0..1
            action = np.argmax(q_table[discrete_state])  # pobranie wartosci akcji z q_table
        else:
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def update_q_table(self, done, q_table, discrete_state, new_discrete_state, action, reward):
        if not done:
            try:
                max_future_q = np.max(q_table[new_discrete_state])
            except IndexError as ie:
                max_future_q = np.max(q_table[new_discrete_state - 1])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif reward == 0:
            self.successes += 1
            q_table[discrete_state + (action,)] = 0
        return q_table

    def print_episode(self, episode, reward):
        success_prefix = '\033[1;32;49m'
        failure_prefix = '\033[1;31;49m'
        if reward == 0:
            print(f'{success_prefix} Success at episode number: {episode}')
        else:
            print(f'{failure_prefix} {episode}')

    def magic(self):
        q_table = np.random.uniform(low=-2, high=0, size=(self.discrete_os_size + [self.env.action_space.n]))
        for episode in range(1, self.episodes + 1):
            done = False
            discrete_state = self.get_discrete_state(self.env.reset())

            while not done:
                action = self.get_action(q_table, discrete_state)
                new_state, reward, done, info = self.env.step(action)
                new_discrete_state = self.get_discrete_state(new_state)
                self.check_if_render(episode)
                q_table = self.update_q_table(done, q_table, discrete_state, new_discrete_state, action, reward)
                discrete_state = new_discrete_state

            self.print_episode(episode, reward)
            self.show_stats(episode)
            self.update_epsilon(episode)

        self.env.close()


bot = Acrobot(episodes=10000)
bot.magic()
print('\n Wszystkie statystyki:')
print('\n'.join([str(x) for x in bot.all_stats]))

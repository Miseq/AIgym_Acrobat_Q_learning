import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
done = False
steps_to_print = 5
step = 1
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space)
print(len(env.reward_range))
while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if step <= steps_to_print:
        print(f"step: {step}")
        print(f"action: {action}")
        print(f"observation: {observation}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info} \n")
        step += 1

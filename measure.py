import matplotlib.pyplot as plt
import numpy as np
from environment.setup_env import setup_env
from stable_baselines3 import PPO
import time
from default_config import default_config
from train import train

config_property_to_change = "training_time"
values = list(range(2*60_000, 3*60_000, 4*60_000))

base_config = default_config
del base_config[config_property_to_change]

measurement_iterations_per_value = 1000

plot_y = []

for value in values:
    config = base_config.copy()
    config[config_property_to_change] = value

    train(config)

    env = setup_env(False, True, config)

    model = PPO.load("trained/new_agent.zip")

    model.set_env(env)

    random_seed = int(time.time())
    model.set_random_seed(random_seed)

    obs = env.reset()

    rewards = []

    for i in range(measurement_iterations_per_value):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    average_reward = sum(rewards) / len(rewards)
    plot_y.append(average_reward)

fig = plt.figure()
plt.plot(values, plot_y)
fig.savefig('plot.png')
np.savetxt("plot_data.csv", (values, plot_y), delimiter=",")

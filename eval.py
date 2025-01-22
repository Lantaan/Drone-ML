from environment.setup_env import setup_env

from stable_baselines3 import PPO
import time

env = setup_env(True, True, 0.5)

model = PPO.load("trained/new_agent.zip")

model.set_env(env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        env.render()
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        if done is True:
            state = env.reset()

finally:
    env.close()

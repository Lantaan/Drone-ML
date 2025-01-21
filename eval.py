from stable_baselines3 import PPO
import time
from environment.setup_env import setup_env

env = setup_env(True)

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

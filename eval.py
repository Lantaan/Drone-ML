from environment.setup_env import setup_env
from stable_baselines3 import PPO
import time
from default_config import default_config
from train import train

agent_name = "new_agent"
train(default_config, agent_name)

env = setup_env(True, True, default_config)

model = PPO.load(f"trained/{agent_name}.zip", env=env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        env.render()
        action, _states = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            state = env.reset()
finally:
    env.close()

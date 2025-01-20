from stable_baselines3 import PPO
import time
from environment.setup_env import setup_env

continuous_mode = True  # if True, after completing one episode the next one will start automatically
random_action = False  # if True, the agent will take actions randomly

render_sim = True  # if True, a graphic is generated

env = setup_env(render_sim)

model = PPO.load("trained/new_agent.zip")

model.set_env(env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        if done is True:
            if continuous_mode is True:
                state = env.reset()
            else:
                break

finally:
    env.close()

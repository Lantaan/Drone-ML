from environment.drone import Drone
from environment.event_handler import pygame_events

import gym
import gym.spaces
import pygame
import pymunk
import pymunk.pygame_util
import gstools
import matplotlib.pyplot as plt
import numpy as np
import random
import os


class Drone2dEnv(gym.Env):
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    render_shade: (bool) if true, the drone's shade is drawn
    shade_distance: (int) distance between consecutive drone's shades
    n_steps: (int) number of time steps
    n_fall_steps: (int) the number of initial steps for which the drone can't do anything
    change_target: (bool) if true, mouse click change target positions
    initial_throw: (bool) if true, the drone is initially thrown with random force
    wind_intensity: (float) factor with which the wind force is multiplied (0.0 - 1.0)
    wind_len_scale: (int) determines how fast the gradient of the wind field changes
    render_wind: (bool) if true, the wind field is drawn, else the borders are drawn
    """

    def __init__(self, render_sim=False, render_path=True, render_shade=True, shade_distance=70,
                 n_steps=500, n_fall_steps=10, change_target=False, initial_throw=True, wind_intensity=0.0, 
                 wind_len_scale=100, render_wind=True):

        if wind_intensity < 0 or wind_intensity > 1:
            raise ValueError("Wind intensity must be between 0 and 1")

        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade
        self.wind_len_scale = wind_len_scale

        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []

        self.init_pymunk()
        
        self.wind_intensity = wind_intensity
        self.render_wind = render_wind

        if wind_intensity > 0: self.init_wind()

        # Parameters
        self.max_time_steps = n_steps
        self.stabilisation_delay = n_fall_steps
        self.drone_shade_distance = shade_distance
        self.force_scale = 1000
        self.initial_throw = initial_throw
        self.change_target = change_target

        # Initial values
        self.first_step = True
        self.done = False
        self.info = {}
        self.current_time_step = 0
        self.left_force = -1
        self.right_force = -1
        self.wind_force = np.array([0, 0])

        # Generating target position
        self.x_target = random.uniform(50, 750)
        self.y_target = random.uniform(50, 750)

        # Defining spaces for action and observation
        action_lim = np.ones(2, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-action_lim, high=action_lim, dtype=np.float32)

        observation_lim = np.ones(8, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-observation_lim, high=observation_lim, dtype=np.float32)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Drone2d Environment")
        self.clock = pygame.time.Clock()

        script_dir = os.path.dirname(__file__)

        icon_path = os.path.join("..", "img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        pygame.display.set_icon(pygame.image.load(icon_path))

        img_path = os.path.join("..", "img", "shade.png")
        img_path = os.path.join(script_dir, img_path)
        self.shade_image = pygame.image.load(img_path)

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0, -981)

        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True

        # Generating drone's starting position
        random_x = random.uniform(200, 600)
        random_y = random.uniform(200, 600)
        angle_rand = random.uniform(-np.pi / 4, np.pi / 4)
        self.drone = Drone(random_x, random_y, angle_rand, 20, 100, 0.2, 0.4, 0.4, self.space)

        self.drone_radius = self.drone.drone_radius

    def init_wind(self):
        self.wind_field = self.generate_wind_field()
        wind_modifier = np.array([random.choice((-1, 1)), -1])
        self.wind_scale = wind_modifier.reshape(-1, 1) * self.wind_intensity * 250

        if self.render_sim is True and self.render_wind is True:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

            x = np.linspace(0, 800, 100)
            y = np.linspace(0, 800, 100)

            field = self.wind_field((x, y), mesh_type="structured")
            field = wind_modifier.reshape(-1, 1, 1) * field
            norm = np.sqrt(field[0, :].T ** 2 + field[1, :].T ** 2)

            ax.streamplot(x, y, field[0, :].T, field[1, :].T, color=norm, linewidth=0.75*norm)
            fig.savefig("img/wind.png")

            plt.close()

            script_dir = os.path.dirname(__file__)
            img_path = os.path.join("..", "img", "wind.png")
            img_path = os.path.join(script_dir, img_path)
            self.wind_image = pygame.image.load(img_path)
            self.wind_image = pygame.transform.scale(self.wind_image, (800, 800))

    def step(self, action):
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_shade is True: self.add_drone_shade()
            self.info = self.initial_movement()

        self.left_force = (action[0] / 2 + 0.5) * self.force_scale
        self.right_force = (action[1] / 2 + 0.5) * self.force_scale

        self.drone.frame_shape.body.apply_force_at_local_point(pymunk.Vec2d(0, self.left_force), (-self.drone_radius, 0))
        self.drone.frame_shape.body.apply_force_at_local_point(pymunk.Vec2d(0, self.right_force), (self.drone_radius, 0))

        if self.wind_intensity > 0:
            self.wind_force = self.wind_field(self.drone.frame_shape.body.position) * self.wind_scale
            self.drone.frame_shape.body.apply_force_at_local_point(pymunk.Vec2d(*self.wind_force), (0, 0))

        self.space.step(1.0 / 60)
        self.current_time_step += 1

        # Saving drone's position for drawing
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
            self.first_step = False

        else:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()

        if self.render_sim is True and self.render_shade is True:
            x, y = self.drone.frame_shape.body.position
            if np.abs(self.shade_x - x) > self.drone_shade_distance or np.abs(self.shade_y - y) > self.drone_shade_distance:
                self.add_drone_shade()

        # Calulating reward function
        obs = self.get_observation()
        reward = (1.0 / (np.abs(obs[4]) + 0.1)) + (1.0 / (np.abs(obs[5]) + 0.1))

        # Stops episode, when drone is out of range or overlaps
        if np.abs(obs[3]) == 1 or np.abs(obs[6]) == 1 or np.abs(obs[7]) == 1:
            self.done = True
            reward = -10

        # Stops episode, when time is up
        if self.current_time_step == self.max_time_steps:
            self.done = True

        return obs, reward, self.done, self.info

    def get_observation(self):
        velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
        velocity_x = np.clip(velocity_x / 1330, -1, 1)
        velocity_y = np.clip(velocity_y / 1330, -1, 1)

        omega = self.drone.frame_shape.body.angular_velocity
        omega = np.clip(omega / 11.7, -1, 1)

        alpha = self.drone.frame_shape.body.angle
        alpha = np.clip(alpha / (np.pi / 2), -1, 1)

        x, y = self.drone.frame_shape.body.position

        if x < self.x_target:
            distance_x = np.clip((x / self.x_target) - 1, -1, 0)

        else:
            distance_x = np.clip((-x / (self.x_target - 800) + self.x_target / (self.x_target - 800)), 0, 1)

        if y < self.y_target:
            distance_y = np.clip((y / self.y_target) - 1, -1, 0)

        else:
            distance_y = np.clip((-y / (self.y_target - 800) + self.y_target / (self.y_target - 800)), 0, 1)

        pos_x = np.clip(x / 400.0 - 1, -1, 1)
        pos_y = np.clip(y / 400.0 - 1, -1, 1)

        return np.array([velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y])

    def render(self, mode='human', close=False):
        if self.render_sim is False: return

        pygame_events(self.space, self, self.change_target)
        self.screen.fill((243, 243, 243))

        # Drawing wind field
        if self.wind_intensity > 0 and self.render_wind is True:
            self.screen.blit(self.wind_image, (0, 0))
        else:    
            pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, 800, 800), 8)
            pygame.draw.rect(self.screen, (33, 158, 188), pygame.Rect(50, 50, 700, 700), 4)
            pygame.draw.rect(self.screen, (142, 202, 230), pygame.Rect(200, 200, 400, 400), 4)

        # Drawing done's shade
        if len(self.path_drone_shade):
            for shade in self.path_drone_shade:
                image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2] * 180.0 / np.pi)
                shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], 800 - shade[1]))
                self.screen.blit(image_rect_rotated, shade_image_rect)

        self.space.debug_draw(self.draw_options)

        # Drawing vectors of motor forces
        vector_scale = 0.05
        l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 0))
        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.force_scale * vector_scale))
        pygame.draw.line(self.screen, (179, 179, 179), (l_x_1, 800 - l_y_1), (l_x_2, 800 - l_y_2), 4)

        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force * vector_scale))
        pygame.draw.line(self.screen, (255, 0, 0), (l_x_1, 800 - l_y_1), (l_x_2, 800 - l_y_2), 4)

        r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.force_scale * vector_scale))
        pygame.draw.line(self.screen, (179, 179, 179), (r_x_1, 800 - r_y_1), (r_x_2, 800 - r_y_2), 4)

        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force * vector_scale))
        pygame.draw.line(self.screen, (255, 0, 0), (r_x_1, 800 - r_y_1), (r_x_2, 800 - r_y_2), 4)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, 800 - self.y_target), 8)

        # Drawing drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        if len(self.drop_path) > 2:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(self.render_sim, self.render_path, self.render_shade, self.drone_shade_distance,
                      self.max_time_steps, self.stabilisation_delay, self.change_target, self.initial_throw,
                      self.wind_intensity)
        return self.get_observation()

    def close(self):
        pygame.quit()

    def initial_movement(self):
        if self.initial_throw is True:
            throw_angle = random.random() * 2 * np.pi
            throw_force = random.uniform(0, 25000)
            throw = pymunk.Vec2d(np.cos(throw_angle) * throw_force, np.sin(throw_angle) * throw_force)

            self.drone.frame_shape.body.apply_force_at_world_point(throw, self.drone.frame_shape.body.position)

            throw_rotation = random.uniform(-3000, 3000)
            self.drone.frame_shape.body.apply_force_at_local_point(pymunk.Vec2d(0, throw_rotation), (-self.drone_radius, 0))
            self.drone.frame_shape.body.apply_force_at_local_point(pymunk.Vec2d(0, -throw_rotation), (self.drone_radius, 0))

            self.space.step(1.0 / 60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()

        else:
            throw_angle = None
            throw_force = None
            throw_rotation = None

        initial_stabilisation_delay = self.stabilisation_delay
        while self.stabilisation_delay != 0:
            self.space.step(1.0 / 60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True: self.render()
            self.stabilisation_delay -= 1

        self.stabilisation_delay = initial_stabilisation_delay

        return {'throw_angle': throw_angle, 'throw_force': throw_force, 'throw_rotation': throw_rotation}

    def add_postion_to_drop_path(self):
        x, y = self.drone.frame_shape.body.position
        self.drop_path.append((x, 800 - y))

    def add_postion_to_flight_path(self):
        x, y = self.drone.frame_shape.body.position
        self.flight_path.append((x, 800 - y))

    def add_drone_shade(self):
        x, y = self.drone.frame_shape.body.position
        self.path_drone_shade.append([x, y, self.drone.frame_shape.body.angle])
        self.shade_x = x
        self.shade_y = y

    def change_target_point(self, x, y):
        self.x_target = x
        self.y_target = y

    def generate_wind_field(self):
        model = gstools.Gaussian(dim=2, var=1, len_scale=self.wind_len_scale, anis=0.75, angles=random.uniform(0, 2*np.pi))
        srf = gstools.SRF(model, generator='VectorField')
        return srf

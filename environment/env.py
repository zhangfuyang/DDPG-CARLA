from carla import make_carla_client
from carla.settings import CarlaSettings
from carla import TCPConnectionError
from carla import Camera
from carla.carla_server_pb2 import Control
from imitation.imitation_learning import ImitationLearning
import random
import time
import os
from PIL import Image
import numpy as np


class action_space(object):
    def __init__(self, dim, high, low, seed):
        self.shape = (dim,)
        self.high = high
        self.low = low
        self.seed = seed
        assert(dim == len(high) == len(low))
        np.random.seed(self.seed)

    def sample(self):
        return np.random.uniform(self.low, self.high)


class Env(object):
    def __init__(self, MONITOR_DIR, SEED, FPS):
        self.MONITOR_DIR = MONITOR_DIR
        self.client = None
        self.connected()
        self.Image_agent = ImitationLearning()
        self.action_space = action_space(2, (1.0, 1.0), (-1.0, -1.0), SEED)
        self.render_ = False
        self.image_dir_ = None
        self.image_i_ = 0
        self.FPS = FPS
        self.reward_time = 0

    def connected(self):
        self.render_ = False
        self.reward_time = 0
        while True:
            try:
                client = make_carla_client('localhost', 2000)
                self.client = client
            except TCPConnectionError as error:
                time.sleep(1.0)

    def step(self, action):
        steer = action['steer']
        acc = action['acc']
        brake = action['brake']
        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0
        try:
            prev_measurements, _ = self.client.read_data()
            self.client.send_control(control)
            measurements, sensor_data = self.client.read_data()
            if self.render_:
                im = sensor_data['CameraRGB'].data[115:510, :]
                im = Image.fromarray(im)
                im.save(os.path.join(self.image_dir_, str(self.image_i_) + '.jpg'))
                self.image_i_ += 1

            feature_vector = self.Image_agent.compute_feature(sensor_data)
            speed = measurements.player_measurements.forward_speed
            speed = speed / 10.0
            offroad = measurements.player_measurements.intersection_offroad
            other_lane = measurements.player_measurements.intersection_otherlane

            reward, done = self.reward(measurements, prev_measurements)
            info = 0
        except TCPConnectionError as error:
            done = True
            feature_vector = None
            speed = None
            offroad = None
            other_lane = None
            reward = 0
            info = -1
            self.connected()

        return (feature_vector, (speed, offroad, other_lane)), reward, done, info

    def reset(self):
        self.image_i_ = 0
        self.image_dir_ = None
        self.render_ = False
        while True:
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=20,
                NumberOfPedestrians=40,
                WeatherId=random.choice([1, 3, 7, 8, 14]),
                QualityLevel='Epic'
            )
            settings.randomize_seeds()
            camera = Camera('CameraRGB')
            camera.set(FOV=100)
            camera.set_image_size(800, 600)
            camera.set_position(2.0, 0.0, 1.4)
            camera.set_rotation(-15.0, 0, 0)
            settings.add_sensor(camera)
            observation = None
            try:
                scene = self.client.load_settings(settings)
                number_of_player_starts = len(scene.player_start_spots)
                player_start = random.randint(0, max(0, number_of_player_starts - 1))
                self.client.start_episode(player_start)
                for i in range(20):
                    action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
                    observation, _, _, _ = self.step(action)
                break
            except TCPConnectionError as error:
                self.connected()
        return observation

    def render(self, image_dir):
        self.render_ = True
        self.image_dir_ = image_dir
        self.image_i_ = 0

    def reward(self, measurements, prev_measurements):
        """
        :param measurements:
        :param prev_measurements: due to the bug in the carla platform, we need to use this ugly
                                  parameter to alarm collision in low speed.
        :return: reward, done
        """
        done = False
        reward = 0.0

        """collision"""
        if measurements.player_measurements.collision_other != 0 or \
                measurements.player_measurements.collision_pedestrians != 0 or \
                measurements.player_measurements.collision_vehicles != 0:
            reward = reward - 10

        """road"""
        reward = reward - 5 * (measurements.player_measurements.intersection_offroad +
                               measurements.player_measurements.intersection_otherlane)

        if reward < -3:
            done = True

        reward = reward + measurements.player_measurements.forward_speed

        x, y = measurements.player_measurements.transform.location.x, \
               measurements.player_measurements.transform.location.y
        prev_x, prev_y = prev_measurements.player_measurements.transform.location.x, \
                         prev_measurements.player_measurements.transform.location.y
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)

        if distance < 1.0 / self.FPS * 1:
            reward = reward - 1
            self.reward_time += 1
        else:
            self.reward_time = 0

        if self.reward_time == 30:
            done = True
            self.reward_time = 0

        return reward, done


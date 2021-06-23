import numpy as np
import gym


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_name, pomdp_type='remove_velocity',
                 flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1):
        """

        :param env_name:
        :param pomdp_type:
            1.  remove_velocity: remove velocity related observation
            2.  flickering: obscure the entire observation with a certain probability at each time step with the
                   probability flicker_prob.
            3.  random_noise: each sensor in an observation is disturbed by a random noise Normal ~ (0, sigma).
            4.  random_sensor_missing: each sensor in an observation will miss with a relatively low probability sensor_miss_prob
            5.  remove_velocity_and_flickering:
            6.  remove_velocity_and_random_noise:
            7.  remove_velocity_and_random_sensor_missing:
            8.  flickering_and_random_noise:
            9.  random_noise_and_random_sensor_missing
            10. random_sensor_missing_and_random_noise:

        """
        super().__init__(gym.make(env_name))
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob

        if pomdp_type == 'remove_velocity':
            # Remove Velocity info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif pomdp_type == 'flickering':
            pass
        elif self.pomdp_type == 'random_noise':
            pass
        elif self.pomdp_type == 'random_sensor_missing':
            pass
        elif self.pomdp_type == 'remove_velocity_and_flickering':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'remove_velocity_and_random_noise':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'remove_velocity_and_random_sensor_missing':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'flickering_and_random_noise':
            pass
        elif self.pomdp_type == 'random_noise_and_random_sensor_missing':
            pass
        elif self.pomdp_type == 'random_sensor_missing_and_random_noise':
            pass
        else:
            raise ValueError("pomdp_type was not specified!")

    def observation(self, obs):
        # Single source of POMDP
        if self.pomdp_type == 'remove_velocity':
            return obs.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == 'flickering':
            # Note: flickering is equivalent to:
            #   flickering_and_random_sensor_missing, random_noise_and_flickering, random_sensor_missing_and_flickering
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(obs.shape)
            else:
                return obs.flatten()
        elif self.pomdp_type == 'random_noise':
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        elif self.pomdp_type == 'random_sensor_missing':
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            return obs.flatten()
        # Multiple source of POMDP
        elif self.pomdp_type == 'remove_velocity_and_flickering':
            # Note: remove_velocity_and_flickering is equivalent to flickering_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(new_obs.shape)
            else:
                return new_obs
        elif self.pomdp_type == 'remove_velocity_and_random_noise':
            # Note: remove_velocity_and_random_noise is equivalent to random_noise_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'remove_velocity_and_random_sensor_missing':
            # Note: remove_velocity_and_random_sensor_missing is equivalent to random_sensor_missing_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'flickering_and_random_noise':
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape)
            else:
                new_obs = obs
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'random_noise_and_random_sensor_missing':
            # Random noise
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'random_sensor_missing_and_random_noise':
            # Random sensor missing
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            # Random noise
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        else:
            raise ValueError("pomdp_type was not in ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']!")

    def _remove_velocity(self, env_name):
        # OpenAIGym
        #  1. Classic Control
        if env_name == "Pendulum-v0":
            remain_obs_idx = np.arange(0, 2)
        elif env_name == "Acrobot-v1":
            remain_obs_idx = list(np.arange(0, 4))
        elif env_name == "MountainCarContinuous-v0":
            remain_obs_idx = list([0])
        #  1. MuJoCo
        elif env_name == "HalfCheetah-v3" or env_name == "HalfCheetah-v2":
            remain_obs_idx = np.arange(0, 8)
        elif env_name == "Ant-v3" or env_name == "Ant-v2":
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        elif env_name == 'Walker2d-v3' or env_name == "Walker2d-v2":
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'Hopper-v3' or env_name == "Hopper-v2":
            remain_obs_idx = np.arange(0, 5)
        elif env_name == "InvertedPendulum-v2":
            remain_obs_idx = np.arange(0, 2)
        elif env_name == "InvertedDoublePendulum-v2":
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
        elif env_name == "Swimmer-v3" or env_name == "Swimmer-v2":
            remain_obs_idx = np.arange(0, 3)
        elif env_name == "Thrower-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Striker-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Pusher-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Reacher-v2":
            remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))
        elif env_name == 'Humanoid-v3' or env_name == "Humanoid-v2":
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        elif env_name == 'HumanoidStandup-v2':
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        # PyBulletEnv:
        #   The following is not implemented:
        #       HumanoidDeepMimicBulletEnv - v1
        #       CartPoleBulletEnv - v1
        #       MinitaurBulletEnv - v0
        #       MinitaurBulletDuckEnv - v0
        #       RacecarBulletEnv - v0
        #       RacecarZedBulletEnv - v0
        #       KukaBulletEnv - v0
        #       KukaCamBulletEnv - v0
        #       PusherBulletEnv - v0
        #       ThrowerBulletEnv - v0
        #       StrikerBulletEnv - v0
        #       HumanoidBulletEnv - v0
        #       HumanoidFlagrunBulletEnv - v0
        #       HumanoidFlagrunHarderBulletEnv - v0
        elif env_name == 'HalfCheetahBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 26)) - set(np.arange(3, 6)))
        elif env_name == 'AntBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 28)) - set(np.arange(3, 6)))
        elif env_name == 'HopperBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 15)) - set(np.arange(3, 6)))
        elif env_name == 'Walker2DBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 22)) - set(np.arange(3, 6)))
        elif env_name == 'InvertedPendulumBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 5)) - set([1, 4]))
        elif env_name == 'InvertedDoublePendulumBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([1, 5, 8]))
        elif env_name == 'InvertedPendulumSwingupBulletEnv-v0':
            pass
        elif env_name == 'ReacherBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([6, 8]))
        # PyBulletGym
        #  1. MuJoCo
        elif env_name == 'HalfCheetahMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'AntMuJoCoEnv-v0':
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        elif env_name == 'Walker2DMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'HopperMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 7)
        elif env_name == 'InvertedPendulumMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 3)
        elif env_name == 'InvertedDoublePendulumMuJoCoEnv-v0':
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
        #  2. Roboschool
        elif env_name == 'HalfCheetahPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 26)) - set(np.arange(3, 6)))
        elif env_name == 'AntPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 28)) - set(np.arange(3, 6)))
        elif env_name == 'Walker2DPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 22)) - set(np.arange(3, 6)))
        elif env_name == 'HopperPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 15)) - set(np.arange(3, 6)))
        elif env_name == 'InvertedPendulumPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 5)) - set([1, 4]))
        elif env_name == 'InvertedDoublePendulumPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([1, 5, 8]))
        elif env_name == 'ReacherPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([6, 8]))
        else:
            raise ValueError('POMDP for {} is not defined!'.format(env_name))

        # Redefine observation_space
        obs_low = np.array([-np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        obs_high = np.array([np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        observation_space = gym.spaces.Box(obs_low, obs_high)
        return remain_obs_idx, observation_space

if __name__ == '__main__':
    import pybulletgym
    import gym
    env = POMDPWrapper("AntPyBulletEnv-v0", 'remove_velocity_and_flickering')
    obs = env.reset()
    print(env.action_space)
    print(env.observation_space)
    print(obs)

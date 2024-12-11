import pybullet as p
import pybullet_data
import gym
import numpy as np
from stable_baselines3 import PPO
import time  



class HumanoidEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(HumanoidEnv, self).__init__()
        
        # Action space: For simplicity, assume we control certain joints with continuous torques
        # Replace with actual number of controllable joints and action range
        # ok im defining the actions here, 10 means 10 controlable joints 
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Observation space: This might include joint positions, velocities, torso orientation, etc.
        # For now, pick a placeholder dimension and refine later
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        
        # Connect to PyBullet (in DIRECT mode if we don't want a pop-up every env creation)
        # self.physics_client = p.connect(p.DIRECT)
        self.physics_client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        orientation = p.getQuaternionFromEuler([1, 1, 1])  # Try different angles if needed


        self.humanoid_id = p.loadURDF("humanoid/humanoid.urdf", [0,0,1],orientation, useFixedBase=False)
        
        self._max_steps = 1000
        self._step_counter = 0

    def reset(self):
        # Reset the simulation
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

    # Set humanoid initial position and orientation
        start_pos = [0, 0, 3.3]  # Standing 1 meter above the ground
        start_orientation = p.getQuaternionFromEuler([1.5, 0, 0])  # Rotate 90 degrees (π/2 radians)
        self.humanoid_id = p.loadURDF("humanoid/humanoid.urdf", start_pos, start_orientation, useFixedBase=False)

        
        self._step_counter = 0
        
        obs = self._get_obs()
        return obs


#executes a single step in the environment, simulating the given action
    def step(self, action):
        # Apply action (torques) to humanoid joints
        # For now, just do nothing or apply a placeholder
        # You will need to identify joint indices and apply torque with `p.setJointMotorControl2`.
        
        # Step simulation, advances sim in time. like wolfram defines time. computational irreducibility 
        p.stepSimulation()
        self._step_counter += 1
        
        # Compute reward, done, info
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._step_counter > self._max_steps
        info = {}
        
        return obs, reward, done, info

    def _get_obs(self):
        # Extract joint states, base position/orientation, velocities, etc.
        # Just return zeros as placeholder
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _compute_reward(self, obs):
        # Reward for moving forward: measure humanoid base x-position or x-velocity
        # For now, return dummy value
        return 0.0






if __name__ == "__main__":
    # Create the environment
    env = HumanoidEnv()
    print("Environment created successfully.")

    # Reset the environment and get the initial observation
    obs = env.reset()
    print("Environment reset. Initial observation:", obs)

    # Run a few steps with random actions
    for step in range(10):  # Take 10 steps
        action = env.action_space.sample()  # Generate a random action
        obs, reward, done, info = env.step(action)  # Perform the action in the environment
        print(f"Step {step + 1}:")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

        time.sleep(0.1)  # Pause for 0.1 seconds between steps for better visualization

        if done:
            print("Environment terminated early.")
            break

    # Keep the GUI open for 10 seconds
    print("Simulation complete. Keeping GUI open for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 3:
        time.sleep(0.1)  # Prevent high CPU usage
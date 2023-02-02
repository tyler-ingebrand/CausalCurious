from causal_world.envs import CausalWorld
from src.custom_task import MyOwnTask

use_sphere = False
task = MyOwnTask(shape='Sphere' if use_sphere else 'Cube')
env = CausalWorld(task=task, enable_visualization=True)
print("SS: ", env.observation_space.shape)
print("AS: ", env.action_space.shape)
env.reset()
for _ in range(20000):
    obs, reward, done, info = env.step(env.action_space.sample())
env.close()
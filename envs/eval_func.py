import RLBench.rlbench.gym
from PIL import Image
import cv2
import numpy as np

def d4rl_evaluate(env,policy,num_episodes):
    rewards = []
    for n in range(num_episodes):
        obs = env.reset()
        returns = 0

        for t in range(env._max_episode_steps):
            action = policy.predict(obs)
            obs,rew,done,info = env.step(action)
            returns += rew
            if done:
                break

        rewards.append(returns)

    return rewards

def rlbench_evaluate(env,policy,num_episodes):
    episode_length = 120
    num_success = 0
    frames = {f"episode{k}":[] for k in range(num_episodes)}
    for i in range(num_episodes):
        obs = env.reset()
        for j in range(episode_length):
            action = policy.predict(obs)
            obs, reward, terminate, _ = env.step(action)
            # state = obs['state'][-6:]
            img = env.render(mode = 'rgb_array')  
            frames[f'episode{i}'].append(img)
            
            if terminate:
                num_success += 1
                break
        print(f"episode{i}: success: {terminate} ")
    succecss_rate = num_success/num_episodes

    return succecss_rate,frames

def rlbench_image_evaluate(env,policy,num_episodes):
    episode_length = 120
    num_success = 0
    frames = {f"episode{k}":[] for k in range(num_episodes)}
    for i in range(num_episodes):
        obs_list = env.reset()
        observation = np.expand_dims(obs_list["front_rgb"],axis=0)
        state = np.expand_dims(obs_list["state"],axis=0)
        for j in range(episode_length):
            action = np.squeeze(policy.predict(observation,state), axis=0)
            obs_list, reward, terminate, _ = env.step(action)
            observation = np.expand_dims(obs_list["front_rgb"],axis=0)
            state = np.expand_dims(obs_list["state"],axis=0)
            # state = obs['state'][-6:]
            img = env.render(mode = 'rgb_array')  
            frames[f'episode{i}'].append(img)
            
            if terminate:
                num_success += 1
                break
        print(f"episode{i}: success: {terminate} ")
    succecss_rate = num_success/num_episodes

    return succecss_rate,frames
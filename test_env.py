import gymnasium as gym

def make_environment():
    # Create the environment
    env = gym.make("FetchSlideDense-v2", render_mode="human")
    return env

# Create the environment
env = make_environment()

# Set the number of episodes to run
episodes = 5

# Run the episodes
for episode in range(1, episodes + 1):
    state, info = env.reset()
    done = False
    score = 0
    
    while not done:
        # Render the environment live
        env.render()
        
        # Take a random action
        action = env.action_space.sample()
        
        # Step through the environment
        n_state, reward, done, truncated, info = env.step(action)
        # print(f'Reward: {reward}')
        score += reward
        
        # Check if the episode is done or truncated
        done = done or truncated
    
    print(f'Episode: {episode} Score: {score}')

# Close the environment
env.close()

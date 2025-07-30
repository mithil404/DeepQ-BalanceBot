#!/usr/bin/env python3
"""
Script to test a trained DQN model for the self-balancing robot.
This will run the model in evaluation mode with no exploration.
"""

import torch
import numpy as np
from model import Linear_QNet
from game import Game
import time
import matplotlib.pyplot as plt

def test_model(model_path='./model/model.pth', episodes=10):
    """
    Test a trained model on multiple episodes and analyze performance
    
    Args:
        model_path: Path to the trained model
        episodes: Number of test episodes to run
    """
    # Load the model
    model = Linear_QNet(4, 24, 9)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model from {model_path}")
    
    # Initialize the game
    game = Game()
    
    # Metrics to track
    scores = []
    episode_lengths = []
    balance_metrics = []  # Will store mean absolute angle deviation
    
    # Run test episodes
    for i in range(episodes):
        game.reset()
        
        # Use random initial angle for robustness testing
        # This depends on your Game implementation supporting this
        
        done = False
        score = 0
        steps = 0
        angle_deviations = []
        
        start_time = time.time()
        print(f"\nStarting test episode {i+1}/{episodes}...")
        
        while not done:
            # Get current state
            state = game.get_state()
            angle_deviations.append(abs(state[0]))  # Absolute angle deviation
            
            # Get action from model (no exploration)
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = model(state_tensor)
            
            # Convert to one-hot encoding
            move = torch.argmax(prediction).item()
            final_move = [0] * 9
            final_move[move] = 1
            
            # Execute action
            reward, done, episode_score = game.play_step(final_move)
            score += reward
            steps += 1
            
            # Show periodic updates for long episodes
            if steps % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {steps}, Reward: {score:.1f}, Time: {elapsed:.1f}s")
        
        # Episode complete - record metrics
        elapsed = time.time() - start_time
        scores.append(score)
        episode_lengths.append(steps)
        
        # Calculate balance metrics
        if angle_deviations:
            mean_deviation = np.mean(angle_deviations)
            balance_metrics.append(mean_deviation)
            
            # Calculate time spent within different angle thresholds
            within_2deg = np.mean([1 if abs(angle) < 0.035 else 0 for angle in angle_deviations]) * 100
            within_5deg = np.mean([1 if abs(angle) < 0.087 else 0 for angle in angle_deviations]) * 100
            
            print(f"Episode {i+1} results:")
            print(f"  Score: {score:.1f}")
            print(f"  Duration: {steps} steps ({elapsed:.2f} seconds)")
            print(f"  Mean angle deviation: {mean_deviation:.4f} radians")
            print(f"  Time within 2° balance: {within_2deg:.1f}%")
            print(f"  Time within 5° balance: {within_5deg:.1f}%")
        
    # Print overall statistics
    print("\n===== Testing Complete =====")
    print(f"Episodes: {episodes}")
    print(f"Mean score: {np.mean(scores):.1f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Mean angle deviation: {np.mean(balance_metrics):.4f} radians")
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(range(1, episodes+1), episode_lengths)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length (steps)')
    ax1.set_title('Test Episode Durations')
    
    ax2.bar(range(1, episodes+1), balance_metrics, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Angle Deviation (radians)')
    ax2.set_title('Balance Performance')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()

if __name__ == '__main__':
    test_model()
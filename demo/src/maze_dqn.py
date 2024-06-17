import pygame as pg
import matplotlib.pyplot as plt
import os
import torch
from environments.src.maze.environment import MazeEnvironment
from rl.src.dqn.dqn import DQN
import logging
import json


def main():
    # Init logging
    logging.basicConfig(level=logging.INFO)

    env = MazeEnvironment(
        "Maze", "A simple maze environment", 10, 10, goal_x=3, goal_y=3)
    # Assuming env.get_state() returns a numpy array
    input_dim = env.get_state().shape[0] * env.get_state().shape[1]
    action_space = 4  # Assuming there are 4 possible actions

    # Define the model here
    nn_agent = DQN(
        input_dim=input_dim,
        output_dim=action_space,
        hidden_dims=[128, 128],
        lr=.001,
        gamma=.99,
        epsilon=1.0,
        epsilon_decay=.995,
        epsilon_min=.01,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_frequency=100,
    )

    save_path = "maze_model.pth"

    if os.path.exists(save_path):
        nn_agent.load(save_path)

    tick = 10
    render_every = 100
    save_every = 1000

    width = 800
    height = 600

    pg.font.init()
    screen = pg.display.set_mode((width, height))
    env.set_screen(screen)

    env.rewards = {
        "goal": 100.0,
        "move": -1.0,
        "wall": -1.0,
    }

    no_progress_steps = 100  # Number of steps before no progress reward
    no_progress_reward = -10.0  # Reward for no progress

    clock = pg.time.Clock()

    parameter_file = "maze_parameters.json"

    game_lengths = [0]
    iterations = []
    iteration = 0

    if os.path.exists(parameter_file):
        logging.info("Loading parameters from file")
        with open(parameter_file, 'r') as json_file:
            data = json.load(json_file)
            game_lengths = data['game_lengths']
            iterations = data['iterations']
            nn_agent.epsilon = data['epsilon']

    def save_to_json(filename):
        data = {
            'iterations': iterations,
            'game_lengths': game_lengths,
            'epsilon': nn_agent.epsilon,
        }
        with open(filename, 'w') as json_file:
            json.dump(data, json_file)

    fig, ax = plt.subplots()
    plt.ion()

    def plot():
        ax.clear()
        ax.set_xlabel('Iterations')
        ax.set_title('Game Length Over Time')
        # ax.plot(game_lengths[:-1])

        n = 100
        avg_game_lengths = [
            sum(game_lengths[i:i+n]) / n for i in range(0, len(game_lengths)-n)]
        ax.plot(avg_game_lengths)

        plt.draw()

    running = True
    while running:
        state = torch.tensor(
            env.get_state().flatten(), dtype=torch.float32).unsqueeze(0)
        action = nn_agent.choose_action(state)

        done, reward = env.step(action)

        if game_lengths[-1] != 0 and game_lengths[-1] % no_progress_steps == 0:
            reward = no_progress_reward
            done, reward = env.step(action, reward)
            done = True

        next_state = torch.tensor(
            env.get_state(), dtype=torch.float32).unsqueeze(0)

        state = state.flatten()
        next_state = next_state.flatten()

        nn_agent.update(state, action, reward, next_state, done)

        game_lengths[-1] += 1

        if done:
            iteration += 1
            game_lengths.append(0)
            iterations.append(iteration)
            env.reset()

        if iteration % save_every == 0:
            nn_agent.save(save_path)
            save_to_json(parameter_file)

        if iteration % render_every != 0:
            continue

        env.screen.fill((0, 0, 0))

        # Display the number of iterations
        env.screen.blit(pg.font.SysFont(
            "Arial", 24).render(f"Iteration: {iteration}", True, (255, 255, 255)), (10, 10))

        env.render(pg.display.get_surface())
        for event in pg.event.get():
            if event.type == pg.QUIT:
                nn_agent.save(save_path)
                save_to_json(parameter_file)
                running = False
                pg.quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    nn_agent.save(save_path)
                    save_to_json(parameter_file)
                    running = False
                    pg.quit()

        if not running:
            break

        pg.display.flip()
        clock.tick(tick)

    plot()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()

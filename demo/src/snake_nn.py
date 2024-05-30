import pygame
import matplotlib.pyplot as plt
import os
import torch
from environments.src.snake.environment import SnakeEnvironment
from rl.src.nn.nn import NeuralNetwork
import logging
import json


def main():
    # Init logging
    logging.basicConfig(level=logging.INFO)

    env = SnakeEnvironment("Maze", "A simple maze environment", 10, 10, 1)
    # Assuming env.get_state() returns a numpy array
    input_dim = env.get_state().shape[0] * env.get_state().shape[1]
    action_space = 4  # Assuming there are 4 possible actions

    # Define the model here
    nn_agent = NeuralNetwork(input_dim=input_dim, output_dim=action_space, hidden_dims=[
                             24, 24], lr=0.01, gamma=0.9, epsilon=0.1)

    save_path = "model.pth"

    if os.path.exists(save_path):
        nn_agent.load(save_path)

    tick = 10
    render_every = 200
    save_every = 1000

    width = 800
    height = 600

    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    env.set_screen(screen)

    env.rewards = {
        "move": 1000.0,
        "eat": 100.0,
        "collision": -1000.0
    }

    clock = pygame.time.Clock()

    parameter_file = "parameters.json"

    snake_lengths = []
    game_lengths = [0]
    iterations = []
    iteration = 0

    if os.path.exists(parameter_file):
        logging.info("Loading parameters from file")
        with open(parameter_file, 'r') as json_file:
            data = json.load(json_file)
            snake_lengths = data['snake_lengths']
            game_lengths = data['game_lengths']
            iterations = data['iterations']

    def save_to_json(filename):
        data = {
            'iterations': iterations,
            'snake_lengths': snake_lengths,
            'game_lengths': game_lengths
        }
        with open(filename, 'w') as json_file:
            json.dump(data, json_file)

    fig, ax = plt.subplots()
    plt.ion()

    def update_plot():
        ax.clear()
        ax.plot(iterations, snake_lengths)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Snake Length')
        ax.set_title('Snake Length Over Time')

        ax2 = ax.twinx()
        ax2.plot(iterations, game_lengths[:-1], 'r.', label='Game Length')
        ax2.set_ylabel('Game Length', color='r')
        ax2.tick_params('y', colors='r')

        plt.draw()

    while True:
        current_state = torch.tensor(
            env.get_state().flatten(), dtype=torch.float32).unsqueeze(0)
        action = nn_agent.choose_action(current_state)

        game_over, reward = env.step(action)

        next_state = torch.tensor(
            env.get_state(), dtype=torch.float32).unsqueeze(0)
        nn_agent.update(current_state, action, reward, next_state)

        game_lengths[-1] += 1

        if game_over:
            iteration += 1
            snake_lengths.append(len(env.snake))
            game_lengths.append(0)
            iterations.append(iteration)
            env.reset()

        if iteration % render_every != 0:
            continue

        if iteration % save_every == 0:
            nn_agent.save(save_path)
            save_to_json(parameter_file)

        env.screen.fill((0, 0, 0))

        # Display the number of iterations
        env.screen.blit(pygame.font.SysFont(
            "Arial", 24).render(f"Iteration: {iteration}", True, (255, 255, 255)), (10, 10))

        env.render(pygame.display.get_surface())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                update_plot()
                pygame.quit()
                plt.ioff()
                plt.show()
                nn_agent.save(save_path)
                save_to_json(parameter_file)
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    update_plot()
                    pygame.quit()
                    plt.ioff()
                    plt.show()
                    nn_agent.save(save_path)
                    save_to_json(parameter_file)
                    return

        pygame.display.flip()
        clock.tick(tick)


if __name__ == '__main__':
    main()

import pygame
from environments.src.snake.environment import SnakeEnvironment
from rl.src.qlearning.qlearning import QLearning
import matplotlib.pyplot as plt
import os


def main():
    env = SnakeEnvironment("Maze", "A simple maze environment", 10, 10)
    action_space = 4
    ql = QLearning(action_space)

    save_path = "q_table.npy"

    if os.path.exists(save_path):
        ql.load(save_path)

    tick = 10
    render_every = 100

    width = 800
    height = 600

    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    env.set_screen(screen)

    clock = pygame.time.Clock()

    lengths = []
    iterations = []

    fig, ax = plt.subplots()
    plt.ion()

    iteration = 0

    def update_plot():
        ax.clear()
        ax.plot(iterations, lengths)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Snake Length')
        ax.set_title('Snake Length Over Time')
        plt.draw()

    while True:
        current_state = env.state_to_index()
        action = ql.choose_action(current_state)

        game_over, reward = env.step(action)

        next_state = env.state_to_index()
        ql.update(current_state, action, reward, next_state)

        if game_over:
            iteration += 1
            lengths.append(len(env.snake))
            iterations.append(iteration)
            update_plot()
            env.reset()

        if iteration % render_every != 0:
            continue

        env.screen.fill((0, 0, 0))

        # Display the number of iterations
        env.screen.blit(pygame.font.SysFont(
            "Arial", 24).render(f"Iteration: {iteration}", True, (255, 255, 255)), (10, 10))

        env.render(pygame.display.get_surface())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                plt.ioff()
                plt.show()
                ql.save(save_path)
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    plt.ioff()
                    plt.show()
                    ql.save(save_path)
                    return

        pygame.display.flip()
        clock.tick(tick)
        # pygame.time.wait(1000)

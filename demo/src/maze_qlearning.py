import pygame
from environments.src.maze.environment import MazeEnvironment
from rl.src.qlearning.qlearning import QLearning
import matplotlib.pyplot as plt
import os


def main():
    env = MazeEnvironment("Maze", "A simple maze environment",
                          10, 10, goal_x=9, goal_y=9)
    action_space = 4
    ql = QLearning(action_space)

    env.rewards = {
        "goal": 0.0,
        "move": 0.0,
        "wall": 0.0,
    }

    save_path = "q_table_maze.npy"

    if os.path.exists(save_path):
        ql.load(save_path)

    tick = 60
    render_every = 100

    width = 800
    height = 600

    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    env.set_screen(screen)

    clock = pygame.time.Clock()

    path_iterations = []
    iterations = [0]
    iteration = 0

    no_progress_steps = 100
    no_progress_reward = -10.0

    fig, ax = plt.subplots()
    plt.ion()

    def update_plot():
        ax.clear()
        ax.plot(iterations)
        ax.set_xlabel('Iterations')
        plt.draw()

    while True:
        iteration += 1
        current_state = env.state_to_index()
        action = ql.choose_action(current_state)

        game_over, reward = env.step(action)

        # We will add reward for closer to goal
        if not game_over:
            reward += 1 / (abs(env.goal.pos_x - env.agent.pos_x) +
                           abs(env.goal.pos_y - env.agent.pos_y))

        next_state = env.state_to_index()
        ql.update(current_state, action, reward, next_state)

        if iteration % no_progress_steps == 0:
            ql.update(current_state, action, no_progress_reward, next_state)
            game_over = True

        if game_over:
            path_iterations.append(iteration)
            iterations.append(iterations[-1]+1)
            iteration = 0
            update_plot()
            env.reset()

        if iterations[-1] % render_every != 0:
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

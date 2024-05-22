import pygame
from src.snake.environment import SnakeEnvironment


def main():
    env = SnakeEnvironment("Maze", "A simple maze environment", 10, 10)

    width = 800
    height = 600

    screen = pygame.display.set_mode((width, height))
    env.set_screen(screen)

    clock = pygame.time.Clock()

    while True:
        env.screen.fill((0, 0, 0))

        action = None
        env.render(pygame.display.get_surface())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

                if event.key == pygame.K_UP:
                    action = "UP"
                elif event.key == pygame.K_DOWN:
                    action = "DOWN"
                elif event.key == pygame.K_LEFT:
                    action = "LEFT"
                elif event.key == pygame.K_RIGHT:
                    action = "RIGHT"

        env.step(action)

        pygame.display.flip()
        clock.tick(2)
        # pygame.time.wait(1000)

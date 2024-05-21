import pygame
from src.snake.environment import SnakeEnvironment


def main():
    env = SnakeEnvironment("Maze", "A simple maze environment", 10, 10)

    width = 800
    height = 600

    screen = pygame.display.set_mode((width, height))
    env.set_screen(screen)

    while True:
        env.screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        env.render(pygame.display.get_surface())
        input = pygame.key.get_pressed()
        action = None
        if input[pygame.K_UP]:
            action = "UP"
        elif input[pygame.K_DOWN]:
            action = "DOWN"
        elif input[pygame.K_LEFT]:
            action = "LEFT"
        elif input[pygame.K_RIGHT]:
            action = "RIGHT"

        env.step(action)

        pygame.display.flip()
        pygame.time.wait(1000)

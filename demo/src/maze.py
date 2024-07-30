import matplotlib
import matplotlib.pyplot as plt
import torch
import logging
from itertools import count
import gymnasium as gym

from rl.src.dqn.dqn_module import DQNModule
from environments.gymnasium.envs.maze.utils.preprocess_state import preprocess_state


gym.register(
    id='Maze-v0',
    entry_point='environments.gymnasium.envs.maze.maze:MazeEnv',
)

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_durations = []


def main():
    env = gym.make('Maze-v0', render_mode='rgb_array')

    _, _ = env.reset()
    state = env.render()
    state = preprocess_state(state)

    model_path = 'maze_dqn.pth'
    dqn = DQNModule(model_path, state.shape, env.action_space.n, seed=4)

    plt.ion()

    num_episodes = 100
    render_every = 5
    try:
        for i_episode in range(num_episodes):
            _, _ = env.reset()
            state = env.render()
            state = preprocess_state(state)

            for t in count():
                if i_episode % render_every == 0:
                    env.render(render_mode='human')

                action = dqn.select_action(state)
                _, reward, terminated, truncated, _ = env.step(
                    action.item())

                observation = env.render()
                observation = preprocess_state(observation)

                done, state = dqn.train(state, action, observation,
                                        reward, terminated, truncated)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()

        logging.info('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == "__main__":
    main()

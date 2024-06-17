import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from itertools import count

from rl.src.dqn.dqn import DQN

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_durations = []


def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)

    plt.ion()

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 200

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        for t in count():
            action = dqn.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(
                action.item())

            done = dqn.train(state, action, observation,
                             reward, terminated, truncated)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

            env.render()

    print('Complete')
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

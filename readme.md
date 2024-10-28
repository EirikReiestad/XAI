## Run
### Install Poetry
pip install poetry

### Install dependencies
poetry install

### Run the project
#### Run the demo
`poetry run python demo`

[tag demo](./demo/src/demos/multi_agent/tag.py)

- WandB is used for logging. If you do not have an account or do not want to use it, the `wandb_active` parameter for `MultiAgentDQN` should be set to False.
- `render_mode` should be set to False if you do not want to render the environment.

#### Tag
- random positions is changed in [tag state](./environments/gymnasium/envs/tag/tag_state.py)

##### DQN
The parameters can be changed in the [dqn](./rl/src/dqn/dqn.py) or when creating the DQN object.

#### Run the GUI
`poetry run python gui`

__Note__: GUI loads the models from WandB, so WandB is required.

Because I am lazy, there is no labels. The left figure represents the seeker (agent 0) and the right figure represents the hider (agent 1).

#### Run the environment generator
`poetry run python utils`

#### Run the data handler (generate data)
`poetry run python data_handler`

#### Run the CAV generator and plotter
`poetry run python methods/src/cav/`

__Note__: CAV generator loads the models from WandB, so WandB is required.

### Errors that may occur
- If you get model shape error, there can be multiple reasons for this.
  - The environment is the wrong shape.
  - Hidden layers / convolution layers are the wrong shape.
  - Dueling is enabled but the model does not support it.

## Test
`poetry run python -m unittest discover`

## Naming conventions for git
### Git branch prefixes
- `feature/` - these branches are used to develop new features 
- `bugfix/` - these branches are used to make fixes 
- `release/` - these branches prepare the codebase for new releases
- `hotfix/` - these branches addresses urgent issues in production

### Issues examples
feature/add-user-authentication

bugfix/fix-login-page

release/v1.0

hotfix/fix-login-bug

### Commit messages
- `feat:` - new feature
- `fix:` - bug fix
- `refactor:` - code refactoring
- `docs:` - changes in documentation
- `model:` - changes to the model parameters
- `idun:` - changes to IDUN 

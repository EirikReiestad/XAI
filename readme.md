## Run
### Install Poetry
pip install poetry

### Install dependencies
poetry install

### Run the project
#### Run the demo
´poetry run python demo´

[tag demo](./demo/src/demos/multi_agent/tag.py)

__Note__: 
- WandB is used for logging. If you do not have an account or do not want to use it, the ´wandb_active´ parameter for ´MultiAgentDQN´ should be set to False.
- ´render_mode´ should be set to False if you do not want to render the environment.

#### Run the environment generator
´poetry run python utils´

#### Run the data handler (generate data)
´poetry run python data_handler´

#### Run the CAV generator and plotter
´poetry run python methods/src/cav/´

## Test
´poetry run python -m unittest discover´

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

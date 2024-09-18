## Run
### Install Poetry
pip install poetry

### Install dependencies
poetry install

### Run the project
poetry run python demo

## Test
poetry run python -m unittest discover

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
- `idun:` - changes to IDUN 

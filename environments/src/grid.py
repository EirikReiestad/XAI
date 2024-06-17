import pygame


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.line_width = 0.01  # % of cell width
        self._cells = [[0 for _ in range(width)] for _ in range(height)]

    def render_grid(self, screen: pygame.Surface, cell_width: int, cell_height: int, color: (int, int, int) = (255, 255, 255), grid: bool = True, env: str = None):
        line_width = cell_width * self.line_width
        if grid:
            for y in range(self.height):
                for x in range(self.width):
                    pygame.draw.rect(screen, color, (x * cell_width + line_width // 2,
                                                     y * cell_height, cell_width - line_width, cell_height - line_width))

        if env == 'maze':
            self._render_maze(screen, cell_width, cell_height, (0, 0, 0))

    def _render_maze(self, screen: pygame.Surface, cell_width: int, cell_height: int, color: (int, int, int) = (0, 0, 0)):
        """
        The maze consists of walls and empty spaces.
        0: no walls (empty space)
        1: wall on the right
        2: wall on the bottom

        """
        for y in range(self.height):
            for x in range(self.width):
                if self.cells[y][x] == 1:
                    pygame.draw.rect(
                        screen, color, (x * cell_width, y * cell_height, cell_width, cell_height))

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, grid: list):
        """
        Parameters:
            grid (list): 2D list of integers
        """
        if len(grid) != self.height:
            raise ValueError(
                f'Grid state must have the same height as the grid {self.height}, not {len(grid)}')
        if all(len(row) != self.width for row in grid):
            raise ValueError(
                f'Maze width must be {self.grid.width}, not {len(grid[0])}')

        if grid is None:
            raise ValueError('Grid state must not be None')
        if len(grid) != self.height or len(grid[0]) != self.width:
            raise ValueError(
                f'Grid state must have the same shape as the grid {self.height, self.width}, not {len(grid), len(grid[0])}')
        self._cells = grid

    def __str__(self):
        return '\n'.join([''.join([str(cell) for cell in row]) for row in self.cells])

    def __getitem__(self, index):
        return self.cells[index]

    def __setitem__(self, index, value):
        self.cells[index] = value

    def __iter__(self):
        for row in self.cells:
            for cell in row:
                yield cell

    def __len__(self):
        return self.width * self.height

    def __eq__(self, other):
        return self.cells == other.cells

    def __ne__(self, other):
        return self.cells != other.cells

    def __contains__(self, item):
        return item in self.cells

    def __add__(self, other):
        if self.width != other.width or self.height != other.height:
            raise ValueError(
                'Grids must be the same size to add them together')
        new_grid = Grid(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                new_grid[y][x] = self[y][x] + other[y][x]
        return new_grid

    def __sub__(self, other):
        if self.width != other.width or self.height != other.height:
            raise ValueError('Grids must be the same size to subtract them')
        new_grid = Grid(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                new_grid[y][x] = self[y][x] - other[y][x]
        return new_grid

    def __mul__(self, other):
        new_grid = Grid(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                new_grid[y][x] = self[y][x] * other
        return new_grid

    def __rmul__(self, other):
        return self * other

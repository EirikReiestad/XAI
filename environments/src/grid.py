import pygame


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.line_width = 0.01  # % of cell width
        self.cells = [[0 for _ in range(width)] for _ in range(height)]

    def render(self, screen: pygame.Surface, cell_width: int, cell_height, color: (int, int, int) = (255, 255, 255)):
        line_width = cell_width * self.line_width
        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(screen, color, (x * cell_width + line_width // 2,
                                 y * cell_height, cell_width - line_width, cell_height - line_width))

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

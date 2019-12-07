from rubiksolver.cube import Cube
import abc


class CubeSolver(abc.ABC):

    def __init__(self):
        pass

    def solve_cube(self, cube):
        pass


class GeneticAlgorithmSolver(CubeSolver):

    def __init__(self):
        super().__init__()
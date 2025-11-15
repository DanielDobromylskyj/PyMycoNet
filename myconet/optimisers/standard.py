from .default import Optimiser as DefaultOptimiser

class Standard(DefaultOptimiser):
    def __init__(self, layers):
        super().__init__(layers)

    def create_deltas(self, gradients: list) -> list:
        """ Just uses the gradients as-is, doesn't change them at all. """
        return gradients


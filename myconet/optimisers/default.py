from ..file_api import encode_dict
from ..logger import Logger

class Optimiser:
    def __init__(self, layers):
        self.layers = layers
        self.logger = Logger()

        self.optimiser_data = {}

        self.logger.log(f"Created new '{self.__class__.__name__}' Optimiser")

    def create_deltas(self, gradients: list) -> list:
        """ Returns a list of deltas to be applied to the network - Internal Function (Override)"""
        raise NotImplementedError()

    def __apply_deltas(self, deltas: list):
        for layer_index, layer_deltas in enumerate(deltas):
            layer = self.layers[layer_index]
            layer.apply_gradient_changes(*layer_deltas)

    def apply_gradients(self, gradients: list):
        deltas = self.create_deltas(gradients)
        self.__apply_deltas(deltas)

    def write_to_file(self, file, compress):
        values = self.optimiser_data
        values["*optimiser_name*"] = self.__class__.__name__
        encode_dict(values, file, compress)

    def load_from_dict(self, values: dict):
        values.pop("*optimiser_name*")
        self.optimiser_data = values

        self.logger.log(f"Loaded '{self.__class__.__name__}' Optimiser values from disc")
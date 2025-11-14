from .default import Layer as DefaultLayer
from .util import weight_init, bias_init
from ..buffer import NetworkBuffer


"""

Gpu Weights are stored in a "1D" array.
Assume that weights are stored so that all of Input Node 1s
Weights are first, then Node 2s, then 3s, ....

"""


class FullyPopulated(DefaultLayer):
    def __init__(self, input_size, output_size, activation, loading=False):
        super().__init__()

        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation
        self.__loading = loading

        self.__weights = None
        self.__biases = None


    def load_values(self):
        if not self.__loading:
            self.__weights = NetworkBuffer(
                self.cl,
                weight_init(self.__activation, (self.__input_size,), (self.__output_size,))
            )
            self.__biases = NetworkBuffer(
                self.cl,
                bias_init(self.__activation, (self.__input_size,), (self.__output_size,))
            )

    @property
    def input_node_count(self):
        return self.__input_size

    @property
    def output_node_count(self):
        return self.__output_size

    @property
    def weight_count(self):
        return self.__input_size * self.__output_size

    @property
    def bias_count(self):
        return self.__output_size

    def serialize_to_dict(self):
        return {
            'input_size': self.__input_size,
            'output_size': self.__output_size,
            'activation': self.__activation,

            'weights': self.__weights.np,
            'biases': self.__biases.np
        }

    @staticmethod
    def load_from_dict(cl_instance, values):
        layer = FullyPopulated(values['input_size'], values['output_size'], values['activation'], loading=True)
        layer.__weights = NetworkBuffer(cl_instance, values['weights'])
        layer.__biases  = NetworkBuffer(cl_instance, values['biases'])

        return layer


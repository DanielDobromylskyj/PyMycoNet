import numpy as np

from .default import Layer as DefaultLayer
from .util import weight_init, bias_init
from ..buffer import NetworkBuffer, EmptyNetworkBuffer


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

    def forward(self, inputs: NetworkBuffer, batch: int=1) -> EmptyNetworkBuffer | NetworkBuffer:
        unreduced_outputs = EmptyNetworkBuffer(self.cl, self.weight_count * batch, np.float32)
        outputs = EmptyNetworkBuffer(self.cl, self.output_node_count * batch, np.float32)

        stage_1 = self.forward_kernel.forward(
            self.cl.queue, (batch, self.input_node_count, self.output_node_count), None,
            inputs.cl,
            unreduced_outputs.cl,
            self.__weights.cl,
            np.int32(self.input_node_count),
            np.int32(self.output_node_count)
        )

        self.forward_kernel.reducer(
            self.cl.queue, (batch, self.output_node_count), None,
            unreduced_outputs.cl,
            outputs.cl,
            self.__biases.cl,
            np.int32(self.__activation),
            np.int32(self.input_node_count),
            np.int32(self.output_node_count),
            wait_for=[stage_1]
        ).wait()

        return outputs


    def backward(self, inputs: NetworkBuffer, outputs: NetworkBuffer, previous_error: NetworkBuffer, learning_rate: float, batch: int=1):
        unreduced_next_error_gradients = EmptyNetworkBuffer(self.cl, self.weight_count * batch, np.float32)
        reduced_next_error_gradients = EmptyNetworkBuffer(self.cl, self.input_node_count * batch, np.float32)

        weight_gradients = EmptyNetworkBuffer(self.cl, self.weight_count * batch, np.float32)
        bias_gradients = EmptyNetworkBuffer(self.cl, self.bias_count * batch, np.float32)

        stage_1 = self.backward_kernel.backward(
            self.cl.queue, (batch, self.input_node_count, self.output_node_count), None,
            inputs.cl,
            outputs.cl,
            self.__weights.cl,

            previous_error.cl,
            unreduced_next_error_gradients.cl,

            weight_gradients.cl,
            bias_gradients.cl,

            np.int32(self.input_node_count),
            np.int32(self.output_node_count),

            np.int32(self.__activation),
            np.float32(learning_rate)
        )

        stage_1.wait()

        self.backward_kernel.reducer(
            self.cl.queue, (batch, self.input_node_count), None,
            unreduced_next_error_gradients.cl,
            reduced_next_error_gradients.cl,

            np.int32(self.input_node_count),
            np.int32(self.output_node_count),

            wait_for=[stage_1]
        ).wait()

        return reduced_next_error_gradients, weight_gradients, bias_gradients

    def apply_gradient_changes(self, weight_deltas, bias_deltas):
        self.__weights.np += weight_deltas
        self.__biases.np += bias_deltas

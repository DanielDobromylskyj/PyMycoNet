from .default import DefaultLayer
from ..buffer import NetworkBuffer, create_empty_buffer, mul_array
from ..file_api import encode_dict, decode_dict
from ..util import weight_init, bias_init, format_with_batching


import numpy as np


def ensure_input_is_3D(shape):
    if len(shape) == 3:
        return shape

    if len(shape) == 2:
        return shape[0], shape[1], 1

    raise Exception("Cannot Make Convoluted Layer with input shape that is not 2D or 3D")

def ensure_kernel_is_2D(shape):
    if len(shape) == 2:
        return shape

    raise Exception("Cannot Make Convoluted Layer with kernel shape that is not 2D")

def calculate_output_shape(input_shape, kernel_shape, stride):
    input_width, input_height, channels = input_shape
    kernel_width, kernel_height = kernel_shape

    output_width = ((input_width - kernel_width) // stride) + 1
    output_height = ((input_height - kernel_height) // stride) + 1

    return channels, output_width, output_height


def ensure_stride_is_int(stride):
    if type(stride) is int:
        return stride

    raise Exception("Cannot Make Convoluted Layer with stride that is not int")


class Convoluted(DefaultLayer):
    def __init__(self, input_shape: tuple[int, int, int] | tuple[int, int], kernel_shape: tuple[int, int], filter_count: int, stride:int, activation:int, is_loading=False):
        super().__init__()

        self.__input_shape = ensure_input_is_3D(input_shape)
        self.__kernel_shape = ensure_kernel_is_2D(kernel_shape)
        self.__stride = ensure_stride_is_int(stride)
        self.__filter_count = filter_count

        self.__output_shape = calculate_output_shape(self.__input_shape, self.__kernel_shape, self.__stride)
        self.__activation = activation
        self.__is_loading = is_loading

        self.weights = None
        self.bias = None

    def init_values(self):
        if not self._cl:
            raise Exception("No OpenCL (cl) parsed before attempting to initialize")

        if self.__is_loading:
            return  # do not init

        def duplicate(data, count):
            return np.tile(data, count)

        self.weights = NetworkBuffer(
            self._cl,
            duplicate(
                weight_init(self.__kernel_shape[0] * self.__kernel_shape[1] * self.__input_shape[2], 1, self.__activation),
                self.__filter_count,
            ),
            (self.__kernel_shape[0] * self.__kernel_shape[1] * self.__input_shape[2], self.__filter_count),
        )

        self.bias = NetworkBuffer(
            self._cl,
            duplicate(
                bias_init((self.__output_shape[1] * self.__output_shape[2]), 1, self.__activation),
                self.__filter_count,
            ),
            (self.__output_shape[1] * self.__output_shape[2], self.__filter_count),
        )

    def forward(self, inputs: NetworkBuffer | list, wait=True, batch=False) -> NetworkBuffer:
        inputs, batch_size = format_with_batching(self._cl, inputs, batch)

        outputs = create_empty_buffer(self._cl, (batch_size, self.__filter_count * self.__output_shape[1] * self.__output_shape[2]))

        self.execute_forward_kernel("forward",
                                    (self.__output_shape[1], self.__output_shape[2], batch_size * self.__filter_count),
                                    inputs.get_as_buffer(),
                                    outputs.get_as_buffer(),

                                    self.weights.get_as_buffer(),
                                    self.bias.get_as_buffer(),

                                    np.int32(self.__input_shape[0]),
                                    np.int32(self.__input_shape[1]),
                                    np.int32(self.__kernel_shape[0]),
                                    np.int32(self.__kernel_shape[1]),
                                    np.int32(self.__output_shape[1]),  # Output shape[0] is the channel count
                                    np.int32(self.__output_shape[2]),
                                    np.int32(self.__stride),
                                    np.int32(self.__input_shape[2]),
                                    np.int32(self.__activation),
                                    np.int32(batch_size),
                                    np.int32(self.__filter_count),
                                    ).wait()

        return outputs

    def forward_train(self, inputs: NetworkBuffer, batch=False):
        inputs, batch_size = format_with_batching(self._cl, inputs, batch)

        outputs = create_empty_buffer(self._cl, (batch_size, self.__filter_count * self.__output_shape[1] * self.__output_shape[2]))
        unactivated_outputs = create_empty_buffer(self._cl, (batch_size, self.__filter_count * self.__output_shape[1] * self.__output_shape[2]))

        self.execute_training_kernel("forward",
                                    (self.__output_shape[1], self.__output_shape[2], batch_size * self.__filter_count),
                                    inputs.get_as_buffer(),
                                     outputs.get_as_buffer(),
                                     unactivated_outputs.get_as_buffer(),

                                     self.weights.get_as_buffer(),
                                     self.bias.get_as_buffer(),

                                     np.int32(self.__input_shape[0]),
                                     np.int32(self.__input_shape[1]),
                                     np.int32(self.__kernel_shape[0]),
                                     np.int32(self.__kernel_shape[1]),
                                     np.int32(self.__output_shape[1]),  # Output shape[0] is the channel count
                                     np.int32(self.__output_shape[2]),
                                     np.int32(self.__stride),
                                     np.int32(self.__input_shape[2]),
                                     np.int32(self.__activation),
                                     np.int32(batch_size),
                                     np.int32(self.__filter_count),
                                    ).wait()

        return outputs, unactivated_outputs

    def backward(self, input_values: np.ndarray, error_gradients: NetworkBuffer, values: list, learning_rate: float, batch=False):
        input_values, batch_size = format_with_batching(self._cl, input_values, batch)

        outputs, unactivated_outputs = values

        # This is a LOT of data...
        input_error_gradients_unreduced = create_empty_buffer(self._cl,
                                        (batch_size, self.__filter_count * self.__output_shape[0] * self.__output_shape[1] * self.__output_shape[2] *
                                        self.__kernel_shape[0] * self.__kernel_shape[1]))

        weight_error_gradients_unreduced = create_empty_buffer(self._cl,
                                                              (batch_size, self.__filter_count * self.__output_shape[0] * self.__output_shape[1] *
                                                              self.__output_shape[2] *
                                                              self.__kernel_shape[0] * self.__kernel_shape[1]))

        weight_gradients = create_empty_buffer(self._cl, (batch_size, self.__filter_count * mul_array(self.weights.get_shape())))
        bias_gradients = create_empty_buffer(self._cl, (batch_size, self.__filter_count * self.__output_shape[1] * self.__output_shape[2]))

        event = self.execute_training_kernel("backwards",
                                     (self.__output_shape[1], self.__output_shape[2], batch_size * self.__filter_count),
                                     NetworkBuffer(self._cl, input_values, input_values.shape).get_as_buffer(),
                                     outputs.get_as_buffer(),
                                     self.weights.get_as_buffer(),

                                     error_gradients.get_as_buffer(),

                                     input_error_gradients_unreduced.get_as_buffer(),
                                     weight_error_gradients_unreduced.get_as_buffer(),
                                     bias_gradients.get_as_buffer(),

                                     np.int32(self.__input_shape[0]),
                                     np.int32(self.__input_shape[1]),

                                     np.int32(self.__kernel_shape[0]),
                                     np.int32(self.__kernel_shape[1]),

                                     np.int32(self.__output_shape[1]),  # Output shape[0] is the channel count need width
                                     np.int32(self.__output_shape[2]),

                                     np.int32(self.__stride),
                                     np.int32(self.__input_shape[2]),

                                     np.int32(self.__activation),
                                     np.float32(learning_rate),

                                     np.int32(batch_size),
                                     np.int32(self.__filter_count),
        )

        # TODO - Make this work with filter count
        self.execute_training_kernel("reduce_weight_gradients",  # __input_shape[2] is channel count
                                     (self.__kernel_shape[0] *  self.__kernel_shape[1], self.__input_shape[2], batch_size * self.__filter_count),
                                     weight_error_gradients_unreduced.get_as_buffer(),
                                     weight_gradients.get_as_buffer(),

                                     np.int32(self.__kernel_shape[0]),
                                     np.int32(self.__kernel_shape[1]),
                                     np.int32(self.__input_shape[2]),
                                     np.int32(self.__output_shape[1] * self.__output_shape[2]),
                                     np.int32(batch_size),
                                     np.int32(self.__filter_count),
                                     wait_for=event
        ).wait()


        if batch_size == 1:
            return None, weight_gradients.get_and_release()[0], bias_gradients.get_and_release()[0]
        else:
            return None, weight_gradients.get_and_release(), bias_gradients.get_and_release()

    def get_node_count(self):
        return self.__input_shape[0] * self.__input_shape[1] * self.__input_shape[2], self.__output_shape[1] * self.__output_shape[2]

    def save(self, file, compress):
        return encode_dict({
            "input_shape": self.__input_shape,
            "kernel_shape": self.__kernel_shape,
            "stride": self.__stride,
            "filter": self.__filter_count,
            "activation": self.__activation,

            "weights": self.weights.get_as_array(),
            "bias": self.bias.get_as_array(),
        }, file, compress)

    @staticmethod
    def load(cl, file, compressed):
        data = decode_dict(file, compressed)

        layer = Convoluted(data["input_shape"], data["kernel_shape"], data["filter"], data["stride"], data["activation"], is_loading=True)

        layer.weights = NetworkBuffer(cl, data["weights"], data["weights"].shape)
        layer.bias = NetworkBuffer(cl, data["bias"], data["bias"].shape)

        return layer

    @staticmethod
    def get_kernel_name():
        return "kernel"

    def release(self):
        self.weights.release()
        self.bias.release()

    def __str__(self):
        return f"{self.__class__.__name__}(input={self.__input_shape}, output={self.__output_shape}, kernel={self.__kernel_shape}, stride={self.__stride}, activation={self.__activation})"


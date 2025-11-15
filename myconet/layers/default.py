import os
from importlib import resources as importlib_resources
import pyopencl as cl

from ..buffer import NetworkBuffer
from ..logger import Logger
from ..file_api import encode_dict

import warnings
from pyopencl import CompilerWarning

warnings.filterwarnings("ignore", category=CompilerWarning)


class Layer:
    def __init__(self):
        self.cl = None
        self.__logger = Logger()

        self.__forward_kernel = None
        self.__backward_kernel = None

    def assign_cl_instance(self, cl_instance):
        self.cl = cl_instance

    def set_kernels(self, k1, k2):
        self.__forward_kernel = k1
        self.__backward_kernel = k2

    @property
    def forward_kernel(self):
        return self.__forward_kernel

    @property
    def backward_kernel(self):
        return self.__backward_kernel

    def __load_kernel(self, direction: str, file_name: str) -> cl.Program | None:
        if self.cl is None:
            raise Exception(f"Layer '{self.__class__.__name__}' has not been assigned a OpenCl instance!")

        path = os.path.join(str(importlib_resources.files("myconet")),
                            "kernel", direction, file_name)

        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            program = cl.Program(self.cl.ctx, f.read()).build()

        self.__logger.log(f"Loaded kernel -> '{os.path.join("kernel", direction, file_name)}'")
        return program

    def load_kernels(self) -> tuple[cl.Program | None, cl.Program | None]:
        file_name = self.__class__.__name__ + ".cl"
        self.__logger.log(f"Loading kernels for '{self.__class__.__name__}' Layer...")
        return self.__load_kernel("forward", file_name), self.__load_kernel("backward", file_name)

    def forward(self, inputs: NetworkBuffer, batch: int=1) -> NetworkBuffer:
        raise NotImplementedError

    def backward(self, inputs: NetworkBuffer, outputs: NetworkBuffer, previous_error: NetworkBuffer, learning_rate: float, batch: int=1):
        """ Must return: (next_error_gradients, weight_gradients, bias_gradients) as NetworkBuffer(s)"""
        raise NotImplementedError

    def apply_gradient_changes(self, weight_deltas, bias_deltas):
        raise NotImplementedError

    @property
    def input_node_count(self):
        raise NotImplementedError

    @property
    def output_node_count(self):
        raise NotImplementedError

    @property
    def weight_count(self):
        raise NotImplementedError

    @property
    def bias_count(self):
        raise NotImplementedError

    def serialize_to_dict(self):
        raise NotImplementedError

    def write_to_file(self, file, compress):
        values = self.serialize_to_dict()
        values["*layer_name*"] = self.__class__.__name__
        encode_dict(values, file, compress)


import pyopencl as cl
import numpy as np
import time

from .job_manager import enqueue_many
from .core.load import load_kernel, load_training_kernel
from . import file_api, buffer
from .layer import loader
from .logger import Logger
from .util import concatenate_batch_to_buffer, format_with_batching

from .optimisers import standard


class InvalidNetwork(Exception):
    pass



class OpenCL_Instance:
    def __init__(self, log=None):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx, None)

        if log:
            device = self.ctx.devices[0]
            log.debug(f"Created OpenCL Context")
            log.debug(f"Device: {device.name}")
            log.debug(f"Vendor: {device.vendor}")
            log.debug(f"Version: {device.version}")
            log.debug(f"Driver Version: {device.driver_version}")
            log.debug(f"Max Compute Units: {device.max_compute_units}")
            log.debug(f"Max Work Group Size: {device.max_work_group_size}")
            log.debug(f"Max Work Item Dimensions: {device.max_work_item_dimensions}")
            log.debug(f"Max Work Item Sizes: {device.max_work_item_sizes}")
            log.debug(f"Global Memory Size: {device.global_mem_size // (1024 * 1024)} MB")
            log.debug(f"Local Memory Size: {device.local_mem_size // 1024} KB")
            log.debug(f"Max Constant Buffer Size: {device.max_constant_buffer_size // 1024} KB")
            log.debug(f"Max Allocatable Memory: {device.max_mem_alloc_size // (1024 * 1024)} MB")
            log.debug(f"Extensions: {device.extensions}")


class ThreadedOpenCL_Instance:
    def __init__(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue


class Network:
    def __init__(self, layout: tuple, verify=True, cl_instance=None, log_level=0, load_data=None):
        self.log = Logger(log_level)
        self.cl = OpenCL_Instance(self.log) if not cl_instance else cl_instance
        self.__kernels = {}
        self.layout = layout

        self.version = (2, 0)
        self.pyn_version = (1, 2)

        self.creation_date = round(time.time() * 1e9) if load_data is None else load_data["creation_date"]

        self.pyn_config = {
            "use_compression": True,
        } if load_data is None else load_data["pyn_config"]

        if verify:
            self.log.debug("Verifying Network Layout")
            self.validate_layout()

        self.__ready_kernels()
        self.__optimiser = standard.Optimiser(self)

    def force_load_all_kernels(self):
        self.__ready_kernels(load_training_kernels=True)

    def __ready_kernels(self, load_training_kernels=False):
        self.log.debug(f"Loading Kernels (Loading Training Kernels = {load_training_kernels})")
        kernels_previously_loaded = self.__kernels != {}
        self.__kernels = {}

        for layer in self.layout:
            if layer.__class__.__name__ not in self.__kernels:
                self.__kernels[layer.__class__.__name__] = (
                    load_kernel(self.cl, layer.get_kernel_name()),
                    load_training_kernel(self.cl, layer.get_kernel_name()) if load_training_kernels else None
                )

            layer.set_logger(self.log)
            layer.set_kernels(self.cl, self.__kernels[layer.__class__.__name__])

            if not kernels_previously_loaded:
                layer.init_values()

    def validate_layout(self):
        for i in range(len(self.layout) - 1):
            nodes_out = self.layout[i].get_node_count()[1]
            nodes_in = self.layout[i + 1].get_node_count()[0]
            if nodes_in != nodes_out:
                raise InvalidNetwork(
                    f"Layout Invalid -> Layer {i+1} outputs {nodes_out}, Yet Layer {i+2} takes {nodes_in} inputs."
                )


    def capture_forward(self, input_buffer: buffer.NetworkBuffer, batch=False):  # Use for training, returns extra data
        extra_data = []
        node_values = [input_buffer.get_as_array()]

        for layer in self.layout:
            values = layer.forward_train(input_buffer, batch)
            input_buffer = values[0]

            extra_data.append(values)
            node_values.append(values[0].get_as_array())

        return input_buffer.get_as_array(), extra_data, node_values


    def forward(self, inputs, batch=False):  # Not for training. Optimised for speed, use capture_forward(input)
        if batch:
            batch = len(inputs)
            input_buffer = concatenate_batch_to_buffer(self.cl, inputs)
        else:
            input_buffer: buffer.NetworkBuffer = buffer.create_network_buffer_from_input(self.cl, inputs)

        for layer in self.layout:
            input_buffer = layer.forward(input_buffer, batch=batch)

        outputs = input_buffer.get_as_array()

        if batch:
            final_output_size = self.layout[-1].get_node_count()[1]
            outputs = outputs.reshape(-1, final_output_size)

        return outputs


    @staticmethod
    def __average_grads(data):
        sample_count = len(data)
        layer_count = len(data[0])

        averaged = data[0]

        for i in range(sample_count-1):
            layers = data[i+1]

            for layerIndex, layer in enumerate(layers):
                averaged[layerIndex][0] += layer[0]
                averaged[layerIndex][1] += layer[1]

        for layerIndex in range(layer_count):
            averaged[layerIndex][0] /= sample_count
            averaged[layerIndex][1] /= sample_count

        return averaged

    def backward(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float, batch: bool | int=False):
        x, batch_size = format_with_batching(self.cl, inputs, batch)

        inputs = buffer.create_network_buffer_from_input(self.cl, inputs)

        output, backprop_data, layer_node_values = self.capture_forward(inputs, batch)

        layer_error = target - output

        error_gradient = buffer.NetworkBuffer(self.cl, layer_error.astype(np.float32), layer_error.shape)

        if batch:
            backprop_gradients = [[] for _ in range(batch_size)]
        else:
            backprop_gradients = []

        for i in range(len(self.layout)):
            layer_index = len(self.layout) - i - 1
            layer = self.layout[layer_index]

            next_error_gradient, weight_gradients, bias_gradients = layer.backward(
                layer_node_values[layer_index],
                error_gradient,
                backprop_data[layer_index],
                learning_rate,
                batch=batch
            )

            error_gradient = next_error_gradient

            if batch is not False:
                for j in range(batch_size):
                    backprop_gradients[j].append([
                        np.array(weight_gradients[j]),
                        np.array(bias_gradients[j])
                    ])

            else:
                backprop_gradients.append([
                    weight_gradients,
                    bias_gradients
                ])

        return backprop_gradients

    def set_optimiser(self, optimiser):
        self.__optimiser = optimiser(self)

    def apply_gradients(self, gradients):
        self.__optimiser.apply_gradients(gradients)

    def score(self, inputs, targets):
        outputs = self.forward(inputs)

        return sum([
            abs(error) for error in outputs - targets
        ])

    def validate(self, validation_data):
        return sum([self.score(sample, sample.output) for sample in validation_data]) / len(validation_data)

    def train(self, training_data, validation_data, epoches, learning_rate, batch: bool | int =False):
        self.__ready_kernels(load_training_kernels=True)

        last_error = self.validate(validation_data)

        for epoch in range(epoches):
            gradients = enqueue_many(
                self.backward,
                [(sample, sample.output, learning_rate, batch) for sample in training_data],
                f"Epoch {epoch+1} | Error: {last_error} | Calculating Gradients"
            )


            averaged_gradients = self.__average_grads(gradients)
            self.apply_gradients(averaged_gradients)

            last_error = self.validate(validation_data)

    def __get_layout_types(self):
        layer_types = []

        for layer in self.layout:
            code = loader.layer_to_code(layer)

            if code not in layer_types:
                layer_types.append(code)

        return layer_types

    @staticmethod
    def __get_optimiser_id():  # todo
        return 0

    def __create_header(self, f):
        self.log.debug("Creating header")
        file_api.encode_intx(self.version[0], 1, f)  # Version - Major
        file_api.encode_intx(self.version[1], 1, f)  # Version - Minor
        self.log.debug("Added Myconet Version")


        file_api.encode_intx(self.pyn_version[0], 1, f)  # Pyn Version - Major
        file_api.encode_intx(self.pyn_version[1], 1, f)  # Pyn Version - Minor
        self.log.debug("Added pyn Version")

        flags = [self.pyn_config["use_compression"], 0, 0, 0, 0, 0, 0, 0]
        flags_int = sum([2**n for n in range(8) if flags[n]])

        file_api.encode_intx(flags_int, 1, f)  # Flags
        self.log.debug("Added Config Flags")

        layer_types = self.__get_layout_types()
        layer_types_int = sum([2^x for x in layer_types])
        file_api.encode_intx(layer_types_int, 8, f)  # All types of layer used (Checking for support)
        self.log.debug("Added Used Layer Types")

        file_api.encode_intx(self.creation_date, 8, f)  # Creation Date
        file_api.encode_intx(self.__get_optimiser_id(), 1, f)  # Get optimiser Used  (Not yet fully supported)
        self.log.debug("Added Misc Data")


    def write_network(self, f):
        self.__create_header(f)
        file_api.encode_number(len(self.layout), f)

        for layer in self.layout:
            file_api.encode_number(
                loader.layer_to_code(layer), f
            )
            layer.save(f, compress=self.pyn_config["use_compression"])
            self.log.debug("Saved Layer {}".format(layer.__class__.__name__))

    def save(self, path):
        self.log.debug("Saving...")
        open(path, "w").close()  # truncate

        with open(path, 'ab') as f:
            self.write_network(f)

        print("Saved Network")

    @staticmethod
    def __decode_flags(flags):
        return [
            (flags & 2^(7-n)) > 0
            for n in range(8)
        ]

    @staticmethod
    def __decode_header(f):
        myconet_version = (file_api.decode_intx(1, f), file_api.decode_intx(1, f))
        pyn_version = (file_api.decode_intx(1, f), file_api.decode_intx(1, f))
        flags = file_api.decode_intx(1, f)
        layer_types = file_api.decode_intx(8, f)
        creation_date = file_api.decode_intx(8, f)
        optimiser_id = file_api.decode_intx(1, f)

        return myconet_version, pyn_version, flags, layer_types, creation_date, optimiser_id

    @staticmethod
    def read_network(f, log_level):
        cl_instance = OpenCL_Instance()

        header = Network.__decode_header(f)
        myconet_version, pyn_version, flags_int, layer_types, creation_date, optimiser_id = header
        flags = Network.__decode_flags(flags_int)

        is_compressed = flags[0]

        layer_count = file_api.decode_int(f)

        layout = tuple([
            loader.code_to_layer(file_api.decode_int(f)).load(cl_instance, f, is_compressed)
            for _ in range(layer_count)
        ])

        return Network(layout, cl_instance=cl_instance, log_level=log_level)

    @staticmethod
    def load(path, log_level=1):
        with open(path, 'rb') as f:
            return Network.read_network(f, log_level)

    def __str__(self):
        inside = [str(layer) for layer in self.layout]
        return f"Myconet.Network(\n  {'\n  '.join(inside)}\n)"

    def release(self):
        for layer in self.layout:
            layer.release()

        self.cl.queue.finish()

        self.__del__()

    def __del__(self):
        self.log.close()
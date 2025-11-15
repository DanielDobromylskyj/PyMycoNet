import time

import numpy as np

from . import file_api
from .logger import Logger
from .buffer import ClInstance, NetworkBuffer, EmptyNetworkBuffer
from .file_api import decode_dict
from .layers.lookup import lookup_table as layer_lookup_table


class NetworkValidationException(Exception):
    pass


class Network:
    def __init__(self, layout: tuple, cl_instance=None, validate=True):
        self.cl = ClInstance() if not cl_instance else cl_instance
        self.__layout = layout

        self.__log = Logger()
        self.__log.log("Initializing network")

        self.version = (2, 4)
        self.pyn_version = (1, 3)

        self.creation_date = round(time.time() * 1e9)

        self.pyn_config = {
            "use_compression": True,
        }

        self.__kernels = {}
        self.__load_kernels()

        if validate:
            self.__network_validator()


    def __network_validator(self):
        self.__log.log("Validating network...")
        if len(self.__layout) == 0:
            raise NetworkValidationException("Empty network!")

        last_value = self.__layout[0].output_node_count

        for i in range(1, len(self.__layout)):
            layer = self.__layout[i]

            if layer.input_node_count != last_value:
                raise NetworkValidationException(f"Layer {i} ({layer.__class__.__name__}) has {layer.input_node_count} input nodes, expected {last_value} nodes")

            last_value = layer.output_node_count


    def __load_kernels(self):
        self.__log.log("Loading kernels")

        for i, layer in enumerate(self.__layout):
            layer.assign_cl_instance(self.cl)
            self.__log.log(f"Setting Cl-Instance for '{layer.__class__.__name__}:{i+1}' Layer...")

        for layer in self.__layout:
            if layer.__class__.__name__ not in self.__kernels:
                self.__kernels[layer.__class__.__name__] = layer.load_kernels()

            layer.set_kernels(*self.__kernels[layer.__class__.__name__])

        for layer in self.__layout:
            layer.load_values()

    @staticmethod
    def unflatten_samples(flat, shapes):
        samples = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            chunk = flat[idx: idx + size]
            samples.append(chunk.reshape(shape))
            idx += size
        return samples

    @property
    def output_shape(self):
        return self.__layout[-1].output_node_count

    def forward(self, inputs: list | tuple | np.ndarray, is_batch=False, get_all_data=False):
        batch = len(inputs) if is_batch is True else 1

        if type(inputs) is not np.ndarray:  # todo - ensure even if they pass a ndarray, its float32
            inputs = np.array(inputs, dtype=np.float32)

        if is_batch:
            self.__log.log(f"Processing batch of {len(inputs)} inputs...")
            inputs = np.concatenate([s.ravel() for s in inputs])

        next_inputs = NetworkBuffer(self.cl, inputs)
        data = [next_inputs] if get_all_data else []

        for i, layer in enumerate(self.__layout):
            next_inputs = layer.forward(next_inputs, batch=batch)

            if get_all_data:
                data.append(next_inputs)

        if is_batch:
            final_values = self.unflatten_samples(next_inputs.np, [self.output_shape for _ in range(batch)])

            if get_all_data:
                return final_values, data
            return final_values

        if get_all_data:
            return next_inputs.np,  data
        return next_inputs.np

    def backward(self, inputs: list | tuple | np.ndarray,
                 targets: list | tuple | np.ndarray,
                 learning_rate, is_batch=False) -> list[list[None | np.ndarray]]:
        """
        Returns the gradients calculated from backpropagation in the form of np arrays,
        Formated so they can be indexed by: [layer_index][0/1 for weights/biases][batch_index]

        :param inputs:
        :param targets:
        :param learning_rate:
        :param is_batch:
        :return:
        """
        batch = len(inputs) if is_batch is True else 1

        if type(targets) is not np.ndarray:  # todo - Ensure even if they pass a ndarray, its float32
            targets = np.array(targets, dtype=np.float32)

        if is_batch:
            targets = np.concatenate([s.ravel() for s in targets])

        outputs, all_layer_values = self.forward(inputs, is_batch, get_all_data=True)
        outputs = np.concatenate([s.ravel() for s in outputs], dtype=np.float32)

        # todo - Ensure this is the correct way around (targets - outputs) vs (outputs - targets)
        previous_errors = NetworkBuffer(self.cl, targets - outputs)

        gradients = [[None, None] for _ in self.__layout]
        for layer_index in range(len(self.__layout)-1, -1, -1):
            layer = self.__layout[layer_index]

            layer_inputs = all_layer_values[layer_index]
            layer_outputs = all_layer_values[layer_index+1]

            previous_errors, weight_gradients, bias_gradients = layer.backward(
                layer_inputs, layer_outputs, previous_errors, learning_rate, batch
            )

            batched_weights = self.unflatten_samples(weight_gradients.np, [layer.weight_count for _ in range(batch)])
            batched_biases = self.unflatten_samples(bias_gradients.np, [layer.bias_count for _ in range(batch)])

            gradients[layer_index] = [batched_weights, batched_biases]

        return gradients

    @staticmethod
    def __get_optimiser_id():  # todo
        return 0

    def __create_header(self, f):
        self.__log.log("Writing Header")

        file_api.encode_intx(self.version[0], 1, f)  # Version - Major
        file_api.encode_intx(self.version[1], 1, f)  # Version - Minor
        self.__log.log("Writing Myconet Version")

        file_api.encode_intx(self.pyn_version[0], 1, f)  # Pyn Version - Major
        file_api.encode_intx(self.pyn_version[1], 1, f)  # Pyn Version - Minor
        self.__log.log("Writing Pyn Version")

        flags = [self.pyn_config["use_compression"], 0, 0, 0, 0, 0, 0, 0]
        flags_int = sum([2 ** n for n in range(8) if flags[n]])
        file_api.encode_intx(flags_int, 1, f)  # Flags
        self.__log.log("Writing Config Flags")

        # todo
        #layer_types = self.__get_layout_types()
        layer_types_int = 0 #sum([2 ^ x for x in layer_types])
        file_api.encode_intx(layer_types_int, 8, f)  # All types of layer used (Checking for support)
        self.__log.log("Writing Used Layer Types")

        file_api.encode_intx(self.creation_date, 8, f)  # Creation Date
        file_api.encode_intx(self.__get_optimiser_id(), 1, f)  # Get optimiser Used  (Not yet fully supported)
        self.__log.log("Writing Misc Data")


    def save(self, path):
        self.__log.log(f"Saving network to '{path}'...")

        with open(path, "wb") as file:
            self.__create_header(file)
            self.__log.log(f"Written Header. {file.tell()} Bytes")

            file_api.encode_number(len(self.__layout), file)
            for i, layer in enumerate(self.__layout):
                old_pointer = file.tell()
                layer.write_to_file(file, compress=True)
                bytes_written = file.tell() - old_pointer

                self.__log.log(f"Written layer '{layer.__class__.__name__}:{i+1}' to file. {round(bytes_written / 1024, 1)}KB")

    @staticmethod
    def __decode_flags(flags):
        return [
            (flags & 2 ^ (7 - n)) > 0
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
    def load(path):
        log = Logger()
        log.log(f"Loading network from '{path}'...")

        cl_instance = ClInstance()

        def read_layer_from_file(cl_instance, file, compressed):
            values = decode_dict(file, compressed)

            layer_name = values["*layer_name*"]
            layer_class = layer_lookup_table[layer_name]

            return layer_class.load_from_dict(cl_instance, values)

        with open(path, "rb") as f:
            header = Network.__decode_header(f)
            myconet_version, pyn_version, flags_int, layer_types, creation_date, optimiser_id = header
            log.log(f"Loaded header | Myconet Version: {myconet_version}, Pyn Version: {pyn_version}")

            flags = Network.__decode_flags(flags_int)
            is_compressed = flags[0]
            layer_count = file_api.decode_int(f)

            layout = tuple([
                read_layer_from_file(cl_instance, f, is_compressed)
                for _ in range(layer_count)
            ])

        return Network(layout, cl_instance, True)

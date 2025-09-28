import pyopencl as pycl
import numpy as np


mf = pycl.mem_flags

def mul_array(shape):
    if type(shape) is int:
        return shape

    total = shape[0]
    for size in shape[1:]:
        total *= size
    return total

def create_network_buffer_from_input(cl, values):
    """ A Standard way of converting input values into a network buffer object"""
    np_array = np.array(values)
    return NetworkBuffer(cl, np_array.ravel().astype(np.float32), np_array.shape)


def combine_buffers(cl, buffers):  # todo - find a better / faster way of doing this
    """ Appends several buffers one after another to make a larger single buffer """
    summed = np.array([], dtype=np.float32)
    for buffer in buffers:
        summed = np.append(summed, buffer.get_as_array())
    return NetworkBuffer(cl, summed, summed.shape)


def convert_gradients_to_buffer_list(cl, buffer, chunk_size):
    """ Helper function to convert gradients to a buffer list object for easier usage"""
    data = buffer.get_as_array()
    chunks = [Gradients(cl, data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]
    return BufferList(cl, chunks)


def rearrange_feature_map_output(cl, outputs):
    """ Converts a feature map output into a single bufferlist object"""
    input_gradients = []
    weight_gradients = []
    bias_gradients = []

    for kernel in outputs:
        input_gradients.append(kernel[0])
        weight_gradients.append(kernel[1])
        bias_gradients.append(kernel[2])

    return BufferList(cl, input_gradients), BufferList(cl, weight_gradients), BufferList(cl, bias_gradients)


def create_empty_buffer(cl, shape, dtype=np.float32):
    return EmptyNetworkBuffer(cl, shape, dtype)

class EmptyNetworkBuffer:
    def __init__(self, cl, shape, dtype, create_buffer=True):
        self.cl = cl
        self.__shape = shape
        self.__dtype = dtype
        self.size = mul_array(shape) * np.dtype(dtype).itemsize

        if create_buffer:
            self.buffer = pycl.Buffer(self.cl.ctx, mf.READ_WRITE, size=self.size, hostbuf=None)

    def write_to_buffer(self, array, offset=0):
        pycl.enqueue_copy(self.cl.queue, self.buffer, array, device_offset=offset * np.dtype(self.__dtype).itemsize)

    def force_shape(self, shape):
        self.__shape = shape

    def get_dtype(self):
        return self.__dtype

    def get_shape(self):
        """ Returns the shape of the buffer """
        return self.__shape

    def is_image(self):
        """ Returns True if the buffer contains an image (Or has more than 1 dimension)"""
        return len(self.get_shape()) > 1

    def is_rgb_image(self):
        """ Returns True if the buffer contains an RGB image (Or has more than 2 dimension)"""
        return len(self.get_shape()) > 2

    def get_as_buffer(self):
        """ Returns the data as a GPU Buffer"""
        return self.buffer

    def get_as_array(self):
        """ Converts the buffer into a numpy array for CPU usage"""
        output = np.empty(self.get_shape(), dtype=np.float32)
        pycl.enqueue_copy(self.cl.queue, output, self.get_as_buffer()).wait()
        return output

    def release(self):
        self.buffer.release()

    def get_and_release(self):
        array = self.get_as_array()
        self.release()
        return array

    def __add__(self, other):
        if isinstance(other, NetworkBuffer) or isinstance(other, Gradients):
            return NetworkBuffer(self.cl, self.get_as_array() + other.get_as_array(), self.__shape)
        return NetworkBuffer(self.cl, self.get_as_array() + other, self.__shape)

    def __truediv__(self, other):
        return NetworkBuffer(self.cl, self.get_as_array() / other, self.__shape)

    def __len__(self):
        return self.get_shape()[0]



class NetworkBuffer(EmptyNetworkBuffer):
    def __init__(self, cl, data: np.ndarray, shape: tuple[int, ...]):
        super().__init__(cl, shape, data.dtype, create_buffer=False)
        self.buffer = pycl.Buffer(self.cl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, data.nbytes, hostbuf=data)


class BufferList:
    def __init__(self, cl, buffers):
        self.__cl = cl
        self.__buffers = buffers

    def get_as_buffer(self, index=None):
        """ Return a single buffer at a given index. If a index is not given, return every buffer concatenated together (As a GPU Buffer)"""
        if index is None:
            return combine_buffers(self.__cl, self.__buffers).get_as_buffer()
        else:
            return self.__buffers[index].get_as_buffer()

    def get_network_buffer(self, index):
        """ Returns the raw buffer class at a given index"""
        return self.__buffers[index]

    def get_as_array(self, index=None):
        """ Returns array form of buffer list (or index of if given)"""
        if index is None:
            return tuple([
                buffer.get_as_array()
                for buffer in self.__buffers
            ])

        else:
            return self.__buffers[index].get_as_array()

    def __len__(self):
        return len(self.__buffers)

    def __iter__(self):
        for buffer in self.__buffers:
            yield buffer

    def __getitem__(self, item):
        return self.__buffers[item]

    def __setitem__(self, key, value):
        self.__buffers[key] = value


class Gradients:
    def __init__(self, cl, gradients: np.ndarray):
        self.__cl = cl
        self.__gradients = pycl.Buffer(self.__cl.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=gradients.astype(np.float32))
        self.__shape = gradients.shape

    def get_shape(self):
        """ Returns the shape of the buffer"""
        return self.__shape

    def get_as_buffer(self):
        """ Returns the data as a GPU Buffer"""
        return self.__gradients

    def get_as_array(self):
        """ Returns the data as a CPU numpy array"""
        output = np.empty(self.get_shape(), dtype=np.float32)
        pycl.enqueue_copy(self.__cl.queue, output, self.get_as_buffer()).wait()
        return output

    def __add__(self, other_gradients):
        return Gradients(self.__cl, self.get_as_array() + other_gradients.get_as_array())

    def __truediv__(self, other):
        return Gradients(self.__cl, self.get_as_array() / other)

    def __neg__(self):
        return Gradients(self.__cl, self.get_as_array() * -1)

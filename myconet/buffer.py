import numpy
import numpy as np
import pyopencl as cl

mf = cl.mem_flags


def multiply_array(shape):
    if type(shape) == int:
        return shape

    total = 1

    for x in shape:
        total *= x

    return total


class ClInstance:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx, device=None)



class EmptyNetworkBuffer:
    def __init__(self, cl_instance: ClInstance, shape, item_dtype):
        self.__cl = cl_instance
        self.__shape = shape
        self.__dtype = item_dtype
        self.__item_count = multiply_array(shape)
        self.__size = self.__item_count * np.dtype(item_dtype).itemsize

        self.__cl_buffer = cl.Buffer(self.__cl.ctx, mf.READ_WRITE, size=self.__size, hostbuf=None)
        self.__np_buffer = numpy.empty(self.__item_count, dtype=self.__dtype)

        self.__last_sync = None
        self.__sync_np_to_cl()


    @property
    def cl(self):
        if self.__last_sync != "cl":
            self.__sync_np_to_cl()
        return self.__cl_buffer

    @property
    def np(self):
        if self.__last_sync != "np":
            self.__sync_cl_to_np()
        return self.__np_buffer


    @property
    def dtype(self):
        return self.__dtype

    @property
    def size(self):
        return self.__size

    @property
    def shape(self):
        return self.__shape

    @property
    def sync_location(self):
        return self.__last_sync


    def __sync_np_to_cl(self):  # todo - see about making a function that returns the enqueue event (for async stuff)
        cl.enqueue_copy(self.__cl.queue, self.__cl_buffer, self.__np_buffer).wait()
        self.__last_sync = "cl"

    def __sync_cl_to_np(self):  # todo - see about making a function that returns the enqueue event (for async stuff)
        cl.enqueue_copy(self.__cl.queue, self.__np_buffer, self.__cl_buffer).wait()
        self.__last_sync = "np"


    def write_to_buffer(self, array: numpy.ndarray, offset=0):
        if self.__last_sync == "np":
            end = offset + array.size
            self.__np_buffer[offset:end] = array.astype(self.__dtype, copy=False)

        if self.__last_sync == "cl":
            cl.enqueue_copy(self.__cl.queue, self.cl, array, device_offset=offset * numpy.dtype(self.__dtype).itemsize)


class NetworkBuffer(EmptyNetworkBuffer):
    def __init__(self, cl_instance: ClInstance, array: numpy.ndarray):
        super().__init__(cl_instance, array.shape, array.dtype)
        self.write_to_buffer(array, offset=0)

from io import BytesIO
import numpy as np
import socket
import time
import math

from .network import Network

class Instructions:
    IS_COMPLETE = b"IC"
    GET_NETWORK = b"GN"
    GET_TRAINING_DATA = b"GT"
    GET_PROCESS_TIME = b"PT"
    SET_LEARNING_RATE = b"LR"
    COMPUTE_GRADIENTS = b"CG"


class HostNodeConnection:
    conn = None
    addr = None
    complete = False
    samples_per_sec = 0

class Host:
    def __init__(self, ip, port, network, training_data, validation_data, learning_rate=0.01, max_nodes=200):
        self.ip = ip
        self.port = port

        self.max_nodes = max_nodes
        self.learning_rate = learning_rate

        self.network = network
        self.training_data = [[np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)] for inputs, outputs in training_data]
        self.validation_data = [[np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)] for inputs, outputs in validation_data]

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.setblocking(False)

        self.connections = []
        self.awaiting_connections = False
        self.processing = False
        self.total_samples_per_sec = 0
        self.worse_sample_rate = math.inf

    def await_connections(self):
        self.awaiting_connections = True
        self.socket.listen(5)

        while len(self.connections) < self.max_nodes and self.awaiting_connections:
            try:
                conn, addr = self.socket.accept()
                print("Accepting Node:", addr)

                node = HostNodeConnection()
                node.conn = conn
                node.addr = addr

                self.connections.append(node)

            except BlockingIOError:
                pass # No one tried to connect yet

    def set_all_nodes_to_not_complete(self):
        for node in self.connections:
            node.complete = False


    def is_recv_data(self, node):
        try:
            data = node.conn.recv(1, socket.MSG_PEEK)
            if data:
                return True
            else:
                self.connections.remove(node)
                print("Dropping Node - Disconnected:", node.addr)
                node.conn.close()

        except BlockingIOError:
            return False

        return False

    def are_all_nodes_ready(self):
        for node in self.connections:
            if not node.complete:
                return False
        return True

    def await_all_clients_completed(self, timeout: int | float = 30.0):
        self.set_all_nodes_to_not_complete()
        start = time.time()

        while (time.time() - start) < timeout and not self.are_all_nodes_ready():
            for node in self.connections:
                if node.complete:
                    continue

                if self.is_recv_data(node):
                    response = node.conn.recv(2)
                    node.complete = (response == Instructions.IS_COMPLETE)

        # Drop non-responsive nodes
        for node in self.connections:
            if not node.complete:
                self.connections.remove(node)
                print("Dropping Node - Timeout:", node.addr)
                node.conn.close()


    def send_to_all_nodes(self, msg):
        for node in self.connections:
            node.conn.send(msg)

    @staticmethod
    def send_np_array(conn, array: np.ndarray):
        byte_data = array.tobytes()
        conn.send(len(byte_data).to_bytes(4))
        conn.send(byte_data)


    def recv_np_array(self, node):
        size = int.from_bytes(self.recv_with_timeout(node, 4))
        byte_data = self.recv_with_timeout(node, size)
        return np.frombuffer(byte_data, dtype=np.float32)

    def send_sample_training_data(self):
        self.send_to_all_nodes(Instructions.GET_TRAINING_DATA)

        data = self.training_data[:min(len(self.training_data), 20)]

        for node in self.connections:
            node.conn.send(len(data).to_bytes(4))

            for inputs, outputs in data:
                self.send_np_array(node.conn, inputs)
                self.send_np_array(node.conn, outputs)

        self.await_all_clients_completed()

    def send_network(self):
        buff = BytesIO()
        self.network.write_network(buff)
        network_data = buff.getvalue()
        buff.close()

        self.send_to_all_nodes(Instructions.GET_NETWORK)
        self.send_to_all_nodes(len(network_data).to_bytes(8))
        self.send_to_all_nodes(network_data)

        self.await_all_clients_completed(timeout=3)

    def test_node_speeds(self):
        self.send_to_all_nodes(Instructions.GET_PROCESS_TIME)
        self.await_all_clients_completed(timeout=60)

        self.total_samples_per_sec = 0
        for node in self.connections:
            if self.is_recv_data(node):
                response = node.conn.recv(4)
                time_per_sample_ms = int.from_bytes(response)
                sample_rate = round(1000 / time_per_sample_ms, 2)
                node.samples_per_sec = sample_rate
                self.total_samples_per_sec += sample_rate

                if sample_rate < self.worse_sample_rate:
                    self.worse_sample_rate = sample_rate

            else:
                self.connections.remove(node)
                print("Dropping Node - Bad Speed Test:", node.addr)
                node.conn.close()

        print(f"Expected Samples/s: {round(self.total_samples_per_sec)}, Over {len(self.connections)} Nodes")


    def set_settings(self):
        # Learning Rate:
        self.send_to_all_nodes(Instructions.SET_LEARNING_RATE)

        a, b = float(self.learning_rate).as_integer_ratio()
        self.send_to_all_nodes(a.to_bytes(8))
        self.send_to_all_nodes(b.to_bytes(8))

    def distribute_work_force(self):
        counter = 0

        for node in self.connections:
            work_force_perc = node.samples_per_sec / self.total_samples_per_sec
            sample_count = len(self.training_data) * work_force_perc

            index1 = math.ceil(counter)
            index2 = math.ceil(counter + sample_count)
            samples = self.training_data[index1:index2]

            node.conn.send(Instructions.GET_TRAINING_DATA)

            node.conn.send(len(samples).to_bytes(4))

            for inputs, outputs in samples:
                self.send_np_array(node.conn, inputs)
                self.send_np_array(node.conn, outputs)

            counter += sample_count

        pre_wait_count = len(self.connections)
        self.await_all_clients_completed()

        if len(self.connections) < pre_wait_count:
            print("[WARNING] Lost some training data due to dropped nodes!")
        else:
            print("Distributed samples across nodes")

    def recv_with_timeout(self, node, buffer_size, timeout=3):
        start = time.time()
        while time.time() - start < timeout:
            if self.is_recv_data(node):
                return node.conn.recv(buffer_size)

        raise TimeoutError("Failed to receive data from node")


    def recv_gradients(self, timeout=30.0):
        start = time.time()
        gradients: list[None] | list[list] = [None for _ in range(len(self.network.layout))]

        self.set_all_nodes_to_not_complete()
        while (start + timeout) > time.time() and not self.are_all_nodes_ready():
            for node in self.connections:
                if node.complete:
                    continue

                if self.is_recv_data(node):
                    node.complete = True
                    for layer_index in range(len(gradients)):
                        weights = self.recv_np_array(node)
                        bias = self.recv_np_array(node)

                        if gradients[layer_index] is None:
                            gradients[layer_index] = [weights, bias]

                        else:
                            gradients[layer_index][0] = np.add(gradients[layer_index][0], weights)
                            gradients[layer_index][1] = np.add(gradients[layer_index][1], bias)

        if not self.are_all_nodes_ready():
            for node in self.connections:
                if not node.complete:
                    # Drop node
                    self.connections.remove(node)
                    print("\nDropping Node - Gradiant Timeout:", node.addr)
                    node.conn.close()

        # Average
        for layer in range(len(gradients)):
            gradients[layer_index][0] = np.divide(gradients[layer_index][0], len(self.connections))
            gradients[layer_index][1] = np.divide(gradients[layer_index][1], len(self.connections))



        return gradients





    def run(self):
        self.await_connections()
        self.set_settings()

        self.send_network()
        self.send_sample_training_data()
        self.test_node_speeds()
        self.distribute_work_force()

        max_wait_time = (1 / self.worse_sample_rate) * (len(self.training_data) / len(self.connections)) * 2

        self.processing = True
        epoch = 1
        symbols = list("|/-\\")

        while self.processing:
            self.send_to_all_nodes(Instructions.COMPUTE_GRADIENTS)
            last_error = self.network.validate(self.validation_data)
            gradients = self.recv_gradients(timeout=max(3.0, max_wait_time))
            self.network.apply_gradients(gradients)
            self.send_network()

            print(f"\r| {symbols[epoch % len(symbols)]} | Epoch: {epoch}, Nodes: {len(self.connections)}, Error: {round(last_error, 3)}", end="")
            epoch += 1





class Node:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.network = None
        self.learning_rate = 0
        self.training_data = []


    def recv_np_array(self):
        size = int.from_bytes(self.conn.recv(4))
        byte_data = self.conn.recv(size)
        return np.frombuffer(byte_data, dtype=np.float32)

    def ready(self):
        self.conn.send(Instructions.IS_COMPLETE)

    def recv_training_data(self):
        count = int.from_bytes(self.conn.recv(4))

        self.training_data = [
            (
                self.recv_np_array(),  # Inputs
                self.recv_np_array(),  # Outputs / targets
            )
            for _ in range(count)
        ]

        self.ready()

    def recv_network(self):
        if self.network:
            self.network.release()

        size = int.from_bytes(self.conn.recv(8))
        byte_data = self.conn.recv(size)

        buffer = BytesIO(byte_data)
        self.network = Network.read_network(buffer, log_level=-1)
        self.network.force_load_all_kernels()
        self.network.log.disable()
        buffer.close()

        self.ready()

    def test_speed(self):
        start = time.time()

        inputs, outputs = zip(*self.training_data)
        self.network.backward(inputs, outputs, learning_rate=self.learning_rate, batch=True)

        elapsed = time.time() - start
        time_per_sample = round((elapsed * 1000) / len(self.training_data))  # ms

        self.ready()

        self.conn.send(time_per_sample.to_bytes(4))

    def set_learning_rate(self):
        a = int.from_bytes(self.conn.recv(8))
        b = int.from_bytes(self.conn.recv(8))
        self.learning_rate = a / b

    def send_gradients(self, gradients):
        for layer in gradients:
            weights, biases = layer

            Host.send_np_array(self.conn, weights)
            Host.send_np_array(self.conn, biases)

    def compute_gradients(self):
        inputs, outputs = zip(*self.training_data)

        gradients_unaveraged = self.network.backward(inputs, outputs, learning_rate=self.learning_rate, batch=True)
        gradients = self.network.average_grads(gradients_unaveraged)

        self.send_gradients(gradients)


    def start(self):
        self.conn.connect((self.ip, self.port))

        while True:
            try:
                instruction = self.conn.recv(2)
            except ConnectionResetError:
                print("Node Dropped :(")
                return

            if instruction == Instructions.GET_NETWORK:
                self.recv_network()

            elif instruction == Instructions.GET_TRAINING_DATA:
                self.recv_training_data()

            elif instruction == Instructions.GET_PROCESS_TIME:
                self.test_speed()

            elif instruction == Instructions.SET_LEARNING_RATE:
                self.set_learning_rate()

            elif instruction == Instructions.COMPUTE_GRADIENTS:
                self.compute_gradients()

            else:
                raise Exception("Unknown Instruction:", instruction)





from io import BytesIO
import numpy as np
import socket
import time

from .network import Network

class Instructions:
    IS_COMPLETE = b"IC"
    GET_NETWORK = b"GN"
    GET_TRAINING_DATA = b"GT"
    GET_PROCESS_TIME = b"PT"
    SET_LEARNING_RATE = b"LR"


class HostNodeConnection:
    conn = None
    addr = None
    complete = False
    samples_per_sec = 0

class Host:
    def __init__(self, ip, port, network, training_data, learning_rate=0.1, max_nodes=200):
        self.ip = ip
        self.port = port

        self.max_nodes = max_nodes
        self.learning_rate = learning_rate

        self.network = network
        self.training_data = [[np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)] for inputs, outputs in training_data]

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.setblocking(False)

        self.connections = []
        self.awaiting_connections = False

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

    def await_all_clients_completed(self, timeout=30):
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

        self.await_all_clients_completed()

    def test_node_speeds(self):
        self.send_to_all_nodes(Instructions.GET_PROCESS_TIME)
        self.await_all_clients_completed(timeout=60)

        total_samples_per_sec = 0
        for node in self.connections:
            if self.is_recv_data(node):
                response = node.conn.recv(4)
                time_per_sample_ms = int.from_bytes(response)
                sample_rate = round(1000 / time_per_sample_ms, 2)
                node.samples_per_sec = sample_rate
                total_samples_per_sec += sample_rate

            else:
                self.connections.remove(node)
                print("Dropping Node - Bad Speed Test:", node.addr)
                node.conn.close()

        print(f"Expected Samples/s: {round(total_samples_per_sec)}, Over {len(self.connections)} Nodes")


    def set_settings(self):
        # Learning Rate:
        self.send_to_all_nodes(Instructions.SET_LEARNING_RATE)

        a, b = float(self.learning_rate).as_integer_ratio()
        self.send_to_all_nodes(a.to_bytes(8))
        self.send_to_all_nodes(b.to_bytes(8))

    def run(self):
        self.await_connections()
        self.set_settings()

        self.send_network()
        self.send_sample_training_data()
        self.test_node_speeds()

        print("DONE")




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
        size = int.from_bytes(self.conn.recv(8))
        byte_data = self.conn.recv(size)

        buffer = BytesIO(byte_data)
        self.network = Network.read_network(buffer, log_level=1)
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

    def start(self):
        self.conn.connect((self.ip, self.port))

        while True:
            instruction = self.conn.recv(2)

            if instruction == Instructions.GET_NETWORK:
                self.recv_network()

            elif instruction == Instructions.GET_TRAINING_DATA:
                self.recv_training_data()

            elif instruction == Instructions.GET_PROCESS_TIME:
                self.test_speed()

            elif instruction == Instructions.SET_LEARNING_RATE:
                self.set_learning_rate()

            else:
                raise Exception("Unknown Instruction:", instruction)





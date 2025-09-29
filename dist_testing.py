from myconet.distrobuted import Host, Node
from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected
import threading, time

print("Creating network")
net = Network((
    FullyConnected(1, 2, 1),
), log_level=1)

print("Created Network")

training_data = [
    [[1], [2, 4]],
    [[2], [4, 8]],
    [[3], [6, 12]],
    [[4], [8, 16]],
    [[5], [10, 20]]
]

host = Host("localhost", 8080, net, training_data)

def _host():
    host.run()


threading.Thread(target=_host, daemon=True).start()
time.sleep(1)

node1 = Node("localhost", 8080)
threading.Thread(target=node1.start, daemon=True).start()

node2 = Node("localhost", 8080)
threading.Thread(target=node2.start, daemon=True).start()

node3 = Node("localhost", 8080)
threading.Thread(target=node3.start, daemon=True).start()

time.sleep(1)
host.awaiting_connections = False  # Force start

while True:
    time.sleep(1)






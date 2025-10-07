from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected


net = Network((
    FullyConnected(1, 2, 1),
), log_level=1, monitor=True)

training_data = [
    [[(i / 100)], [(i / 100) * 2, (i / 100) * 4]]
    for i in range(5, 100)
]

verification_data = [
    [[(i / 100)], [(i / 100) * 2, (i / 100) * 4]]
    for i in range(5)
]

epoches = 1000
learning_rate = 0.0001

net.train(training_data, verification_data, epoches, learning_rate, False)


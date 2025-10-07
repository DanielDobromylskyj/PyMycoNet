from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted

import numpy as np

net = Network((
    Convoluted((4, 4, 1), (2, 2), 2, 1),
), log_level=1)



training_data = [
    [np.full((4, 4), 0, dtype=np.float32), [0, 0, 0, 0]],
    [np.full((4, 4), 0.4, dtype=np.float32), [0.4, 0.4, 0.4, 0.4]],
    [np.full((4, 4), 0.6, dtype=np.float32), [0.6, 0.6, 0.6, 0.6]],
    [np.full((4, 4), 1, dtype=np.float32), [1, 1, 1, 1]],
]

validation_data = [
    [np.full((4, 4), 0.2, dtype=np.float32), [0.2, 0.2, 0.2, 0.2]],
    [np.full((4, 4), 0.5, dtype=np.float32), [0.5, 0.5, 0.5, 0.5]],
]


net.train(training_data, validation_data, 10, 0.01, batch=False)

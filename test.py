import myconet.network
from myconet.layers.populated import FullyPopulated
from myconet.optimisers.standard import Standard as StandardOptimiser


# todo - re-write FileAPI Specification


net = myconet.network.Network((
    FullyPopulated(2, 3, 1),  # Both ReLU (activation = 1)
    FullyPopulated(3, 1, 1)
), optimiser=StandardOptimiser)

net.save("test.pyn")

net2 = myconet.network.Network.load("test.pyn")

values = [[5, 5], [3, 3], [2, 5], [7, 2], [3, 0]]
targets = [[5], [3], [3.5], [5], [1.5]]
learning_rate = 0.01

outputs_1 = [net2.forward(value, is_batch=False) for value in values]
outputs_2 = net2.forward(values, is_batch=True)

if outputs_1[0] != outputs_2[0]:
    print("Forward Batch vs Linear Output Mismatch!")
    print(outputs_1)
    print(outputs_2)


gradients_1 = [
    net2.backward(values[i], targets[i], learning_rate, is_batch=False)
    for i in range(len(values))
]
gradients_2 = net2.backward(values, targets, learning_rate, is_batch=True)

batched_item = 1
bias_or_weight = 1
layer_index = 0

if gradients_1[batched_item][layer_index][bias_or_weight].all() != gradients_2[layer_index][bias_or_weight][batched_item].all():
    print("Backward Batch vs Linear Output Mismatch!")
    print(gradients_1[batched_item][layer_index][bias_or_weight])
    print(gradients_2[layer_index][bias_or_weight][batched_item])

average_1 = net2.average_gradients(gradients_1, is_batch=False)
average_2 = net2.average_gradients(gradients_2, is_batch=True)

print(average_1)
print(average_2)

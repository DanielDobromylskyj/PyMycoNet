from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted

from tabulate import tabulate
import numpy as np
import random
import time

BATCH_SIZE = 300

test_data = [
    np.random.randn(30_000).astype(np.float32)
    for _ in range(BATCH_SIZE)
]

test_data_conv = [
    np.random.randn(30_000).astype(np.float32).reshape((100, 100, 3))
    for _ in range(BATCH_SIZE)
]

print("Creating Training Data...  (50%)")

training_data = [
    (
        data,
        np.array((random.randint(0, 1),), dtype=np.float32)
     )
    for data in test_data
]

conv_training_data = [
    (
        data,
        np.random.randn(2304).astype(np.float32)
    )
    for data in test_data
]

print("Created training Data")

net_dense = Network((
    FullyConnected(30_000, 10, 2),  # Sigmoid
), log_level=1)

net_convoluted = Network((
    Convoluted((100, 100, 3), (5, 5), 1, 2, 2),  # Sigmoid
), log_level=1)

net_convoluted.log.disable()

net_dense.force_load_all_kernels()  # Force load kernels as we are not going though normal .train() method
net_convoluted.force_load_all_kernels()


# Forward Test -> Standard Execution

standard_start = time.time()


outputs1 = [
    net_dense.forward(item, batch=False) for item in test_data
]

standard_end = time.time()

print("Completed Forward Standard Tests")

# Forward Test -> Batch Execution

batched_start = time.time()

outputs2 = net_dense.forward(test_data, batch=True)

batched_end = time.time()

print("Completed Forward Batched Tests")

# Training Test -> Standard Execution


training_standard_start = time.time()

training_outputs_standard = [
    net_dense.backward(inputs, outputs, 0.01)
    for inputs, outputs in training_data
]

training_standard_end = time.time()

print("Completed Backward Standard Tests")

# Training Test -> Batch Execution

training_batch_start = time.time()

inputs, outputs = zip(*training_data)
training_outputs_batched = net_dense.backward(inputs, outputs, 0.01, batch=True)

training_batch_end = time.time()

print("Completed Backward Batched Tests")

net_dense.log.disable()
net_convoluted.log.disable()

standard_conv_start = time.time()

conv_net_forward_outputs = [
    net_convoluted.forward(sample, batch=False)[0]
    for sample in test_data_conv
]

standard_conv_end = time.time()

print("Completed Conv Forward Standard Tests")

batched_conv_start = time.time()

conv_net_forward_outputs_batched = net_convoluted.forward(test_data_conv, batch=True)

batched_conv_end = time.time()

print("Completed Conv Forward Batched Tests")

training_conv_standard_start = time.time()

training_conv_outputs_standard = [
    net_convoluted.backward(inputs, outputs, 0.01)
    for inputs, outputs in conv_training_data
]

training_conv_standard_end = time.time()

print("Completed Conv Backwards Standard Tests")

training_conv_batched_start = time.time()

inputs, outputs = zip(*conv_training_data)
training_conv_outputs_batched = net_convoluted.backward(inputs, outputs, 0.01, batch=True)

training_conv_batched_end = time.time()

print("Completed Conv Backwards Standard Tests")

# Test Output / Calculations

forward_standard_elapsed = standard_end - standard_start
forward_batched_elapsed = batched_end - batched_start

backward_standard_elapsed = training_standard_end - training_standard_start
backward_batch_elapsed = training_batch_end - training_batch_start

conv_forward_standard_elapsed = standard_conv_end - standard_conv_start
conv_forward_batched_elapsed = batched_conv_end - batched_conv_start

conv_backward_standard_elapsed = training_conv_standard_end - training_conv_standard_start
conv_backwards_batched_elapsed = training_conv_batched_end - training_conv_batched_start



def get_shape(obj):
    """Recursively get the shape of a list or ndarray."""
    if isinstance(obj, np.ndarray):
        return obj.shape
    elif isinstance(obj, list):
        if not obj:  # empty list
            return (0,)
        return (len(obj),) + get_shape(obj[0])
    else:
        return ()  # scalar

def combined_shape(*objs):
    """Get the total combined shape of multiple lists/arrays."""
    shapes = [get_shape(o) for o in objs]
    return tuple(max(s[i] if i < len(s) else 1 for s in shapes)
                 for i in range(max(map(len, shapes))))

def same_shape(a, b):
    """Recursively checks if the structure of a and b are the same."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(same_shape(x, y) for x, y in zip(a, b))
    else:
        # Base case: both must be scalar types
        return not isinstance(a, (list, np.ndarray)) and not isinstance(b, (list, np.ndarray))

def compare_values(a, b, tol=1e-8):
    """Recursively compare values of a and b."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.allclose(a, b, atol=tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(compare_values(x, y, tol=tol) for x, y in zip(a, b))
    else:
        return a == b

def compare_nested_lists(a, b):
    """Wrapper function to compare shape and values."""
    if not same_shape(a, b):
        return False, "Shape mismatch"

    if not compare_values(a, b):
        return False, "Value mismatch"

    return True, None

outputs1 = np.array(outputs1)  # Just for formats / comparing
output_match_1, err1 = compare_nested_lists(outputs1, outputs2)
output_match_2, err2 = compare_nested_lists(training_outputs_standard, training_outputs_batched)

conv_net_forward_outputs = np.array(conv_net_forward_outputs)
conv_output_match_1, err3 = compare_nested_lists(conv_net_forward_outputs, conv_net_forward_outputs_batched)
conv_output_match_2, err4 = compare_nested_lists(training_conv_outputs_standard, training_conv_outputs_batched)

print("\n\n>> PERFORMANCE BREAK DOWN <<")
print("Batch Size ->", len(test_data))

batch_increase_forward = round((forward_standard_elapsed - forward_batched_elapsed) / forward_standard_elapsed * 100, 1)
batch_increase_backward = round((backward_standard_elapsed - backward_batch_elapsed) / backward_standard_elapsed * 100, 1)

conv_batch_increase_forward = round((conv_forward_standard_elapsed - conv_forward_batched_elapsed) / conv_forward_standard_elapsed * 100, 1)
conv_batch_increase_backward = round((conv_backward_standard_elapsed - conv_backwards_batched_elapsed) / conv_backward_standard_elapsed * 100, 1)

print(f"Batch Speed Increase (Dense) -> {batch_increase_forward}%, {batch_increase_backward}%")
print(f"Batch Speed Increase (Convoluted) -> {conv_batch_increase_forward}%, {conv_batch_increase_backward}%")
print("Forward Output Match ->", output_match_1, f"-> {err1} error" if err1 else "")
print("Backward Output Match ->", output_match_2, f"-> {err2} error" if err2 else "")
print("Forward (Convoluted) Output Match ->", conv_output_match_1, f"-> {err3} error" if err3 else "")
print("Backward (Convoluted) Output Match ->", conv_output_match_2, f"-> {err4} error" if err4 else "")

if err1:
    print("Dense Forward Output Error!")
    print(outputs1)
    print(outputs2)

if err2 and not err1:
    print("Dense backward Output Error!")
    print(training_outputs_standard[2])
    print(training_outputs_batched[2])

if err3 and not (err1 or err2):
    print("Conv Forward Output Error!")
    print(get_shape(conv_net_forward_outputs), "vs", get_shape(conv_net_forward_outputs_batched))
    print(conv_net_forward_outputs.shape, "vs", conv_net_forward_outputs_batched.shape)


    #print(conv_net_forward_outputs)
    #print(conv_net_forward_outputs_batched)

if err4 and not (err1 or err2 or err3):
    print("Conv Backward Output Error!")
    print(get_shape(training_conv_outputs_standard), "vs", get_shape(training_conv_outputs_batched))

    print(training_conv_outputs_standard[2])
    print(training_conv_outputs_batched[2])


print("\n")
data = [
    ["Dense", "Forward", "Iterative", round(forward_standard_elapsed, 3), round(forward_standard_elapsed / len(test_data) * 1000, 2), round(forward_standard_elapsed / len(test_data) * 10_000, 1)],
    ["Dense", "Forward", "Grouped", round(forward_batched_elapsed, 3), round(forward_batched_elapsed / len(test_data) * 1000, 2), round(forward_batched_elapsed / len(test_data) * 10_000, 1)],

    ["Dense", "Backward", "Iterative", round(backward_standard_elapsed, 3), round(backward_standard_elapsed / len(test_data) * 1000, 2), round(backward_standard_elapsed / len(test_data) * 10_000, 1)],
    ["Dense", "Backward", "Grouped", round(backward_batch_elapsed, 3), round(backward_batch_elapsed / len(test_data) * 1000, 2), round(backward_batch_elapsed / len(test_data) * 10_000, 1)],

    ["Convoluted", "Forward", "Iterative", round(conv_forward_standard_elapsed, 3),
     round(conv_forward_standard_elapsed / len(test_data_conv) * 1000, 2),
     round(conv_forward_standard_elapsed / len(test_data_conv) * 10_000, 1)],

    ["Convoluted", "Forward", "Grouped", round(conv_forward_batched_elapsed, 3),
     round(conv_forward_batched_elapsed / len(test_data_conv) * 1000, 2),
     round(conv_forward_batched_elapsed / len(test_data_conv) * 10_000, 1)],

    ["Convoluted", "Backward", "Iterative", round(conv_backward_standard_elapsed, 3),
     round(conv_backward_standard_elapsed / len(test_data_conv) * 1000, 2),
     round(conv_backward_standard_elapsed / len(test_data_conv) * 10_000, 1)],

    ["Convoluted", "Backward", "Grouped", round(conv_backwards_batched_elapsed, 3),
     round(conv_backwards_batched_elapsed / len(test_data_conv) * 1000, 2),
     round(conv_backwards_batched_elapsed / len(test_data_conv) * 10_000, 1)],

]

headers = ["Layer Type", "Direction", "Type", "Total Time (s)", "Time Per Item (ms)", "Time Per Epoch (10k Training) (s)"]

net_convoluted.log.disable()
net_dense.log.disable()
print(tabulate(data, headers=headers, tablefmt="grid"))



__kernel void forward(
    __global float* inputs,
    __global float* unreduced_outputs,
    __global float* weights,
    int input_size,
    int output_size
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);
    int output_index = get_global_id(2);

    int input_offset = input_size * batch_index;
    int bias_offset = output_size * batch_index;
    int unreduced_output_offset = input_size * output_size * batch_index;

    int weight_index = input_index * output_size + output_index; // Same as "unreduced_output_index"

    float value = inputs[input_offset + input_index];
    float weight = weights[weight_index];

    float weighted_value = value * weight;
    unreduced_outputs[unreduced_output_offset + weight_index] = weighted_value;
}

__kernel void reducer(
     __global float* unreduced_outputs,
     __global float* reduced_outputs,
     __global float* biases,
     int activation_id,
     int input_size,
     int output_size
) {
    int batch_index = get_global_id(0);
    int output_index = get_global_id(1);

    int unreduced_output_offset = input_size * output_size * batch_index;
    int reduced_output_offset = output_size * batch_index;


    float total = biases[output_index]; // Sum up all values for a given output node
    for (int input_index=0; input_index < input_size; input_index++) {
        int unreduced_output_index = input_index * output_size + output_index;
        total += unreduced_outputs[unreduced_output_offset + unreduced_output_index];
    }

    float activated; // Activate value
    switch (activation_id) {
        case 1: // ReLU
            activated = total > 0.0f ? total : 0.0f;
            break;
        case 2: // Sigmoid
            activated = 1.0f / (1.0f + exp(-total));
            break;
        default: // No activation - Not recommended
            activated = total;
            break;
    }

    reduced_outputs[reduced_output_offset + output_index] = activated;
}
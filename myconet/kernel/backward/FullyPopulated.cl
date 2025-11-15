
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float derivative_sigmoid(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

inline float derivative_relu(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

inline float clip(float value, float clip_value) {
    return fmin(fmax(value, -clip_value), clip_value);
}

__kernel void backward(
    __global float* inputs,
    __global float* outputs,
    __global float* weights,

    __global float* previous_error_gradients,
    __global float* unreduced_next_error_gradients,

    __global float* weight_gradients,
    __global float* bias_gradients,

    int input_node_count,
    int output_node_count,

    int activation_id,
    float learning_rate
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);
    int output_index = get_global_id(2);

    int weight_index = input_index * output_node_count + output_index;

    int input_offset = input_node_count * batch_index;
    int output_offset = output_node_count * batch_index;

    int weight_gradient_offset = input_node_count * output_node_count * batch_index;
    //int next_error_gradients_offset = input_node_count * output_node_count * batch_index;
    int previous_error_gradients_offset = output_node_count * batch_index;
    int bias_gradient_offset = output_node_count * batch_index;

    float activated_output = outputs[output_offset + output_index];
    float weight = weights[weight_gradient_offset + weight_index];

    float derivative = 1.0f;
    switch (activation_id) {
        case 1: // ReLU activation
            derivative = activated_output > 0 ? 1.0f : 0.0f;
            break;
        case 2: // Sigmoid activation
            derivative = activated_output * (1.0f - activated_output);
            break;
        default:
            derivative = 1.0f; // Linear activation (default)
            break;
    }

    float delta = previous_error_gradients[previous_error_gradients_offset + output_index] * derivative;
    float weight_gradient = delta * inputs[input_offset + input_index] * learning_rate;

    weight_gradients[weight_gradient_offset + weight_index] = clip(weight_gradient, 1.0f);

    unreduced_next_error_gradients[weight_gradient_offset + weight_index] = clip(weights[weight_index] * delta, 1.0f); // Yes, we use weight index, its reduced later

    if (input_index == 0) { // Only run this once per output node
        bias_gradients[bias_gradient_offset + output_index] = delta * learning_rate;
    }
}


__kernel void reducer(
    __global float* unreduced_error_gradients,
    __global float* reduced_error_gradients,
    int input_size,
    int output_size
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);

    int unreduced_gradient_offset = input_size * output_size * batch_index;
    int reduced_gradient_offset = input_size * batch_index;

    float total = 0.0f;
    for (int output_index=0; output_index<output_size; output_index++) {
        int weight_index = input_index * output_size + output_index;
        total += unreduced_error_gradients[unreduced_gradient_offset + weight_index];
    }

    reduced_error_gradients[reduced_gradient_offset + input_index] = total;
}

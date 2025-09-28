
inline float relu(float x) {
    return fmax(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float clip(float value, float clip_value) {
    return fmin(fmax(value, -clip_value), clip_value);
}

__kernel void forward(__global float* inputs,
                      __global float* outputs,
                      __global float* unactivated_outputs,

                      __global float* weights,
                      __global float* biases,

                      int input_width,
                      int input_height,
                      int kernel_width,
                      int kernel_height,
                      int output_width,
                      int output_height,

                      int stride,
                      int channels,
                      int activation_type,
                      int max_batches,
                      int filter_count
) {
    int output_x = get_global_id(0);
    int output_y = get_global_id(1);
    int mixed_index = get_global_id(2);

    int batch_index  = mixed_index % max_batches;
    int filter_index = mixed_index / max_batches;

    int batch_output_offset = output_width * output_height * filter_count * batch_index;
    int batch_input_offset = input_width * input_height * channels * filter_count * batch_index;

    int filter_output_offset = output_width * output_height * filter_index;
    int filter_weight_offset = kernel_width * kernel_height * channels * filter_index;

    int output_index = output_y * output_width + output_x;
    int input_x_anchor = output_x * stride;
    int input_y_anchor = output_y * stride;

    float total_sum = biases[filter_index];
    for (int channel=0; channel<channels; channel++) {
        int base_weight_index = kernel_width * kernel_height * channel + filter_weight_offset;
        int base_input_index = (input_width * input_height * channel) + batch_input_offset;

        for (int dx=0; dx<kernel_width; dx++) {
            for (int dy=0; dy<kernel_height; dy++) {
                int weight_index = base_weight_index + (dy * kernel_width) + dx;
                int input_index = base_input_index + ((input_y_anchor + dy) * input_height) + (input_x_anchor + dx);

                float weighted_value = weights[weight_index] * inputs[input_index];
                total_sum += weighted_value;
            }
        }
    }

    float activated = 0.0f;
    switch (activation_type) {
            case 1:  // ReLU
                activated = relu(total_sum);
                break;
            case 2:  // Sigmoid
                activated = sigmoid(total_sum);
                break;
            default: // Default
                activated = total_sum;
                break;
    }

    unactivated_outputs[output_index + batch_output_offset + filter_output_offset] = total_sum;
    outputs[output_index + batch_output_offset + filter_output_offset] = activated;
}

__kernel void backwards(
            __global float* inputs, // Input
            __global float* outputs, // Input
            __global float* weights, // Input

            __global float* output_error_gradients, // Input

            __global float* input_error_gradients_unreduced, // Output
            __global float* weight_gradients_unreduced, // Output
            __global float* bias_gradients, // Output

            int input_width, // Static / Input
            int input_height, // Static / Input
            int kernel_width, // Static / Input
            int kernel_height, // Static / Input
            int output_width, // Static / Input
            int output_height,
            int stride, // Static / Input
            int channels, // Static / Input
            int activation_type, // Static / Input
            float learning_rate, // Static / Input
            int max_batches,
            int filter_count
) {
    int output_x = get_global_id(0);
    int output_y = get_global_id(1);
    int mixed_index = get_global_id(2);

    int batch_index  = mixed_index % max_batches;
    int filter_index = mixed_index / max_batches;

    int filter_output_offset = output_width * output_height * filter_index;
    int filter_weight_offset = kernel_width * kernel_height * channels * filter_index;

    int kernel_weight_count = kernel_width * kernel_height * channels;
    int output_count = output_width * output_height;

    int output_batch_offset = output_count * filter_count * batch_index;
    int input_batch_offset = input_width * input_height * channels * filter_count * batch_index;
    int unreduced_buffer_offset = kernel_weight_count * output_count * filter_count * batch_index;

    int relative_output_index = output_y * output_width + output_x;
    int output_index = relative_output_index + output_batch_offset + filter_output_offset;

    float activated_value = outputs[output_index];

    float derivative = 1.0f;
    switch (activation_type) {
        case 1: // ReLU activation
            derivative = activated_value > 0 ? 1.0f : 0.0f;
            break;
        case 2: // Sigmoid activation
            derivative = activated_value * (1.0f - activated_value);
            break;
        default:
            derivative = 1.0f; // Linear activation (default)
            break;
    }

    float delta = output_error_gradients[output_index] * derivative;
    bias_gradients[output_index] = delta * learning_rate;

    int input_x_anchor = output_x * stride;
    int input_y_anchor = output_y * stride;

    for (int channel=0; channel<channels; channel++) {
        int base_weight_index = kernel_width * kernel_height * channel;
        int base_input_index = input_width * input_height * channel;

        for (int dx=0; dx<kernel_width; dx++) {
            for (int dy=0; dy<kernel_height; dy++) {
                int weight_index = base_weight_index + (dy * kernel_width) + dx + filter_weight_offset; // No need to add batching logic, ony 1 set of weights/biases
                int input_index = base_input_index + ((input_y_anchor + dy) * input_width) + (input_x_anchor + dx);

                float weight_gradient = delta * inputs[input_index + input_batch_offset] * learning_rate;
                float input_error_gradient = clip(weights[weight_index] * delta, 1.0f);  // Dont *learning rate, as this makes the gradients disappear faster (Bad)

                // Here comes the silly (and probably wrong) indexing
                int max_output_weight_size = kernel_width * kernel_height * channels;
                int unreduced_index = (max_output_weight_size * relative_output_index) + weight_index + unreduced_buffer_offset;  // Its the same for weights & input gradients

                weight_gradients_unreduced[unreduced_index] = weight_gradient;
                input_error_gradients_unreduced[unreduced_index] = input_error_gradient;
            }
        }
    }
}


__kernel void reduce_weight_gradients(
    __global float* weight_gradients_unreduced,
    __global float* weight_gradients,

    int kernel_width,
    int kernel_height,
    int channels,

    int output_size,
    int max_batches,
    int filter_count
) {
    int kernel_index = get_global_id(0);
    int channel = get_global_id(1);
    int mixed_index = get_global_id(2);

    int batch_index  = mixed_index % max_batches;
    int filter_index = mixed_index / max_batches;

    int kernel_x = kernel_index / kernel_width;
    int kernel_y = kernel_index % kernel_width;

    int values_per_output = kernel_width * kernel_height * channels;
    int values_per_filter = values_per_output * output_size;
    int values_per_batch = values_per_filter * filter_count;

    int batch_offset = values_per_batch * batch_index;
    int filter_offset = values_per_filter * filter_index;
    // Cant calculate output offset here (Missing values)
    int weight_offset = (kernel_y * kernel_width) + kernel_x;

    // Buffer Layout: [batch_index][filter_index][output_index][weight_index]
    // [batch_index][filter_index]
    // We want to sum along "output_index"?
    // [batch_index][filter_index] will be "static" / not change over our loop
    // [weight_index] Should also be "static" over our loop, but NOT the output index/offset

    float weight_gradient_sum = 0.0f;
    for (int output_index = 0; output_index < output_size; output_index++) {
        // fixme / todo / Broken - weight_index might be wrong too but idk this is a future me problem. (I hate past me)
        int output_offset = values_per_output * output_index;
        int unreduced_index = batch_offset + filter_offset + output_offset + weight_offset;

        weight_gradient_sum += weight_gradients_unreduced[unreduced_index];
    }

    weight_gradients[weight_offset] = weight_gradient_sum;
}

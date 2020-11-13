# another attempt at building basic neural net to understand what they do

import random
import numpy as np
import matplotlib.pyplot as plt


# try doing it just with basic data structures first, not numpy arrays


def get_neuron_output_raw(inputs, weights, bias):
    value = bias
    assert len(inputs) == len(weights)
    for inp, w in zip(inputs, weights):
        value += inp * w
    return value


def get_weight_adjustment(learning_rate, gradient_of_loss_wrt_weight):
    return -1 * learning_rate * gradient_of_loss_wrt_weight



if __name__ == "__main__":
    sigmoid = lambda x: 1/(1+np.exp(-x))  # = sigma(x)
    derivative_of_sigmoid = lambda x: np.exp(-x) / (1+np.exp(-x))**2  # = sigma(x) * (1 - sigma(x))
    relu = lambda x: 0 if x <= 0 else x  # rectified linear
    derivative_of_relu = lambda x: 0 if x <= 0 else 1  # f'(0) undefined, just set it to 0

    activation_function = relu
    derivative_of_activation_function = derivative_of_relu

    squared_error = lambda y_predicted, y: (y_predicted - y)**2
    gradient_of_squared_error = lambda y_predicted, y: 2*(y_predicted - y)  # differentiated with respect to y_predicted, which is what we want to change; y_observed is given
    loss_function = squared_error
    gradient_of_loss_function = gradient_of_squared_error
    # loss_multiple_data = lambda ys_p, ys: sum(loss_function(y_p, y) for y_p, y in zip(ys_p, ys)) / len(ys)
    # gradient_loss_multiple_data = lambda ys_p, ys: sum(gradient_of_loss_function(y_p, y) for y_p, y in zip(ys_p, ys)) / len(ys)
    learning_rate = 0.00001  # usually between 1e-6 and 0.1


    # try fitting a function of three input variables to one output variable
    # secret_data_function = lambda x0,x1,x2: np.sin(x0)**2 + np.cos(np.sin(x1)**2)**2 + -4*np.sin(np.cos(x0*x2)**2)
    secret_data_function = lambda x0,x1,x2: x0*x1 + x1*x2**2 + np.sqrt(np.abs(x0+x1+x2))
    n_data_points = 1000
    X_sample = [np.random.uniform(-10, 10, (3,)) for i in range(n_data_points)]
    y_sample = [secret_data_function(x0,x1,x2) for x0,x1,x2 in X_sample]

    n_input_neurons = 3
    n_receiving_neurons = 1
    inputs = X_sample
    weights = [[random.random() for i in range(n_input_neurons)] for j in range(n_receiving_neurons)]
    biases = [random.random() for i in range(n_receiving_neurons)]
    loss_record = []  # loss per round

    n_rounds = 20

    for round_i in range(n_rounds):
        print("beginning round {}".format(round_i))
        outputs = []  # for each output neuron, a list of outputs (predicted ys) for each sample X
        raw_outputs = []  # without activation function, for later computation of f'(wx+b)
        losses = []
        gradient_losses = []

        for receiving_neuron_i in range(n_receiving_neurons):
            print("receiving neuron {}".format(receiving_neuron_i))
            weight_vector = weights[receiving_neuron_i]
            bias = biases[receiving_neuron_i]
            print("w={}, b={}".format(weight_vector, bias))
            outputs_this_neuron = []  # one for each sample input
            raw_outputs_this_neuron = []
            losses_this_neuron = []
            gradient_losses_this_neuron = []

            for sample_i in range(n_data_points):
                input_vector_this_sample = inputs[sample_i]  # one value for each neuron in the input layer
                inputs_to_this_neuron = input_vector_this_sample  # final values from last layer

                raw_output_this_sample = get_neuron_output_raw(inputs_to_this_neuron, weight_vector, bias)
                raw_outputs_this_neuron.append(raw_output_this_sample)

                output_this_sample = activation_function(raw_output_this_sample)
                outputs_this_neuron.append(output_this_sample)

                correct_output = y_sample[sample_i]

                loss_this_sample = loss_function(output_this_sample, correct_output)
                losses_this_neuron.append(loss_this_sample)

                gradient_loss_this_sample = gradient_of_loss_function(output_this_sample, correct_output)
                gradient_losses_this_neuron.append(gradient_loss_this_sample)

                print("input {}, raw out {}, activated {}, correct {}, loss {}, gradloss {}".format(input_vector_this_sample, raw_output_this_sample, output_this_sample, correct_output, loss_this_sample, gradient_loss_this_sample))

            outputs.append(outputs_this_neuron)
            raw_outputs.append(raw_outputs_this_neuron)

            mean_loss_over_samples_this_neuron = sum(losses_this_neuron) / len(losses_this_neuron)
            losses.append(mean_loss_over_samples_this_neuron)

            mean_gradient_loss_over_samples_this_neuron = sum(gradient_losses_this_neuron) / len(gradient_losses_this_neuron)
            gradient_losses.append(mean_gradient_loss_over_samples_this_neuron)

        loss_this_round = sum(losses)  # don't mean over neurons, but do mean over samples (within each neuron)
        loss_record.append(loss_this_round)
        print("round {} has loss {}".format(round_i, loss_this_round))

        # adjust the weights
        # I'm just going to do it for each sample point one at a time, for now
        # accumulate the adjustments here and then add them all at the end
        weight_adjustments = [[0 for w in weight_vector] for weight_vector in weights]
        bias_adjustments = [0 for bias in biases]

        for sample_i in range(n_data_points):
            X_vector = X_sample[sample_i]
            y_vector = y_sample[sample_i]

            for receiving_neuron_i in range(n_receiving_neurons):
                raw_output = raw_outputs[receiving_neuron_i][sample_i]
                y_pred = outputs[receiving_neuron_i][sample_i]
                y = y_sample[sample_i]
                weight_vector = weights[receiving_neuron_i]
                bias = biases[receiving_neuron_i]

                for weight_i in range(len(weight_vector)):
                    weight = weight_vector[weight_i]
                    input_from_previous_neuron = X_vector[weight_i]

                    # adjust the weight by (learning rate times negative of gradient of loss wrt this weight)
                    # by chain rule, dL/dw = dL/dy_predicted * dy_predicted/dw
                    d_loss_d_ypred = gradient_of_loss_function(y_pred, y)
                    # ypred = activation(w*x+b)
                    w = weight
                    x = input_from_previous_neuron
                    b = bias
                    d_ypred_d_w = derivative_of_activation_function(raw_output) * x   # chain rule on activation(w*x+b)
                    d_loss_d_w = d_loss_d_ypred * d_ypred_d_w
                    gradient_of_loss_wrt_weight = d_loss_d_w

                    weight_adjustment = get_weight_adjustment(learning_rate, gradient_of_loss_wrt_weight)
                    weight_adjustments[receiving_neuron_i][weight_i] += weight_adjustment

                # bias adjustment
                # same logic as weight adjustment, follow negative gradient
                d_loss_d_ypred = gradient_of_loss_function(y_pred, y)
                d_ypred_d_b = derivative_of_activation_function(raw_output) * 1  # chain rule on activation(w*x+b)
                d_loss_d_b = d_loss_d_ypred * d_ypred_d_b
                gradient_of_loss_wrt_bias = d_loss_d_b

                bias_adjustment = get_weight_adjustment(learning_rate, gradient_of_loss_wrt_bias)
                bias_adjustments.append(bias_adjustment)

        # now actually change the weights and biases
        for weight_vector_i in range(len(weights)):
            weight_vector = weights[weight_vector_i]
            for weight_i in range(len(weight_vector)):
                weight_vector[weight_i] += weight_adjustments[weight_vector_i][weight_i]
        for bias_i in range(len(biases)):
            biases[bias_i] += bias_adjustments[bias_i]

        # go to next round
        print("done with round {}".format(round_i))


    print("done training network")
    plt.plot(loss_record)
    plt.gca().set_yscale('log')
    plt.show()

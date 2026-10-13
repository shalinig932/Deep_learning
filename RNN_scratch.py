# Implement Vanilla RNN from scratch
import string
import numpy as np


# Keep the same input and expected sequences as the original code.
inputs = np.array([
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
    ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
    ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
    ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
])

expected = np.array([
    ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"],
    ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
    ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
])


def string_to_one_hot(sequences):
    """Convert each character to a (26, 1) one-hot column vector."""
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}
    encoded_sequences = []

    for sequence in sequences:
        encoded_sequence = []
        for char in sequence:
            vector = np.zeros((len(string.ascii_uppercase), 1))
            vector[char_to_index[str(char).upper()], 0] = 1.0
            encoded_sequence.append(vector)
        encoded_sequences.append(encoded_sequence)

    return np.array(encoded_sequences, dtype=np.float64)


def softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


class InputLayer:
    def __init__(self, inputs, hidden_size):
        self.inputs = inputs
        self.hidden_size = hidden_size
        self.input_size = inputs.shape[1]
        rng = np.random.default_rng()
        self.U = rng.uniform(-0.1, 0.1, (hidden_size, self.input_size))
        self.delta_U = np.zeros_like(self.U)

    def get_input(self, time_step):
        return self.inputs[time_step]

    def weighted_sum(self, time_step):
        return self.U @ self.get_input(time_step)

    def calculate_deltas_per_step(self, time_step, delta_weighted_sum):
        self.delta_U += delta_weighted_sum @ self.get_input(time_step).T

    def update_weights_and_bias(self, learning_rate):
        self.U -= learning_rate * self.delta_U


class HiddenLayer:
    def __init__(self, sequence_length, hidden_size):
        rng = np.random.default_rng()
        self.W = rng.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.bias = np.zeros((hidden_size, 1))
        self.states = np.zeros((sequence_length, hidden_size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_W = np.zeros_like(self.W)
        self.next_delta_activation = np.zeros((hidden_size, 1))

    def get_hidden_state(self, time_step):
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step, hidden_state):
        self.states[time_step] = hidden_state

    def activate(self, weighted_input, time_step):
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        weighted_sum = weighted_input + self.W @ previous_hidden_state + self.bias
        activation = np.tanh(weighted_sum)
        self.set_hidden_state(time_step, activation)
        return activation

    def calculate_deltas_per_step(self, time_step, delta_output):
        current_hidden = self.get_hidden_state(time_step)
        delta_activation = delta_output + self.next_delta_activation
        delta_weighted_sum = delta_activation * (1.0 - current_hidden ** 2)

        previous_hidden = self.get_hidden_state(time_step - 1)
        self.delta_W += delta_weighted_sum @ previous_hidden.T
        self.delta_bias += delta_weighted_sum
        self.next_delta_activation = self.W.T @ delta_weighted_sum
        return delta_weighted_sum

    def update_weights_and_bias(self, learning_rate):
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias


class OutputLayer:
    def __init__(self, vocab_size, hidden_size, sequence_length):
        rng = np.random.default_rng()
        self.V = rng.uniform(-0.1, 0.1, (vocab_size, hidden_size))
        self.bias = np.zeros((vocab_size, 1))
        self.states = np.zeros((sequence_length, vocab_size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_V = np.zeros_like(self.V)

    def predict(self, hidden_state, time_step):
        logits = self.V @ hidden_state + self.bias
        prediction = softmax(logits)
        self.states[time_step] = prediction
        return prediction

    def get_state(self, time_step):
        return self.states[time_step]

    def calculate_deltas_per_step(self, expected_vector, hidden_state, time_step):
        delta_output = self.get_state(time_step) - expected_vector
        self.delta_V += delta_output @ hidden_state.T
        self.delta_bias += delta_output
        return self.V.T @ delta_output

    def update_weights_and_bias(self, learning_rate):
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias


class VanillaRNN:
    def __init__(self, vocab_size, hidden_size, alpha):
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None

    def feed_forward(self, inputs):
        sequence_length = len(inputs)
        self.input_layer = InputLayer(inputs, self.hidden_size)
        self.hidden_layer = HiddenLayer(sequence_length, self.hidden_size)
        self.output_layer = OutputLayer(len(string.ascii_uppercase),
                                        self.hidden_size, sequence_length)

        for step in range(sequence_length):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input, step)
            self.output_layer.predict(activation, step)

        return self.output_layer

    def backpropagation(self, expected):
        # Reset accumulated gradients before backpropagating this sequence.
        self.input_layer.delta_U.fill(0)
        self.hidden_layer.delta_W.fill(0)
        self.hidden_layer.delta_bias.fill(0)
        self.hidden_layer.next_delta_activation.fill(0)
        self.output_layer.delta_V.fill(0)
        self.output_layer.delta_bias.fill(0)

        for step in reversed(range(len(expected))):
            delta_output = self.output_layer.calculate_deltas_per_step(
                expected[step],
                self.hidden_layer.get_hidden_state(step),
                step,
            )
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(
                step, delta_output
            )
            self.input_layer.calculate_deltas_per_step(step, delta_weighted_sum)

        # Average gradients over the sequence length for a stable update.
        sequence_length = len(expected)
        for layer in (self.input_layer, self.hidden_layer, self.output_layer):
            for name in ("delta_U", "delta_W", "delta_bias", "delta_V"):
                if hasattr(layer, name):
                    getattr(layer, name)[:] /= sequence_length

        self.output_layer.update_weights_and_bias(self.alpha)
        self.hidden_layer.update_weights_and_bias(self.alpha)
        self.input_layer.update_weights_and_bias(self.alpha)

    def loss(self, predictions, expected):
        epsilon = 1e-12
        total_loss = 0.0
        for step in range(len(expected)):
            total_loss -= np.sum(
                expected[step] * np.log(np.clip(predictions[step], epsilon, 1.0))
            )
        return float(total_loss / len(expected))

    def train(self, inputs, expected, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            for idx, sequence in enumerate(inputs):
                output_layer = self.feed_forward(sequence)
                sequence_loss = self.loss(output_layer.states, expected[idx])
                self.backpropagation(expected[idx])
                total_loss += sequence_loss

            print(f"epoch={epoch + 1}")
            print(f"Loss round: {total_loss / len(inputs):.6f}")

    def predict_sequence(self, sequence):
        one_hot_sequence = string_to_one_hot(np.array([sequence]))[0]
        output_layer = self.feed_forward(one_hot_sequence)
        predicted_indices = np.argmax(output_layer.states, axis=1).flatten()
        return "".join(string.ascii_uppercase[index] for index in predicted_indices)


if __name__ == "__main__":
    one_hot_inputs = string_to_one_hot(inputs)
    one_hot_expected = string_to_one_hot(expected)

    rnn = VanillaRNN(
        vocab_size=len(string.ascii_uppercase),
        hidden_size=128,
        alpha=0.0001,
    )
    rnn.train(one_hot_inputs, one_hot_expected, epochs=10)

    new_inputs = np.array([["B", "C", "D"]])
    for sequence in string_to_one_hot(new_inputs):
        predictions = rnn.feed_forward(sequence)
        output = np.argmax(predictions.states[-1])
        print(output)
        print(string.ascii_uppercase[output])

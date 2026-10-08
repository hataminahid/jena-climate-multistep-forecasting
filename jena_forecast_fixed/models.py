

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_naive_baseline(label_width: int, num_label_features: int):

    class NaiveLastValue(tf.keras.Model):
        def __init__(self, label_width, num_label_features):
            super().__init__()
            self.label_width = label_width
            self.num_label_features = num_label_features

        def call(self, inputs):
            last = inputs[:, -1:, : self.num_label_features]
            return tf.tile(last, [1, self.label_width, 1])

    return NaiveLastValue(label_width, num_label_features)


def build_lstm(input_width, label_width, num_features, num_label_features, units=32):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_width, num_features)),
        layers.LSTM(units, return_sequences=False),
        layers.Dense(label_width * num_label_features),
        layers.Reshape([label_width, num_label_features]),
    ])
    return model


def build_gru(input_width, label_width, num_features, num_label_features, units=32):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_width, num_features)),
        layers.GRU(units, return_sequences=False),
        layers.Dense(label_width * num_label_features),
        layers.Reshape([label_width, num_label_features]),
    ])
    return model


class Seq2Seq(Model):

    def __init__(self, num_label_features, units=32, label_width=24):
        super().__init__()
        self.num_label_features = num_label_features
        self.label_width = label_width
        self.units = units

        self.encoder = layers.GRU(units, return_state=True)
        self.decoder_cell = layers.GRUCell(units)
        self.dense = layers.Dense(num_label_features)

    def call(self, inputs, training=False):
        # inputs: (batch, input_width, num_features)
        _, state = self.encoder(inputs)
        last_obs = inputs[:, -1, : self.num_label_features]

        predictions = []
        dec_input = last_obs
        for _ in range(self.label_width):
            dec_output, state = self.decoder_cell(dec_input, states=state)
            pred = self.dense(dec_output)
            predictions.append(pred)
            dec_input = pred  # autoregressive

        return tf.stack(predictions, axis=1)  # (batch, label_width, num_label_features)


def compile_and_fit(model, window, patience=3, max_epochs=20, learning_rate=1e-3,
                     loss=None, sample_weight_train=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", restore_best_weights=True
    )
    model.compile(
        loss=loss or tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history

from tensorflow.keras import layers, models
import tensorflow as tf

def make_double_model(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inp)

    x = layers.Dense(384, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(192, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
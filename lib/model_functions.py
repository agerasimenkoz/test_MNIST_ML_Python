import os
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop
from pandas import DataFrame
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from matplotlib.pyplot import plot as plt
from lib.model import MnistModel
import tensorflow as tf


class ModelFunctions:
    def __init__(self, model: Model, checkpoint_dir="tmp/checkpoint/"):
        self.history = None
        self.model = model
        self._checkpoint_dir = checkpoint_dir

        self.fit_setting = {"batch_size": 64,
                            "epochs": 20,
                            # "validation_split": 0.15,
                            "shuffle": True,
                            # "use_multiprocessing": True,
                            # "workers": 2,
                            "verbose": 1,
                            }
        self._optimizer = Adam()
        # self._optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # self._loss = SparseCategoricalCrossentropy(
        #     from_logits=True)
        self._loss = "categorical_crossentropy"
        # self._metrics = [SparseCategoricalAccuracy()]
        self._metrics = ["accuracy"]
        # Create a callback
        self._callback = [ModelCheckpoint(
            filepath=os.path.join(self._checkpoint_dir, "cp-{epoch:04d}-{val_accuracy:.3f}.ckpt"),
            verbose=1,
            save_weights_only=True,
            save_freq="epoch",
        ),
            ReduceLROnPlateau(monitor='val_accuracy',
                              patience=3,
                              verbose=1,
                              factor=0.5,
                              min_lr=0.00001),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
            #                   verbose=1, mode='auto'),
            CSVLogger(f"{self._checkpoint_dir}history.csv", separator=",", append=True)
        ]

        self.compile()

    def train_model_all_layers(self, valid_generator, train_generator):
        step_size_train = train_generator.n // train_generator.batch_size
        step_size_valid = valid_generator.n // valid_generator.batch_size
        print(f"{step_size_train=} {step_size_valid=}")
        self.history = self.model.fit(train_generator, steps_per_epoch=step_size_train,
                                      validation_data=valid_generator, validation_steps=step_size_valid,
                                      callbacks=self._callback,
                                      **self.fit_setting
                                      )
        return self.history

    def compile(self):
        # compile the model
        self.model.compile(optimizer=self._optimizer,
                           loss=self._loss,
                           metrics=self._metrics)

    def inference(self):
        pass

    def evaluate(self, test_generator):
        self.model.evaluate(test_generator, verbose=2)

    def _save_history(self):
        pass

    def read_history(self):
        pass

    def save_model(self, model_path):
        self.model.save(model_path)

    def restore_model(self, model_path):
        # checkpoint_path = tf.train.latest_checkpoint(self._checkpoint_path)
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
        return self.model
        # self.model.load_weights(self._checkpoint_path)

    def load_latest_checkpoint(self):
        checkpoint_path = tf.train.latest_checkpoint(self._checkpoint_dir)
        self.model.load_weights(checkpoint_path)
        self.model.summary()

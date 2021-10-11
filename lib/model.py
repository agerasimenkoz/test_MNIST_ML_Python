import tensorflow as tf
from keras import Sequential


class MnistModel(tf.keras.Model):
    """
    A class to apply and run style transfer for a Content Image from a given Style Image
    Attributes
    ----------

    Methods
    -------
        model() :
            Loads an image as a numpy array and normalizes it from the given image path
    """
    def __init__(self):
        super().__init__()
        # first layer
        self.layer1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                                             input_shape=(28, 28, 1))
        self.layer2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')
        self.layer3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer4 = tf.keras.layers.Dropout(0.20)

        # second layer
        self.layer5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer7 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.layer8 = tf.keras.layers.Dropout(0.25)

        # third layer
        self.layer9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.layer10 = tf.keras.layers.Dropout(0.25)

        # output layer
        self.layer11 = tf.keras.layers.Flatten()
        self.layer12 = tf.keras.layers.Dense(128, activation='relu')
        self.layer13 = tf.keras.layers.Dropout(0.3)
        self.layer14 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def model(self):
        model = Sequential([
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.layer5, self.layer6, self.layer7, self.layer8,
            self.layer9, self.layer10, self.layer11, self.layer12,
            self.layer13, self.layer14
        ])
        model.summary()
        return model


if __name__ == '__main__':
    a = MnistModel().model()





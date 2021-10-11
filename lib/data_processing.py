import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataProcessing:
    """
    Class for preprocessing a dataset of images
    Attributes
    ----------
        dataset_dir : Path to data
        BATCH_SIZE: pandas dataset
        image_size: data into features (pixel values)
        data_generator : data labels
    Methods
    -------
        get_test_generator() :
            Split out the data into features (pixel values) and categorical labels (digit values 0-9)
        get_train_generators() :
            Split out the data into features (pixel values) and categorical labels (digit values 0-9)
    """

    def __init__(self, path_to_data: str):
        """
        Constructs data processing

        Parameters
        ----------
            path_to_data: Path to data directory
        """
        self.dataset_dir = path_to_data
        # self.dataset = self._open_dataset(self.dataset_path)
        # self.dataset_x = None
        # self.dataset_y = None
        self.BATCH_SIZE = 86
        self.image_size = (28, 28)
        self.data_generator = ImageDataGenerator(rescale=1. / 255.)

        self._dataflow_kwargs = dict(
            directory=self.dataset_dir,
            class_mode='categorical',
            target_size=self.image_size,
            batch_size=self.BATCH_SIZE,
            color_mode="grayscale",
            seed=42)

    @staticmethod
    def _open_dataset(dataset_path):
        return pd.read_csv(dataset_path)

    def get_test_generator(self):
        """
        Preprocess for test data

        Returns
        -------
            test_generator : test data generator
        """
        self.data_generator = ImageDataGenerator(rescale=1. / 255.)
        return self._get_generator_from_dir(shuffle=False)

    def get_train_generators(self):
        """
        Split dataset

        Returns
        -------
            train_generator : train data generator
            valid_generator : valid data generator
        """

        datagen_kwargs = dict(validation_split=0.2,
                              featurewise_center=False,  # set input mean to 0 over the dataset
                              samplewise_center=False,  # set each sample mean to 0
                              featurewise_std_normalization=False,  # divide inputs by std of the dataset
                              samplewise_std_normalization=False,  # divide each input by its std
                              zca_whitening=False,  # apply ZCA whitening
                              rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                              zoom_range=0.1,  # Randomly zoom image
                              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                              horizontal_flip=False,  # randomly flip images
                              vertical_flip=False)  # randomly flip images

        # train_datagen = ImageDataGenerator(**datagen_kwargs)

        # train_generator = train_datagen.flow_from_dataframe(**dataflow_kwargs, shuffle=True, subset="training")
        # valid_generator = train_datagen.flow_from_dataframe(**dataflow_kwargs, subset="validation")
        self.data_generator = ImageDataGenerator(**datagen_kwargs)
        # train_generator = train_datagen.flow_from_directory(**self.dataflow_kwargs, shuffle=True, subset="training")
        train_generator = self._get_generator_from_dir(shuffle=True, subset="training")
        # valid_generator = train_datagen.flow_from_directory(**self.dataflow_kwargs, subset="validation")
        valid_generator = self._get_generator_from_dir(shuffle=False, subset="validation")
        return train_generator, valid_generator

    def _get_generator_from_dir(self, **kwargs):
        return self.data_generator.flow_from_directory(**self._dataflow_kwargs, **kwargs)


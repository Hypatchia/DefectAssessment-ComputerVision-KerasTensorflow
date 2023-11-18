
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create tf generators
def create_data_generators(data_dir, test_dir, image_size, batch_size=32, seed=42):
    """
    Creates and configures data generators for training, validation, and testing using Keras' ImageDataGenerator.

    Parameters:
    - data_dir (str): Path to the directory containing training and validation data.
    - test_dir (str): Path to the directory containing test data.
    - image_size (tuple): Target size for input images in the format (height, width).
    - batch_size (int, optional): Number of samples per batch. Defaults to 32.
    - seed (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
    - train_generator (DirectoryIterator): Generator for training data.
    - validation_generator (DirectoryIterator): Generator for validation data.
    - test_generator (DirectoryIterator): Generator for test data.
    """
    # Train and validation datagen
    datagen = ImageDataGenerator(
        validation_split=0.3,
        rescale=1.0 / 255.0
    )

    # Test datagen
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # Train generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=seed
    )

    # Validation generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        seed=seed
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='binary',
        shuffle=False,
        seed=seed
    )

    return train_generator, validation_generator, test_generator



def create_dataset_from_generator(generator, image_size):
    """
    Create a TensorFlow dataset from a generator function.

    Parameters:
    - generator (generator): A generator function that yields tuples of images and labels.
    - image_size (tuple): The size of the input images in the format (height, width).

    Returns:
    - tf.data.Dataset: A TensorFlow dataset containing image-label pairs.
    """
    # Define the output signature for the dataset based on image size and data types
    # Create a TensorFlow dataset from the generator with the specified output signature

    return tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
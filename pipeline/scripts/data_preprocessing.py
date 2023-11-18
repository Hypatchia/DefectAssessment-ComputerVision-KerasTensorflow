
import tensorflow as tf



def is_grayscale(image):
    """
    Check if an image is grayscale.

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array.

    Returns:
    - bool: True if the image is grayscale, False otherwise.
    """
    return image.shape[-1] == 1


def assert_grayscale(dataset, dataset_name):
    """
    Assert that all images in a dataset are grayscale.

    Parameters:
    - dataset (tf.data.Dataset): The input dataset containing image-label pairs.
    - dataset_name (str): The name of the dataset used for error messages.

    Returns:
    - bool: True if all images are grayscale, False otherwise.
    """
    for image, label in dataset:
        # Check if each image in the dataset is grayscale
        if not is_grayscale(image):
            print(f"Error: {dataset_name} contains non-grayscale images.")
            return False

    # If no non-grayscale images are found, print a success message
    print(f"All images in {dataset_name} are grayscale.")
    return True


# Create function to convert rgb images to grayscale using tf
def convert_to_grayscale(image, label=None):
    grayscale_image = tf.image.rgb_to_grayscale(image)

    return grayscale_image, label

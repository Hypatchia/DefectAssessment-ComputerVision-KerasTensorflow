import numpy as np




def get_labels_from_generator(generator):
    """
    Get labels from a data generator.

    Parameters:
    - generator (generator): A data generator yielding batches of data and labels.

    Returns:
    - list: A list containing all the labels from the generator.

    Note:
    This function iterates through the given generator to extract labels from each batch and
    appends them to a list, which is then returned.
    """
    # Initialize an empty list to store labels
    labels = []

    # Iterate through the generator
    for _ in range(len(generator)):
        # Get the next batch from the generator
        _, batch_labels = next(generator)

        # Extend the list of labels with the batch labels
        labels.extend(batch_labels)

    # Return the accumulated labels
    return labels




#
def make_predictions(model, test_dataset, num_test_samples):
    """
    Make predictions using the given model on the provided test dataset.

    Parameters:
    - model (object): The trained machine learning model.
    - test_dataset (object): The dataset on which predictions will be made.
    - num_test_samples (int): The number of samples in the test dataset.

    Returns:
    - numpy.ndarray: An array of rounded predictions (0 or 1).
    """

    # Predict on test data
    pred = model.predict(test_dataset, steps=num_test_samples)

    # Round predictions to 0 or 1
    predicted_labels = np.round(pred).astype(int)

    return predicted_labels




import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
# Build CNN model

def build_cnn_model(img_shape, num_color_layers):
    """
    Build a Convolutional Neural Network (CNN) model using TensorFlow's Sequential API.

    Parameters:
    - img_shape (tuple): The shape of the input images in the format (height, width, channels).
    - num_color_layers (int): The dilation rate for Conv2D layers.

    Returns:
    - tf.keras.Sequential: The constructed CNN model.
    """
    # Convolutional Neural Network Sequential Model
    model = tf.keras.Sequential([
        # Input layer with the specified input image shape
        tf.keras.layers.Input(shape=img_shape),

        # Conv2D layer with 16 filters, a 3 x 3 kernel size, and a relu activation
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), dilation_rate=num_color_layers, padding='same', activation='relu'),
        # Downsampling to reduce the spatial dimensions of the input while retaining important features
        tf.keras.layers.MaxPooling2D(),

        # Conv2D layer with 32 filters, a 3 x 3 kernel size, and a relu activation
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=num_color_layers, padding='same', activation='relu'),
        # Default downsampling
        tf.keras.layers.MaxPooling2D(),

        # Conv2D layer with 64 filters, a 3 x 3 kernel size, and a relu activation
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=num_color_layers, padding='same', activation='relu'),
        # Default downsampling
        tf.keras.layers.MaxPooling2D(),

        # Flatten feature map into a 1D vector
        tf.keras.layers.Flatten(),

        # Fully connected network layer with 128 neurons and a relu activation
        tf.keras.layers.Dense(128, activation='relu'),

        # Output layer of the fully connected network with 1 neuron representing a binary output (sigmoid function)
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return model

def compile_train_model(model, train_data,steps_per_epoch, validation_data,validation_steps, epochs,batch_size,learning_rate):
    """
    Compile and train the CNN model.

    Parameters:
    - model (tf.keras.Sequential): The CNN model to compile and train.
    - train_dataset (tf.data.Dataset): The training dataset.
    - validation_dataset (tf.data.Dataset): The validation dataset.
    - epochs (int): The number of epochs to train the model for.

    Returns:
    - tf.keras.Sequential: The trained CNN model.
    """
    # Compile model using adam ,binary cross entropy and accuracy as a performance measure.
   

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', # Binary cross entropy for binary classification
        metrics=[
            'accuracy', #
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc_roc'),
            AUC(name='auc_pr', curve='PR')
        ]
    )

    # Train Model

    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        epochs = epochs)


    return model, history
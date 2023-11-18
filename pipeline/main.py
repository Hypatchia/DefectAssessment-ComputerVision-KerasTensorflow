import mlflow
from scripts.data_loading import load_from_local
from scripts.data_preprocessing import is_grayscale, assert_grayscale, convert_to_grayscale
from scripts.train_val_test_creation import create_data_generators, create_dataset_from_generator
from scripts.model_scoring import make_confusion_matrix, make_classification_report, make_roc_auc_curve
from scripts.evaluate import plot_training_metrics
from scripts.train import build_cnn_model, compile_train_model
from scripts.test import get_labels_from_generator, make_predictions

def main():
    

    # Set the paths to the dataset file and the extraction path
    print("Loading data...")
    dataset_file_path = '../Data/Impellers.zip'
    extraction_path = '../Data/'

    # Load data
    data = load_from_local(dataset_file_path, extraction_path)
    print("Data loaded successfully.")
    # Define the path to the training and validation directory
    data_dir ='../Data/dataset'
    # Define the path to test directory

    test_dir = "../Data/test_dataset/"
    # Set datagenerator parameters
    # Set target size for images
    image_size = (300, 300)

    # Set batch_size for tf dataset
    batch_size = 32
    # Set the seed for random operations.
    seed = 42
    print("Creating data generators...")
    # Create data generators for training, validation, and testing
    train_generator, validation_generator, test_generator = create_data_generators(
        data_dir,
        test_dir,
        image_size,
        batch_size,
        seed)
    print("Data generators created successfully.")
    # Create tf datasets from generators
    train_dataset = create_dataset_from_generator(train_generator, image_size)
    validation_dataset = create_dataset_from_generator(validation_generator, image_size)
    test_dataset = create_dataset_from_generator(test_generator, image_size)
    print("Datasets created successfully.")
    # Get the class names and the corresponding indices.
    class_indices = train_generator.class_indices
    class_names = list(train_generator.class_indices.keys())
    print("Class names: ", class_names)

    # Check if images in datasets are grayscale (height,width,1)
    assert_grayscale(train_dataset, "train_dataset")
    assert_grayscale(validation_dataset, "validation_dataset")

    # Check if images in datasets are grayscale (height,width,1)
    # Convert to grayscale
    grayscale_train_data = train_dataset.map(convert_to_grayscale)
    grayscale_validation_data = validation_dataset.map(convert_to_grayscale)
    grayscale_test_data = test_dataset.map(convert_to_grayscale)
    print("Datasets converted to grayscale successfully.")
    # Set input image properties for CNN Model
    num_color_layers =1
    img_shape = image_size +(num_color_layers,)

    
    # Set training parameters
    n_epochs=10
    batch_size=32
    learning_rate = 0.0001
    # Get size of train, validation and test samples
    num_train_samples = train_generator.n
    num_val_samples = validation_generator.n
    num_test_samples = test_generator.n

    # Set epochs properties
    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size
    test_steps=num_test_samples//batch_size
    print("Training parameters set successfully.")
    print("Training model...")

    # Set the tracking URI to localhost
    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("your_experiment_name")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_color_layers", num_color_layers)
        mlflow.log_param("img_shape", img_shape)

        # Log dataset information
        mlflow.log_param("num_train_samples", num_train_samples)
        mlflow.log_param("num_val_samples", num_val_samples)
        mlflow.log_param("num_test_samples", num_test_samples)

        # Build CNN model
        model = build_cnn_model(img_shape, num_color_layers)
         # Compile and train model using MLflow
        mlflow.tensorflow.autolog()
        # Compile and train model using adam ,binary cross entropy and accuracy as a performance measure.
        model, history= compile_train_model(model, 
                                            grayscale_train_data,
                                            steps_per_epoch,
                                            grayscale_validation_data,
                                            validation_steps,
                                            epochs=n_epochs,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate)
        
        # Plot training & validation metrics

        plot_training_metrics(history)

        # Evaluate model on validation data
        evaluation_res =model.evaluate(grayscale_validation_data,steps=validation_steps)

        # Print evaluation results
        print("Evaluation results: ", evaluation_res)


        # Get labels of test data
        test_labels = get_labels_from_generator(test_generator)
        # Make predictions on test data
        test_predictions = make_predictions(model,grayscale_test_data,num_test_samples)
        # Print classification report
        make_classification_report(test_labels, test_predictions, class_names)
        # Print confusion matrix
        make_confusion_matrix(test_labels, test_predictions)
        # Plot ROC curve
        make_roc_auc_curve(test_labels, test_predictions)



if __name__ == "__main__":
    main()

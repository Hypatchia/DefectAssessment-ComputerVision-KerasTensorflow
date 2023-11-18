
import matplotlib.pyplot as plt

def plot_training_metrics(history):
    """
    Plot training metrics and loss from a model's training history.

    Parameters:
    - history (History): The training history of a model containing metrics and loss information.

    Returns:
    None
    """
    # Extract metrics from history
    acc = history.history['accuracy']
    precision = history.history['precision']
    recall = history.history['recall']
    auc_roc = history.history['auc_roc']
    auc_pr = history.history['auc_pr']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))
       # Set a larger figure size
    plt.figure(figsize=(10, 10))


    # Plot accuracy metrics
    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, precision, label='Training Precision')
    plt.plot(epochs_range, recall, label='Training Recall')
    plt.plot(epochs_range, auc_roc, label='Training AUC ROC')
    plt.plot(epochs_range, auc_pr, label='Training AUC PR')
    plt.legend(loc='lower right')
    plt.title('Training Metrics')

    # Plot loss metrics
    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()


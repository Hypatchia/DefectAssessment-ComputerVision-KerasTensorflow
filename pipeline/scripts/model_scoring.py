from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def make_confusion_matrix(true_labels, predicted_labels):
    """
    Calculate and print the confusion matrix based on true and predicted labels.

    Parameters:
    - true_labels (array-like): The true labels of the data.
    - predicted_labels (array-like): The predicted labels of the data.

    Returns:
    - numpy.ndarray: The confusion matrix.
    """

    # Calculate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix


    return confusion

def make_classification_report(true_labels, predicted_labels, class_names):
    """
    Generate and print a classification report based on true and predicted labels.

    Parameters:
    - true_labels (array-like): The true labels of the data.
    - predicted_labels (array-like): The predicted labels of the data.
    - class_names (list): List of class names for better readability in the report.

    Returns:
    - str: The classification report.
    """

    # Create a classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_names)

    print(report)



def make_roc_auc_curve(true_labels, predicted_labels):
    """
    Generate and display the Receiver Operating Characteristic (ROC) curve and its area under the curve (AUC).

    Parameters:
    - true_labels (array-like): The true labels of the data.
    - predicted_labels (array-like): The predicted labels of the data.

    Returns:
    - None
    """

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

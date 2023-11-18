
#from google.colab import drive
import zipfile
import os
from google.colab import drive


# Define function to load the dataset from local storage
def load_from_local(dataset_file_path, extraction_path):
  """
  Load a dataset from local storage, extract it to the specified path.

  Parameters:
  - dataset_file_path (str): The path to the zipped dataset file on local storage.
  - extraction_path (str): The path where the dataset should be extracted.

  Returns:
  None
  """
  # Create the extraction directory if it doesn't exist
  os.makedirs(extraction_path, exist_ok=True)
  # Extract the zipped dataset
  with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
      zip_ref.extractall(extraction_path)



# Define function to load the dataset from Google Drive
def load_from_drive(dataset_file_path,extraction_path):
  """
  Load a dataset from Google Drive, extract it to the specified path.

  Parameters:
  - dataset_file_path (str): The path to the zipped dataset file on Google Drive.
  - extraction_path (str): The path where the dataset should be extracted.

  Returns:
  None
  """
  # Mount Google Drive to access the dataset file
  drive.mount('/content/drive')
  # Create the extraction directory if it doesn't exist
  os.makedirs(extraction_path, exist_ok=True)
  # Extract the zipped dataset
  with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
      zip_ref.extractall(extraction_path)
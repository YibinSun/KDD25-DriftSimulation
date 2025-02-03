import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import arff


def load_data(file_path):
    """
    Load data from CSV, ARFF, or directly return the DataFrame if it's already a Pandas DataFrame.
    """
    if isinstance(file_path, pd.DataFrame):
        return file_path
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.arff'):
        with open(file_path, 'r') as f:
            arff_data = arff.load(f)
        df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
        return df
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, ARFF, or Pandas DataFrame.")

# GPU trained model can be only used with GPUs.
def train_ctgan(input_data, epochs=300, batch_size=500, generator_lr=1e-3, discriminator_lr=1e-3, save_path=None):
    """
    Train a CTGAN model using SDV and optionally save the model using SDV's built-in save method.

    Parameters:
        input_data (str or pd.DataFrame): Path to CSV/ARFF or Pandas DataFrame.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        generator_lr (float): Learning rate for the generator.
        discriminator_lr (float): Learning rate for the discriminator.
        save_path (str): Path to save the trained model.

    Returns:
        Trained CTGAN model.
    """
    # Load data
    data = load_data(input_data)

    # Initialize the CTGAN model

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    model = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, generator_lr=generator_lr, discriminator_lr=discriminator_lr)
    # Train the model
    print("Training CTGAN model...")
    model.fit(data)
    print("Training completed.")

    # Save the model if a save path is provided
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

        model.save(save_path)  # SDV built-in method to save the model
        print(f"Model saved to {save_path}")

    return model


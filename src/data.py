# src/data.py
import numpy as np
import pickle
import urllib.request
import tarfile
import gzip
import struct
import os
from pathlib import Path
from typing import Tuple, Optional, Union


class DataLoader:
    """
    Data loading utilities for various datasets.
    """
    
    @staticmethod
    def download_cifar10(data_dir: Union[str, Path] = "./data") -> Path:
        """
        Downloads and extracts CIFAR-10 dataset.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Path to the extracted dataset directory
        """
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        
        filename = data_dir / "cifar-10-python.tar.gz"
        extract_dir = data_dir / "cifar-10-batches-py"
        
        # Download if not exists
        if not filename.exists():
            print(f"Downloading CIFAR-10 from {url}...")
            urllib.request.urlretrieve(url, filename)
            print("Download complete!")
        
        # Extract if not already extracted
        if not extract_dir.exists():
            print("Extracting CIFAR-10...")
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(data_dir)
            print("Extraction complete!")
        
        return extract_dir
    
    @staticmethod
    def load_cifar10_batch(batch_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single CIFAR-10 batch file.
        
        Args:
            batch_path: Path to the batch file
            
        Returns:
            Tuple of (images, labels)
            - images: shape (10000, 32, 32, 3), dtype uint8
            - labels: shape (10000,), dtype int
        """
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        # Convert keys from bytes to strings
        batch = {k.decode() if isinstance(k, bytes) else k: v 
                 for k, v in batch.items()}
        
        data = batch["data"]               # shape: (10000, 3072)
        labels = batch["labels"]           # list of 10000 integers
        
        # Reshape to images: (10000, 32, 32, 3)
        # CIFAR-10 stores in [R, G, B] channel order
        images = data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        
        return images, np.array(labels)
    
    @staticmethod
    def load_cifar10(
        data_dir: Union[str, Path] = "./data",
        normalize: bool = True,
        flatten: bool = False,
        one_hot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load complete CIFAR-10 dataset (train + test).
        
        Args:
            data_dir: Directory containing the dataset
            normalize: If True, normalize pixel values to [0, 1]
            flatten: If True, flatten images from (32, 32, 3) to (3072,)
            one_hot: If True, convert labels to one-hot encoding
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Download/extract if needed
        cifar_dir = DataLoader.download_cifar10(data_dir)
        
        # Load training batches
        print("Loading training data...")
        X_train_batches = []
        y_train_batches = []
        
        for i in range(1, 6):
            batch_path = cifar_dir / f"data_batch_{i}"
            imgs, labels = DataLoader.load_cifar10_batch(batch_path)
            X_train_batches.append(imgs)
            y_train_batches.append(labels)
        
        X_train = np.concatenate(X_train_batches, axis=0)
        y_train = np.concatenate(y_train_batches, axis=0)
        
        # Load test batch
        print("Loading test data...")
        test_path = cifar_dir / "test_batch"
        X_test, y_test = DataLoader.load_cifar10_batch(test_path)
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Apply transformations
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        if one_hot:
            y_train = one_hot_encode(y_train, num_classes=10)
            y_test = one_hot_encode(y_test, num_classes=10)
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def download_mnist(data_dir: Union[str, Path] = "./data") -> Path:
        """
        Downloads MNIST dataset files.
        
        Args:
            data_dir: Directory to store the dataset
            
        Returns:
            Path to the dataset directory
        """
        # Using ossci mirror as the original site is unreliable
        base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        
        data_dir = Path(data_dir)
        mnist_dir = data_dir / "mnist"
        mnist_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in files:
            filepath = mnist_dir / filename
            if not filepath.exists():
                url = base_url + filename
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
        
        return mnist_dir
    
    @staticmethod
    def _read_mnist_images(filepath: Union[str, Path]) -> np.ndarray:
        """
        Read MNIST image file in IDX format.
        
        Args:
            filepath: Path to the gzipped IDX image file
            
        Returns:
            Numpy array of images
        """
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and dimensions
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in MNIST image file")
            
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        
        return images
    
    @staticmethod
    def _read_mnist_labels(filepath: Union[str, Path]) -> np.ndarray:
        """
        Read MNIST label file in IDX format.
        
        Args:
            filepath: Path to the gzipped IDX label file
            
        Returns:
            Numpy array of labels
        """
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and number of labels
            magic, num_labels = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in MNIST label file")
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels
    
    @staticmethod
    def load_mnist(
        data_dir: Union[str, Path] = "./data",
        normalize: bool = True,
        flatten: bool = False,
        one_hot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load complete MNIST dataset (train + test).
        
        Args:
            data_dir: Directory containing the dataset
            normalize: If True, normalize pixel values to [0, 1]
            flatten: If True, flatten images from (28, 28) to (784,)
            one_hot: If True, convert labels to one-hot encoding
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            - X_train: shape (60000, 28, 28) or (60000, 784) if flattened
            - y_train: shape (60000,) or (60000, 10) if one-hot
            - X_test: shape (10000, 28, 28) or (10000, 784) if flattened
            - y_test: shape (10000,) or (10000, 10) if one-hot
        """
        # Download if needed
        mnist_dir = DataLoader.download_mnist(data_dir)
        
        # Load training data
        print("Loading MNIST training data...")
        X_train = DataLoader._read_mnist_images(
            mnist_dir / "train-images-idx3-ubyte.gz"
        )
        y_train = DataLoader._read_mnist_labels(
            mnist_dir / "train-labels-idx1-ubyte.gz"
        )
        
        # Load test data
        print("Loading MNIST test data...")
        X_test = DataLoader._read_mnist_images(
            mnist_dir / "t10k-images-idx3-ubyte.gz"
        )
        y_test = DataLoader._read_mnist_labels(
            mnist_dir / "t10k-labels-idx1-ubyte.gz"
        )
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Apply transformations
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        if one_hot:
            y_train = one_hot_encode(y_train, num_classes=10)
            y_test = one_hot_encode(y_test, num_classes=10)
        
        return X_train, y_train, X_test, y_test


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        labels: Array of integer labels, shape (N,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array, shape (N, num_classes)
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


# Optional: Add more dataset loaders
def generate_circles_data(n_samples: int = 500, 
                         radius_threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic concentric circles dataset for binary classification.
    """
    np.random.seed(42)
    X = (np.random.rand(n_samples, 2) - 0.5) * 2.5
    radius_sq = np.sum(X**2, axis=1)
    y_indices = (radius_sq > radius_threshold**2).astype(int)
    y = np.zeros((n_samples, 2))
    y[np.arange(n_samples), y_indices] = 1
    return X, y
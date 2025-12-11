from utils import *
import numpy as np

feature_amount = 400  # Number of features to extract
neighbors = 7 # Number of neighbors for KNN

def image_to_reduced_feature(images, split='test'):
    from sklearn.decomposition import PCA
    # Get lablels only if training for LDA
    processed_images = gaussian_noise(images)
    if split == 'train':
        pca = PCA(n_components=feature_amount)
        pca.fit(processed_images)
        pca_reduced_image = pca.transform(processed_images)
        save_model(pca, "pca_model.pkl")

    else:
        pca = load_model("pca_model.pkl")
        pca_reduced_image = pca.transform(images)

    return pca_reduced_image

def training_model(train_features, train_labels):
    # use scikit-learns knn implementation
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
    model.fit(train_features, train_labels)
    return model

DENOISING_SIGMA = 1.0 
def gaussian_noise(images):
    """
    Applies Gaussian filtering (blur) to the input images to reduce high-frequency
    Gaussian noise, then flattens them back into the feature vector format.

    Args:
        images (np.ndarray): The input image data with shape (N, 784), 
                             where N is the number of samples.

    Returns:
        np.ndarray: The denoised images with the same shape (N, 784).
    """
    from scipy.ndimage import gaussian_filter

    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("Input images must be a 2D array of shape (N, 784).")

    # Reshape from (N, 784) to (N, 28, 28) for 2D filtering
    N = images.shape[0]
    images_2d = images.reshape(N, 28, 28)

    # Apply Gaussian filter to each image
    filtered_images = np.array([
        gaussian_filter(img, sigma=DENOISING_SIGMA) for img in images_2d
    ])

    # Flatten them back to (N, 784) for PCA
    images_flat = filtered_images.reshape(N, 784)
    
    return images_flat
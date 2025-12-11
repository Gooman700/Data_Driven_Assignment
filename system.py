from utils import *
import numpy as np

feature_amount = 49  # Number of features to extract
neighbors = 9 # Number of neighbors for KNN

def image_to_reduced_feature(images, split='test'):
    # Get lablels only if training for LDA
    if split == 'train':
        _, labels = get_dataset('train')
        standardised_images = standardise_features(images, split='train')
        pca_reduced_image = pca(standardised_images, split='train')
        reduced_image = lda(pca_reduced_image, labels, split='train')
    else:
        standardised_images = standardise_features(images, split='test')
        pca_reduced_image = pca(standardised_images, split='test')
        reduced_image = lda(pca_reduced_image, None, split='test')

    return reduced_image

def training_model(train_features, train_labels):
    # use scikit-learns knn implementation
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=neighbors, metric='cosine')
    model.fit(train_features, train_labels)
    return model

def standardise_features(data, split='test'):
    """Center data and standardise to unit variance based on training data."""
    if split == 'train':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        save_model((mean, std), "standardisation_params.pkl")
    else:
        mean, std = load_model("standardisation_params.pkl")
    
    data_standardised = (data - mean) / std
    return data_standardised

def pca(data_standardised, n_components=feature_amount, split='test'):
    """Perform PCA on the data and return the reduced feature set based on training data."""

    if split == 'train':
        # Compute covariance matrix
        covariance_matrix = np.cov(data_standardised, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
        save_model(eigenvector_subset, "topEigenvectors.pkl")
    else:
        eigenvector_subset = load_model("topEigenvectors.pkl")

    # Transform the data
    reduced_data = np.dot(data_standardised, eigenvector_subset)
    return reduced_data

def lda(data_standardised, labels, n_components=None, split='test'):
    """
    Performs Linear Discriminant Analysis (LDA) to find the dimensions
    that maximise class separability.

    Args:
        data (np.ndarray): 2D array of standardised data (N_samples x D_features).
        labels (np.ndarray): 1D array of class labels for each sample.
        n_components (int): The number of components to keep (default: D_features).

    Returns:
        np.ndarray: The transformed feature set (N_samples x k_components).
    """

    if split == 'train':
        # Determine dimensions
        unique_classes = np.unique(labels)
        n_features = data_standardised.shape[1]

        # Set the number of components to keep (C - 1 is the theoretical max for separation)
        if n_components is None:
            n_components = n_features
        # LDA's useful dimensionality is typically min(dimensions, classes-1)
        k_components = min(n_components, len(unique_classes) - 1)

        # Compute Overall Mean
        overall_mean_vector = np.mean(data_standardised, axis=0)

        # Compute the Within-Class Scatter Matrix (S_W)
        # S_W measures the scatter (spread) within each class. It should be minimised.
        # S_W = sum(Cov_c * (N_c - 1)) across all classes c
        scatter_within_class_matrix = np.zeros((n_features, n_features))
        
        for class_c in unique_classes:
            # Extract data points for the current class
            class_data = data_standardised[labels == class_c]
            n_samples_c = class_data.shape[0]

            # Calculate the class covariance matrix and add to S_W
            # np.cov includes division by (N_c - 1), so multiply by (N_c - 1) to get the true scatter matrix
            class_covariance_matrix = np.cov(class_data, rowvar=False)
            scatter_within_class_matrix += class_covariance_matrix * (n_samples_c - 1)


        # --- 3. Compute the Between-Class Scatter Matrix (S_B) ---
        # S_B measures the scatter (distance) between class means. It should be maximised.
        # S_B = sum(N_c * (Mean_c - Mean_overall) * (Mean_c - Mean_overall).T) across all classes c
        scatter_between_class_matrix = np.zeros((n_features, n_features))
        
        for class_c in unique_classes:
            class_data = data_standardised[labels == class_c]
            class_mean_vector = np.mean(class_data, axis=0)
            n_samples_c = class_data.shape[0]

            # Difference vector (Mean_c - Mean_overall)
            mean_difference_vector = (class_mean_vector - overall_mean_vector).reshape(n_features, 1)

            # Outer product: (Mean_diff) * (Mean_diff).T
            outer_product = mean_difference_vector.dot(mean_difference_vector.T)
            
            scatter_between_class_matrix += n_samples_c * outer_product


        # --- 4. Solve the Generalised Eigenvalue Problem ---
        # We solve for: inv(S_W) * S_B * w = lambda * w
        # np.linalg.pinv is used for the inverse to handle the SSSP (Small Sample Size Problem)
        
        # Fisher's discriminant criteria
        discriminant_matrix = np.linalg.pinv(scatter_within_class_matrix).dot(scatter_between_class_matrix)
        
        discriminant_values, discriminant_vectors = np.linalg.eig(discriminant_matrix)

        discriminant_vectors = discriminant_vectors.real

        # --- 5. Sort and Select Components ---

        # Sort indices by the absolute value of eigenvalues (descending order)
        # The absolute value is used because numpy.linalg.eig can return complex numbers for non-symmetric matrices
        sorted_indices = np.argsort(np.abs(discriminant_values))[::-1]
        
        # Sort the eigenvectors according to the sorted indices
        sorted_discriminant_vectors = discriminant_vectors[:, sorted_indices]
        
        # Select the final subset of eigenvectors (the Fisher Discriminant Axes)
        discriminant_axis_subset = sorted_discriminant_vectors[:, 0:k_components]
        save_model(discriminant_axis_subset, "ldaDiscriminantAxes.pkl")
    else:
        discriminant_axis_subset = load_model("ldaDiscriminantAxes.pkl")

    # --- 6. Transform the Data ---
    # The transformation projects the standardised data onto the selected discriminant axes
    reduced_feature_set = np.dot(data_standardised, discriminant_axis_subset)
    return reduced_feature_set
# import numpy
import numpy as np
import warnings

# define functions to be used in kmeans algorithm
def manhatten_dist(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    computes the Manhatten distance between all points and the centroids
    """
    return np.abs(points - centroids[:, np.newaxis]).sum(axis=2)
def calc_new_centroid(points: np.ndarray, distances: np.ndarray, k: int) -> np.ndarray:
    """
    calculates the new centroids based on clusters assignment in earlier step
    """
    closest_centroids = np.argmin(distances, axis=0)
    return np.array([points[np.where(closest_centroids == i)[0]].mean(axis=0) for i in range(k)])
def k_means(init_centroids: np.ndarray, points: np.ndarray, number_of_iterations: int 
            , number_of_centroids: int, print_results: bool = True):
    #TODO should return for each point to which centroid it belongs
    #TODO add early stopping
    #TODO add plots for each substep
    '''
    number_of_iterations -- set the number of iterations for the k-means algorithm
    stopping -- defines after how many unchanged results the k-means algorithm stops (i.e. convergence)
    returns -- final centroids 
    '''
    assert init_centroids.shape[0] == number_of_centroids, warnings.warn('number_of_centroids does not match # initial centroids')
    centroids_stored = []
    centroids = init_centroids
    for i in range(number_of_iterations):
        centroids_stored.append(centroids)
        distances = manhatten_dist(points, centroids)
        centroids = calc_new_centroid(points, distances, number_of_centroids)
        if print_results == True:
            print(f'Iteration number: {i}')
            print(centroids)
            print('**************************************************')
    return centroids
import numpy as np

    
class MaxEdgeRandomSplitter(object):
    """Random max-edge splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.

    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self):
        pass
    def __call__(self, X_range):
        edge_ratio = X_range[1] - X_range[0]

        rd_dim =  np.random.choice(np.where(edge_ratio == edge_ratio.max())[0])
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min + rddim_max)/2
        return rd_dim, rd_split
    
    
    
    
import numpy as np

def global_normalize(all_coords):
    coords = np.vstack(all_coords).astype(np.float32)
    mean = coords.mean(axis=0)
    coords -= mean
    span = max(coords[:,0].ptp(), coords[:,1].ptp(), 1e-6)
    coords /= span
    coords -= coords.mean(axis=0)
    return coords.flatten()
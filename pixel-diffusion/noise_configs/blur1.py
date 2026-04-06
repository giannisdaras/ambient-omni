import numpy as np

def uniform_blur_sigma():
    return np.random.uniform(1.0, 1.0)

corruptions_dict = {
    "blur": {
        "sigma": uniform_blur_sigma
    },
}
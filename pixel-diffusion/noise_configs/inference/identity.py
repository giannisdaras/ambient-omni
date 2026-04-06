import numpy as np

def uniform_blur_sigma():
    return np.random.uniform(0.0, 0.0)

corruptions_dict = {
    "blur": {
        "sigma": uniform_blur_sigma
    },
}
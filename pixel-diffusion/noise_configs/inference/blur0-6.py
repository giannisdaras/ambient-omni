import numpy as np

def uniform_blur_sigma():
    return np.random.uniform(0.6, 0.6)

corruptions_dict = {
    "blur": {
        "sigma": uniform_blur_sigma
    },
}
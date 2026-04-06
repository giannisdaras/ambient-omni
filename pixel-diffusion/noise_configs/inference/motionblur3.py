import numpy as np

def return_name():
    return "motion_blur"

def return_severity():
    return 3

corruptions_dict = {
    "imagecorruptions": {
        "corruption_name": return_name,
        "severity": return_severity
    },
}
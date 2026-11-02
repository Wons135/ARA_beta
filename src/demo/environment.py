"""Check the actual demo environment before running."""
from importlib.metadata import version
import sys


def verify() -> None:
    expected = {
        "torch": "2.5.1" if sys.platform == "darwin" else "2.5.1+cpu",
        "transformers": "4.41.1", "huggingface-hub": "0.33.1", "tokenizers": "0.19.1",
        "safetensors": "0.5.3", "xgboost": "2.0.3", "scikit-learn": "1.4.2", "pandas": "2.2.2",
        "numpy": "1.26.4", "scipy": "1.16.0", "nltk": "3.9.1", "joblib": "1.5.1",
        "matplotlib": "3.9.0", "textblob": "0.19.0", "tqdm": "4.66.4", "threadpoolctl": "3.6.0",
    }
    for package, pin in expected.items():
        actual = version(package)
        if actual != pin:
            raise RuntimeError(f"{package}: expected {pin}, found {actual}")
    import torch
    import transformers
    import xgboost
    import sklearn
    import pandas
    import scipy
    import nltk
    import matplotlib
    if torch.version.cuda is not None:
        raise RuntimeError("Demo environment must use the CPU torch wheel.")

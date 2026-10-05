# Mac/anaconda only: its torchvision build is broken. Put this folder on PYTHONPATH so
# transformers skips vision imports. Not needed on the CUDA training PC.
import importlib.util, sys
_find_spec = importlib.util.find_spec
def _hidden(name, package=None):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _find_spec(name, package)
importlib.util.find_spec = _hidden
sys.modules["torchvision"] = None  # direct imports raise ImportError

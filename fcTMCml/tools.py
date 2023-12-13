import importlib.util


def openbabel_available() -> bool:
    # checks if openbabel is installed
    openbabel_available = importlib.util.find_spec("openbabel")
    return openbabel_available is not None
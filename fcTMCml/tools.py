import importlib.util
import os


def openbabel_available() -> bool:
    # return False
    # checks if openbabel is installed
    openbabel_available = importlib.util.find_spec("openbabel")
    return openbabel_available is not None


def make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def unpack_tar(path: str) -> None:
    if not os.path.exists(path):
        os.system("tar -xcvf " + path + ".tar.gz")


def remove_dir(path: str) -> None:
    os.removedirs(path)


def load_ligand_dict(path: str) -> dict:
    # load molSimplify ligand dict from ligands.dict
    with open(path, "r") as f:
        lines = f.readlines()
    return {x.split(":")[0]: x.split(":")[1][:-1].split(",") for x in lines[2:]}

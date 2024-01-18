import importlib.util


def openbabel_available() -> bool:
    # checks if openbabel is installed
    openbabel_available = importlib.util.find_spec("openbabel")
    return openbabel_available is not None


def load_ligand_dict(path: str) -> dict:
    # load molSimplify ligand dict from ligands.dict
    with open(path, "r") as f:
        lines = f.readlines()
    return {x.split(":")[0]: x.split(":")[1][:-1].split(",") for x in lines[2:]}

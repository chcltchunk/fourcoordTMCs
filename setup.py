from setuptools import setup

setup(
    name='fourCoordTMCsMl',
    version='1.0',
    description='Binary Geometry Classification and SSE prediction for four-coordinate TMCs',
    author='Jonas A. Oldensteadt',
    author_email='joldenstaedt@gmail.com',
    packages=['fcTMCml'],
    install_requires=[
        'matplotlib==3.8.2',
        'networkx==3.2.1',
        'numpy==1.26.2',
        'pandas==2.1.4',
        'ase==3.22.1',
        'scipy==1.11.4',
        'scikit_learn==1.3.2',
        'imbalanced-learn==0.11.0',
        'hyperopt==0.2.7',
        'umap-learn==0.5.5',
        'tensorflow[and-cuda]',
        'hyperas==0.4.1'

    ],
    extras_require={
        'bond_order_matrix_MCDL46_features': ['openbabel==3.0.0', 'pybel==0.15.5']
    }
)

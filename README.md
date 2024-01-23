# tetrahedral_geom_sse
## installation instructions
clone the repository and change to its root directory:
```
git clone git@github.com:chcltchunk/tetrahedral_geom_sse.git
cd tetrahedral_geom_sse
```
Install the package to your local pip environment:
```
pip install -e .
``` 



## Classifier Results
## classification after feature selection (1:1 balanced dataset)
| Feature Set               | ML Model | Average Score (K-Fold) | PPV   | Sensitivity | F_score |
|---------------------------|----------|------------------------|-------|-------------|---------|
| MCDL53_classification     | RR       | 0.792                  | 0.776 | 0.825       | 0.799   |
| MCDL53_classification     | RFC      | 0.958                  | 0.954 | 0.964       | 0.958   |
| MCDL53_cff_classification | RR       | 0.85                   | 0.817 | 0.905       | 0.858   |
| MCDL53_cff_classification | RFC      | 0.964                  | 0.962 | 0.966       | 0.964   |
| RAC_classification        | RR       | 0.726                  | 0.722 | 0.739       | 0.73    |
| RAC_classification        | RFC      | 0.829                  | 0.819 | 0.846       | 0.832   |
| RAC_cff_classification    | RR       | 0.845                  | 0.812 | 0.9         | 0.853   |
| RAC_cff_classification    | RFC      | 0.953                  | 0.948 | 0.96        | 0.954   |
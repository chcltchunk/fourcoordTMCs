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
### classification after feature selection (1:1 balanced dataset)
| Feature Set               | ML Model | Average Score (K-Fold) | PPV   | Sensitivity | F_score |
|---------------------------|----------|------------------------|-------|-------------|---------|
| MCDL53_classification     | RR       | 0.78                   | 0.761 | 0.818       | 0.788   |
| MCDL53_classification     | RFC      | 0.959                  | 0.96  | 0.958       | 0.959   |
| MCDL53_cff_classification | RR       | 0.863                  | 0.825 | 0.922       | 0.871   |
| MCDL53_cff_classification | RFC      | 0.964                  | 0.957 | 0.972       | 0.964   |
| RAC_classification        | RR       | 0.715                  | 0.718 | 0.71        | 0.713   |
| RAC_classification        | RFC      | 0.828                  | 0.811 | 0.855       | 0.832   |
| RAC_cff_classification    | RR       | 0.849                  | 0.813 | 0.906       | 0.857   |
| RAC_cff_classification    | RFC      | 0.959                  | 0.953 | 0.966       | 0.959   |
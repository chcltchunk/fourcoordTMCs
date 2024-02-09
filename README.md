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



after RFS
| Feature Set          | ML Model | MAE   | MAE_train | R2    | MSE (K-Fold) | MSE_train |
|----------------------|----------|-------|-----------|-------|--------------|-----------|
| MCDLF_regression     | KRR      | 6.807 | 4.312     | 0.569 | 9.266        | 5.868     |
| MCDLF_cff_regression | KRR      | 5.874 | 4.228     | 0.651 | 8.318        | 5.983     |
| RAC_regression       | KRR      | 6.522 | 5.178     | 0.588 | 9.076        | 7.134     |
| RAC_cff_regression   | KRR      | 6.438 | 4.956     | 0.589 | 9.048        | 6.892     |

without RFS
| Feature Set          | ML Model | MAE   | MAE_train | R2    | MSE (K-Fold) | MSE_train |
|----------------------|----------|-------|-----------|-------|--------------|-----------|
| MCDLF_regression     | KRR      | 6.597 | 4.356     | 0.594 | 9.036        | 6.064     |
| MCDLF_cff_regression | KRR      | 6.252 | 5.135     | 0.631 | 8.557        | 7.073     |
| RAC_regression       | KRR      | 6.946 | 5.23      | 0.577 | 9.314        | 7.146     |
| RAC_cff_regression   | KRR      | 6.352 | 5.033     | 0.604 | 8.921        | 7.128     |
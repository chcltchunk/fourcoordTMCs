# tetrahedral_geom_sse
## Classification w/o feature selection
| Feature Set                          | ML Model | Average Score (K-Fold)  | PPV            | Sensitivity | F_score |
|--------------------------------------|----------|-------------------------|----------------|-------------|---------|
| MCDL53_classification                | RR (TPE) | 0.818                   | 0.776          | 0.786       | 0.78    |
| MCDL53_classification                | RFC (TPE)| 0.932                   | 0.925          | 0.908       | 0.916   |
| MCDL53_cff_classification            | RR (TPE) | 0.888                   | 0.840          | 0.899       | 0.868   |
| MCDL53_cff_classification            | RFC (TPE)| 0.948                   | 0.932          | 0.941       | 0.936   |
| RAC_classification                   | RR (TPE) | 0.738                   | 0.682          | 0.689       | 0.683   |
| RAC_classification                   | RFC (TPE)| 0.774                   | 0.715          | 0.753       | 0.732   |
| RAC_cff_classification               | RR (TPE) | 0.887                   | 0.823          | 0.927       | 0.872   |
| RAC_cff_classification               | RFC (TPE)| 0.903                   | 0.892          | 0.872       | 0.881   |

## classification after feature selection
| Feature Set               | ML Model  | Average Score (K-Fold) | PPV     | Sensitivity  | F_score |
|---------------------------|-----------|------------------------|---------|--------------|---------|
| MCDL53_classification     | RR (TPE)  |   0.802                |   0.758 |   0.767      |   0.761 |
| MCDL53_classification     | RFC (TPE) |   0.946                |   0.944 |   0.923      |   0.933 |
| MCDL53_cff_classification | RR (TPE)  |   0.848                |   0.799 |   0.845      |   0.821 |
| MCDL53_cff_classification | RFC (TPE) |   0.954                |   0.944 |   0.945      |   0.944 |
| RAC_classification        | RR (TPE)  |   0.719                |   0.676 |   0.611      |   0.641 |
| RAC_classification        | RFC (TPE) |   0.801                |   0.773 |   0.733      |   0.751 |
| RAC_cff_classification    | RR (TPE)  |   0.844                |   0.793 |   0.843      |   0.817 |
| RAC_cff_classification    | RFC (TPE) |   0.940                |   0.922 |   0.933      |   0.927 |

## classification after feature selection less oversampling (0.5 instead of 0.7)
| Feature Set               | ML Model | Average Score (K-Fold) | PPV   | Sensitivity | F_score |
|---------------------------|----------|------------------------|-------|-------------|---------|
| MCDL53_classification     | RR       | 0.814                  | 0.746 | 0.676       | 0.708   |
| MCDL53_classification     | RFC      | 0.932                  | 0.921 | 0.873       | 0.895   |
| MCDL53_cff_classification | RR       | 0.843                  | 0.755 | 0.787       | 0.77    |
| MCDL53_cff_classification | RFC      | 0.942                  | 0.925 | 0.901       | 0.912   |
| RAC_classification        | RR       | 0.729                  | 0.633 | 0.435       | 0.515   |
| RAC_classification        | RFC      | 0.787                  | 0.713 | 0.607       | 0.654   |
| RAC_cff_classification    | RR       | 0.845                  | 0.758 | 0.784       | 0.77    |
| RAC_cff_classification    | RFC      | 0.934                  | 0.909 | 0.891       | 0.9     |

## classification after feature selection (any number of features)
| Feature Set               | ML Model | Average Score (K-Fold) | PPV   | Sensitivity | F_score |
|---------------------------|----------|------------------------|-------|-------------|---------|
| MCDL53_classification     | RR       | 0.811                  | 0.768 | 0.777       | 0.772   |
| MCDL53_classification     | RFC      | 0.945                  | 0.939 | 0.927       | 0.933   |
| MCDL53_cff_classification | RR       | 0.847                  | 0.797 | 0.846       | 0.82    |
| MCDL53_cff_classification | RFC      | 0.953                  | 0.937 | 0.948       | 0.942   |
| RAC_classification        | RR       | 0.719                  | 0.674 | 0.617       | 0.643   |
| RAC_classification        | RFC      | 0.79                   | 0.75  | 0.738       | 0.743   |
| RAC_cff_classification    | RR       | 0.844                  | 0.793 | 0.843       | 0.817   |
| RAC_cff_classification    | RFC      | 0.943                  | 0.929 | 0.933       | 0.931   |

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
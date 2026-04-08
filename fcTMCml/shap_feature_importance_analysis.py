import shap
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from fcTMCml.tools import load_features, make_dir, add_feature_inportance_pie_to_plot
from fcTMCml.constants import feature_target_dir, SHAP_directory

make_dir("results/feature_importance_analysis")
make_dir(SHAP_directory)

classification_in_subdir = "classification_rff_selection/"
classification_targets, feature_dict, feature_names_dict, feature_importances_dict = load_features(feature_target_dir, classification_in_subdir,
                                                                                                   "classification", load_importances=True)

for i, run_ident in enumerate(feature_importances_dict):
    fig, ax = plt.subplots()
    with open(SHAP_directory + run_ident + "_RFC_shap_values.pkl", 'rb') as f:
        shap_values = pkl.load(f)
    importances = shap_values.abs.mean((0, 2)).values
    print(importances)
    names = shap_values.feature_names
    print(names)
    add_feature_inportance_pie_to_plot(ax, importances, names, run_ident, path=SHAP_directory)
    plt.close()
    shap.plots.beeswarm(shap_values[..., 1], show=False, max_display=20)
    plt.savefig(SHAP_directory + run_ident + "_RFC_shap.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    shap.plots.bar(shap_values.abs.mean((0, 2)), show=False, max_display=20)
    plt.savefig("results/feature_importance_analysis/SHAP/" + run_ident + "_RFC_shap_mean.pdf", bbox_inches="tight", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
from fcTMCml.tools import load_features, make_dir, add_feature_inportance_pie_to_plot
from fcTMCml.constants import feature_target_dir

make_dir("results/feature_importance_analysis")

classification_in_subdir = "classification_rff_selection/"
classification_targets, feature_dict, feature_names_dict, feature_importances_dict = load_features(feature_target_dir, classification_in_subdir,
                                                                                                   "classification", load_importances=True)

for i, run_ident in enumerate(feature_importances_dict):
    fig, ax = plt.subplots()
    importances = feature_importances_dict[run_ident]
    names = feature_names_dict[run_ident]
    add_feature_inportance_pie_to_plot(ax, importances, names, run_ident)


regression_in_subdir = "regression_rff_selection/"
regression_targets, feature_dict, feature_names_dict, feature_importances_dict = load_features(feature_target_dir, regression_in_subdir,
                                                                                               "regression", load_importances=True)

for i, run_ident in enumerate(feature_importances_dict):
    fig, ax = plt.subplots()
    importances = feature_importances_dict[run_ident]
    names = feature_names_dict[run_ident]
    add_feature_inportance_pie_to_plot(ax, importances, names, run_ident)

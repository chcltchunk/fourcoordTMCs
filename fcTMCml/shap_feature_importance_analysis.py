import re
import shap
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from fcTMCml.tools import load_features, make_dir, add_feature_inportance_pie_to_plot
from fcTMCml.constants import feature_target_dir, SHAP_directory

make_dir("results/feature_importance_analysis")
make_dir(SHAP_directory)

classification_in_subdir = "classification_rff_selection/"
classification_targets, feature_dict, feature_names_dict, feature_importances_dict = load_features(feature_target_dir, classification_in_subdir,
                                                                                                   "classification", load_importances=True)

# define TeX style identifier for plots
replacements = [
               ['CA', 'I(M)', 'L#A', 'LC', 'Ox', 'S', 'TK', r'max($\chi$)',
                r'min($\chi$)', r'sum($\chi$)'],
               ['CA', 'I(M)', 'L#A', 'LC', 'Ox', 'S', 'TK', r'$\Delta E$', r'max($\chi$)',
                r'sum($\chi$)'],
               [r'$^\mathrm{lc}_\mathrm{ax}\chi_{3}^D$', r'$^\mathrm{all}_\mathrm{all}Z_{1}^D$',
                r'$^\mathrm{mc}_\mathrm{all}\chi_{1}^D$', 'Ox', r'$^\mathrm{f}_\mathrm{ax}S_{1}$',
                r'$^\mathrm{lc}_\mathrm{ax}Z_{2}$', r'$^\mathrm{mc}_\mathrm{all}S_{0}$',
                r'$^\mathrm{mc}_\mathrm{all}Z_{0}$', r'$^\mathrm{mc}_\mathrm{all}\chi_{0}$',
                r'$^\mathrm{mc}_\mathrm{all}\chi_{2}$'],
               [r'$^\mathrm{lc}_\mathrm{ax}\chi_{3}^D$', r'$^\mathrm{all}_\mathrm{all}Z_{1}^D$',
                'Ox', r'$\Delta E$', r'$^\mathrm{f}_\mathrm{ax}\chi_{0}$',
                r'$^\mathrm{lc}_\mathrm{ax}Z_{2}$', r'$^\mathrm{mc}_\mathrm{all}S_{0}$',
                r'$^\mathrm{mc}_\mathrm{all}Z_{1}$', r'$^\mathrm{mc}_\mathrm{all}\chi_{1}$',
                r'$^\mathrm{mc}_\mathrm{all}\chi_{2}$'],
]


for i, run_ident in enumerate(feature_importances_dict):
    print(run_ident, i)
    fig, ax = plt.subplots()
    with open(SHAP_directory + run_ident + "_RFC_shap_values.pkl", 'rb') as f:
        shap_values = pkl.load(f)
    importances = shap_values.abs.mean((0, 2)).values
    names = shap_values.feature_names
    add_feature_inportance_pie_to_plot(ax, importances, names, run_ident, path=SHAP_directory)
    plt.close()
    diff_explanation = shap.Explanation(
        values=shap_values[..., 1].values - shap_values[..., 0].values,
        base_values=shap_values[..., 0].base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names
    )

    df = pd.DataFrame(diff_explanation.values, columns=[re.sub(r"ax\d+", "ax", name) for name in diff_explanation.feature_names])

    grouped_df_expl = df.T.groupby(level=0, by=lambda col: "-".join(col.split("-")[:-1])).sum().T
    df = pd.DataFrame(diff_explanation.data, columns=[re.sub(r"ax\d+", "ax", name) for name in diff_explanation.feature_names])
    grouped_df_data = df.T.groupby(level=0, by=lambda col: "-".join(col.split("-")[:-1])).sum().T

    print("diff expl")
    print(grouped_df_expl.columns)

    grouped_expl = shap.Explanation(
        values=grouped_df_expl.values,
        base_values=diff_explanation.base_values,
        data=grouped_df_data.values,
        feature_names=replacements[i]
    )
    shap.plots.beeswarm(grouped_expl, show=False, max_display=20)
    ax = plt.gca()
    ax.set_xlabel("SHAP value difference (tetrahedral vs. square planar)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16, direction='in')
    ax.set_xlim(-1.15, 1.15)
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Feature value", fontsize=16)
    cbar.tick_params(labelsize=12)
    plt.savefig(SHAP_directory + run_ident + "_RFC_shap_diff.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    shap.plots.bar(shap_values.abs.mean((0, 2)), show=False, max_display=20)
    plt.savefig("results/feature_importance_analysis/SHAP/" + run_ident + "_RFC_shap_mean.pdf", bbox_inches="tight", dpi=300)
    plt.close()

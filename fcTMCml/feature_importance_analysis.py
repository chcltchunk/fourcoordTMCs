import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from fcTMCml.tools import load_features, make_dir
from fcTMCml.constants import feature_target_dir


def add_feature_inportance_pie_to_plot(ax, importances, feature_names, run_ident):
    print(run_ident)
    size = 0.3
    # grouped in coordination spheres (up to 3rd) by most metal distant atom (RACs); reminder: {start}-{prop}-{d}-{scope}
    # not complete (may append on demand)
    feature_classes = {"Metal": ["I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", "delE", 'mc-chi-0-all',
                                 'mc-S-0-all', 'mc-Z-0-all', ],
                       "Ligand" : ["CA", "LC", "LD", "#A", "L#A", "max_LBO", "K", "TK"],
                       "Counts" : ["#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I",
                                         "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"],
                       "1st" : ['D_mc-chi-1-all', 'D_mc-Z-1-all', 'mc-Z-1-all'],
                       "2nd" : ['mc-chi-2-all', 'D_mc-chi-2-all'],
                       "3rd" : ['D_lc-S-3-ax1', 'D_lc-S-3-ax2', 'D_lc-S-3-ax3', 'D_lc-S-3-ax4',
                                'D_lc-T-3-ax1', 'D_lc-T-3-ax2', 'D_lc-T-3-ax3', 'D_lc-T-3-ax4',
                                'lc-Z-2-ax1', 'lc-Z-2-ax2', 'lc-Z-2-ax3', 'lc-Z-2-ax4',
                                'D_lc-S-3-ax1', 'D_lc-S-3-ax2', 'D_lc-S-3-ax3', 'D_lc-S-3-ax4',
                                'mc-Z-3-all'],
                       "Global" : ['f-chi-0-ax1', 'f-chi-0-ax2', 'f-chi-0-ax3', 'f-chi-0-ax4',
                                   'f-Z-1-all', 'f-Z-1-ax1' 'f-Z-1-ax2' 'f-Z-1-ax3' 'f-Z-1-ax4']
                       }
    """
    blues:  ['#f7fbff', '#ecf4fb', '#e1edf8', '#d6e6f4', '#ccdff1', '#bdd7ec', '#abd0e6', '#99c7e0', '#82bbdb', '#6aaed6', '#58a1cf', '#4695c8', '#3787c0', '#2979b9', '#1b69af', '#105ba4', '#084d96', '#083e81', '#08306b']
    greens:  ['#f7fcf5', '#ebf7e7', '#dbf1d6', '#c7e9c0', '#aedea7', '#92d28f', '#73c476', '#52b365', '#37a055', '#228a44', '#0b7734', '#005f26', '#00441b']
    oranges:  ['#fff5eb', '#fee8d2', '#fdd5ad', '#fdb97d', '#fd9c51', '#f87d29', '#e95e0d', '#cd4401', '#a13403', '#7f2704']
    reds:  ['#fff5f0', '#fee3d6', '#fcc4ad', '#fca082', '#fb7c5c', '#f6553c', '#e32f27', '#c2161b', '#9d0d14', '#67000d']
    greys:  ['#ffffff', '#f9f9f9', '#f4f4f4', '#ededed', '#e4e4e4', '#dcdcdc', '#d1d1d1', '#c6c6c6', '#bbbbbb', '#adadad', '#9e9e9e', '#8f8f8f', '#828282', '#757575', '#686868', '#5c5c5c', '#4d4d4d', '#3c3c3c', '#2b2b2b', '#1c1c1c', '#0e0e0e', '#000000']
    """
    inner_chart_colors = {"Global" : {"S" : "#BABABA", "Z" : "#D1D1D1", "chi" : "#DDDDDD", "ox" : "#F4F4F4"},
                          "Counts" : {'#B': '#ffffff', '#C': '#f9f9f9', '#N': '#f4f4f4', '#O': '#ededed', '#F': '#e4e4e4',
                                      '#P': '#dcdcdc', '#S': '#d1d1d1', '#Cl': '#c6c6c6', '#Br': '#bbbbbb', '#I': '#adadad',
                                      'T#B': '#9e9e9e', 'T#C': '#8f8f8f', 'T#N': '#828282', 'T#O': '#757575', 'T#F': '#686868',
                                      'T#P': '#5c5c5c', 'T#S': '#4d4d4d', 'T#Cl': '#3c3c3c', 'T#Br': '#2b2b2b', 'T#I': '#1c1c1c'},
                          "Metal" : {"S" : "#6666FF", "T" : "#9898FF", "Z" : "#B1B2FF", "chi" : "#CCCCFF",
                                     "I(M)": '#f7fbff', "Ox": '#e1edf8', r"sum($\chi$)": '#d6e6f4', r"min($\chi$)": '#ccdff1',
                                     r"max($\chi$)": '#bdd7ec', "SS": '#3787c0', "delE": '#084d96'},
                          "1st" : {"S" : "#FF6666", "T" : "#FF8080", "Z" : "#FF9999", "chi" : "#FFB2B3"},
                          "Ligand" : {"CA" : "#FF6666", "LC" : "#FF8080", "LD" : "#FF9999", "#A" : "#FFB2B3", "L#A": '#fee3d6',
                                      "max_LBO": '#fdb97d', "K": '#e32f27', "TK": '#c2161b'},
                          "2nd" : {"S" : "#65DC8C", "T" : "#80E19F", "Z" : "#99E7B3", "chi" : "#B2EEC6"},
                          "3rd" : {"S" : "#FFC966", "T" : "#FFD280", "Z" : "#FFDC99", "chi" : "#FFDC99"}
                          }
    outer_chart_colors = {"Global" : '#999CA1',
                          "Counts" : '#999CA1',
                          "Metal" : '#4356A3',
                          "1st" : '#F04646',
                          "Ligand" : '#F04646',
                          "2nd" : '#4EBA70',
                          "3rd" : '#FEB137'
                          }

    outer_chart_colors = {'Global' : 0, 'Counts': 0, 'Metal': 0, 'Ligand': 0, '1st' : 0, '2nd' : 0, '3rd' : 0}
    inner_chart_colors = {"Global" : {"S" : 0, "Z" : 0, "chi" : 0, "ox" : 0},
                          "Counts" : {'#B': 0, '#C': 0, '#N': 0, '#O': 0, '#F': 0,
                                      '#P': 0, '#S': 0, '#Cl': 0, '#Br': 0, '#I': 0,
                                      'T#B': 0, 'T#C': 0, 'T#N': 0, 'T#O': 0, 'T#F': 0,
                                      'T#P': 0, 'T#S': 0, 'T#Cl': 0, 'T#Br': 0, 'T#I': 0},
                          "Metal" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0,
                                     "I(M)": 0, "Ox": 0, r"sum($\chi$)": 0, r"min($\chi$)": 0,
                                     r"max($\chi$)": 0, "SS": 0, "delE": 0},
                          "1st" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
                          "Ligand" : {"CA" : 0, "LC" : 0, "LD" : 0, "#A" : 0, "L#A": 0,
                                      "max_LBO": 0, "K": 0, "TK": 0},
                          "2nd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
                          "3rd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0}
                          }

    colormap_selection = ["Greys_r", "Purples_r", "Blues_r", "Greens_r", "Greens_r", "Reds_r", "Oranges_r"]
    for colormap, key in zip(colormap_selection, outer_chart_colors):
        cmap = cm.get_cmap(colormap, len(inner_chart_colors[key].keys()) + 5)

        outer_chart_colors[key] = colors.rgb2hex(cmap(2))
        for i, key2 in enumerate(inner_chart_colors[key]):
            inner_chart_colors[key][key2] = colors.rgb2hex(cmap(i + 3))

    # since features in eaach subclass (meatal, Ligand, Counts) are not equally long
    # and feature labels are of type string, we use an intermediate helper dictionary
    # for the feature grouping
    # vals = {'Metal': {}, 'Ligand': {}, 'Counts': {}, '1st' : {}, '2nd' : {}, '3rd' : {}, 'Global' : {}}
    vals = {"Global" : {"S" : 0, "Z" : 0, "chi" : 0, "ox" : 0},
            "Counts" : {'#B': 0, '#C': 0, '#N': 0, '#O': 0, '#F': 0,
                        '#P': 0, '#S': 0, '#Cl': 0, '#Br': 0, '#I': 0,
                        'T#B': 0, 'T#C': 0, 'T#N': 0, 'T#O': 0, 'T#F': 0,
                        'T#P': 0, 'T#S': 0, 'T#Cl': 0, 'T#Br': 0, 'T#I': 0},
            "Metal" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0,
                       "I(M)": 0, "Ox": 0, r"sum($\chi$)": 0, r"min($\chi$)": 0,
                       r"max($\chi$)": 0, "SS": 0, "delE": 0},
            "1st" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
            "Ligand" : {"CA" : 0, "LC" : 0, "LD" : 0, "#A" : 0, "L#A": 0,
                        "max_LBO": 0, "K": 0, "TK": 0},
            "2nd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
            "3rd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0}
            }
    assert len(importances) == len(feature_names)
    print("feature names", feature_names)
    for i, feature_name in enumerate(feature_names):
        for class_i in feature_classes:
            if feature_name in feature_classes[class_i]:
                split = feature_name.split('-')
                feature_name = feature_name if len(split) < 2 else split[1]
                vals[class_i][feature_name] += importances[i]
    # remove zero valued feature names
    vals_cleaned = {'Metal': {}, 'Ligand': {}, 'Counts': {}, '1st' : {}, '2nd' : {}, '3rd' : {}, 'Global' : {}}
    for key in vals:
        for key2 in vals[key]:
            if vals[key][key2] != 0:
                print(f"deleting {key2} from {key}")
                vals_cleaned[key][key2] = vals[key][key2]
        if vals_cleaned[key] == {}:
            del vals_cleaned[key]
    vals = vals_cleaned
    # count number of features that are in each subclass
    outer_values = []
    outer_colors = []
    outer_names = []
    inner_values = []
    inner_colors = []
    inner_names = []
    for i, class_i in enumerate(vals):
        lock = False
        for j, feature_name in enumerate(vals[class_i]):
            if not lock:
                outer_colors += [outer_chart_colors[class_i]]
                outer_names += [class_i]
                outer_values += [0]
                lock = True
            outer_values[-1] += vals[class_i][feature_name]
            inner_values += [vals[class_i][feature_name]]
            inner_colors += [inner_chart_colors[class_i][feature_name]]
            inner_names += [feature_name if feature_name not in ["chi"] else r"$\chi$"]
    print(outer_names)
    print(inner_names)

    size = 0.3
    radius = 1
    outer_pie = ax.pie(outer_values, radius=radius, colors=outer_colors,
                       wedgeprops=dict(width=size, edgecolor='w'), labeldistance=radius - size * 0.5,
                       rotatelabels=True, textprops={'fontsize': 10})

    inner_pie = ax.pie(inner_values, radius=radius - size, colors=inner_colors,
                       labeldistance=0.9, wedgeprops=dict(width=0.4, edgecolor='w'), rotatelabels=True)

    plt.setp(inner_pie[1], rotation_mode="anchor", ha="left", va="center")
    plt.setp(outer_pie[1], rotation_mode="anchor", ha="center", va="center")
    inner_values_norm = np.array(inner_values) / np.max(inner_values)
    outer_values_norm = np.array(outer_values) / np.max(outer_values)

    for val, label, c, t, w in zip(inner_values_norm, inner_names, inner_colors, inner_pie[1], inner_pie[0]):
        x, y = t.get_position()
        angle = int(np.degrees(np.arctan2(y, x)))
        ha = "right"

        if x < 0:
            angle -= 180
            ha = "left"
        # if val < 0.05:
        #     ang = (w.theta2 - w.theta1) / 2. + w.theta1
        #     y = np.sin(np.deg2rad(ang))
        #     x = np.cos(np.deg2rad(ang))
        #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        #     connectionstyle = f"angle,angleA=0,angleB={ang}"
        #     bbox_props = dict(boxstyle="square,pad=0.3", fc=c, ec=c, lw=0.72)
        #     kw = dict(arrowprops=dict(arrowstyle="-", color=c, linewidth=2),
        #               bbox=bbox_props, zorder=1, va="center")
        #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #     ax.annotate(label, xy=(x, y), xytext=(text_xscale * x, text_yscale * y),
        #                 horizontalalignment=horizontalalignment, **kw)
        # else:
        print("adding label: ", label)
        ax.annotate(label, xy=(x, y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=6)
    for val, c, label, t, w in zip(outer_values_norm, outer_colors, outer_names, outer_pie[1], outer_pie[0]):
        if val < 0.05:
            ang = (w.theta2 - w.theta1) / 2. + w.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            bbox_props = dict(boxstyle="square,pad=0.3", fc=c, ec=c, lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-", color=c, linewidth=2),
                      bbox=bbox_props, zorder=0, va="center")
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(label, xy=(x, y), xytext=(1.2 * np.sign(x), y),
                        horizontalalignment=horizontalalignment, **kw)
        else:
            x, y = t.get_position()
            angle = int(np.degrees(np.arctan2(y, x)) + 90)
            if x < 0:
                angle -= 180
            ha = "center"
            ax.annotate(label, xy=(x, y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=10)
    ax.set_axis_off()
    plt.savefig(f"results/feature_importance_analysis/{run_ident}.pdf", dpi=300, bbox_inches='tight')


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

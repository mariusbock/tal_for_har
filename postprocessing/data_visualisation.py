# ------------------------------------------------------------------------
# Scripts used for creating color-based prediction timeline visualization
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import colorcet as cc

dataset = 'sbhar'
sample_sbj = 3

if dataset == 'wear':
    sbjs = 18
    sens_axes = 12
    label_dict = {
        'null': 0,
        'jogging': 1,
        'jogging (rotating arms)': 2,
        'jogging (skipping)': 3,
        'jogging (sidesteps)': 4,
        'jogging (butt-kicks)': 5,
        'stretching (triceps)': 6,
        'stretching (lunging)': 7,
        'stretching (shoulders)': 8,
        'stretching (hamstrings)': 9,
        'stretching (lumbar rotation)': 10,
        'push-ups': 11,
        'push-ups (complex)': 12,
        'sit-ups': 13,
        'sit-ups (complex)': 14,
        'burpees': 15,
        'lunges': 16,
        'lunges (complex)': 17,
        'bench-dips': 18
    }
elif dataset == 'rwhar':
    sbjs = 15
    sens_axes = 21
    label_dict = {
    'climbingdown': 0,
    'climbingup': 1,
    'jumping': 2,
    'lying': 3,
    'running': 4,
    'sitting': 5,
    'standing': 6,
    'walking': 7
}
elif dataset == 'wetlab':
    sbjs = 22
    sens_axes = 3
    label_dict = {
    'null': 0,
    'cutting': 1,
    'inverting': 2,
    'peeling': 3,
    'pestling': 4,
    'pipetting': 5,
    'pouring': 6,
    'stirring': 7,
    'transfer': 8
}
elif dataset == 'opportunity':
    sbjs = 4
    sens_axes = 113
    label_dict = {
    'null': 0,
    'open_door_1': 1,
    'open_door_2': 2,
    'close_door_1': 3,
    'close_door_2': 4,
    'open_fridge': 5,
    'close_fridge': 6,
    'open_dishwasher': 7,
    'close_dishwasher': 8,
    'open_drawer_1': 9,
    'close_drawer_1': 10,
    'open_drawer_2': 11,
    'close_drawer_2': 12,
    'open_drawer_3': 13,
    'close_drawer_3': 14,
    'clean_table': 15,
    'drink_from_cup': 16,
    'toggle_switch': 17
}
elif dataset == 'sbhar':
    sbjs = 30
    sens_axes = 3
    label_dict = {
    'null': 0,
    'walking': 1,
    'walking_upstairs': 2,
    'walking_downstairs': 3,
    'sitting': 4,
    'standing': 5,
    'lying': 6,
    'stand-to-sit': 7,
    'sit-to-stand': 8,
    'sit-to-lie': 9,
    'lie-to-sit': 10,
    'stand-to-lie': 11,
    'lie-to-stand': 12,
}
elif dataset == 'hangtime':
    sbjs = 24
    sens_axes = 3
    label_dict = {
    'null': 0,
    'dribbling': 1,
    'shot': 2,
    'pass': 3,
    'rebound': 4,
    'layup': 5,
}

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def compare_timelines(gt, timeline_1, timeline_2, timeline_3, timeline_4, timeline_5,timeline_6, timeline_7, timeline_8):
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter

    n_classes = len(np.unique(gt))

    # plot 1:
    # ax3, ax4, ax5, ax6
    fig, (gt_ax, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(9, 1, sharex=True, figsize=(9, 6), layout="constrained")

    colors1 = sns.color_palette(palette=cc.glasbey_warm, n_colors=n_classes).as_hex()
    if dataset != 'rwhar':
        colors1[0] = '#F8F8F8'
    cmap1 = LinearSegmentedColormap.from_list(name='My Colors1', colors=colors1, N=len(colors1))

    gt_ax.set_yticks([])
    gt_ax.pcolor([gt], cmap=cmap1, vmin=0, vmax=n_classes)

    ax1.set_yticks([])
    ax1.pcolor([timeline_1], cmap=cmap1, vmin=0, vmax=n_classes)

    ax2.set_yticks([])
    ax2.pcolor([timeline_2], cmap=cmap1, vmin=0, vmax=n_classes)

    ax3.set_yticks([])
    ax3.pcolor([timeline_3], cmap=cmap1, vmin=0, vmax=n_classes)
   
    ax4.set_yticks([])
    ax4.pcolor([timeline_4], cmap=cmap1, vmin=0, vmax=n_classes)
    
    ax5.set_yticks([])
    ax5.pcolor([timeline_5], cmap=cmap1, vmin=0, vmax=n_classes)

    ax6.set_yticks([])
    ax6.pcolor([timeline_6], cmap=cmap1, vmin=0, vmax=n_classes)

    ax7.set_yticks([])
    ax7.pcolor([timeline_7], cmap=cmap1, vmin=0, vmax=n_classes)

    ax8.set_yticks([])
    ax8.pcolor([timeline_8], cmap=cmap1, vmin=0, vmax=n_classes)
    
    print(colors1)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.xticks([])
    plt.savefig('test.png')
    plt.close()


def get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt

    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

time_1 = np.load('predictions/unprocessed_deepconvlstm_{}.npy'.format(dataset))
time_2 = np.load('predictions/deepconvlstm_{}.npy'.format(dataset))
time_3 = np.load('predictions/unprocessed_shallowdeepconvlstm_{}.npy'.format(dataset))
time_4 = np.load('predictions/shallowdeepconvlstm_{}.npy'.format(dataset))
time_5 = np.load('predictions/unprocessed_aandd_{}.npy'.format(dataset))
time_6 = np.load('predictions/aandd_{}.npy'.format(dataset))
time_7 = np.load('predictions/unprocessed_tinyhar_{}.npy'.format(dataset))
time_8 = np.load('predictions/tinyhar_{}.npy'.format(dataset))

all_s_data = np.empty((0, sens_axes + 2))
for sbj in range(sbjs):
    t_data = pd.read_csv(os.path.join('data/{}/raw/inertial'.format(dataset), 'sbj_' + str(int(sbj)) + '.csv'), index_col=False, low_memory=False).replace({"label": label_dict}).fillna(0).to_numpy()
    all_s_data = np.append(all_s_data, t_data, axis=0)

gt_viz = all_s_data[(all_s_data[:, 0] == sample_sbj)][:, -1]
time_1_viz = time_1[(all_s_data[:, 0] == sample_sbj)]
time_2_viz = time_2[(all_s_data[:, 0] == sample_sbj)]
time_3_viz = time_3[(all_s_data[:, 0] == sample_sbj)]
time_4_viz = time_4[(all_s_data[:, 0] == sample_sbj)]
time_5_viz = time_5[(all_s_data[:, 0] == sample_sbj)]
time_6_viz = time_6[(all_s_data[:, 0] == sample_sbj)]
time_7_viz = time_7[(all_s_data[:, 0] == sample_sbj)]
time_8_viz = time_8[(all_s_data[:, 0] == sample_sbj)]

compare_timelines(gt_viz, time_1_viz, time_2_viz, time_3_viz, time_4_viz, time_5_viz, time_6_viz, time_7_viz, time_8_viz)

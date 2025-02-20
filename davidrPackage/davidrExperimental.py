#!/usr/bin/env python

"""
Description:  

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""
import random
import threading
import time
import pandas as pd
import sys
from tqdm import tqdm

import scanpy.get
import anndata as ad
# Function to add stat annotation to seaborn plots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import numpy as np
from matplotlib import patches

import davidrUtility
from scipy.stats import mannwhitneyu, false_discovery_control
from torch.distributed.elastic.agent.server.local_elastic_agent import logger

davidrUtility.iOn()


class _StatPlotter:
    """
    Class to add stats to barplots. Ass of now, it cannot handle plots with hue.
    TODO --> Add the option to handle hue; Add the option to handle other type of plots
    TODO --> like violinplot, boxplot, etc
    TODO --> Only plot significant option
    """
    def __init__(self, ax: plt.axis, data: pd.DataFrame, x: str, y:str,
                 hue:str, pairs:list, offset:float, text_offset:float, text_size:str,
                 test:str):
        """
        Initialise the class
        :param ax: matplotlib axis
        :param data: dataframe with the data that has been plotted
        :param x: column in data which correspond to the x-axis
        :param y:  column in data which correspond to the y-axis
        :param hue: column in data which is use for hue
        :param pairs: pairs to be tested [(catg1, catg2), (catg1, catg3) ...]
        :param offset: offset to add the bar with pvals
        """

        self.axis = ax
        self.data = data
        self.x_axis = x
        self.y_axis = y
        self.hues = hue
        self.offset = offset
        self.text_size = text_size
        self.text_offset = text_offset
        self.type_test = test
        # Correct pairs --> Convert to strings
        self.pairs = [tuple(map(str, t)) for t in pairs]

        # Exract X ticks from the plot
        x_ticks = self.axis.get_xmajorticklabels()
        self.x_positions = [_tick.get_position()[0] for _tick in x_ticks]
        self.labels = [_tick.get_text() for _tick in x_ticks]

        return

    def _get_box_height(self):
        """
        For barplots, extract the height of all the bars and save in the heights attribute
        :return:
        """
        error_bars = [child for child in self.axis.get_children() if isinstance(child, plt.Line2D)]
        if len(error_bars) == 0:
            heights = np.fromiter((np.nanmax(patch.get_height())
                                   if  patch.get_height() >=0  else np.nanmin(patch.get_height())
                                   for patch in self.axis.patches), dtype=float)
        else:
            heights = np.fromiter((np.nanmax(error_bar.get_data()[1])
                                   if all(b >= 0 for b in (val for val in error_bars[0].get_data()[1] if not np.isnan(val))) else np.nanmin(error_bar.get_data()[1])
                                   for error_bar in error_bars), dtype=float)
        self.heights = heights
        return

    def _do_wilcoxon(self):
        """
        Run Wilcoxon test on the pairs specified by the user
        :return:
        """
        pvals = []
        for catg1, catg2 in self.pairs:  # pairs --> [(cat1, cat2), (cat1, cat3), ... (cat1, catn)]
            group1 = self.data[self.data[self.x_axis].astype(str) == catg1][self.y_axis]
            group2 = self.data[self.data[self.x_axis].astype(str) == catg2][self.y_axis]
            _, pval = sp.stats.mannwhitneyu(group1, group2, use_continuity=True)
            if pval > 0.05:
                pval = str(round(pval, 2))
            elif pval > 0.009:
                pval = str(round(pval, 4))
            else:
                if pval == 0:
                    pval = sys.float_info.min
                pval = '{:0.2e}'.format(pval)
            pvals.append(pval)  # Round Pval in scientific notation
        self.pvals = pvals
        return

    def _do_ttest(self):
        """
        Run t-test on the pairs specified by the user
        :return:
        """
        pvals = []
        for catg1, catg2 in self.pairs:  # pairs --> [(cat1, cat2), (cat1, cat3), ... (cat1, catn)]
            group1 = self.data[self.data[self.x_axis].astype(str) == catg1][self.y_axis]
            group2 = self.data[self.data[self.x_axis].astype(str) == catg2][self.y_axis]
            _, pval = sp.stats.ttest_ind(group1, group2)
            if pval > 0.05:
                pval = str(round(pval, 2))
            elif pval > 0.009:
                pval = str(round(pval, 4))
            else:
                if pval == 0:
                    pval = sys.float_info.min
                pval = '{:0.2e}'.format(pval)
            pvals.append(pval)  # Round Pval in scientific notation
        self.pvals = pvals


    def _get_offset(self, offset_pct=0.05):  # 5 % offset taking the largest height of both groups
        """
        Determine the y positions where the box with pval will be plotted. Take the highest
        height across two bars and add an offset of 5 %
        :param offset_pct:
        :return:
        """
        offsets, h = [], []
        for pair in self.pairs:
            pair_height = max([self.heights[self.labels.index(catg)] for catg in pair])
            offset = pair_height * offset_pct
            offsets.append(round(offset, 2))
            h.append(pair_height)
        self.y_offsets = offsets
        self.y_positions = h
        return

    def _get_pair_xpos(self):
        """
        Return the x pair positions of the box with the pval
        :return:
        """
        pairs_positions = []
        for pair in self.pairs:
            pair_xpos = [self.x_positions[self.labels.index(catg)] for catg in pair]
            pairs_positions.append(tuple(pair_xpos))
        self.pair_positions = pairs_positions
        return

    def _set_ylim(self, offset=0.1):
        """
        Adjust y limit of the axis in case the box with pval is on the top
        :param offset:
        :return:
        """
        h = np.array(self.y_positions)
        if all(b < 0 for b in h):
            h_max = h.min() + h.min() * offset * 2
        else:
            h_max = h.max() + h.max() * offset * 2
        self.axis.set_ylim(0, h_max + h_max * offset/4)
        return

    def _do_rectangles_overlap(self, rect1, rect2):
        """
        For a pair of patches, check if they overlap
        :param rect1:
        :param rect2:
        :return:
        """
        x1, y1 = rect1.xy
        w1, h1 = rect1.get_width(), rect1.get_height()
        x2, y2 = rect2.xy
        w2, h2 = rect2.get_width(), rect2.get_height()
        if np.abs(y1 + h1) < np.abs(y2 + h2):
            if (x1 < x2) and (x1 + w1 > x2 + w2):
                return True
            return True
        else:
            return False

    def _check_overlap_rectangles(self, rect, current_pos):
        """
        If a pair of patches overlap, adjust the height of the patch for the pval so it does
        not overlap
        :param rect:
        :param current_pos:
        :return:
        """

        for existing_rect in self.axis.patches:
            if isinstance(existing_rect, patches.Rectangle) and self._do_rectangles_overlap(rect, existing_rect):
                add_offset, overlap = 0.05, True
                while overlap:
                    pos_copy = self.y_positions.copy()
                    #pos_copy = pos_copy[current_pos]
                    del pos_copy[current_pos]

                    new_offset = existing_rect.get_height() * add_offset
                    new_y = existing_rect.get_height() + new_offset * 2

                    diff = np.abs(np.array(np.abs(pos_copy) - np.abs(new_y)))
                    overlap = np.any(np.array(diff) < np.abs(new_y) * 0.05)

                    if overlap:
                        add_offset += 0.01
                        continue

                    self.y_offsets[current_pos] = new_offset
                    self.y_positions[current_pos] = new_y
        return

    def _add_stats(self):
        """
        Main function that adds the stats to the plot
        :return:
        """
        rects = []
        old_offset = self.y_offsets.copy()
        for _stat in range(len(self.pairs)):
            # Generate Patch
            rect = patches.Rectangle((self.pair_positions[_stat][0], self.y_positions[_stat] + self.y_offsets[_stat]),
                                     self.pair_positions[_stat][1] - self.pair_positions[_stat][0],
                                     0.000000000001,
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor='none')

            # Check if it overlaps with other patches
            self._check_overlap_rectangles(rect, current_pos=_stat)

            # Re-generate Patch avoiding overlap
            rect = patches.Rectangle((self.pair_positions[_stat][0], self.y_positions[_stat] + self.y_offsets[_stat]),
                                     self.pair_positions[_stat][1] - self.pair_positions[_stat][0],
                                     0.000000000001,
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor='none')
            rects.append(rect)

        for _stat, rect in enumerate(rects):
            self.axis.add_patch(rect)
            # Add text in the center of the box
            text_x = (self.pair_positions[_stat][0] + self.pair_positions[_stat][1]) / 2
            text_y = self.y_positions[_stat] + self.y_offsets[_stat] + self.text_offset
            if self.type_test == 'wilcox':
                txt = 'Wilcox'
            elif self.type_test == 'ttest':
                txt = 'ttest'
            else:
                txt = 'Wilcox'
            self.axis.text(text_x, text_y, f'{txt} p = ' + self.pvals[_stat], ha="center", va="bottom", fontsize=self.text_size)

        self._set_ylim() # Adjust limit with the new y_positions
        return


def plot_stats(ax, df, x, y, pairs, offset=0.05, text_offset=0.05, hue=None, text_size=12, test='wilcoxon'):

    plotter = _StatPlotter(ax=ax, data=df, x=x, y=y, hue=hue, pairs=pairs, offset=offset, text_offset = text_offset, text_size=text_size,
                           test=test)

    plotter._get_box_height()
    plotter._get_pair_xpos()
    plotter._get_offset(offset_pct=offset)
    if test == 'wilcoxon':
        plotter._do_wilcoxon()
    elif test == 'ttest':
        plotter._do_ttest()
    else:
        logger.warn('Test not implemented, running the default Wilcoxon')
        plotter._do_wilcoxon()
    plotter._add_stats()

    return


def loading_bar(stop_event):
    """Simulates a long-running task."""
    fun_messages = [
        "Crunching numbers...",
        "Finding hidden patterns...",
        "Making magic happen...",
        "Just a moment...",
        "Hold tight, we're almost there!",
        "Calculating the uncalculable...",
        "Don't worry, be happy!",
        "Loading your results...",
        "Almost done, keep calm!",
        "Success is just around the corner!"
    ]

    with tqdm(desc="Processing", unit='') as pbar:
        last_message_time = time.time()  # Track the last time a message was displayed
        message_interval = 20  # Interval in seconds to display messages

        while not stop_event.is_set():
            pbar.update(1)  # Update the progress bar
            current_time = time.time()

            # Check if 20 seconds have passed since the last message
            if current_time - last_message_time >= message_interval:
                tqdm.write('\n' + random.choice(fun_messages))  # Cycle through messages
                last_message_time = current_time  # Update the last message time

            time.sleep(1)
    return

class _ScStatPlotter:
    """
    Class to add stats to barplots. Ass of now, it cannot handle plots with hue.
    """

    def __init__(self, ax: plt.axis,
                 data: ad.AnnData,
                 x: str,
                 gene: str,
                 ctrl: str,
                 groups: str,
                 offset: float,
                 text_offset: float,
                 step_offset: float,
                 text_size: str,
                 pvals: list,
                 max_offset: float,
                 ):
        """
        Initialise the class
        :param ax: matplotlib axis
        :param data: dataframe with the data that has been plotted
        :param x: column in data which correspond to the x-axis
        :param y:  column in data which correspond to the y-axis
        :param hue: column in data which is use for hue
        :param pairs: pairs to be tested [(catg1, catg2), (catg1, catg3) ...]
        :param offset: offset to add the bar with pvals
        """

        self.axis = ax
        self.data = data
        self.x_axis = x
        self.y_axis = gene
        self.ctrl = ctrl
        self.groups = tuple(groups)  # Make the list unmutable to avoid problems
        self.offset = offset
        self.text_size = text_size
        self.text_offset = text_offset
        self.step_offset = step_offset
        self.max_offset = max_offset

        if pvals is not None:
            self.pvals = [
                str(np.round(p, 2)) if p > 0.05 else
                str(np.round(p, 4)) if p > 0.009 else
                '{:0.2e}'.format(sys.float_info.min if p == 0 else p)
                for p in pvals
            ]

        # Exract X ticks from the plot
        x_ticks = self.axis.get_xmajorticklabels()
        self.x_positions = [_tick.get_position()[0] for _tick in x_ticks]
        self.labels = [_tick.get_text() for _tick in x_ticks]

        return


    def _get_bar_height(self):
        """
        For barplots, extract the height of all the
        bars and save in the heights attribute
        :return:
        """
        height = np.array([np.nanmax(line.get_ydata()) for line in
                           self.axis.lines])  # If showing the error bar, extract the height of this
        if len(height) == 0:
            height = np.array([bar.get_height() for bar in self.axis.patches])
        self.heights = height
        return

    def _do_wilcoxon(self):
        """
        Run Wilcoxon test on the pairs specified by the user
        :return:
        """
        # Get .X + the obs column name with the groups
        df_tmp = pd.concat([self.data.to_df(), self.data.obs[self.x_axis]], axis=1)

        # Precompute masks
        data_mask1 = df_tmp[df_tmp[self.x_axis] == self.ctrl].iloc[:, :-1]  # Last column is the x_axis

        # Calculate wilcox per group
        wilcox_pval = []
        for group in self.groups:
            # Get current group to test against
            data_mask2 = df_tmp[df_tmp[self.x_axis] == group].iloc[:, :-1]

            # Perform Wilcox and Bh correction
            print('Calculating Wilcox with Bh correction')
            stop_event = threading.Event()
            loading_thread = threading.Thread(target=loading_bar, args=(stop_event,))
            loading_thread.start()
            # Dataframe columnwise calculation (Each column is a gene, each row is a cell)
            _, p_vals = mannwhitneyu(data_mask1, data_mask2, axis=0, use_continuity=True, nan_policy='omit')
            pval_adj = false_discovery_control(p_vals)  # Apply Bh correction
            p = pval_adj[np.where(np.array(df_tmp.columns[:-1]) == self.y_axis)].flatten()[
                0]  # Only get the gene of interest

            stop_event.set()
            loading_thread.join()
            wilcox_pval.append(p)  # Round Pval in scientific notation

        self.pvals = [
            str(np.round(p, 2)) if p > 0.05 else
            str(np.round(p, 4)) if p > 0.009 else
            '{:0.2e}'.format(sys.float_info.min if p == 0 else p)
            for p in wilcox_pval
        ]
        return

    def _get_offset(self):  # 5 % offset taking the largest height of both groups
        """
        Determine the y positions where the box with pval will be plotted. Take the highest
        height across two bars and add an offset of 5 %
        :param offset_pct:
        :return:
        """
        offsets = []

        # 1st group
        pair_height = max([self.heights[self.labels.index(self.ctrl)],
                           self.heights[self.labels.index(self.groups[0])]])
        offset = pair_height * self.offset + pair_height
        offsets.append(offset)
        self.offset += self.step_offset
        if len(self.groups) > 1:
            for group in self.groups[1:]:
                if self.offset > self.max_offset:
                    self.offset = self.max_offset
                offset = offset * self.offset + offset
                offsets.append(offset)
                self.offset += self.step_offset
        self.y_positions = offsets
        return

    def _get_pair_xpos(self):
        """
        Return the x pair positions of the box with the pval
        :return:
        """
        pairs_positions = []
        for group in self.groups:
            pair_xpos = [self.x_positions[self.labels.index(self.ctrl)],
                         self.x_positions[self.labels.index(group)]]
            pairs_positions.append(tuple(pair_xpos))
        self.pair_positions = pairs_positions
        return

    def _add_stats(self):
        """
        Main function that adds the stats to the plot
        :return:
        """
        rects = []
        for _stat in range(len(self.groups)):
            rect = patches.Rectangle((self.pair_positions[_stat][0],
                                      self.y_positions[_stat]),
                                     self.pair_positions[_stat][1] - self.pair_positions[_stat][0],
                                     0.0000000000000000001,
                                     linewidth=1, edgecolor='k', facecolor='none')
            rects.append(rect)

        for _stat, rect in enumerate(rects):
            self.axis.add_patch(rect)
            # Add text in the center of the box
            text_x = (self.pair_positions[_stat][0] + self.pair_positions[_stat][1]) / 2
            text_y = self.y_positions[_stat] + self.text_offset
            self.axis.text(text_x, text_y, f'p = ' + self.pvals[_stat], ha="center", va="bottom",
                           fontsize=self.text_size)

        return


def plot_stats_adata(axis: plt.axis,
                     adata: ad.AnnData,
                     x_labels: str,
                     gene: str,
                     ctrl_cond: str,
                     groups_cond: list,
                     groups_pvals: list = None,
                     offset: float = 0.05,
                     text_offset:float = 1e-5,
                     step_offset:float = 0.15,
                     text_size:int = 10,
                     max_offset:float = 0.25):

    plotter = _ScStatPlotter(ax=axis,
                             data=adata,
                             x=x_labels,
                             gene=gene,
                             ctrl=ctrl_cond,
                             groups=groups_cond,
                             offset=offset,
                             text_offset=text_offset,
                             step_offset=step_offset,
                             text_size=text_size,
                             pvals=groups_pvals,
                             max_offset=max_offset)

    plotter._get_bar_height()  # Calculate the height
    plotter._get_pair_xpos()  # Calculate the positions in the X axis
    plotter._get_offset()  # Calculate the offset

    if groups_pvals is None:
        plotter._do_wilcoxon()

    plotter._add_stats()


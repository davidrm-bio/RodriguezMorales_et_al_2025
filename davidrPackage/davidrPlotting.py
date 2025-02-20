#!/usr/bin/env python

"""
Description:  Module for generating Plots for
scRNA or snRNA and Spatial visium data

Author: David Rodriguez Morales
Date Created: 31-01-2024
Date Modified: 16-03-24
Version: 2.0
Python Version: 3.11.8
"""
# Libraries
import os
import warnings
from datetime import datetime

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from adjustText import adjust_text
from typing import Union, Optional

from davidrUtility import get_subplot_shape, ExtractExpression, axis_format, remove_extra, iOn, set_plt_theme, config_verbose
import davidrExperimental
# Global configuration


class CustomWarning(UserWarning):
    pass

warnings.filterwarnings("ignore")  # Scanpy creates many warnings
warnings.filterwarnings('default', category=CustomWarning)
set_plt_theme()
iOn()
logger = config_verbose(True)

# Quality Control Plots
def generate_violin_plot(adata: ad.AnnData,
                         title: str = 'ViolinPlots - Quality Metrics',
                         path: str = None,
                         filename: str = 'ViolinPlots.png',
                         col_obs: list = ['total_counts',
                                          'n_genes_by_counts',
                                          'pct_counts_mt']
                         ) -> None:
    """**Violin Plots showing basic QC stats**

    Generate ViolinPlots to show the distribution of total counts, counts per gene
    and percentage of mitochondrial genes in an anndata object. The following should have
    been run before:
    ``$adata.var['mt'] = adata.var_names.str.startswith('mt')``  # mt for mouse and MT for human
    ``$sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, log1p=True)``

    :param adata: anndata object
    :param title: Title of the Plot. Default ViolinPlot
    :param path: path to figure folder. Default ./ (current folder)
    :param filename: name of the file with the plot. Default ViolinPlot.png
    :param col_obs: obs column name to plot
    :return:
    """

    fig, ax = plt.subplots(1, 3, figsize=(18, 9))
    ax1 = plt.subplot(1, 3, 1)
    sc.pl.violin(adata, col_obs[0], stripplot=True, show=False, ax=ax1)

    ax2 = plt.subplot(1, 3, 2)
    sc.pl.violin(adata, col_obs[1], stripplot=True, show=False, ax=ax2)
    ax2.set_ylim(bottom=0, top=10000)

    ax3 = plt.subplot(1, 3, 3)
    sc.pl.violin(adata, col_obs[2], stripplot=True, show=False, ax=ax3)

    plt.suptitle(title, fontsize=30)

    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    return


def plot_markers(adata: ad.AnnData,
                 list_of_markers: list = None,
                 dict_of_markers: dict = None,
                 typePlot: str = 'umap',
                 groupby: str = 'annotation',
                 path: str = None,
                 filename: str = None,
                 layer: str = None) -> None:
    """**Plots showing Markers genes**

    This function plot marker genes. There are two options available:

    * UMAP
        * If a list of markers is provided, all the genes will be plotted in a plot with subplots of UMAPs
        * If a dictionary of list of markers is provided, a plot will be generated for each key of the dictionary
    * DotPlot
        * If a list of markers is provided, all the genes will be plotted in a plot.
        * If a dictionarz is provided, all the genes will be plotted and grouped by key.

    Plots will be saved in a subfolder called 'Markers_UMAP' in the specified path. Two type of inputs are accepted
    list of markers or a dictionary of  a list of markers

    :param adata: anndata object
    :param list_of_markers: list of marmers to be used for plotting.
    :param dict_of_markers: dictionary of a list of markers to be used for plotting.
    :param typePlot: type of plot. Suported umap and dotplot
    :param groupby: obs column name to group in the dotplot
    :param path: path where to save the plots.
    :param filename: name of the file with the UMAP plot
    :param layer: layer to use for plotting
    :return:
    """
    # Quality Control -- Check the provided arguments are okay
    if list_of_markers is None and dict_of_markers is None:
        assert 'Please provide a list of markers or a dicitionary of a list of markers'
    if list_of_markers is not None and dict_of_markers is not None:
        assert 'Please either provide a list or a dictionary but not both'
    if typePlot.upper() == 'DOTPLOT' and groupby is None:
        assert 'Please provide an obs column in groupby for the dotplot'

    # If a path is provided, set-up to save plots
    if path is not None and filename is not None:
        try:
            os.makedirs(os.path.join(path, 'Markers_UMAP'))  # Save Markers in a subfolder in the final path
        except FileExistsError:
            pass
        path = os.path.join(path, 'Markers_UMAP')  # Update label

    # Filter to keep genes present actually in the anndata object
    if list_of_markers is not None:
        markers_in_data = [m for m in list_of_markers if m in adata.var_names]
    if dict_of_markers is not None:
        markers_in_data = {}
        for key, value in dict_of_markers.items():
            markers_in_data[key] = [m for m in value if m in adata.var_names]

    # Plotting
    if type(markers_in_data) == 'list':
        if typePlot.upper() == 'UMAP':
            with plt.rc_context():
                sc.pl.umap(adata, color=markers_in_data, vmin=0, vmax="p99.2", sort_order=True, frameon=False,
                           cmap="Reds", show=False, layer=layer)
        else:  # DotPlot
            with plt.rc_context():
                sc.pl.dotplot(adata, groupby=groupby, var_names=markers_in_data, show=False, layer=layer)
        # Save plots
        if path is not None and filename is not None:
            plt.savefig(os.path.join(path, filename), bbox_inches='tight')

    else:
        for key, val in markers_in_data.items():
            if typePlot.upper() == 'UMAP':
                with plt.rc_context():
                    sc.pl.umap(adata, color=val, vmin=0, vmax="p99.2", sort_order=True, frameon=False,
                               cmap="Reds", show=False, layer=layer)
            else:  # Dotplot
                with plt.rc_context():
                    sc.pl.dotplot(adata, groupby=groupby, var_names=markers_in_data, show=False, layer=layer)
            # Save Plots
            if path is not None and filename is not None:
                filename = key + '_Marker.png'
                plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    return


def FindTopMarkers(anndata: ad.AnnData,
                   annot: str = 'leiden',
                   top_n: int = 5,
                   layer: str = None,
                   fig_path: str = None,
                   table_path: str = None,
                   suffix: str = '',
                   mouse: bool = True,
                   remove_mt: bool = True) -> None:
    """**Find Top Markers for clusters**

    Find the top genes (markers) for each cluster. Mitochondrial genes can be excluded from the analysis.
    This function can be used to visualise in a DotPlot the top 5 genes (Default) in
    each cluster. If a figure path is provided the DotPlot will be saved under the specified
    folder. If a table path is provided an Excel file with the full list of genes will be generated
    under the specified folder. Genes included pass the default threshold
    of 0.05 (P-value adjusted) and 0.25 (Log-fold change).

    The user can specify a suffix for the filenames. Default is FindMarkers-{DotPlot/Table}-{suffix}.{svg/xlsx}

    :param anndata: anndata object with log normalise counts in .X attribute
    :param annot: obs column with cluster information. Default 'leiden'
    :param top_n: number of genes to plot. Default 5
    :param layer: layer with the log normalise counts. Default None
    :param fig_path: path to folder where to save DotPlot. If provided, DotPlot will not be shown. Default None
    :param table_path: path to folder where to save Excel with Marker list. Default None
    :param mouse: whether the input is mouse or human data. Use to remove Mt-genes. Default True (mouse)
    :param remove_mt: whether to remove or not mitochondrial genes before DGE. Default True
    :param suffix:
    :return:
    """
    # Update .X with the layer (Optional)
    adata = anndata.copy()
    if layer:
        adata.X = anndata.layers[layer].copy()

    # Remove MT genes
    if remove_mt:
        print('\n# ', datetime.now(), ' - Removing Mitochondrial genes...')
        if mouse:
            mt_gene = 'mt-'  # mouse genes are lowercase
        else:
            mt_gene = 'MT-'  # human genes are uppercase
        adata.var['mt'] = adata.var_names.str.startswith(mt_gene)
        adata = adata[:, ~adata.var['mt'].values]

    # DGE per cell type
    print('# ', datetime.now(), ' - Running DGE analysis...')
    key = 'rank'
    sc.tl.rank_genes_groups(adata, groupby=annot, method='wilcoxon',
                            key_added=key, use_raw=False, tie_correct=True)

    # Plotting
    print('# ', datetime.now(), ' - Creating the Plot...')
    sc.pl.rank_genes_groups_dotplot(adata, groupby=annot,
                                    n_genes=top_n, show=False, swap_axes=True,
                                    dendrogram=False, key=key)

    if fig_path is not None:
        print('# ', datetime.now(), ' - Saving the plot...')
        plt.savefig(os.path.join(fig_path, 'FindMarkers-DotPlot-{}.svg'.format(suffix)),
                    bbox_inches='tight', dpi=150)
    if table_path is not None:
        print('# ', datetime.now(), ' - Saving the Table with Markers...')
        sc.get.rank_genes_groups_df(adata, group=None,
                                    key=key, pval_cutoff=0.05,
                                    log2fc_min=0.25).to_excel(os.path.join(table_path,
                                                                           'FindMarkers-Table-{}.xlsx'.format(suffix)),
                                                              index=False)
    return None


# Improvement of Basic plotting functions from scanpy
def split_umap(anndata: ad.AnnData,
               split_by: str,
               ncol: int = 4,
               nrow: int = None,
               path: str = None,
               filename: str = 'UMAP.svg',
               figsize: tuple = (10, 8),
               **kwargs) -> None:
    """**UMAP split in categories**

    This function takes an anndata and a categorical column in .obs and generate a plot of
    subplots of umaps highlighting the different categories of the .obs column

    :param anndata: anndta object
    :param split_by: .obs column with categorical values
    :param ncol: number of subplots per row
    :param nrow: number of rows
    :param path: path to save the plot
    :param filename: filename of the plot
    :param figsize: size of the figure
    :param kwargs: additional arguments for ``sc.pl.umap()``
    :return: None
    """
    assert anndata.obs[split_by].dtypes == 'category', 'Not a categorical column'

    # Get all the categories
    categories = anndata.obs[split_by].cat.categories

    # Set-Up
    if nrow is None:
        nrow = int(np.ceil(len(categories) / ncol))
    extra = nrow * ncol - len(categories)
    figs, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    # Plotting
    for idx, cat in enumerate(categories):
        sc.pl.umap(anndata, color=split_by, groups=[cat],
                   ax=axs[idx],
                   show=False, title=cat, **kwargs)
        axs[idx].get_legend().remove()
        axs[idx].spines[['right', 'top']].set_visible(False)
        axs[idx].set_xlabel('UMAP1', loc='left', fontsize=10)
        axs[idx].set_ylabel('UMAP2', loc='bottom', fontsize=10)
    if extra != 0:
        for jdx in range(extra):
            axs[idx + 1 + jdx].set_visible(False)
    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    return


def pl_umap(adata: ad.AnnData,
            color: str,
            split_by: Optional[str] = None,
            ncols: int = 4,
            title_font: dict = {'size': 18, 'weight': 'bold'},
            figsize: tuple = (12, 6),
            common_legend=False,
            vmax: Optional[float] = None,
            spacing: tuple = (.3, .2),
            path:  Optional[str]  = None,
            filename: str = 'Umap.svg',
            show: bool = True,
            **kwargs) -> plt.Axes:

    """ Create UMAP Plot

    This function builds on `sc.pl.umap()` and add extra functionalities like
    splitting by a categorical column in .obs

    :param adata: anndata object
    :param color: .obs column or .var_names value
    :param split_by: categorical .obs column
    :param ncols: number of columns per row
    :param figsize: figure size (width, heigh) in Inches
    :param common_legend: set a common legend when plotting multiple values, it will automatically scale if plotting continuous values like
                          gene expression if vmax is not specified
    :param title_font: font properties of the title for each subplot
    :param vmax: maximum value for continuos data
    :param spacing: spacing between subplots (height, width) padding between plots
    :param path: path to save plot
    :param filename: filename of the plot
    :param show: when set to False the matplotlib axes will be returned
    :param kwargs: additional parameters pass to ``sc.pl.umap()``
    :return: matplotlib axis
    """

    # TODO work on the efficiency (faster plotting)
    # TODO for some reason it does not work in jupyter notebook?

    # adata = adata.copy()  # We copy to not modify input

    # We consider that the input is always a list;
    if isinstance(color, str):  # If we only provide a string, convert to list
        color = [color]

    # If a .obs column is provided plot will have as many subplots as categories
    ncatgs = 1
    if split_by is not None:
        assert adata.obs[split_by].dtype == 'category', 'split_by is not a categorical column'
        ncatgs = len(adata.obs[split_by].unique())
        catgs = adata.obs[split_by].unique()
        nrows, ncols, nExtra = get_subplot_shape(ncatgs, ncols)
    else: # Otherwise, we have as many subplots as things to plot (len(colors))
        nrows, ncols, nExtra = get_subplot_shape(len(color), ncols)

    # Scale vmax when setting a common legend
    vmax_genes = vmax
    if vmax is None and common_legend is True:
        # We could be plotting different genes
        if len(color) > 1:
            genes = [val for val in color if val in adata.var_names]
            expr = ExtractExpression(adata, features=genes, out_format='wide')  # Extract the expression
            vmax_genes = expr.apply(lambda x: np.percentile(x, 99.2), axis=0).mean()  # the vmax is the mean of 99.2 percentile across genes
        else:  # We could also be plotting one gene splitting by categories
            if split_by is not None and color[0] in adata.var_names:
                def q99_2(x):
                    return x.quantile(0.992)
                # We plot one value but split by something
                expr = ExtractExpression(adata, features=color[0], groups=split_by, out_format='wide')  # Extract the expression
                vmax_genes = expr.groupby(split_by).agg(q99_2).mean()[0] # the vmax is the mean of 99.2 percentile across categories

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Generate the Plot                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], left=.1)  # Spacing between subplots
    cat, cb_loc, cont = None, 'right', 0
    # 1st Case; - We do not split by categories and only 1 thing is plotted
    if ncatgs == 1:
        if len(color) == 1:
            color = color[0] # Color is always a list
            sc.pl.umap(adata, color=color, ax=axs, vmax=vmax, **kwargs)
            axs.set_title(color, fontdict=title_font)
            axis_format(axs)
        # 2nd Case; We do not split by categories and multiple values are plotted
        else:
            axs = axs.flatten()
            for idx, val in enumerate(color):
                if common_legend:
                    # Remove the legend from all subplots except the last one per row
                    if cont != ncols - 1 and idx != len(color) - 1:  # We remove legend from all subplots except last column per row
                        if val in adata.obs.columns:  # Is color in .obs?
                            cat = adata.obs[val].dtype.name  # It can be continuous or categorical
                        if cat != 'category':
                            # Is continuous --> Remove color bar
                            cb_loc = None
                        cont += 1
                    else:
                        # Entered when we are in the last column per row
                        cat, cb_loc, cont = None, 'right', 0

                # If value to plot is a gene, update vmax (if common legend true) otherwise use the vmax provided by user
                vmax = vmax_genes if val in adata.var_names else vmax
                sc.pl.umap(adata, color=val, ax=axs[idx], colorbar_loc=cb_loc, vmax=vmax, **kwargs)
                axis_format(axs[idx])
                axs[idx].set_title(val, fontdict=title_font)
                remove_extra(nExtra, nrows, ncols, axs)

                # Never remove categorical when plotting several values without splitting by
                # if common_legend and cat == 'category':
                #    axs[idx].get_legend().remove()  # Remove legend for categorical values

    else:
        # 3rd Case Multiple Values are plotted and splitting by categories
        # 3rd case plot each category per row
        assert len(color) == 1, 'Not Implemented'

        # 4th Case; One value is plotted splitting by categories
        color = color[0]  # color is always converted to list
        axs = axs.flatten()
        for idx in range(ncatgs):
            adata_subset = adata[adata.obs[split_by] == catgs[idx]]
            if common_legend:
                # Remove the legend from all subplots except the last one per row
                if cont != ncols - 1 and idx != ncatgs - 1:
                    if color in adata.obs.columns:  # Is color in .obs?
                        cat = adata.obs[color].dtype.name  # It can be continuous or categorical
                    if cat != 'category':
                        # Is continuous --> Remove color bar
                        cb_loc = None
                    cont += 1
                else:
                    # Entered when we are in the last column per row
                    cat, cb_loc, cont = None, 'right', 0

            # If value to plot is a gene, update vmax (if common legend true) otherwise use the vmax provided by user
            vmax = vmax_genes if color in adata.var_names else vmax

            sc.pl.umap(adata_subset, color=color, ax=axs[idx], colorbar_loc=cb_loc, vmax=vmax, **kwargs)
            axis_format(axs[idx])
            remove_extra(nExtra, nrows, ncols, axs)
            if common_legend and cat == 'category':
                axs[idx].get_legend().remove()  # Remove legend for categorical values except last column

            # Minimal Text when Splitting by categories
            axs[idx].set_title(catgs[idx], fontdict=title_font)
            fig.supylabel(color, fontsize=23, fontweight='bold')

    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    if show:
        return
    else:
        return axs


# Functions for Spatial Transcriptomics Data
def spatial_split(anndata: ad.AnnData,
                  cat_col: str,
                  fig_path: str = None,
                  sp_size: float = 1.5,
                  ncol: int = 4,
                  filename: str = 'Spatial.svg',
                  figsize=(10, 10),
                  title='Group ',
                  title_font={'size': 12, 'weight': 'bold'},
                  show=True,
                  **kwargs) -> Union[plt.Axes, None]:
    """**Spatial split in categories**

    Plot categorical observation in Spatial Embedding splitting individual categories
    :param anndata: anndata object
    :param cat_col: obs column name with categorical values
    :param fig_path: path to save figure
    :param sp_size: size of the dots
    :param ncol: number of columns in the plot
    :param filename: filename of the plot
    :param figsize: size of the figure
    :param title: title for subplots. Format will be "title [current group]"
    :param title_font: properties of the title font for each subplot
    :param show: if set to True returns axes
    :return:
    """
    assert anndata.obs[cat_col].dtypes == 'category', 'Not a categorical column'

    # Compute number of cols and rows needed to plot categories
    n_catgs = len(anndata.obs[cat_col].unique())
    nrow, ncol, extra = get_subplot_shape(n_catgs, ncol)

    # Generate the Plot
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    ax = axs.flatten()
    for idx, group in enumerate(anndata.obs[cat_col].unique()):
        sc.pl.spatial(anndata, ax=ax[idx],
                      title=title,
                      groups=[group],  # Only plot 1 category
                      color=cat_col,
                      size=sp_size,
                      **kwargs)
        ax[idx].set_title(title + group, fontdict=title_font)
        ax[idx].get_legend().remove()  # Remove legend
        axis_format(ax[idx], txt='SP')
        remove_extra(extra, nrow, ncol, axs)  # Remove extra subplots

        if fig_path is not None:
            plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    if show:
        return
    else:
        return axs


def plot_slides(adata: ad.AnnData,
                color: str,
                obs_col: str = 'sample',
                ncols: int = 4,
                sp_size: float = 1.5,
                fig_path: str = None,
                filename: str = 'Spatial.svg',
                common_legend: bool = True,
                order: list = None,
                figsize: tuple = (15, 8),
                layer: str = None,
                img_key: str = 'hires',
                title_fontsize: int = 15,
                title_fontweight: str = None,
                select_samples: Union[list, str] = None,
                show: bool = True,
                minimal_title: bool = True,
                vmax: float = None,
                verbose: bool = True,
                spacing: tuple = (.3, .2),
                **kwargs) -> plt.Axes:
    """
    **Plot multiple visium slides**

    Plot a feature in .var_names or a column from .obs in
    multiple visium slides.

    :param adata: anndata object
    :param color:  .var_names or .obs column to plot
    :param obs_col: .obs column containing Batch/Sample Information. This column should have the same names system use
                    to save the spatial images in ``adata.uns['Spatial'].keys()``. (Default is **sample**)
    :param ncols:  number of subplots per row. (Default is **4**)
    :param sp_size: size of the dots. (Default is **1.5**)
    :param fig_path: path to save the plot.
    :param filename: filename of the plot. Specify also the file format (E.g., .png, .svg, .pdf, etc.)
    :param common_legend: if set to true only the legend of the last column will be shown. Otherwise, the legend of
                          all the subplots will be shown. (Default is **True**)
    :param order: provide a list with the order of the slides to show. If not set the ``obs_col`` will be sorted
    :param figsize: size of the subplots
    :param layer: layer to use to plot data. If not specified, '.X' will be used.
    :param img_key: image key to use for plotting (hires or lowres). (Default is **hires**)
    :param title_fontsize: fontsize of the title for the subplots
    :param title_fontweight: change fontweight of the title
    :param select_samples: list with a subset of samplename that want to be plotted
    :param show: if False, return axs
    :param minimal_title: if set to true only the sample name will be shown as title, otherwise title + color
    :param vmax: maximum value for continus values (e.g., expression). If common legend is set to True and vmax
                 is not specified, it will be automatically computed taking the p99.2 expression value across
                 all subplots
    :param verbose: show a progress bar when plotting multiple slides
    :param kwargs: additional arguments for the function ``scanpy.pl.spatial()``
    :param spacing: spacing between subplots (height, width) padding between plots
    :return: a matplotlib axes object
    """
    from tqdm import tqdm
    # TODO Consider the case where we only have 1 sample
    # TODO change the minimal text as default (i.e, remove the option and set it as in pl_umap)
    if select_samples is not None:
        if type(select_samples) is str:
            select_samples = [select_samples]
        adata = adata[adata.obs[obs_col].isin(select_samples)].copy()
        adata.obs[obs_col] = pd.Categorical(adata.obs[obs_col].astype(str))

    # Define the number of rows base on the desired number of columns
    n_samples = len(adata.obs[obs_col].unique())
    nrows, ncols, extras = get_subplot_shape(n_samples, ncols)

    # Control the order of the samples
    show_order = order
    if order is None:
        show_order = sorted(adata.obs[obs_col].unique())  # If no order is provided sort the samples

    # Assume we plot non-categorical values
    cat, cb_loc, cont = None, 'right', 0

    # Scale values if common legend
    if vmax is None and common_legend is True:
        if color in adata.var_names:
            expr = ExtractExpression(adata, color)
            vmax = np.percentile(expr['expr'], 99.2)
        if color in adata.obs.columns and adata.obs[color].dtype.name != 'category':
            vmax = np.percentile(adata.obs[color], 99.2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1], left=.05)  # Spacing between subplots
    axs = axs.flatten()
    for idx, sample in tqdm(enumerate(show_order), desc='Slide ', disable=not verbose, total=len(show_order)):
        sdata = adata[adata.obs[obs_col] == sample]

        if common_legend:
            # Remove the legend from all subplots except the last one per row
            if cont != ncols - 1 and idx != n_samples - 1:
                if color in adata.obs.columns:  # Is color in .obs?
                    cat = adata.obs[color].dtype.name  # It can be continuous or categorical
                if cat != 'category':
                    # Is continuous --> Remove color bar
                    cb_loc = None
                cont += 1
            else:
                # Entered when we are in the last column per row
                cb_loc = 'right'
                cont = 0
                cat = None

        # Main Plotting function, based on Scanpy
        sc.pl.spatial(sdata, ax=axs[idx], img_key=img_key,
                      color=color,
                      library_id=sample,
                      size=sp_size,
                      colorbar_loc=cb_loc,
                      layer=layer,
                      vmax=vmax,
                      show=False,
                      **kwargs)

        if common_legend and cat == 'category':
            axs[idx].get_legend().remove()  # Remove legend for categorical values

        # Modify axis
        title_color = '' if color is None else color
        """    
        if color == None:
            title_color = ''
        else:
            title_color = color
        """
        if minimal_title:
            axs[idx].set_title(sample, fontsize=title_fontsize, fontweight=title_fontweight)
            # plt.suptitle(color, fontsize=23, fontweight='bold')
            fig.supylabel(color, fontsize=23, fontweight='bold')
        else:
            axs[idx].set_title(sample + '\n' + title_color, fontsize=title_fontsize, fontweight=title_fontweight)

        axis_format(axs[idx], txt='SP')
        remove_extra(extras, nrows, ncols, axs)  # Remove extra subplots
        if fig_path is not None:
            plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    if show:
        return
    else:
        return axs


# Utility Plots
def create_dotplot(df_size: pd.DataFrame,
                   df_color: pd.DataFrame,
                   figsize: tuple = (6, 8),
                   cmap: str = 'RdBu_r',
                   vmax: float = None,
                   vcenter: float = 0,
                   vmin: float = None,
                   legend_size_title='P-values',
                   legend_color_title='Difference (Old - Young)',
                   title='',
                   xlabel='',
                   ylabel='',
                   path=None,
                   filename='DotPlot.svg',
                   show=True,
                   ) -> None:
    """Create a dotplot from two dataframes

    This functions creates a Dotplot from two dataframes. One will be used
    to determine the size and the other the color. Right now only P-values are accepted as input for
    the dot size.

    :param df_size: dataframe where values determine the size. Right now is optimised to only take P-values
    :param df_color: dataframe where values determine the color
    :param figsize: plot height and width in inches
    :param cmap: colormap
    :param vmax:  upper limit of the color map
    :param vmin: lower limit of the colot map
    :param legend_size_title: title for the dot size
    :param legend_color_title: title for the colorbar
    :param title: title of the plot
    :param xlabel: xlabel of the plot
    :param ylabel: ylabel of the plot
    :param path: path to save the plot
    :param filename: filename
    :return:
    """

    assert df_size.max().max() <= 1 and df_size.min().min() >= 0, 'Size dataframe should be P-values'
    # Compute Max and Minimum Values for the color bar
    if vmax is None:
        vmax = df_color.max(axis=None)  # axis = None in pandas 2.2 agregates
        if ~isinstance(vmax, int):  # For previous versions of pandas
            vmax = df_color.max().max()
        vmax = np.round(vmax, 2)
    if vmin is None:
        vmin = df_color.min(axis=None)  # axis = None in pandas 2.2 agregates
        if ~isinstance(vmin, int):  # For previous versions of pandas
            vmin = df_color.min().min()
        vmin = np.round(vmin, 2)

    # Compute Dot Sizes
    dot_size = pd.DataFrame(np.full(df_size.shape, 50),
                            columns=df_size.columns,
                            index=df_size.index)  # Dot Size 50 (Pval > 0.05)
    dot_size[df_size < .05] = 250  # Dot Size 250 (Pval < 0.05)

    # Plot::Start
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, .25], height_ratios=[1, .15], wspace=.08, hspace=.05)

    # Loop over each value to add the dot
    ax = fig.add_subplot(gs[:, 0])
    x_nchar = 0
    for row in df_size.index:
        for col in df_size.columns:
            size = dot_size.loc[row, col]
            color = df_color.loc[row, col]
            ax.scatter(x=[row], y=[col], s=size, c=[[color]],
                       cmap=cmap, edgecolor='black', linewidth=1,
                       vmax=vmax, vmin=vmin)

        x_nchar = x_nchar if x_nchar > len(row) else len(row)

    x_rotation = 0 if x_nchar < 5 else 90
    plt.xticks(rotation=x_rotation, fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(False)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)  # Make sure to plot all the spines (Default davidrPackage deactivates it)

    # Add some padding around the axis limits
    ax.set_xlim(-0.5, len(df_size.index) - 0.5)  # Adds padding to the x-axis
    ax.set_ylim(-0.5, len(df_size.columns) - 0.5)  # Adds padding to the y-axis

    # Set the labels
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20, fontweight='bold')

    # Add Legend for Dot Size
    size_legend_values = [50, 250]  # Example sizes for the legend
    size_legend_labels = ['>0.05', '<0.05']
    handles = [mlines.Line2D([], [],
                             linestyle='none', marker='o',
                             markersize=np.sqrt(s),
                             color='gray') for s in size_legend_values]

    # Dedicate a portion of the plotting space to the legend
    axl = fig.add_subplot(gs[0, 1])
    axl.axis('off')
    size_legend = axl.legend(handles, size_legend_labels,
                             title=legend_size_title,
                             loc='center left',
                             frameon=False,
                             title_fontproperties={'weight': 'bold'},
                             fontsize=15, ncols=1)

    # Color legend
    axcb = fig.add_subplot(gs[1, 1])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) # Norm based on normalized color range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    color_legend = plt.colorbar(sm, cax=axcb, pad=0.5, fraction=.04, orientation='horizontal' )
    color_legend.set_label(legend_color_title, fontsize=15, fontweight='bold')

    # Manually set tick locations at vmin, vcenter, and vmax
    color_legend.set_ticks([vmin, vcenter, vmax])

    # Optionally set custom tick labels
    tick_labels = [f'{tick:g}' for tick in [vmin, vcenter, vmax]]  # This will format the tick labels
    color_legend.set_ticklabels(tick_labels)

    # axcb.add_artist(size_legend)  # Add size legend manually
    axcb.grid(False)

    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    if show:
        return
    else:
        return {'main_ax': ax, 'size_legend_ax': axl, 'color_legend_ax': axcb}


def volcano_plot(dge,
                 lfc_col: str = 'logfoldchanges',
                 pval_col: str = 'pvals_adj',
                 gene_col: str = 'names',
                 fig_path: str = None,
                 filename: str = 'Volcano.svg',
                 pval_lim: float = 2e-10,
                 lfc_lim: tuple = (-10, 10),
                 title: str = '',
                 figsize: tuple = (18, 9),
                 mygenes: list = None,
                 lfc_cut: float = 0.25,
                 pval_cut: float = 0.05,
                 clean: bool = True,
                 dot_size: float = 2.5,
                 topN: int = 10,
                 textprops: dict = {'weight': 'bold', 'size': 10},
                 **kwargs):
    """**Generate a volcano plot.**

    Genes will be colored differently depending on the p-value (Pval) and logfoldchange (LFC)

    * Genes Pval < pval_cut & LFC > lfc_cut: Red
    * Genes Pval < pval_cut & LFC < lfc_cut: Blue
    * Genes Pval > pval_cut & LFC > lfc_cut: Green
    * Genes Pval > pval_cut & LFC < lfc_cut: Gray

    If not genes are provided (with the mygenes argument) the top 10 genes with highest and lowest LFC that are
    significant will be marked.

    :param dge: pandas dataframe with DGE. Should have at least 3 columns (Genes, Pvalue, Logfoldchange)
    :param lfc_col: name of the column that has the LFC
    :param pval_col: name of the column that has the Pvals
    :param gene_col: name of the column that has the gene names
    :param fig_path: path where to save the figure
    :param filename: name of the file
    :param pval_lim: Y axis limit. Genes with a < p-value will be set to this value
    :param lfc_lim: X axis limit. Genes with a > LFC will be ignored
    :param title: a text to add as the title of the plot. Default nothing
    :param figsize: size of the plot
    :param lfc_cut: significance threshold for the LFC. Default 0.25
    :param pval_cut: significance threshold for the P-value. Default 0.05
    :param mygenes: list of genes to be indicated (adding text)
    :param clean: remove genes with Pval == 1 and LFC > lfc_lim
    :param dot_size: size of the dots
    :param topN: if mygenes is None. The top 10 positive and negative genes are plotted
    :param textprops: properties of the gene labels (See `plt.text <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html>`_)
    :return:
    """
    dge = dge.copy()  # Do not Modify input

    # Replace Pvals & LFC greater than limit to the limit
    dge[pval_col][dge[pval_col] < pval_lim] = pval_lim

    assert lfc_lim[0] < lfc_lim[1], f'{lfc_lim[0]} cannot be greater than {lfc_lim[1]}'
    dge[lfc_col][dge[lfc_col] < lfc_lim[0]] = lfc_lim[0]
    dge[lfc_col][dge[lfc_col] > lfc_lim[1]] = lfc_lim[1]

    if clean:
        # Remove Genes with P adjusted == 1 (Not Informative)
        dge = dge[dge[pval_col] < 1]
        dge = dge[dge[lfc_col] > lfc_lim[0]]
        dge = dge[dge[lfc_col] < lfc_lim[1]]

    # Define 3 Categories: LFC > lfc_cut; Pval < pval_cut & combination
    pvals = dge[pval_col].to_numpy()
    lfcs = dge[lfc_col].to_numpy()
    genes = dge[gene_col].to_numpy()
    cat1 = np.where((pvals < pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))
    cat2 = np.where((pvals < pval_cut) & (lfcs > -lfc_cut) & (lfcs < lfc_cut))
    cat3 = np.where((pvals > pval_cut) & ((lfcs > lfc_cut) | (lfcs < -lfc_cut)))

    # Generate Plot
    # Create scatter Plot
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.scatter(lfcs, -np.log10(pvals), color='gray', alpha=0.7, label='NS', s=dot_size, rasterized=True)
    axs.scatter(lfcs[cat1], -np.log10(pvals[cat1]), color='tomato', alpha=0.7, label='FDR & log2FC', s=dot_size,
                rasterized=True)
    axs.scatter(lfcs[cat2], -np.log10(pvals[cat2]), color='cornflowerblue', alpha=0.7, label='FDR', s=dot_size,
                rasterized=True)
    axs.scatter(lfcs[cat3], -np.log10(pvals[cat3]), color='limegreen', alpha=0.7, label='log2FC', s=dot_size,
                rasterized=True)
    axs.spines[['top', 'right']].set_visible(False)
    axs.grid(False)

    # Add significant lines
    axs.axhline(-np.log10(pval_cut), color='black', linestyle='--', alpha=0.8)
    axs.axvline(-lfc_cut, color='black', linestyle='--', alpha=0.8)
    axs.axvline(lfc_cut, color='black', linestyle='--', alpha=0.8)

    topPos = dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] > lfc_cut)].sort_values(lfc_col, ascending=False)[
        gene_col].head(topN).tolist()
    topNeg = dge[(dge[pval_col] < pval_cut) & (dge[lfc_col] < -lfc_cut)].sort_values(lfc_col, ascending=True)[
        gene_col].head(topN).tolist()
    texts = []
    for x, y, l in zip(lfcs, pvals, genes):
        if mygenes is None:
            if l in topPos:
                texts.append(plt.text(x, -np.log10(y), l, ha='center', va='center', fontdict=textprops))
            if l in topNeg:
                texts.append(plt.text(x, -np.log10(y), l, ha='center', va='center', fontdict=textprops))
        else:
            if l in mygenes:
                texts.append(plt.text(x, -np.log10(y), l, ha='center', va='center', fontdict=textprops))
    adjust_text(texts,
                arrowprops=dict(arrowstyle="-", color='k', lw=0.5), **kwargs)

    # Add Axis labels, Legend, & Title
    axs.set_xlabel('Log2FC', fontsize=20, fontweight='bold')
    axs.set_ylabel(f'-log10(FDR)', fontsize=20, fontweight='bold')
    axs.set_title(title, fontsize=25, fontweight='bold')
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, ncols=2)

    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return

def BarPlot_cell_contribution(anndata, fig_path, groups=['annotation', 'sample']):  # todo improve
    """
    Barplot showing the contribution of each celltype. Should be used on the integrated object with all the sample.
    Groups is a list that specify the obs column name with sample names (batches) and cell types (annotation).
    The final Plot will be called  'BarPlot-ContributionSample.svg'
    :param anndata: integrated anndata object
    :param fig_path: path to figure folder
    :param groups: list with obs column with sample and celltype. Default ['annotation', 'sample']. It should be first the cell type column and second the sample column
    :return:
    """
    df_obs = anndata.obs
    groupby = anndata.obs.groupby(groups)
    df_plot = pd.DataFrame([])

    for name, group in groupby:
        total_n = df_obs[df_obs['sample'] == name[1]].shape[0]
        try:
            contribution = group.shape[0] / total_n * 100
        except ZeroDivisionError:
            contribution = 0
        row = [name[0], name[1], contribution]
        df_tmp = pd.DataFrame(row).T
        df_tmp.columns = ['Annotation', 'Sample', 'Contribution (%)']
        df_plot = pd.concat([df_plot, df_tmp])

    # Create Bar Plot
    g = sns.catplot(df_plot,
                    y='Contribution (%)',
                    x='Sample',
                    col='Annotation',
                    col_wrap=5,
                    kind='bar',
                    sharey=True)

    g.set_titles(fontsize=25, fontweight='bold')
    g.set_xlabels(fontsize=18)
    g.set_ylabels(fontsize=18)
    g.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'BarPlot-ContributionSample.svg'))
    return df_plot  # TODO IMp


def gsea_dotplot(df,  # TODO add an argument to specify how long the lines have to be
                 title: str = None,
                 top_terms: int = 10,
                 pval_col: str = 'Adjusted P-value',
                 term_col: str = 'Term',
                 overlap_col: str = 'Overlap',
                 sort_by: str = 'Combined Score',
                 pval_cutoff: float = 0.05,
                 preprocess: bool = False,
                 gseapy: bool = True,
                 figsize: tuple = (4, 10),
                 legend_size_title: str = 'Gene Overlap',
                 cmap='Reds_r',
                 dot_size_scale: float = 1000,
                 path: str = None,
                 filename: str = 'Dotplot.svg',
                 show=True,
                 ) -> plt.Axes:
    """**Generate a DotPlot for GSEA results**

    The default arguments of this function assumes that results where generated using the
    `GSEAPY package <https://gseapy.readthedocs.io/en/latest/index.html>`_. The top 10 terms will be
    plotted sorting the  combined Score. Optionally, if the pre-process is set to True, the terms
    will be filtered base on the pval_col and pval_cutoff.

    :param df: dataframe containing gese set enrichment analysis results
    :param title: title for the plot
    :param top_terms: the number of terms to plot
    :param pval_col: column name that contains the P-values or Adjusted P-values. Will also be used as title for the colorbar
                     legend
    :param term_col: column name that contains the Go Terms
    :param overlap_col: column name with fraction of genes overlaping with the gene associated with the term #TODO Add an option to compute if no overlap is provided
    :param sort_by: column name to sort by and will also represent the X-axis in the plot.
                    Assuming Enrichr we consider the combined score
    :param pval_cutoff: cutoff for selecting significant terms
    :param preprocess: if set to True, a prefiltering to select terms base on the pval_cutoff will be performed
    :param gseapy: if set to True, assume gseapy has been used and a correction to the Overlap column is applied which is
                   represented as fraction text (e.g., "2/3")
    :param figsize: figure size
    :param legend_size_title: title for the dot size legend
    :param cmap: colormap to be used
    :param dot_size_scale: scale factor to increase the size of the dots
    :param path: path to save the plot
    :param filename: name of the file
    :param show: if set to false a dictionary of the axes will be returned
    :return: plt.Axes
    """
    # Local Function
    from fractions import Fraction
    import re
    if len(df) < top_terms:
        warnings.warn(f'Less than {top_terms} Terms in the DataFrame\nOnly plotting {len(df)} Terms',
                      category=CustomWarning, stacklevel=2)

    # Select only significant terms
    if preprocess:
        df = df[df[pval_col] < pval_cutoff].copy()

    # Rank based on combined score & Select Top N Terms
    df = df.sort_values(sort_by, ascending=False).iloc[:top_terms, :].copy()
    # Correction for gseapy Overlap column which is represented as "2/3"
    if gseapy:
        df[overlap_col] = df[overlap_col].apply(lambda x: float(Fraction(x)))

    # Correction when the Terms are too long
    def remove_whitespace_around_newlines(text):
        # Replace whitespace before and after newlines with just the newline
        return re.sub(r'\s*\n\s*', '\n', text)

    newterms = []
    for text in df[term_col]:
        newterm, text_list_nchar, nchar, limit = [], [], 0, 25
        text_list = text.split(' ')
        for txt in text_list:  # From text_list get a list where we sum nchar from a word + previous word
            nchar += len(txt)
            text_list_nchar.append(nchar)
        for idx, word in enumerate(text_list_nchar):
            if word > limit:  # If we have more than 25 characters in len add a break line
                newterm.append('\n')
                limit += 25
            newterm.append(text_list[idx])
        newterm = ' '.join(newterm)
        cleanterm = remove_whitespace_around_newlines(newterm)  # remove whitespace inserted
        newterms.append(cleanterm)
    df[term_col] = newterms

    # Normalise p-value column for color mapping
    norm = plt.Normalize(df[pval_col].min(), df[pval_col].max())
    cmap = plt.get_cmap(cmap)
    # Reverse order to plot the top one in the top
    df = df.sort_values(sort_by, ascending=True)

    # # # # # # Generate the Plot itself
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, .05], height_ratios=[.5, 2], wspace=.08, hspace=.08)

    # Add Main Axis
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(
        x=df[sort_by], y=df[term_col], s=df[overlap_col] * dot_size_scale,  # Scale the overlap for better visibility
        c=df[pval_col], cmap=cmap, norm=norm, edgecolor='black',)

    # Modify Layout
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    ax.set_xlabel(sort_by, fontsize=18, fontweight='bold')
    ax.set_ylabel(term_col, fontsize=18, fontweight='bold')

    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)

    # Add some padding around the axis limits
    ax.set_xlim(df[sort_by].min() - df[sort_by].max()*0.05, df[sort_by].max() + df[sort_by].max()*0.05 )  # Adds padding to the x-axis
    ax.set_ylim(-0.5, len(df[term_col]) - 0.5)  # Adds padding to the y-axis

    # Add Legend for Dot Size
    size_values = list(df[overlap_col] * dot_size_scale)

    def get_legend_vals(perct, data=size_values):
        return np.where(data >= np.percentile(data, perct))[0][0]

    size_legend_idx = list(set([get_legend_vals(25), get_legend_vals(50), get_legend_vals(75)]))
    size_legend_values = [val for idx, val in enumerate(size_values) if idx in size_legend_idx]
    size_legend_labels = [' ' + str(round(val / dot_size_scale, 2)) for val in size_legend_values]
    handles = [mlines.Line2D([], [], linestyle='none', marker='o',
                             markersize=np.sqrt(s), color='gray') for s in size_legend_values]

    # Dedicate a portion of the plotting space to the legend
    axl = fig.add_subplot(gs[0, 1])
    axl.axis('off')
    axl.legend(handles, size_legend_labels,
               title=legend_size_title, loc='center left',
               frameon=False, title_fontproperties={'weight': 'bold'}, fontsize=15, ncols=1)

    # Add colorbar
    axcb = fig.add_subplot(gs[1, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=axcb, pad=.5, fraction=.04)
    cbar.set_label("Adjusted p-value", fontsize=15, fontweight='bold')
    axcb.grid(False)

    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

    if show:
        return
    else:
        return {'main_ax': ax, 'size_legend_ax': axl, 'color_legend_ax': axcb}


def barplot_nUMI(adata: ad.AnnData, gene: str, x: str, layer: str = None, batch_col='sample',
                 figsize: tuple = (12, 8), palette: str | list = 'tab10', capsize: float = 0.1, xtick_rotation=None,
                 ctrl_cond: str = None,  groups_cond: list = None, groups_pvals: list = None, title: str = None,
                 offset: float = 0.05,  step_offset: float = 0.15, max_offset:float = 0.25,
                 error_text_size: int = 10, text_offset: float = 1e-5,
                 path: str = None, filename: str = 'Barplot.svg', estimator='custom',
                 title_fontproperties: dict = {'size': 20, 'weight':'bold'},
                 show: bool = True, **kwargs):
    """
    Show the mean expression of a gene in a barplot showing stats
    :param adata: adata (We assume that logcounts will be provided)
    :param gene: gene present in .var_names
    :param x: .obs column that will be shown in the X axis
    :param layer: layer in adata to use
    :param batch_col: column in .obs with the batch information
    :param figsize: figure size
    :param palette: palette of colors or a dictionary for each value in x
    :param capsize: capsize of the barplot (see sns.barplot)
    :param xtick_rotation: rotate x ticks
    :param ctrl_cond: ctrl condition to use for testing
    :param groups_cond: list of conditions to test against
    :param groups_pvals: list of pvals for the conditions (same order as groups_cond). Instead of calculating wilcox you can provide a list of pre-calculated p-values
    :param title: title for the plot, if not set the genename will be used
    :param offset: offset of the lines for the pval
    :param step_offset: step size distances between pvals lines
    :param max_offset: maximum offset between lines
    :param error_text_size: size of the pval text
    :param text_offset: offset of the text with respect to the line
    :param path: path to save the plot
    :param filename: filename of the plot
    :param estimator: how to calculate the mean expression. Default is log1p(mean(exp1m(values)))
    :param show: if false it returns the axis
    :param title_fontproperties: size and fontweight of the title
    :param kwargs: other arguments provided to sns.barplot
    :return:
    """
    import davidrUtility
    assert isinstance(groups_cond, list | None), 'groups_cond should be a list'
    assert  isinstance(groups_pvals, list | None), 'groups_pvals should be a list'

    # Mean Expr needs to be in the unlogrithmize space
    def custom_estimator(values):
        return np.log1p(np.mean(np.expm1(values)))

    davidrUtility.set_plt_theme()  # Set the theme

    df = davidrUtility.ExtractExpression(adata, gene, groups=x, layer=layer)  # Extract expression
    df_batch = davidrUtility.AverageExpression(adata, group_by=[x, batch_col], feature=gene,  layer=layer)

    # Create Figure
    fig, axs = plt.subplots(1, 1, figsize=figsize)

    if estimator == 'custom':
        bp = sns.barplot(df, x=x, y='expr', estimator=custom_estimator,
                         capsize=capsize, ax=axs, palette=palette, **kwargs)
    else:
        bp = sns.barplot(df, x=x, y='expr', estimator=estimator,
                         capsize=capsize, ax=axs, palette=palette, **kwargs)

    sns.stripplot(df_batch, x='group0', y='expr', alpha=0.75, color='k', s=2.5)
    # If pairs are provided apply wilcoxon test
    if ctrl_cond is not None and groups_cond is not None:
        davidrExperimental.plot_stats_adata(axis = bp,
                                            adata =adata,
                                            x_labels=x,
                                            gene=gene,
                                            ctrl_cond=ctrl_cond,
                                            groups_cond=groups_cond,
                                            groups_pvals=groups_pvals,
                                            offset=offset,
                                            step_offset=step_offset,
                                            text_size=error_text_size,
                                            text_offset=text_offset,
                                            max_offset=max_offset)
    if xtick_rotation is not None:
        bp.set_xticklabels(bp.get_xticklabels(), rotation=xtick_rotation, ha='right', va='top', fontweight='bold')
    else:
        bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')

    bp.set_xlabel('')  # Remove xlabel
    bp.set_ylabel('LogMean nUMI', fontsize=18, fontweight='bold')  # Y label is Mean(nUMI)
    try:
        title_size = title_fontproperties['size']
        title_font = title_fontproperties['weight']
    except ValueError:
        title_size = 20
        title_fontproperties = 'bold'

    if title is None:
        bp.set_title(gene, fontsize=title_size, fontweight=title_font)  # Title is the genename
    else:
        bp.set_title(title, fontsize=title_size, fontweight=title_font)
    if path is not None:  # If the path is provided we save it
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    if show is False:  # if show is false we return the axes
        return bp
    return

def pl_distr_abundance(anndata: ad.AnnData,
                       cell_types: list,
                       cmap: plt.cm = 'coolwarm',
                       path: None = None,
                       cols: int = 4,
                       filename: str = 'Abundance.svg'):
    """
    Plot the distribution of the abundance for cell types in anndata. To be used
    after cell2location estimation. The function shows the plot per default, specify
    path to save as well as a filename.

    Usage:
        pl_distr_abundance(anndata, cell_types, cmap, path, cols, filename)

    :param anndata: anndata object
    :param cell_types: cell types to plot. Should be a .obs column
    :param cmap: color map for the bars. Default coolwarm
    :param path: path to save plot
    :param cols: number of columns in the subplot
    :param filename: name of the file with the plot. Default Abundance.svg
    :return:
    """

    df = anndata.obs[cell_types]
    n_cells, cols, cmap = len(cell_types), cols, cmap
    row = int(np.ceil(n_cells / cols))
    extra = cols * row - n_cells
    # Create Plot
    fig, axs = plt.subplots(row, cols, figsize=(16, 10))
    if row > 1:
        axs = axs.flatten()

    for idx, cell in enumerate(df.columns):
        sns.histplot(df[cell], ax=axs[idx], bins=25)  # Histogram Plot
        sns.despine()  # Remove Top and Right Frame (default)
        axs[idx].grid(False)  # Remove Grid

        # Color Bars using a colormap
        colormap = cmap
        for i, patch in enumerate(axs[idx].patches):
            axs[idx].patches[i].set_facecolor(colormap(i / 20))

        # Set x/y label and titlte
        axs[idx].set_xlabel('Cell Abundance', fontsize=12)
        axs[idx].set_ylabel('Spot Count', fontsize=12)
        axs[idx].set_title(cell, fontsize=16, fontweight='bold')

        # Add text with Max & Min Abundance
        axs[idx].text(.8, 0.8, 'Abundance', fontsize=10, fontweight='bold', ha='left', va='bottom',
                      transform=axs[idx].transAxes)
        axs[idx].text(.8, 0.65, 'Min ' + str(np.round(np.min(df[cell]), 3)), fontsize=10, ha='left', va='bottom',
                      transform=axs[idx].transAxes)
        axs[idx].text(.8, 0.50, 'Max ' + str(np.round(np.max(df[cell]), 3)), fontsize=10, ha='left', va='bottom',
                      transform=axs[idx].transAxes)
        # control x and y lim
        # axs[idx].set_ylim(top=500)

    if extra != 0:
        for idx in range(extra):
            idx_new = row * cols - extra + idx
            axs[idx_new].axis('off')

    plt.tight_layout()
    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    return

def plot_layers(adata, color, layers, ncols=4, normalise=True, **kwargs):
    from tqdm import tqdm
    adata = adata.copy()
    if normalise:
        for layer in tqdm(layers, desc='Normalised Layers'):
            sc.pp.normalize_total(adata, layer=layer)
            sc.pp.log1p(adata, layer=layer)

    nCatgs = len(layers)
    nrows = int(np.ceil(nCatgs / ncols))
    nExtra = nrows * ncols - nCatgs
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))
    axs = axs.flatten()
    for idx, ly in enumerate(layers):
        sc.pl.spatial(adata, color=color, ax=axs[idx], layer=ly, **kwargs)
        axs[idx].set_title(ly + '\n' + color, fontsize=15)
        axs[idx].spines[['right', 'top']].set_visible(False)
        axs[idx].set_xlabel('SP1', loc='left', fontsize=8, fontweight='bold')
        axs[idx].set_ylabel('SP2', loc='bottom', fontsize=8, fontweight='bold')

    if nExtra > 0:  # Hide extra subplots
        for check in range(nrows * ncols - nExtra, nrows * ncols):
            axs[check].set_visible(False)

    return


def format_terms_gsea(df, term_col, cutoff=35):
    import re
    def remove_whitespace_around_newlines(text):
        # Replace whitespace before and after newlines with just the newline
        return re.sub(r'\s*\n\s*', '\n', text)

    newterms = []
    for text in df[term_col]:
        newterm, text_list_nchar, nchar, limit = [], [], 0, cutoff
        text_list = text.split(' ')
        for txt in text_list:  # From text_list get a list where we sum nchar from a word + previous word
            nchar += len(txt)
            text_list_nchar.append(nchar)
        for idx, word in enumerate(text_list_nchar):
            if word > limit:  # If we have more than cutoff characters in len add a break line
                newterm.append('\n')
                limit += cutoff
            newterm.append(text_list[idx])
        newterm = ' '.join(newterm)
        cleanterm = remove_whitespace_around_newlines(newterm)  # remove whitespace inserted
        newterms.append(cleanterm)
    df[term_col] = newterms

    return df


def split_bar_gsea(df: pd.DataFrame,
                   term_col: str,
                   col_split: str,
                   cond_col: str,
                   pos_cond: str,
                   cutoff: int = 40,
                   log10_transform: bool = True,
                   figsize: tuple = (12, 8),
                   topN: int = 10,
                   path: str = None,
                   spacing: float = 5,
                   filename: str = 'SplitBar.svg',
                   title: str = 'Top 10 GO Terms in each Condition',
                   show: bool = True) -> Union[None, plt.axis]:
    """ **Split BarPlot for GO terms**

    This function generates a split barplot. This is a plot where the top 10 Go terms
    are shown, sorted based on a column ('col_split'). Two conditions are shown at the same
    time. One condition is shown in the positive axis, while the other in the negative one.
    The condition to be shown as positive is set with 'pos_col'.

    The GO terms will be shown inside the bars, if the term is too long, using 'cutoff',
    you can control the maximum number of characters per line.

    :param df: dataframe with the results of a gene set enrichment analysis
    :param term_col: column in the dataframe that contains the terms
    :param col_split: column in the dataframe that will be used to sort and split the plot
    :param cond_col: column in the dataframe that contains the condition information
    :param pos_cond: condition that will be shown in the positive side of the plot
    :param cutoff: maximum number of characters per line
    :param log10_transform: if col_split contains values between 0 and 1, assume they are pvals and apply a -log10 transformation
    :param figsize: figure size
    :param topN: how many terms are shown
    :param title: title of the plot
    :param show: if False, the axis is return
    :return: None or the axis
    """

    assert len(df[cond_col].unique()) == 2, 'Not implement - Only 2 conditions can be used'

    df = df.copy()
    jdx = list(df.columns).index(cond_col)  # Get index of the condition column

    # Update the col_split values; Positive values for one condition and
    # negative for the other positive. The positive is set by the 'pos_cond' argument
    min_val = df[col_split].min()
    max_val = df[col_split].max()
    is_pval = True if (min_val >= 0) and (max_val <= 1) else False
    if is_pval and log10_transform:
        logger.warn('Assuming col_split contains Pvals, apply -log10 transformation')
        df['-log10(Padj)'] = -np.log10(df[col_split])
        col_split = '-log10(Padj)'
        spacing = .5
    df[col_split] = [val if df.iloc[idx, jdx] == pos_cond else -val for idx, val in enumerate(df[col_split])]
    # Format the Terms
    df[term_col] = df[term_col].str.capitalize()  # Capitalise
    df = format_terms_gsea(df, term_col, cutoff)  # Split terms too long in several rows

    df_pos = df[df[cond_col] == pos_cond].sort_values(col_split, ascending=False).head(topN)
    df_neg = df[df[cond_col] != pos_cond].sort_values(col_split).head(topN)

    # Actual Plot
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    y_pos = range(topN)

    # Plot bars for "Down" condition (positive values) on the left side
    bars_down = axs.barh(y_pos,
                         df_neg[col_split].sort_values(ascending=False),
                         left=-spacing, color='darkorange',
                         align='center', alpha=.25)

    # Plot bars for "Up" condition (negative values) on the right side
    bars_up = axs.barh(y_pos,
                       df_pos[col_split].sort_values(),
                       left=spacing, color='royalblue',
                       align='center', alpha=.25)

    # Layout
    axs.spines[['left', 'top', 'right']].set_visible(False)
    axs.set_yticks([])
    axs.set_xlim(-np.abs(df[col_split]).max(),
                 np.abs(df[col_split]).max())
    axs.set_xlabel(col_split, fontsize=18)
    axs.set_title(title, fontsize=20)
    axs.grid(False)
    plt.vlines(0, -1, topN - .5, color='k', lw=1)
    axs.set_ylim(-.5, topN)

    # Add text labels for each bar (GO term name)
    for i, bar in enumerate(bars_up):
        # Add the GO term for "Up" bars (positive)
        axs.text(spacing * 2, bar.get_y() + bar.get_height() / 2,
                 df_pos.sort_values(col_split)[term_col].iloc[i],
                 va='center', ha='left', color='k', fontweight='bold')

    for i, bar in enumerate(bars_down):
        # Add the GO term for "Down" bars (negative)
        axs.text(-spacing * 2, bar.get_y() + bar.get_height() / 2,
                 df_neg.sort_values(col_split, ascending=False)[term_col].iloc[i],
                 va='center', ha='right', color='k', fontweight='bold')
    # Save Plot
    if path is not None:
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

    # If show is False, return axs
    if not show:
        return axs
    else:
        return

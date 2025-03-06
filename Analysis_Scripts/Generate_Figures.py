#!/usr/bin/env python

"""
Description: Generate Figures for the Manuscript

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""

# <editor-fold desc="Sep-Up">
import os
from tqdm import tqdm
import itertools

import anndata as ad
import scanpy as sc
import bbknn as bkn
import decoupler as dc
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import mannwhitneyu, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines
import seaborn as sns

import davidrPlotting
import davidrUtility
import davidrExperimental
import davidrScanpy

np.random.seed(13)

main_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/'
result_path = os.path.join(main_path, 'Results/')
object_path = os.path.join(result_path, 'Objects/Scanpy/')
table_path = os.path.join(result_path, 'Tables/')
figure_path = os.path.join(result_path, 'Figures/_Submission/')
misty_path = os.path.join(result_path, 'Figures/4_FunctionalAnalysis/mistyR_Analysis')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
ref = sc.read_h5ad(os.path.join(object_path, 'Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()
adata_ccc = sc.read_h5ad(
    os.path.join(object_path, 'HoloNet/Visium_YoungOld_HoloNet_CCC_230624.h5ad'))  # AnnData with Communication events
adata_ccc.X = adata_ccc.layers['centrality_eigenvector'].copy()
adata_ccc.obs = aging.obs.reindex(adata_ccc.obs_names)

# Generate a dictionary of celltype (keys) and colors (values)
ct_catgs = list(ref.obs.annotation.cat.categories)
ct_colors = ref.uns['annotation_colors']
ct_zip = dict(zip(ct_catgs, ct_colors))
# </editor-fold>


# <editor-fold desc="Figure 1a. UMAP snRNA across celltypes">
ax = davidrPlotting.pl_umap(ref, 'annotation', size=12, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('CellTypes')
ax.legend(handles=[mlines.Line2D([0], [0], marker=".", color=c,
                                 lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in zip(ct_catgs, ct_colors)],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'}, bbox_to_anchor=(1.35, .5), fontsize=13, ncols=1)
plt.savefig(os.path.join(figure_path, 'Fig1a_UMAP_snRNA_Annotation.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 1b. UMAP ST across clusters">
ax = davidrPlotting.pl_umap(aging, 'clusters', size=12, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('Clusters')
ax.legend(handles=[mlines.Line2D([0], [0], marker=".", color=c,
                                 lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(aging.obs.clusters.cat.categories), aging.uns['clusters_colors'])],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'}, bbox_to_anchor=(1.35, .5), fontsize=13, ncols=1)
plt.savefig(os.path.join(figure_path, f'Fig1b_UMAP_ST_Niches.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 1c. Heatmap of marker gene expression in snRNA">
markers = {'ArtEC': ['Stmn2', 'Fbln5', 'Sox17'],
           'VeinEC': ['Vwf', 'Vcam1', 'Tmem108'],
           'CapEC': ['Car4', 'Rgcc'],
           'EndoEC': ['Npr3'],
           'LymphEC': ['Lyve1', 'Mmrn1'],
           'Pericytes': ['Rgs5', 'Pdgfrb'],
           'SMC': ['Acta2', 'Tagln'],
           'Fibroblasts': ['Tnxb', 'Gsn'],
           'Fibro_activ': ['Postn'],
           'MP': ['Lyz2', 'C1qc'],
           'Ccr2+MP': ['Ccr2'],
           'B_cells': ['Cd74', 'Cd79a'],
           'T_cells': ['Cd3d', 'Themis'],
           'Epi_cells': ['Wwc1', 'C3'],
           'Adip': ['Adipoq', 'Plin1'],
           'CM': ['Ttn', 'Tnni3']}

# Calculate AverageExpression per celltype
genes = list(itertools.chain(*markers.values()))
table = davidrUtility.AverageExpression(ref, group_by='annotation', feature=genes, out_format='wide')
table = table.reindex(index=genes, columns=markers.keys())  # Sort the dataframe by index

# Reverse the marker dict (celltype:genes --> gene: celltype) to match genes to celltype colors
reversed_markers = {gene: [ct for ct, genes in markers.items() if gene in genes][0]
                    for gene in {gene for genes in markers.values() for gene in genes}}
row_colors = [ct_zip[reversed_markers[gene]] for gene in table.index]
col_colors = [ct_zip[col] for col in markers.keys()]

# Plot: Heatmap showing marker genes
cm = sns.clustermap(table, cmap='Reds',
                    yticklabels=1, xticklabels=1,  # Show all ticks
                    row_colors=row_colors,  # Indicate what celltype the marker is from
                    col_cluster=False, row_cluster=False,  # Do not cluster
                    z_score=0, robust=True,  # Z-score over genes
                    figsize=(9, 12), square=True,
                    cbar_pos=([0.85, .15, 0.03, 0.15]))

# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
colorbar_ax = cm.cax

# Set black border around the outer edge of the heatmap
heatmap_ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)

# Replace xticks with circles colored by the annotation
xtick_pos, xtick_lab = heatmap_ax.get_xticks(), heatmap_ax.get_xticklabels()
ytick_pos, ytick_lab = heatmap_ax.get_yticks(), heatmap_ax.get_yticklabels()
heatmap_ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
for x, color in zip(xtick_pos, col_colors):
    heatmap_ax.plot(x, 31.5, 'o', color=color, markersize=8, clip_on=False)

# Adjust Xticks and Yticks to correct positions
heatmap_ax.set_yticklabels(ytick_lab, rotation=0, ha='right', fontsize=15)
heatmap_ax.set_xticklabels(xtick_lab, rotation=45, ha='right', fontsize=15, fontweight='bold')

heatmap_ax.tick_params(axis='x', pad=13)
heatmap_ax.tick_params(axis='y', pad=20, labelright=False, labelleft=True)

# Set Colorbar Ticks
colorbar_ax.set_title('Z score', fontweight='bold', loc='left', fontsize=18)
colorbar_ax.set_yticks([-0.9, np.percentile(cm.data, 99)])
colorbar_ax.set_yticklabels(['Min', 'Max'], fontweight='bold')

# Remove X and Y labels
heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')
plt.savefig(os.path.join(figure_path, f'Fig1c_Heatmap_Markers_snRNA.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 1d. Spatial distribution of celltypes in Young and Old representative sample">
old5 = davidrUtility.select_slide(aging, 'Old_5')
young3 = davidrUtility.select_slide(aging, 'Young_3')

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
plt.subplots_adjust(hspace=0, wspace=.08, left=.05)  # Spacing between subplots
for idx, color in enumerate(['SMC', 'EndoEC', 'Epi_cells', 'MP']):
    sc.pl.spatial(young3, color=color, bw=True, size=1.5, cmap='Reds', vmax='p99.2', ncols=4,
                  ax=axs[0, idx], colorbar_loc=None, show=False)
    sc.pl.spatial(old5, color=color, bw=True, size=1.5, cmap='Reds', vmax='p99.2', ncols=4,
                  ax=axs[1, idx], colorbar_loc=None, show=False)

    # Modify Layout of Spines
    davidrUtility.axis_format(axs[0, idx], 'SP')
    davidrUtility.axis_format(axs[1, idx], 'SP')

    # Specify figure title
    axs[0, idx].set_title(color, fontsize=18, fontweight='bold')
    axs[1, idx].set_title(None, fontsize=18, fontweight='bold')

# Add Y Label to specify what is Young and Old
fig.text(0.03, 0.75, 'Young', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'Old', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.05, .12, 0.1, 0.015])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('Cell Proportions', fontweight='bold', loc='center', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min', 'Max'], fontweight='bold', fontsize=12)
plt.savefig(os.path.join(figure_path, f'Fig1d_SpatialPlot_SpatialPlot_YoungOld.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 2a. Stacked barplot of niche proportion across spatial locations">
# Calculate proportion of niches across anatomic regions
df = aging.obs['clusters'].to_frame().copy()
df['region'] = aging.obs['AnatomicRegion'].copy()
df = df.value_counts(['region', 'clusters']).reset_index()
df_pivot = df.pivot(index='region', columns='clusters', values='count')
df_pivot['total'] = df_pivot.sum(axis=1)
df_pivot = df_pivot.iloc[1:, :]  # Exclude BG --> First row is BG
df_prop = df_pivot.iloc[:, :-1].div(df_pivot['total'], axis=0)  # Compute proportion
df_prop = df_prop.reindex(
    columns=['Niche 0', 'Niche 1', 'Niche 2', 'Niche 3', 'Niche 4', 'Niche 5', 'Niche 6', 'Niche 7',
             'Niche 8', 'Niche 9', 'Niche 10'])  # Sort Niches

# Get colors associated to each niche
cmap = {key: aging.uns['clusters_colors'][idx] for idx, key in enumerate(aging.obs['clusters'].cat.categories)}
colors = [cmap[col] for col in df_prop.columns]

fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
ax1 = df_prop.plot(kind='bar', stacked=True, grid=False, rot=75, ax=ax1, color=colors, width=.95)
sns.despine()
ax1.set_ylabel('Proportions', fontsize=18, fontweight='bold')
ax1.set_xlabel('')
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15, fontweight='bold')
sns.move_legend(ax1, loc='center right', frameon=False, title='Clusters',
                title_fontproperties={'weight': 'bold', 'size': 15},
                ncols=1, bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'Fig2a_StackedBarPlot.svg'), bbox_inches='tight')

# Generate a representative image to show the spatial distribution of the niches
fig, axs = plt.subplots(1, 1, figsize=(15, 8))
ax = sc.pl.spatial(old5, color='clusters', bw=True, size=1.5, show=False, ax=axs, legend_loc=None)[0]
davidrUtility.axis_format(ax, 'SP')
ax.set_title('Old_5', fontweight='bold', fontsize=25)
plt.savefig(os.path.join(figure_path, 'Fig2a_ST_Old5_Clusters.svg'))
# </editor-fold>


# <editor-fold desc="Figure 2b. Stacked barplot of niche proportion across condition">
df = aging.obs.value_counts(['clusters', 'condition']).sort_index().reset_index()
df['prop'] = df['count'] / df.groupby('condition')['count'].transform('sum')  # Calculate proportions

df_pivot = df.pivot(index='condition', columns='clusters', values='prop')
df_pivot = df_pivot.sort_index(ascending=False)  # Sort to have Young, Old

sbp = df_pivot.plot(kind='bar', stacked=True, grid=False, rot=75, width=0.9, figsize=(2, 6))
sbp.set_ylabel('Proportions', fontsize=18, fontweight='bold')
sbp.set_xlabel('', fontsize=18, fontweight='bold')
sbp.set_xticklabels(sbp.get_xticklabels(), fontsize=15, fontweight='bold')
sns.move_legend(sbp, loc='center right', frameon=False, title='Clusters',
                title_fontproperties={'weight': 'bold', 'size': 15},
                ncols=1, bbox_to_anchor=(1.6, .5))
plt.savefig(os.path.join(figure_path, 'Fig2b_stackedBarPlot.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 2c. Clustermap showing cellttype enriched in each niche">
df = aging.obsm['c2l'].copy()  # Take the predicted celltype abundance
df['clusters'] = aging.obs['clusters'].copy()
df = df.groupby('clusters').agg('sum')
df['total'] = df.sum(axis=1)
df_prop = df.iloc[:, :-1].div(df['total'], axis=0)  # estimate cell proportion
df_prop = df_prop.reindex(columns=df_prop.sum().sort_values(ascending=False).index)

# Sort celltypes
df_prop = df_prop.reindex(columns=['ArtEC', 'VeinEC', 'LymphEC', 'CapEC', 'EndoEC',
                                   'MP', 'Ccr2+MP', 'B_cells', 'T_cells', 'Pericytes', 'SMC',
                                   'Fibroblasts', 'Fibro_activ', 'CM', 'Adip', 'Epi_cells'])
# Hirarchical clustering of niches
cm = sns.clustermap(df_prop.T, cmap='RdBu_r', xticklabels=1, yticklabels=1, z_score='col',
                    row_cluster=False, center=0, vmax=2.5, vmin=-2.5, cbar_pos=None, square=False)

# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
# Set black border around the outer edge of the heatmap
heatmap_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)

# Replace xticks with circles colored by the annotation
xtick_pos, xtick_lab = heatmap_ax.get_xticks(), heatmap_ax.get_xticklabels()
ytick_pos, ytick_lab = heatmap_ax.get_yticks(), heatmap_ax.get_yticklabels()
heatmap_ax.tick_params(axis='y', which='both', length=0)  # Remove ticks
# Adjust Xticks and Yticks to correct positions
heatmap_ax.set_yticklabels(ytick_lab, rotation=0, ha='right')
heatmap_ax.set_xticklabels(xtick_lab, rotation=75, ha='right', va='top', fontweight='bold')
heatmap_ax.tick_params(axis='y', pad=5, labelright=False, labelleft=True)

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.83, .15, 0.03, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=-2.5, vmax=2.5)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Scale Prop', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([-2.5, 0, 2.5])
cbar.set_ticklabels(['-2.5', '0', '2.5'], fontweight='bold')
cbar.ax.grid(False)
plt.subplots_adjust(right=0.8)

# Adjustment of how the plot looks like
heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')
plt.savefig(os.path.join(figure_path, 'Fig2c_Clustermap_celltype_across_niches.svg'), bbox_inches='tight')

# </editor-fold>


# Figure 2d --> CytosScape


# Figure 3a --> Drawing made


# <editor-fold desc="Figure 3b, Extended Figure 6a and 6b - MistyR results">
df = pd.read_csv(os.path.join(misty_path, 'importances_samples_c2l_v3.csv'))
df_contrib = pd.read_csv(os.path.join(misty_path, 'importances_samples_c2l_r2vals_v3.csv'))

#  Remove importances if R2 is < 5
cutoff = 5

# Intraview filtering
df_contrib_intra = df_contrib[df_contrib.measure == 'intra.R2']
df_contrib_intra = df_contrib_intra[df_contrib_intra['mean'] < cutoff]
df_intra = df[df['view'] == 'intra']
ndf_intra = pd.DataFrame()
for batch in tqdm(df_intra['sample'].unique()):
    rm_target = df_contrib_intra[df_contrib_intra['sample'] == batch]['target'].tolist()
    sdf_intra = df_intra[df_intra['sample'] == batch]
    sdf_intra = sdf_intra[~sdf_intra.Target.isin(rm_target)]
    ndf_intra = pd.concat([ndf_intra, sdf_intra])

# Juxta View uses multi.R2
df_contrib_juxta = df_contrib[df_contrib.measure == 'multi.R2']
df_contrib_juxta = df_contrib_juxta[df_contrib_juxta['mean'] < cutoff]
df_juxta = df[df['view'].str.startswith('juxta')]
ndf_juxta = pd.DataFrame()
for batch in tqdm(df_juxta['sample'].unique()):
    rm_target = df_contrib_juxta[df_contrib_juxta['sample'] == batch]['target'].tolist()
    sdf_juxta = df_juxta[df_juxta['sample'] == batch]
    sdf_juxta = sdf_juxta[~sdf_juxta.Target.isin(rm_target)]
    ndf_juxta = pd.concat([ndf_juxta, sdf_juxta])

# Para View uses multi.R2
df_contrib_para = df_contrib[df_contrib.measure == 'multi.R2']
df_contrib_para = df_contrib_para[df_contrib_para['mean'] < cutoff]
df_para = df[df['view'].str.startswith('para')]
ndf_para = pd.DataFrame()
for batch in tqdm(df_para['sample'].unique()):
    rm_target = df_contrib_para[df_contrib_para['sample'] == batch]['target'].tolist()
    sdf_para = df_para[df_para['sample'] == batch]
    sdf_para = sdf_para[~sdf_para.Target.isin(rm_target)]
    ndf_para = pd.concat([ndf_para, sdf_para])

# <editor-fold desc="Figure 3b - Clustermap on Intraview">
ndf_intra.loc[ndf_intra.Importance < 0, 'Importance'] = 0  # Negative Importance set 0
ndf_intra.loc[ndf_intra.Importance.isna(), 'Importance'] = 0  # Nan Importance set to 0
ndf_intra['Predictor'] = ndf_intra['Predictor'].str.replace('Ccr2_MP',
                                                            'Ccr2+MP')  # Correction for running the analysis in R
ndf_intra['Target'] = ndf_intra['Target'].str.replace('Ccr2_MP', 'Ccr2+MP')

cm_intra = ndf_intra.loc[:, ['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg(
    'median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_intra[cm_intra < 0] = 0
cm_intra[cm_intra.isna()] = 0

# Hirarchical clustering of rows and columns
Z = linkage(cm_intra, method='complete')
dg = dendrogram(Z, no_plot=True)
new_idx = cm_intra.index[dg['leaves']]

Z = linkage(cm_intra.T, method='complete')
dg = dendrogram(Z, no_plot=True)
new_cols = cm_intra.columns[dg['leaves']]

# Resort based on the hirarchical clustering
cm_intra = cm_intra.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_intra, cmap='Reds', col_cluster=False, row_cluster=False, robust=True,
                    square=True, vmin=.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation': 'horizontal'},
                    figsize=(6, 6), xticklabels=1, yticklabels=1, linewidths=.1, vmax=3)
cb_ax = cm.ax_cbar

cm.ax_heatmap.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
cb_ax.set_xticks([0.25, 1.5, 3], ['>0.25', '1.5', '3'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(figure_path, 'Fig3b_Clustermap_Intraview.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 6a - Clustermap on JuxtaView">
ndf_juxta.loc[ndf_juxta.Importance < 0, 'Importance'] = 0
ndf_juxta.loc[ndf_juxta.Importance.isna(), 'Importance'] = 0
ndf_juxta['Predictor'] = ndf_juxta['Predictor'].str.replace('Ccr2_MP', 'Ccr2+MP')
ndf_juxta['Target'] = ndf_juxta['Target'].str.replace('Ccr2_MP', 'Ccr2+MP')

cm_juxta = ndf_juxta.loc[:, ['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg(
    'median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_juxta[cm_juxta < 0] = 0
cm_juxta[cm_juxta.isna()] = 0
cm_juxta = cm_juxta.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_juxta, cmap='Reds', col_cluster=False, row_cluster=False, robust=True,
                    square=True, vmin=0.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation': 'horizontal'},
                    figsize=(6, 6), xticklabels=1, yticklabels=1, linewidths=.1, vmax=1.5)
cb_ax = cm.ax_cbar

cm.ax_heatmap.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
cb_ax.set_xticks([0.25, 0.75, 1.5], ['>0.25', '0.75', '1.5'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(figure_path, 'ExtFig6a_Clustermap_JuxtaView.svg'), bbox_inches='tight')
# </editor-fold>

# <editor-fold desc="Extended Figure 6b - Clustermap on Paraview">
ndf_para.loc[ndf_para.Importance < 0, 'Importance'] = 0
ndf_para.loc[ndf_para.Importance.isna(), 'Importance'] = 0
ndf_para['Predictor'] = ndf_para['Predictor'].str.replace('Ccr2_MP', 'Ccr2+MP')
ndf_para['Target'] = ndf_para['Target'].str.replace('Ccr2_MP', 'Ccr2+MP')

cm_para = ndf_para.loc[:, ['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg(
    'median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_para[cm_para < 0] = 0
cm_para[cm_para.isna()] = 0
cm_para = cm_para.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_para, cmap='Reds', col_cluster=False, row_cluster=False, robust=True,
                    square=True, vmin=0.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation': 'horizontal'},
                    figsize=(6, 6), xticklabels=1, yticklabels=1, linewidths=.1, vmax=1.5)
cb_ax = cm.ax_cbar

cm.ax_heatmap.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
cb_ax.set_xticks([0.25, .75, 1.5], ['>0.25', '0.75', '1.5'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(figure_path, 'ExtFig6b_Clustermap_Paraview.svg'), bbox_inches='tight')
# </editor-fold>

# </editor-fold>


# <editor-fold desc="Figure 3c. Spatial distribution of FB and MP">
young1 = davidrUtility.select_slide(aging, 'Young_1')
old5 = davidrUtility.select_slide(aging, 'Old_5')

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young1, color=['Fibroblasts'], size=1.5, bw=True, vmax='p99.2', ax=axs[0],
              title='Fibroblasts', colorbar_loc=None)
sc.pl.spatial(young1, color=['MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[1],
              title='MP', colorbar_loc=None)
sc.pl.spatial(old5, color=['Fibroblasts'], size=1.5, bw=True, vmax='p99.2', ax=axs[2],
              title='', colorbar_loc=None)
sc.pl.spatial(old5, color=['MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[3],
              title='', colorbar_loc=None)
for ax in axs:
    davidrUtility.axis_format(ax, 'SP')

fig.text(0.03, 0.75, 'Young_1', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'Old_5', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Add colorbar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.95, .15, 0.03, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop', fontweight='bold', loc='left', fontsize=12)
cbar_ax.set_yticks([0, 1], ['Min', 'Max'], fontsize=10, fontweight='bold')

plt.savefig(os.path.join(figure_path, 'Fig3c_SpatiatDistribution_YoungOld_MPFB.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 3d. Spatial distribution of C3:C3ar1">
young1_ccc = davidrUtility.select_slide(adata_ccc, 'Young_1')
old5_ccc = davidrUtility.select_slide(adata_ccc, 'Old_5')

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young1_ccc, color=['C3 :C3ar1'], size=1.5, bw=True, vmax='p99.2', ax=axs[0],
              title='C3:C3ar1', colorbar_loc=None)
sc.pl.spatial(old5_ccc, color=['C3 :C3ar1'], size=1.5, bw=True, vmax='p99.2', ax=axs[1],
              title='', colorbar_loc=None)
for ax in axs:
    davidrUtility.axis_format(ax, 'SP')

# Add colorbar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.75, .15, 0.03, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('CE Actv', fontweight='bold', loc='left', fontsize=12)
cbar_ax.set_yticks([0, 1], ['Min', 'Max'], fontsize=10, fontweight='bold')

plt.savefig(os.path.join(figure_path, 'Fig3d_SpatialDistribution_C3C3ar1.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 3e. MatrixPlot Expr/Ct abundance/CE">
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
fig.subplots_adjust(wspace=0.2)

# Axis 1 shows the CE activity
ax1 = sc.pl.matrixplot(adata_ccc, groupby='clusters',
                       var_names=['C3 :C3ar1'], colorbar_title='Scale Mean CE\nActivity in group',
                       standard_scale='var', cmap='Reds', ax=axs[0], show=False, figsize=(2, 5),
                       title='CE\nActivity')
# Axis 2 shows the expression of Ligand-Receptor
ax2 = sc.pl.matrixplot(aging, groupby='clusters', var_names=['C3', 'C3ar1'], standard_scale='var',
                       cmap='Reds', ax=axs[1], show=False, colorbar_title='Scale Mean\nExpression in group',
                       figsize=(3, 5), title='Ligand-Receptor\nExpression')
# Axus 3 shows cell type abundance
ax3 = sc.pl.matrixplot(aging, groupby='clusters', var_names=['MP', 'Fibroblasts', 'Ccr2+MP', 'Fibro_activ'],
                       standard_scale='var',
                       cmap='Reds', ax=axs[2], show=False, colorbar_title='Scale Mean\nProp in group',
                       figsize=(6, 5), title='CellType\nProportion')
# Layout of matrixplot
ax1['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax2['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax3['mainplot_ax'].spines[['top', 'right']].set_visible(True)
for idx, ax in enumerate([ax1['mainplot_ax'], ax2['mainplot_ax'], ax3['mainplot_ax']]):
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', rotation=45, ha='right', va='top', fontsize=15)
    if idx == 0:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    else:
        ax.set_yticklabels([])

# Remove colorbars
ax1['color_legend_ax'].set_visible(False)
ax2['color_legend_ax'].set_visible(False)
ax3['color_legend_ax'].set_visible(False)

# Add Colorbar for each pannel
fig = plt.gcf()
cbar_ax = fig.add_axes([0.25, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_yticks([0, 1], ['Min', 'Max'], fontsize=10, fontweight='bold')
cbar.ax.set_title('CE Actv', fontweight='bold', loc='left', fontsize=12)

# Add Colorbar for each pannel
fig = plt.gcf()
cbar_ax = fig.add_axes([0.60, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_yticks([0, 1], ['Min', 'Max'], fontsize=10, fontweight='bold')
cbar.ax.set_title('Expr', fontweight='bold', loc='left', fontsize=12)

# Add Colorbar for each pannel
fig = plt.gcf()
cbar_ax = fig.add_axes([0.92, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop', fontweight='bold', loc='left', fontsize=12)
cbar.set_yticks([0, 1], ['Min', 'Max'], fontsize=10, fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Fig3e_MatrixPlot_SummaryC3C3ar1.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 3f. Barplot of Median CE Activity of C3C3ar1">
# Do DGE (Young Vs Old)
sc.tl.rank_genes_groups(adata_ccc, groupby='condition', method='wilcoxon', tie_correct=True)
table = sc.get.rank_genes_groups_df(adata_ccc, group='Old')
table = table[table.names == 'C3 :C3ar1']

df = davidrUtility.ExtractExpression(adata_ccc, 'C3 :C3ar1', ['condition'])
fig, axs = plt.subplots(1, 1, figsize=(3, 5))
bp = sns.barplot(df, x='condition', y='expr', palette={'Young': 'sandybrown', 'Old': 'royalblue'},
                 ax=axs, order=['Young', 'Old'], estimator='median', capsize=.1)
# Add Pval stats to the plot
davidrExperimental.plot_stats_adata(bp, adata_ccc, 'condition', 'C3 :C3ar1',
                                    'Young', ['Old'], list(table.pvals_adj), text_offset=4.5e-7)
bp.set_ylim(0, 4.5e-5)
bp.set_xlabel('')
bp.set_ylabel('Median CE strength')
bp.set_title('C3:C3ar1')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Fig3f_Barplot_MedianC3C3ar1.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 3g. Spatial distribution of SMC and Fibroblast activated">
old4 = davidrUtility.select_slide(aging, 'Old_4')
young3 = davidrUtility.select_slide(aging, 'Young_3')

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young3, color=['SMC'], size=1.5, bw=True, vmax='p99.2', ax=axs[0], title='Young_3')
sc.pl.spatial(young3, color=['Fibro_activ'], size=1.5, bw=True, vmax='p99.2', ax=axs[2], title='')
sc.pl.spatial(old4, color=['SMC'], size=1.5, bw=True, vmax='p99.2', ax=axs[1], title='Old_4')
sc.pl.spatial(old4, color=['Fibro_activ'], size=1.5, bw=True, vmax='p99.2', ax=axs[3], title='')

for ax in axs:
    davidrUtility.axis_format(ax, 'SP')

fig.text(0.03, 0.75, 'SMC', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'Fibro_activ', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Fig3g_SpatialDistribution_YoungOld_FBaSMC.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 3h. Heatmap Progeny activities -- Progeny Analysis">
# Pathway Activity Inference with Progeny
progeny = dc.get_progeny(organism='human', top=500)  # Mouse gives problems
progeny['target'] = progeny['target'].str.capitalize()  # Convert in Mouse format

# Inference per sample
dict_adatas = {}
for s in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, s)
    sdata.X = sdata.layers['SCT_norm'].copy()
    dc.run_mlm(mat=sdata, net=progeny,
               source='source', target='target', weight='weight',
               verbose=True, use_raw=False)

    # Store in new obsm keys
    sdata.obsm['progeny_mlm_estimate'] = sdata.obsm['mlm_estimate'].copy()
    sdata.obsm['progeny_mlm_pvals'] = sdata.obsm['mlm_pvals'].copy()

    # adata of activities
    acts = dc.get_acts(sdata, obsm_key='progeny_mlm_estimate')
    dict_adatas[s] = acts

# Concatenate
adata_concat = ad.concat(dict_adatas.values(), join='outer', uns_merge='unique')
del dict_adatas  # Release memory
davidrUtility.free_memory()

# Normalise to be between positive for each gene to do DGE
X = adata_concat.X.copy()
X_shifted = X + np.abs(X.min(axis=0))
adata_concat.layers['X_shifted'] = X_shifted

# DGE between Young and Old based on pathways activities
table = pd.DataFrame()
for clust in adata_concat.obs.clusters.unique():
    sdata = adata_concat[adata_concat.obs.clusters == clust]
    sc.tl.rank_genes_groups(sdata, groupby='condition', method='wilcoxon', tie_correct=True, layer='X_shifted')
    df = sc.get.rank_genes_groups_df(sdata, group='Old', pval_cutoff=0.05)
    df['cluster'] = clust
    table = pd.concat([table, df])

# Calculate the pathways that are overall activated
data = davidrUtility.ExtractExpression(adata_concat, adata_concat.var_names, groups=['condition', 'clusters'])
data_summary = data.groupby(['clusters', 'condition', 'genes']).agg('mean').reset_index().pivot(
    index=['clusters', 'genes'], columns='condition', values='expr')

# Mean activity of the pathways (+ means activated and - inactivated)
table['OldMean'] = [data_summary.loc[(row['cluster'], row['names']), 'Old'] for idx, row in table.iterrows()]
table['YoungMean'] = [data_summary.loc[(row['cluster'], row['names']), 'Young'] for idx, row in table.iterrows()]

# Filter pathways that are inactivated
table_filt = table[~((table['OldMean'] < 0) & (table['YoungMean'] < 0))].sort_values('scores', ascending=False)
table_filt = table_filt[
    ~table_filt.names.isin(['Androgen', 'Estrogen'])]  # Remove androgen and estrogen from the analysis

tmp = table_filt.pivot(index=['cluster'], columns='names', values='logfoldchanges')  # Prepare data for the heatmap
tmp[tmp.isna()] = 0  # If NaN, set to 0
hm_lfc = tmp.reindex(index=['Niche 9', 'Niche 4', 'Niche 5', 'Niche 3', 'Niche 2',
                            'Niche 6', 'Niche 7', 'Niche 1', 'Niche 10', 'Niche 0',
                            'Niche 8'], columns=['EGFR', 'TNFa', 'WNT', 'p53',
                                                 'Hypoxia', 'TGFb'])

#  Clustermap showing Up and Down Pathways in Old with respect to young
cm = sns.clustermap(hm_lfc, cmap='RdBu_r', center=0, row_cluster=False, col_cluster=False, robust=True,
                    cbar_pos=[0.2, .86, .15, .03], cbar_kws={'orientation': 'horizontal'}, square=True,
                    figsize=(4, 6), linewidth=0.1)
# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
colorbar_ax = cm.cax
# Correct layout
heatmap_ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)
heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontweight='bold', fontsize=12, rotation=75, ha='right')
# Modify colorbar
colorbar_ax.set_title('Log2FC', fontweight='bold', loc='center', fontsize=12)
colorbar_ax.set_xticklabels(colorbar_ax.get_xticklabels(), fontweight='bold', fontsize=12)
plt.savefig(os.path.join(figure_path, 'Fig3h_Heatmap_Progeny.svg'), bbox_inches='tight')
# </editor-fold>


# Figure 4a --> Drawing made


# <editor-fold desc="Figure 4b. Histology and vessel annotation">
# Generate individual panels to modify in insckape
young1 = davidrUtility.select_slide(aging, 'Young_1')


# Part 1 - Histology and Zoom-In for Young_1
axs = sc.pl.spatial(young1, show=False, frameon=True)[0]
# Add a zoomed-in inset axes
inset_ax = inset_axes(axs, width="100%", height="50%", loc='lower left', bbox_to_anchor=(0, 1, 1, 1),
                      # [left, bottom, width, height],
                      bbox_transform=axs.transAxes)
sub_axs = sc.pl.spatial(young1, ax=inset_ax, crop_coord=[3800, 6000, 4000, 5500],  # (left, right, top, bottom).
                        frameon=True, show=False)[0]
# Layout
for ax in [axs, sub_axs]:
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.savefig(os.path.join(figure_path, 'Fig4b_Part1_Young1_Histology.pdf'), bbox_inches='tight', dpi=1000)


# Part 2 - Vessels and Zoom-In for Young_1
fig, axs = plt.subplots(1, 2, figsize=(12, 12))
sc.pl.spatial(young1, color='vessels', size=1, legend_loc=None, ax=axs[0], title='')
sc.pl.spatial(young1, color='vessels', ax=axs[1], crop_coord=[3800, 6000, 4000, 5500],  # (left, right, top, bottom).
              frameon=True, size=1, title='')
# Layout
for idx in range(2):
    axs[idx].spines[['top', 'right', 'left', 'bottom']].set_visible(True)
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('')
plt.savefig(os.path.join(figure_path, 'Fig4b_Part2_Young1_Vessels.svg'), bbox_inches='tight')

# Part 3 - Vessels and Zoom-In for Young_1 Extra
fig, axs = plt.subplots(1, 2, figsize=(12, 12))
sc.pl.spatial(young1, ax=axs[0], crop_coord=[2800, 4500, 1000, 2000],  # (left, right, top, bottom).
              frameon=True, size=1, title='')
sc.pl.spatial(young1, color='vessels', ax=axs[1], crop_coord=[2800, 4500, 1000, 2000],  # (left, right, top, bottom).
              frameon=True, size=1, title='')
# Layout
for idx in range(2):
    axs[idx].spines[['top', 'right', 'left', 'bottom']].set_visible(True)
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('')
plt.savefig(os.path.join(figure_path, 'Fig4b_Part3_Young1_Vessels_Extra.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 4c. Dotplot Markers and vessels">
ec_markers = ['Tagln', 'Myh11',  # SMC
              'Stmn2', 'Fbln5',  # ArtEC
              'Vwf', 'Vcam1',  # VeinEC
              'Lyve1', 'Mmrn1']  # LymphEC

cmap_per = davidrUtility.generate_cmap('white', '#ff4d2e')

ax = sc.pl.dotplot(aging, groupby='vessels', var_names=ec_markers,
                   dot_max=.75, cmap=cmap_per,
                   categories_order=['Arteries', 'Art_Lymph', 'MixVasc',
                                     'Veins', 'Vein_Lymph', 'Lymphatics',
                                     'nonVasc'],
                   standard_scale='var', swap_axes=True,
                   size_title='Fraction of spots\nin group (%)',
                   show=False, colorbar_title='Scaled expression\nin group',
                   figsize=(4, 3))
# Layout
main_ax = ax['mainplot_ax']
main_ax.spines[['top', 'right']].set_visible(True)
main_ax.set_xticklabels(main_ax.get_xticklabels(), fontweight='bold')
main_ax.set_yticklabels(main_ax.get_yticklabels(), fontweight='bold')

ax['color_legend_ax'].set_title('Scaled expression\nin group', fontsize=10)
ax['color_legend_ax'].grid(False)
ax['size_legend_ax'].set_title('\n\n\n' + ax['size_legend_ax'].get_title(), fontsize=10)
plt.savefig(os.path.join(figure_path, 'Fig4c_DotplotMarkersVessels.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 4d. Dotplot Vessels and celltypes">
# Generate an AnnData of the cell type proportions
df = aging.obsm['c2l_prop'].copy()
adata = ad.AnnData(X=df.values, obs=list(df.index), var=list(df.columns))
adata.obs_names = adata.obs[0]
adata.var_names = adata.var[0]
adata.obs = aging.obs.copy()

for col in adata.var_names:  # Remove the celltype names in .obs otherwise there is redundancy
    del adata.obs[col]

ax = sc.pl.dotplot(adata, groupby='vessels', cmap=davidrUtility.generate_cmap('white', '#ff4d2e'),
                   expression_cutoff=.01, var_names=['SMC', 'ArtEC', 'VeinEC', 'LymphEC'],
                   categories_order=['Arteries', 'Art_Lymph', 'MixVasc',
                                     'Veins', 'Vein_Lymph', 'Lymphatics', 'nonVasc'],
                   vmax=.1, show=False)

main_ax = ax['mainplot_ax']
main_ax.spines[['top', 'right']].set_visible(True)
main_ax.set_xticklabels(main_ax.get_xticklabels(), fontweight='bold', rotation=70, ha='center', va='top')
main_ax.set_yticklabels(main_ax.get_yticklabels(), fontweight='bold')

ax['color_legend_ax'].set_title('Mean cell prop.\nin group', fontsize=10)
ax['color_legend_ax'].grid(False)
ax['size_legend_ax'].set_title('\n\n\n' + ax['size_legend_ax'].get_title(), fontsize=12)
plt.savefig(os.path.join(figure_path, 'Fig4d_MeanCtProp_Vessels.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 4e. Barplot of Vessel Proportion">
df = aging.obsm['c2l'].copy()
df[['condition', 'type_vessel', 'sample']] = aging.obs[['condition', 'vessels', 'sample']].copy()

# Select LargeVessels
arteries = df[df['type_vessel'].isin(['Arteries', 'MixVasc', 'Art_Lymph'])]
veins = df[df['type_vessel'].isin(['Veins', 'MixVasc', 'Vein_Lymph'])]
lymphatics = df[df['type_vessel'].isin(['Lymphatics', 'Vein_Lymph', 'Art_Lymph'])]

# Clean
del arteries['type_vessel'], veins['type_vessel'], lymphatics['type_vessel']

# Groupby condition and calculate Proportions
vessels = [arteries, veins, lymphatics]
for idx, vals in enumerate([(arteries, 'ArtEC', 'Arteries'),
                            (veins, 'VeinEC', 'Veins'),
                            (lymphatics, 'LymphEC', 'Lymphatics')]):
    data, ct, vessel = vals
    # Groupby condition and calculate Proportions
    data = data.groupby(['condition', 'sample']).agg('sum').T
    # Compute proportions
    data = data / data.sum()
    # Select the CellType we want
    data = data.loc[ct, :].reset_index().dropna()
    # Merge Subpopulations
    data['Type'] = vessel
    data.columns = ['condition', 'sample', 'Proportions', 'Type']
    vessels[idx] = data
VesselsSpots = pd.concat(vessels, ignore_index=True)

# Generate Barplot
fig, axs = plt.subplots(1, 1, figsize=(6, 4))
bp = sns.barplot(VesselsSpots, x='Type', y='Proportions', hue='condition', estimator='mean',
                 palette={'Old': 'royalblue', 'Young': 'darkorange'}, hue_order=['Young', 'Old'],
                 capsize=0.1, errorbar='ci', gap=.1, ax=axs)
sns.stripplot(VesselsSpots, x='Type', y='Proportions', hue='condition', hue_order=['Young', 'Old'],
              ax=bp, color='black', dodge=True, alpha=.6)
# Layout
bp.set_xlabel('', fontsize=15, fontweight='bold')
bp.set_ylabel('Proportion', fontsize=15, fontweight='bold')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
for bar in bp.patches:
    bar.set_zorder(3)
# Remove extra legend handles from plotting strip on top of barplot
handles, labels = bp.get_legend_handles_labels()
bp.legend(handles[:2], labels[:2], title='Condition', bbox_to_anchor=(.5, 1.1), loc='upper center',
          frameon=False, ncols=2, title_fontproperties={'weight': 'bold'})
plt.savefig(os.path.join(figure_path, 'Fig4e_Barplot_VesselsAging.svg'), bbox_inches='tight')

# Test if there is a significant difference
for vessel in VesselsSpots['Type'].unique():
    sdata = VesselsSpots[VesselsSpots['Type'] == vessel]
    s, p = mannwhitneyu(sdata[sdata['condition'] == 'Old']['Proportions'],
                        sdata[sdata['condition'] == 'Young']['Proportions'],
                        use_continuity=True)
    print(vessel, p)
    # Arteries 0.42063492063492064
    # Veins 0.015873015873015872
    # Lymphatics 0.007936507936507936
# </editor-fold>


# <editor-fold desc="Figure 4f. Dotplot celltypes across vessels in aging">
# Generate an Anndata of celltype abundance
df = aging.obsm['c2l_prop'].copy()
adata = ad.AnnData(X=df.values, obs=list(df.index), var=list(df.columns))
adata.obs_names = adata.obs[0]
adata.var_names = adata.var[0]
adata.obs = aging.obs.copy()

for col in adata.var_names:  # Remove the celltype names in .obs otherwise there is redundancy
    del adata.obs[col]

adata.obs['vessel+cond'] = pd.Categorical(adata.obs.vessels.astype(str) + '_' + adata.obs.condition.astype(str))
adata = adata[adata.obs.vessels.isin(['Arteries', 'Veins', 'Lymphatics', 'nonVasc'])]  # Exclude Mix Vessels


# Generate the Dotplot
axs = sc.pl.dotplot(adata, groupby='vessel+cond', swap_axes=True,
                    var_names=['MP', 'Ccr2+MP', 'B_cells', 'T_cells',  # Celltypes of interest
                               'Fibroblasts', 'Fibro_activ', 'Adip'],
                    expression_cutoff=0.01, standard_scale='var',
                    categories_order=['Arteries_Young', 'Arteries_Old',
                                      'Veins_Young', 'Veins_Old',
                                      'Lymphatics_Young', 'Lymphatics_Old',
                                      'nonVasc_Young', 'nonVasc_Old'],
                    show=False, cmap=davidrUtility.generate_cmap('white', '#ff4d2e'),
                    colorbar_title='Scaled Mean prop\n in group', figsize=(5.8, 3.2))
# Layout
ax = axs['mainplot_ax']  # Get Main axs
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)

# Get Current Figure
fig = plt.gcf()
# Add a subplot of top to add the brackets
pos = ax.get_position()
top_ax = fig.add_axes([pos.x0, pos.y1, pos.width, 0.1])  # Adjust height as needed

# Set the categories
labels = ['Arteries', 'Veins', 'Lymphatics', 'nonVasc']
bracket_positions = [(0, 2), (2, 4), (4, 6), (6, 8)]  # Tuples of (start, end) indices of clusters to group

# Add brackets on top for the groups
for (i, (x_start, x_end)) in enumerate(bracket_positions):
    path = davidrUtility.create_bracket(x_start, x_end)
    patch = PathPatch(path, lw=2, fill=False)
    top_ax.add_patch(patch)
    label_position = (x_start + x_end) / 2
    top_ax.text(label_position, 1.5, labels[i], ha='center', va='center', fontsize=10, fontweight='bold')

# Set limits and remove spines
top_ax.set_xlim(-.5, 8.5)  # Adjust based on total number of var_names
top_ax.set_ylim(0, 2)
top_ax.set_xticks([])  # Remove the default ticks
top_ax.set_yticks([])
top_ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)

# Replace xticks of the mainplot
ax.set_xticklabels([txt.get_text().split('_')[-1] for txt in ax.get_xticklabels()], fontweight='bold', rotation=75,
                   ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Fig4f_Dotplot_Ct_AgingVessels.svg'), bbox_inches='tight')

# Test significance
cts = ['MP', 'Ccr2+MP', 'B_cells', 'T_cells', 'Fibroblasts', 'Fibro_activ', 'Adip']
for vessel in adata.obs.vessels.unique():
    sdata = adata[adata.obs.vessels == vessel]
    sc.tl.rank_genes_groups(sdata, groupby='condition', method='wilcoxon', tie_correct=True)
    table = sc.get.rank_genes_groups_df(sdata, group='Old', pval_cutoff=0.05, log2fc_min=0.2)
    table = table[table.names.isin(cts)]
    print(f'Results for {vessel} in Old\n')
    print(table)
    table = sc.get.rank_genes_groups_df(sdata, group='Old', pval_cutoff=0.05, log2fc_max=-0.2)
    table = table[table.names.isin(cts)]
    print(f'Results for {vessel} in Young\n')
    print(table)
# </editor-fold>


# Figure 4g --> Immunostaining


# <editor-fold desc="Figure 4h. Barplot quantification of Cd68+ cells in Aging (Stainings)">
row1 = [12, 10.2, 13, 11.7, 10.8, 14.3, 'Young', 'Adventitial']  # CD68+ (%) Adventitial - Young
row2 = [np.nan, 18, 18.1,  22.7, 18.7, 21,'Old', 'Adventitial']  # Cd68+ (%) Adventitial - Old
row3 = [6.3, 6.7, 6, 5.2, 4.9, 6.4, 'Young', 'Interstitial']  # CD68+ (%) Interstitital - Young
row4 = [np.nan, 5, 9.1, 4.8, 4.9, 6.5, 'Old', 'Interstitial']  # CD68+ (%) Interstitital - Old

df = pd.DataFrame([row1, row2, row3, row4], columns = ['Sample1', 'Sample2', 'Sample3', 'Sample4',
                                                       'Sample5', 'Sample6', 'Condition', 'Region'],
                  ).melt(id_vars=['Condition', 'Region'])


fig, axs = plt.subplots(1, 1, figsize=(12, 8))
bp = sns.barplot(df, x='Region', y='value', hue='Condition', order=['Interstitial', 'Adventitial'],
            hue_order=['Young', 'Old'], palette={'Young':'sandybrown', 'Old':'royalblue'},
            capsize=0.1, ax=axs, gap=0.1)
bp = sns.stripplot(df, x='Region', y='value', hue='Condition', order=['Interstitial', 'Adventitial'],
            hue_order=['Young', 'Old'], ax=axs, dodge=True, color='k', alpha=0.5)
bp.set_xlabel('')
bp.set_ylabel('Relative abundance\nCd68+ cells (%)')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
# Remove extra legend handles from plotting strip on top of barplot
handles, labels = bp.get_legend_handles_labels()
bp.legend(handles[:2], labels[:2], title='Condition', bbox_to_anchor=(.5, 1.1), loc='upper center',
          frameon=False, ncols=2, title_fontproperties={'weight': 'bold'})
plt.savefig(os.path.join(figure_path, 'Fig4g_Quantification_MP.svg'), bbox_inches='tight')

# Test significance
model = ols('value ~ C(Condition) * C(Region)', data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)

"""
                            sum_sq    df           F        PR(>F)
C(Condition)             72.662857   1.0   28.896831  3.460000e-05
C(Region)               555.137175   1.0  220.768988  6.503405e-12
C(Condition):C(Region)   89.157143   1.0   35.456340  9.890563e-06
Residual                 47.776667  19.0         NaN           NaN
"""

_, p = shapiro(row1[:6])  # pval = 0.911 --> does not follow normality
_, p = mannwhitneyu(row1[:6], row2[:6], nan_policy='omit')  # Test Adventitial
# Pval = 0.004329 --> 4.33e-03
_, p = mannwhitneyu(row3[:6], row4[:6], nan_policy='omit')  # Test Interstitital
# Pval = 0.7143 --> 0.71
# </editor-fold>


# <editor-fold desc="Figure 4i. Quantification of Cd68+ Lyve1+ cells in Aging Adventitial (Stainings)">
row1 = [85.2, 89.4, 91.9, 87.5, 83.3, 91.4, 'Young', 'Lyve1+', 'Adventitial']  #  Adventitial - Young
row2 = [72.6, 77.2, 73.9,86.1, 76.6, 68.1, 'Old', 'Lyve1+', 'Adventitial']  #  Adventitial - Old

df = pd.DataFrame([row1, row2], columns = ['Sample1', 'Sample2', 'Sample3', 'Sample4',
                                                       'Sample5', 'Sample6', 'Condition', 'Case', 'Region'],
                  ).melt(id_vars=['Condition', 'Case', 'Region'])

# Test for significance
_, p = shapiro(row1[:6])  # pval = 0.661  --> Does not follow normality
_, p = mannwhitneyu(row1[:6], row2[:6])

fig, axs = plt.subplots(1, 1, figsize=(4, 5))
bp = sns.barplot(df, x='Condition', y='value', order=['Young', 'Old'], palette={'Young':'sandybrown', 'Old':'royalblue'},
            capsize=0.1, ax=axs, gap=0.1)
davidrExperimental.plot_stats_adata(bp,
                                    aging, # It will be Ignored, we just take the framework to plot the stats, ignore the calculation part
                                    'condition',
                                    'Acta2', # It will be Ignored, we just take the framework to plot the stats, ignore the calculation part
                                    'Young',
                                    ['Old'],
                                    [p],
                                    text_size=15)
bp = sns.stripplot(df, x='Condition', y='value',  order=['Young', 'Old'], ax=axs, dodge=True, color='k', alpha=0.5)
bp.set_xlabel('')
bp.set_ylabel('Relative abundance\nCd68+/Lyve1+ cells (%)')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold', fontsize=15)
bp.set_title('Adventitial Lyve1+ MP')
plt.savefig(os.path.join(figure_path, 'Fig4h_Quantification_Adventitial_ResidentMP.svg'), bbox_inches='tight')
# </editor-fold>



# <editor-fold desc="Figure 4j . Quantification of Cd68+ Lyve1- cells in Aging Adventitial (Stainings)">
row1 = [14.8, 10.6, 8.1, 12.5, 16.7, 8.6, 'Young', 'Lyve1-', 'Adventitial']  #  Adventitial - Young
row2 = [27.4, 22.8, 26.1, np.nan, 23.4, 31.9, 'Old', 'Lyve1-', 'Adventitial']  #  Adventitial - Old

df = pd.DataFrame([row1, row2], columns = ['Sample1', 'Sample2', 'Sample3', 'Sample4',
                                                       'Sample5', 'Sample6', 'Condition', 'Case', 'Region'],
                  ).melt(id_vars=['Condition', 'Case', 'Region'])

# Test for significance
_, p = shapiro(row1[:6])  # pval = 0.661  --> Does not follow normality
_, p = mannwhitneyu(row1[:6], row2[:6], nan_policy='omit')


fig, axs = plt.subplots(1, 1, figsize=(4, 5))
bp = sns.barplot(df, x='Condition', y='value', order=['Young', 'Old'], palette={'Young':'sandybrown', 'Old':'royalblue'},
            capsize=0.1, ax=axs, gap=0.1)
davidrExperimental.plot_stats_adata(bp,
                                    aging, # It will be Ignored, we just take the framework to plot the stats, ignore the calculation part
                                    'condition',
                                    'Acta2', # It will be Ignored, we just take the framework to plot the stats, ignore the calculation part
                                    'Young',
                                    ['Old'],
                                    [p],
                                    text_size=15)
bp = sns.stripplot(df, x='Condition', y='value',  order=['Young', 'Old'], ax=axs,
                   dodge=True, color='k', alpha=0.5)
bp.set_xlabel('')
bp.set_ylabel('Relative abundance\nCd68+/Lyve1- cells (%)')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
bp.set_title('Adventitial Lyve1- MP')
plt.savefig(os.path.join(figure_path, 'Fig4i_Quantification_Adventitial_RecrutedMP.svg'), bbox_inches='tight')
# </editor-fold>


# Figure 5a --> Drawing made


# <editor-fold desc="Figure 5b. Senescence Score">
young1 = davidrUtility.select_slide(aging, 'Young_1')
old5 = davidrUtility.select_slide(aging, 'Old_5')
vmax = pd.concat([young1.obs.senescence, old5.obs.senescence])

# Young and Old representative for manuscript
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
ax1 = sc.pl.spatial(young1, color='senescence', size=1.5, cmap='RdBu_r', show=False, vmax=np.percentile(vmax, 99.2),
                    ax=axs[0], colorbar_loc=None, title='')[0]
ax2 = sc.pl.spatial(old5, color='senescence', size=1.5, cmap='RdBu_r', show=False, vmax=np.percentile(vmax, 99.2),
                    ax=axs[1], colorbar_loc=None, title='')[0]
# Layout
davidrUtility.axis_format(ax1, 'SP')
davidrUtility.axis_format(ax2, 'SP')

fig.text(0.2, 0.75, 'Young_1', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.2, 0.34, 'Old_5', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Add colorbar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.75, .15, 0.03, 0.13])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=-2.5, vmax=2.5)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Senescence\nScore', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels(['Min', 'Max'], fontweight='bold')
cbar.ax.grid(False)
plt.subplots_adjust(right=0.8)
plt.savefig(os.path.join(figure_path, 'Fig5b_SenescenceScore_YoungOld.svg'), bbox_inches='tight')

# </editor-fold>


# <editor-fold desc="Figure 5c. Density of Hotspots of senescence">
fig, axs = plt.subplots(2, 1, figsize=(15, 8))
fig.subplots_adjust(hspace=.3, wspace=.2, left=.05)
axs = axs.flatten()
for idx, batch in enumerate(['Young_1', 'Old_5']):
    sdata = davidrUtility.select_slide(aging, batch)

    # Replace SenescenceGradient values with number to determine the density
    sss = sdata.obs.senescence_gradient.replace({'Hspot': 12, 'dist100': 10, 'dist200': 8,
                                                 'dist300': 6, 'dist400': 4, 'dist500': 2, 'rest': 0})
    # Scale spatial embedding
    coords = pd.DataFrame(sdata.obsm['spatial']) * sdata.uns['spatial'][batch]['scalefactors']['tissue_hires_scalef']
    max_x, min_x = coords[0].max(), coords[0].min()
    max_y, min_y = coords[1].max(), coords[1].min()
    ax = sc.pl.spatial(sdata, color=None, bw=True, size=1.5, show=False, ax=axs[idx])[0]
    ax = sc.pl.spatial(sdata, color=None, bw=True, size=1.5, show=False, ax=ax, alpha_img=0)[0]

    sns.kdeplot(coords, x=0, y=1, weights=sss.values, fill=True, cmap='Reds', alpha=0.5, ax=ax,
                bw_adjust=.75, levels=8, cut=0)

    sc.pl.spatial(sdata, basis='spatial', color='senescence_gradient',
                  palette={'Hspot': 'firebrick', 'dist100': (1, 1, 1, 0), 'dist200': (1, 1, 1, 0),
                           'dist300': (1, 1, 1, 0), 'dist400': (1, 1, 1, 0), 'dist500': (1, 1, 1, 0),
                           'rest': (1, 1, 1, 0)},
                  size=1.5, show=False, ax=axs[idx], alpha_img=0, legend_loc=None, title='')

    davidrUtility.axis_format(axs[idx], 'SP')

plt.savefig(os.path.join(figure_path, 'Fig5c_DensityHspots.svg'), bbox_inches='tight', dpi=250)
# </editor-fold>


# <editor-fold desc="Figure 5d, Heatmap of celltypes across senescence gradient">
# CellTypes Enriched in Senescence Spots
df = aging.obsm['c2l_prop'].copy()
df[['gradient']] = aging.obs[['senescence_gradient']]

cm = sns.clustermap(
    df.groupby('gradient').agg('median').T.reindex(columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400',
                                                            'dist500', 'rest']),
    z_score=0, cmap='RdBu_r', center=0, cbar_pos=(0.2, 0.9, 0.18, 0.05), col_cluster=False,
    yticklabels=1, xticklabels=1, figsize=(5.2, 6.3), cbar_kws={'orientation': 'horizontal'},
    square=True, robust=True, linewidth=0.1, )
cm.ax_cbar.set_title('Z-score', fontsize=8)  # Adjust font size to 8 or desired value
cm.ax_heatmap.set_xlabel('')
cm.ax_heatmap.grid(False)
cm.ax_cbar.grid(False)
cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), rotation=45, ha='right', va='top', fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Fig5d_Heatmap_Cts_SenescenceGradient.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 5e. StackedBarplot of proportion of Hspots across niches in aging">
df = aging.obs.value_counts(['senescence_gradient', 'condition', 'clusters']).reset_index()
total = df[['condition', 'clusters', 'count']].groupby(['condition', 'clusters']).agg('sum')

# Compute proportions
df['norm'] = [row['count'] / total.loc[(row.condition, row.clusters)].values[0] for idx, row in df.iterrows()]

# SplitBy Condition
df_young = df[df['condition'] == 'Young']
df_old = df[df['condition'] == 'Old']
del df_young['condition'], df_old['condition']

# Sort index
df_young = df_young.pivot(index=['clusters'], columns='senescence_gradient',
                          values='norm').reindex(columns=['Hspot', 'dist100', 'dist200', 'dist300',
                                                          'dist400', 'dist500', 'rest'],
                                                 index=['Niche 7', 'Niche 0', 'Niche 1', 'Niche 2', 'Niche 8',
                                                        'Niche 5',
                                                        'Niche 6', 'Niche 10', 'Niche 3', 'Niche 9', 'Niche 4'])
df_old = df_old.pivot(index=['clusters'], columns='senescence_gradient',
                      values='norm').reindex(columns=['Hspot', 'dist100', 'dist200', 'dist300',
                                                      'dist400', 'dist500', 'rest'],
                                             index=['Niche 7', 'Niche 0', 'Niche 1', 'Niche 2', 'Niche 8', 'Niche 5',
                                                    'Niche 6', 'Niche 10', 'Niche 3', 'Niche 9', 'Niche 4'])
# Stackbarplot for young condition
fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = df_young.plot.bar(stacked=True, color={'Hspot': 'firebrick', 'dist100': 'tomato', 'dist200': 'lightsalmon',
                                            'dist300': 'royalblue', 'dist400': 'cornflowerblue',
                                            'dist500': 'lightsteelblue',
                                            'rest': 'sandybrown'}, ax=axs, width=.9)
sns.move_legend(ax, loc='center right', frameon=False, title='Senescence\nGradient',
                title_fontproperties={'weight': 'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'Fig5e_StackBarplotYoung.svg'), bbox_inches='tight')

# Stackedbarplot for old condition
fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = df_old.plot.bar(stacked=True, color={'Hspot': 'firebrick', 'dist100': 'tomato', 'dist200': 'lightsalmon',
                                          'dist300': 'royalblue', 'dist400': 'cornflowerblue',
                                          'dist500': 'lightsteelblue',
                                          'rest': 'sandybrown'}, ax=axs, width=.9)
sns.move_legend(ax, loc='center right', frameon=False, title='Senescence\nGradient',
                title_fontproperties={'weight': 'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'Fig5e_StackBarplotOld.svg'), bbox_inches='tight')
# </editor-fold>


# Figure 5f --> Staining


# Figure 5g --> Staining


# <editor-fold desc="Figure 5h. Heatmap of Immune inhibitory pathway and recruitment">
genes = ['Cd47', 'Sirpa', 'Cd24a', 'Pilra', 'Clec4a1', 'Clec12a',  # Immune inhibitory signals
         'Ccr2', 'Ccl2', 'Ccl4', 'Ccl5', 'Ccr5', 'Vcam1', 'Icam1']

# Mean Expression over the senescence gradient
df = davidrUtility.AverageExpression(aging, group_by=['senescence_gradient'], feature=genes, out_format='wide')
df = df.reindex(index=genes, columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'])
ndf = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1), axis=0)  # Scale --> (row - row.min) / row.max

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
hm1 = sns.heatmap(ndf, cmap='Reds', square=True, linewidths=.1, ax=axs, cbar=False, yticklabels=1, xticklabels=1)
# Set Axis Label
hm1.set_xlabel('')
hm1.set_ylabel('')
# Set Ticks
hm1.set_yticklabels(hm1.get_yticklabels(), fontsize=15)
hm1.set_xticklabels([txt.get_text().split('_')[0] for txt in hm1.get_xticklabels()], fontsize=15, fontweight='bold',
                    rotation=45, ha='right', va='top')
# Layout
hm1.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
# Create manually the cbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.8, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Expr', fontweight='bold', loc='left', fontsize=12)
plt.savefig(os.path.join(figure_path, 'Fig5h_Heatmap_ImmuneInhibitory.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 1a. ViolinPlot of QC metrics in the snRNA">
batch_colors = [('Young_B1', '#8c564b'), ('Young_B2', '#ffbb78'), ('Young_B3', '#98df8a'),
                ('Young_B4', '#ff9896'), ('Old_B1', '#1f77b4'), ('Old_B2', '#ff7f0e'),
                ('Old_B3', '#279e68'), ('Old_B4', '#d62728'), ('Old_B5', '#aa40fc'),
                ('Old_B6', '#e377c2'), ('Old_B7', '#b5bd61'), ('Old_B8', '#17becf'),
                ('Old_B9', '#aec7e8'), ]

# Get total_counts and nFeatures in log1p
data = ref.obs[['log1p_total_counts', 'log1p_n_genes_by_counts']].copy()
data['log(nUMIs)'] = data['log1p_total_counts']
data['log(nGenes)'] = data['log1p_n_genes_by_counts']
data['batch'] = pd.Categorical(ref.obs['batch'].copy(),  # Set the order
                               categories=['Young_B1', 'Young_B2', 'Young_B3', 'Young_B4',
                                           'Old_B1', 'Old_B2', 'Old_B3', 'Old_B4', 'Old_B5',
                                           'Old_B6', 'Old_B7', 'Old_B8', 'Old_B9', ], ordered=True)
data = data.sort_values('batch')

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for idx, value in enumerate(['log(nUMIs)', 'log(nGenes)']):
    vp = sns.violinplot(data, x='batch', y=value,
                        palette=dict(batch_colors),  # Use same colors as in the UMAP
                        saturation=.9, ax=axs[idx])
    # Layout and labels
    vp.set_ylabel(value, fontsize=18, fontweight='bold')
    vp.grid(False)
    sns.despine()
    if idx == 1:
        vp.set_xticklabels(vp.get_xticklabels(), rotation=75, fontweight='bold', ha='right', rotation_mode='anchor')
        vp.set_xlabel('')
plt.savefig(os.path.join(figure_path, f'ExtFig1a_QC_snRNA.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 1b. UMAP snRNA splitting by age">
random_indices = np.random.permutation(list(range(ref.shape[0])))  # Sort barcodes randomly
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'age', size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('Condition')
ax.legend(handles=[mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None, markersize=18) for lab, c in
                   zip(list(ref.obs.age.cat.categories[::-1]), ref.uns['age_colors'][::-1])],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'ExtFig1b_UMAP_snRNA_Age.svg'), bbox_inches='tight', dpi=300)
# </editor-fold>


# <editor-fold desc="Extended Figure 1c. UMAP snRNA splitting by batches">
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'batch', size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('Sample')
ax.legend(handles=[mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None, markersize=18) for lab, c in batch_colors],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'ExtFig1c_UMAP_snRNA_Batches.svg'), bbox_inches='tight', dpi=300)
# </editor-fold>


# <editor-fold desc="Extended Figure 1d. UMAP snRNA splitting by clusters">
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'leiden5', size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('Clusters')
ax.legend(handles=[mlines.Line2D([0], [0], marker=".", color=c, lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None, markersize=18) for lab, c in
                   zip(list(ref.obs.leiden5.cat.categories), ref.uns['leiden5_colors'])],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13, ncols=2)
plt.savefig(os.path.join(figure_path, f'ExtFig1d_UMAP_snRNA_Clusters.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 1e. UMAPs Subclustering ECs">
ec = ref[ref.obs.annotation.isin(['ArtEC', 'VeinEC', 'CapEC', 'EndoEC'])].copy()
bkn.bbknn(ec, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(ec)
sc.tl.leiden(ec, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)

ax = davidrPlotting.pl_umap(ec, ['annotation', 'leiden', 'Rgcc',
                                 'Stmn2', 'Vwf', 'Npr3'], size=20, ncols=3, figsize=(10, 8), show=False,
                            common_legend=True, legend_loc='on data', legend_fontsize=15, legend_fontweight=750,
                            legend_fontoutline=1.5)
ax[0].set_title('Annotation')
ax[1].set_title('Leiden Clusters')
plt.savefig(os.path.join(figure_path, f'ExtFig1e_UMAP_subclusteringEC.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 1f. UMAPs Subclustering MP">
md = ref[ref.obs.annotation.isin(['MP', 'Ccr2+MP'])].copy()
bkn.bbknn(md, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(md)
sc.tl.leiden(md, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)

ax = davidrPlotting.pl_umap(md, ['annotation', 'leiden', 'Lyve1', 'Ccr2'], size=40, ncols=2, figsize=(10, 8),
                            show=False, common_legend=True, legend_loc='on data', legend_fontsize=15,
                            legend_fontweight=750, legend_fontoutline=1.5)
ax[0].set_title('Annotation')
ax[1].set_title('Leiden Clusters')
plt.savefig(os.path.join(figure_path, f'ExtFig1f_UMAP_SubclusteringMP.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 2a. Violinplots QC ST">
sp_batch_colors = [('Young_1', '#8c564b'), ('Young_2', '#e377c2'), ('Young_3', '#7f7f7f'),
                   ('Young_4', '#bcbd22'), ('Young_5', '#17becf'), ('Old_1', '#1f77b4'),
                   ('Old_2', '#ff7f0e'), ('Old_3', '#2ca02c'), ('Old_4', '#d62728'),
                   ('Old_5', '#9467bd'), ]

data = aging.obs[['log1p_total_counts', 'log1p_n_genes_by_counts']]
data['log(nUMIs)'] = data['log1p_total_counts']  # Do the log as we did for the snRNA
data['log(nGenes)'] = data['log1p_n_genes_by_counts']
data['batch'] = pd.Categorical(aging.obs['sample'], categories=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                                                'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'],
                               ordered=True)
data = data.sort_values('batch')

sp_batch_colors = {key: aging.uns['sample_colors'][idx] for idx, key in enumerate(aging.obs['sample'].cat.categories)}

# Violin Plot showing the QC Metrics
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for idx, value in enumerate(['log(nUMIs)', 'log(nGenes)']):
    vp = sns.violinplot(data, x='batch', y=value,
                        palette=sp_batch_colors,
                        saturation=.9, ax=axs[idx])
    # Layout and labels
    vp.set_ylabel(value, fontsize=18, fontweight='bold')
    vp.grid(False)
    sns.despine()
    if idx == 1:
        vp.set_xticklabels(vp.get_xticklabels(), rotation=75, fontweight='bold', ha='right', rotation_mode='anchor')
        vp.set_xlabel('')
plt.savefig(os.path.join(figure_path, 'ExtFig2a_ViolinPlot_QC_ST.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 2b. Spatial distribution of log(nUMI) in Young">
coord_sample = {'Old_1': (700.0, 6000.0, 1200.0, 6100.0),
                'Old_2': (500, 5000.0, 1400, 5800.0),
                'Old_3': (200.0, 6100.0, 1000.0, 6200.0),
                'Old_4': (900.0, 6800.0, 850.0, 6050.0),
                'Old_5': (400, 5300.0, 800.0, 5100.0),
                'Young_1': (600.0, 6500.0, 600.0, 5800.0),
                'Young_2': (600.0, 6500.0, 800.0, 6000.0),
                'Young_3': (600.0, 6500.0, 800.0, 6000.0),
                'Young_4': (1100, 5400, 1300, 6000),
                'Young_5': (1100, 4600, 1000, 4500),
                }

fig, axs = plt.subplots(2, 5, figsize=(15, 8))
plt.subplots_adjust(hspace=.05, wspace=.15, left=.05)  # Spacing between subplots
axs = axs.flatten()

for idx, sample in enumerate(['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5']):
    slide = davidrUtility.select_slide(aging, sample)
    sc.pl.spatial(slide, color=None, ax=axs[idx], colorbar_loc=None, crop_coord=coord_sample[sample])
    sc.pl.spatial(slide, color='log1p_total_counts', cmap='inferno', size=1.5, ax=axs[idx + 5],
                  vmax=np.max(aging.obs['log1p_total_counts']),
                  vmin=np.min(aging.obs['log1p_total_counts']),
                  colorbar_loc=None, crop_coord=coord_sample[sample], title='')
    for ax in [axs[idx], axs[idx + 5]]:
        davidrUtility.axis_format(ax, txt='SP')
    axs[idx].set_title(sample, fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.43, .075, 0.15, 0.05])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=aging.obs.log1p_total_counts.min(), vmax=aging.obs.log1p_total_counts.max())
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('log1p(nUMI)', fontweight='bold', loc='center')
cbar.set_ticks([aging.obs.log1p_total_counts.min(),
                aging.obs.log1p_total_counts.median(),
                aging.obs.log1p_total_counts.max()])
cbar.set_ticklabels(['{:.2f}'.format(aging.obs.log1p_total_counts.min()),
                     '{:.2f}'.format(aging.obs.log1p_total_counts.median()),
                     '{:.2f}'.format(aging.obs.log1p_total_counts.max())], fontweight='bold')
cbar_ax.grid(False)
plt.savefig(os.path.join(figure_path, f'ExtFig2b_Spatial_log_counts_Young.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 2c. Spatial distribution of log(nUMI) in Old">
fig, axs = plt.subplots(2, 5, figsize=(15, 8))
plt.subplots_adjust(hspace=.05, wspace=.15, left=.05)  # Spacing between subplots
axs = axs.flatten()
cont = 0
for sample in ['Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5']:
    slide = davidrUtility.select_slide(aging, sample)
    sc.pl.spatial(slide, color=None, ax=axs[cont], colorbar_loc=None, crop_coord=coord_sample[sample])
    sc.pl.spatial(slide, color='log1p_total_counts', cmap='inferno', size=1.5, ax=axs[cont + 5],
                  vmax=np.max(aging.obs['log1p_total_counts']),
                  vmin=np.min(aging.obs['log1p_total_counts']),
                  colorbar_loc=None, crop_coord=coord_sample[sample])
    for ax in [axs[cont], axs[cont + 5]]:
        davidrUtility.axis_format(ax, txt='SP')
    axs[cont].set_title(sample, fontsize=18, fontweight='bold')
    cont += 1

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.43, .075, 0.15, 0.05])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=aging.obs.log1p_total_counts.min(), vmax=aging.obs.log1p_total_counts.max())
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('log1p(nUMI)', fontweight='bold', loc='center')
cbar.set_ticks([aging.obs.log1p_total_counts.min(), aging.obs.log1p_total_counts.median(),
                aging.obs.log1p_total_counts.max()])
cbar.set_ticklabels(['{:.2f}'.format(aging.obs.log1p_total_counts.min()),
                     '{:.2f}'.format(aging.obs.log1p_total_counts.median()),
                     '{:.2f}'.format(aging.obs.log1p_total_counts.max())], fontweight='bold')
cbar_ax.grid(False)
plt.savefig(os.path.join(figure_path, f'ExtFig2c_Spatial_log_counts_Old.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Figure 2d. UMAP ST splitting by batch">
random_indices = np.random.permutation(list(range(aging.shape[0])))  # Sort barcodes randomly
ax = davidrPlotting.pl_umap(aging[random_indices, :], 'sample', size=15, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('Sample')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c, lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in sp_batch_colors],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'ExtFig2d_UMAP_ST_splitting_Batch.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 2e. UMAP ST splitting by condition">
ax = davidrPlotting.pl_umap(aging[random_indices, :], 'condition',
                            size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('Condition')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c, lw=0, label=lab, markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(aging.obs.condition.cat.categories[::-1]), aging.uns['condition_colors'][::-1])],
          loc='center right', frameon=False, edgecolor='black', title='',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'ExtFig2e_UMAP_ST_Splitting_Condition.svg'), bbox_inches='tight', dpi=300)
# </editor-fold>


# <editor-fold desc="Extended Figure 2f. Spatial distribution of niches">
ax = davidrPlotting.plot_slides(aging, 'clusters', bw=True, common_legend=True, minimal_title=True,
                                select_samples=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5', 'Old_1', 'Old_2',
                                                'Old_3', 'Old_4', 'Old_5'],
                                order=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5', 'Old_1', 'Old_2', 'Old_3',
                                       'Old_4', 'Old_5'],
                                title_fontweight='bold', show=False, ncols=5)
ax[4].legend().set_visible(False)
plt.savefig(os.path.join(figure_path, 'ExtFig2f_ST_Clusters.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 3g. Pseudobulk">
aging_copy = aging[:, aging.var.highly_variable].copy()
# Generate PseudoBulk
pdata = dc.get_pseudobulk(aging_copy, sample_col='sample', groups_col='clusters',
                          layer='counts', mode='sum', min_cells=10, min_counts=1000)
pdata.layers['counts'] = pdata.X.copy()

#  dc.plot_psbulk_samples(pdata, groupby=['sample', 'clusters'], figsize=(12, 4))

# Normalise, scale and pca
sc.pp.normalize_total(pdata, target_sum=10_000)
sc.pp.log1p(pdata)
sc.pp.scale(pdata, max_value=10)
sc.tl.pca(pdata)

dc.swap_layer(pdata, 'counts', X_layer_key=None, inplace=True)  # Return to counts

sc.pl.pca(pdata, color=['sample', 'clusters'])
sc.pl.pca_variance_ratio(pdata)

dc.get_metadata_associations(pdata,
                             obs_keys=['sample', 'condition', 'clusters', 'psbulk_n_cells', 'psbulk_counts'],
                             # Metadata columns to associate to PCs
                             obsm_key='X_pca',  # Where the PCs are stored
                             uns_key='pca_anova',  # Where the results are stored
                             inplace=True, )

# Plot the association between PCs and metadata
dc.plot_associations(pdata,
                     uns_key='pca_anova',  # Summary statistics from the anova tests
                     obsm_key='X_pca',  # where the PCs are stored
                     stat_col='p_adj',  # Which summary statistic to plot
                     obs_annotation_cols=['sample', 'condition', 'clusters'],  # which sample annotations to plot
                     titles=['Principle component scores', 'Adjusted p-values from ANOVA'],
                     figsize=(7, 5), n_factors=10, cmap_cats='tab20c', )
plt.savefig(os.path.join(figure_path, 'ExtFig2g_PseudoBulk.svg'), bbox_inches='tight')

# Generate a PCA Plot
pca = pdata.obsm['X_pca'].copy()
meta = pdata.obs[['clusters', 'condition']]
niche_color = {pdata.obs['clusters'].cat.categories[i]: pdata.uns['clusters_colors'][i] for i in
               range(len(pdata.obs.clusters.unique()))}
cond_color = {'Young': 'o', 'Old': 'x'}
variance_ratio = pdata.uns['pca']['variance_ratio'][:2] * 100

# Scatterplot of the PCA
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i, bc in enumerate(pdata.obs_names):
    niche = pdata.obs.loc[bc, 'clusters']
    cond = pdata.obs.loc[bc, 'condition']
    ax.scatter(pca[i, 0], pca[i, 1], color=niche_color[niche],
               marker=cond_color[cond], s=100)
# Layout
ax.grid(False)
ax.set_xlabel(f'PC1 ({str(round(variance_ratio[0], 2))} %)')
ax.set_ylabel(f'PC2 ({str(round(variance_ratio[1], 2))} %)')
ax.set_xticks([])
ax.set_yticks([])
ax.legend()  # Remove legend
# Add Manually the legend
color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=niche_color[n], markersize=10, label=n)
                 for n in niche_color]
marker_handles = [plt.Line2D([0], [0], marker=m, color='k', linestyle='None', markersize=10, label=c)
                  for c, m in cond_color.items()]
color_legend = ax.legend(handles=color_handles, title='Clusters', loc='upper left', bbox_to_anchor=(1, .5),
                         title_fontproperties={'weight': 'bold', 'size': 13})
ax.add_artist(color_legend)  # To avoid replacing the first legend
ax.legend(handles=marker_handles, title='Conditions', loc='upper left', bbox_to_anchor=(1, 0.9),
          title_fontproperties={'weight': 'bold', 'size': 13})
ax.set_title('PCA Pseudobulk', fontsize=20)
plt.savefig(os.path.join(figure_path, 'ExtFig2g_PseudoBulk_Scatterplot.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 2h. Barplot proportion celltypes">
df = aging.obsm['c2l'].copy()
df = df.melt().groupby('variable').agg('sum') / df.melt().groupby('variable').agg('sum').sum()
df = df.sort_values('value', ascending=False)

# Create the barplot
plt.figure(figsize=(8, 5))
ax = sns.barplot(df, x='variable', y='value', palette=ct_zip)
# Add annotations
for i, value in enumerate(df['value']):
    ax.text(i + 0.3, value + 0.01, f'{value:.3f}', rotation=45, ha='center', va='bottom', fontsize=15,
            fontweight='bold')

ax.set_xlabel('')
ax.set_ylabel('Proportion')
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right', va='top', fontweight='bold')
plt.savefig(os.path.join(figure_path, 'ExtFig2h_Barplot_prop_cts.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 3a. ST Anatomic Regions">
labels_color = {'LVi': 'lightsteelblue', 'LVm': 'cornflowerblue', 'LVo': 'royalblue',
                'RV': '#f6b278', 'SEP': 'salmon', 'BG': 'whitesmoke'}

ax = davidrPlotting.plot_slides(aging, 'AnatomicRegion', ncols=5, title_fontweight='bold',
                                order=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                       'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'],
                                show=False, palette=labels_color, bw=True)

# Adjust legend manually to add missing BG and only have 1 legend
ax[4].legend(handles=[mlines.Line2D([0], [0],
                                    marker=".", color=c,
                                    lw=0, label=lab,
                                    markerfacecolor=c, markeredgecolor='k',
                                    markersize=20, markeredgewidth=.5) for lab, c in labels_color.items()],
             loc='lower right', frameon=False, edgecolor='black', title='Regions',
             title_fontproperties={'size': 16, 'weight': 'bold'},
             bbox_to_anchor=(1.5, -.75), fontsize=13, ncols=1)
ax[9].legend([]).set_visible(False)
plt.savefig(os.path.join(figure_path, 'ExtFig3a_ST_AnatomicRegions.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 3b. Barplot proportion niches per sample">
# Df --> Clusters per sample
df = aging.obs.value_counts(['clusters', 'sample']).sort_index().reset_index()
df['prop'] = df['count'] / df.groupby('sample')['count'].transform('sum')
df_pivot = df.pivot(index='sample', columns='clusters', values='prop')
df_pivot = df_pivot.reindex(index=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                   'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'])

fig, axs = plt.subplots(1, 1, figsize=(6, 5))
df_pivot.plot(kind='bar', stacked=True, grid=False, rot=75, ax=axs, width=0.9)
axs.set_ylabel('Proportions', fontsize=18, fontweight='bold')
axs.set_xticklabels(axs.get_xticklabels(), fontsize=15, fontweight='bold')
axs.set_xlabel('')
sns.move_legend(axs, loc='center right', frameon=False, title='Clusters',
                title_fontproperties={'weight': 'bold', 'size': 15},
                ncols=1, bbox_to_anchor=(1.2, .5))
plt.savefig(os.path.join(figure_path, 'ExtFig3b_Proportion_Niches_Sample.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 3c. Proportion niches during aging">
df = aging.obs[['clusters', 'sample']].copy()
df_cond = df.value_counts(['sample', 'clusters']).reset_index()

# Normalise by sample
df_groups = df_cond[['sample', 'count']].groupby('sample').agg('sum')
df_cond['norm'] = [cell['count'] / df_groups.loc[cell['sample'], 'count'] for idx, cell in df_cond.iterrows()]
df_cond['condition'] = df_cond['sample'].str.split('_').str[0]

fig, axs = plt.subplots(1, 1, figsize=(25, 9))
bp = sns.barplot(df_cond, x='clusters', y='norm', capsize=.1, palette={'Young': 'darkorange', 'Old': 'royalblue'},
                 hue_order=['Young', 'Old'], hue='condition', ax=axs)
bp.set_xlabel('')
bp.set_ylabel('Mean Proportion', fontsize=18)
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold', fontsize=15)
sns.move_legend(bp, loc='upper right', ncols=2, frameon=False, title='Condition',
                title_fontproperties={'weight': 'bold', 'size': 20}, fontsize=15)
bp.grid(False)
plt.savefig(os.path.join(figure_path, f'ExtFig3c_ProportionNichesAging.svg'), bbox_inches='tight')


# Test significance
for clust in df_cond.clusters.unique():
    sdf = df_cond[df_cond.clusters == clust]
    _, p = mannwhitneyu(sdf[sdf.condition == 'Young'].norm, sdf[sdf.condition == 'Old'].norm)
    print(f'Pval for {clust}: {round(p, 2)}')
# </editor-fold>


# <editor-fold desc="Extended Figure 4a-e and Extended Figure 5c-h">
# Correction when the Terms are too long
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
                   color_up=[],
                   color_down=[],
                   title: str = 'Top 10 GO Terms in each Condition',
                   show: bool = True) -> plt.axis :
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
                         left=-spacing, color=color_down,
                         align='center', alpha=.25)

    # Plot bars for "Up" condition (negative values) on the right side
    bars_up = axs.barh(y_pos,
                       df_pos[col_split].sort_values(),
                       left=spacing, color=color_up,
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


logger = davidrUtility.config_verbose(True)

df = pd.read_excel(os.path.join(table_path, '241017_DecoupleR_GOBP_ORA_PseudoBulk_Young_Vs_Old_Niches.xlsx'))

# Take the top 20 per cluster and condition sorting by combined score
# Additionally, we filter by the FDR < 0.05
top_terms = pd.DataFrame([])
for cluster in df['cluster'].unique():
    for state in df['state'].unique():
        sdf = df[(df['cluster'] == cluster) & (df['state'] == state)]
        sdf_filt = sdf[sdf['FDR p-value'] < 0.05].sort_values('Combined score', ascending=False)
        sdf_filt['Term'] = sdf_filt['Term'].str.replace('_', ' ')
        top_terms = pd.concat([top_terms, sdf_filt.head(10)])

# palegreen --> 'Immune process'
# tomato --> Muscle related
# sandybrown --> ECM and cell juction


colors_dict = {'Niche 0': {'color_up': ['lightgray', 'palegreen', 'palegreen', 'lightgray', 'palegreen',
                                        'lightgray', 'palegreen', 'lightgray', 'lightgray', 'sandybrown'],
                           'color_down': ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                          'lightgray', 'lightgray', 'tomato', 'lightgray', 'lightgray']},
               'Niche 1': {'color_up': ['lightgray', 'lightgray', 'lightgray', 'palegreen', 'lightgray',
                                        'lightgray', 'lightgray', 'palegreen', 'sandybrown', 'palegreen'],
                           'color_down': ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                          'sandybrown', 'sandybrown', 'sandybrown', 'lightgray', 'tomato']},
               'Niche 2': {'color_up': ['lightgray', 'sandybrown', 'lightgray', 'palegreen', 'lightgray',
                                        'palegreen', 'lightgray', 'lightgray', 'lightgray', 'palegreen'],
                           'color_down': ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                          'lightgray', 'palegreen', 'lightgray', 'lightgray', 'lightgray']},
               'Niche 3': {'color_up': ['lightgray', 'lightgray', 'tomato', 'tomato', 'lightgray',
                                        'lightgray', 'lightgray', 'tomato', 'lightgray', 'lightgray'],
                           'color_down': ['sandybrown', 'sandybrown', 'sandybrown', 'tomato', 'lightgray',
                                          'lightgray', 'sandybrown', 'sandybrown', 'lightgray', 'tomato']},
               'Niche 4': {'color_up': ['lightgray', 'tomato', 'palegreen', 'tomato', 'tomato',
                                        'lightgray', 'tomato', 'lightgray', 'lightgray', 'lightgray'],
                           'color_down': ['lightgray', 'lightgray', 'tomato', 'tomato', 'sandybrown',
                                          'lightgray', 'lightgray', 'tomato', 'sandybrown', 'sandybrown']},
               'Niche 5': {'color_up': ['lightgray', 'palegreen', 'palegreen', 'tomato', 'lightgray',
                                        'palegreen', 'sandybrown', 'lightgray', 'lightgray', 'lightgray'],
                           'color_down': ['lightgray', 'lightgray', 'tomato', 'lightgray', 'lightgray',
                                          'lightgray', 'lightgray', 'lightgray', 'lightgray', 'tomato']},
               'Niche 6': {'color_up': ['lightgray', 'lightgray', 'tomato', 'palegreen', 'lightgray',
                                        'lightgray', 'sandybrown', 'lightgray', 'lightgray', 'lightgray'],
                           'color_down': ['lightgray', 'palegreen', 'lightgray', 'tomato', 'palegreen',
                                          'lightgray', 'lightgray', 'sandybrown', 'sandybrown', 'palegreen']},
               'Niche 7': {'color_up': ['lightgray', 'lightgray', 'lightgray', 'palegreen', 'sandybrown',
                                        'lightgray', 'palegreen', 'lightgray', 'lightgray', 'sandybrown'],
                           'color_down': ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                          'lightgray', 'tomato', 'sandybrown', 'sandybrown', 'tomato']},
               'Niche 8': {'color_up': ['sandybrown', 'lightgray', 'tomato', 'sandybrown', 'lightgray',
                                        'lightgray', 'lightgray', 'lightgray', 'sandybrown', 'lightgray'],
                           'color_down': ['lightgray', 'tomato', 'sandybrown', 'sandybrown', 'sandybrown',
                                          'lightgray', 'tomato', 'lightgray', 'sandybrown', 'sandybrown']},
               'Niche 9': {'color_up': ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                        'lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray'],
                           'color_down': ['sandybrown', 'lightgray', 'tomato', 'lightgray', 'sandybrown',
                                          'sandybrown', 'lightgray', 'lightgray', 'lightgray', 'sandybrown']},
               'Niche 10': {'color_up': ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
                                         'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray'],
                            'color_down': ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                                           'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']}}

for figname, cluster in [('ExtFig4a', 'Niche 7'), ('ExtFig4b', 'Niche 0'), ('ExtFig4c', 'Niche 1'),
                         ('ExtFig4d', 'Niche 2'),
                         ('ExtFig4e', 'Niche 10'), ('ExtFig5c', 'Niche 3'), ('ExtFig5d', 'Niche 6'),
                         ('ExtFig5e', 'Niche 4'),
                         ('ExtFig5f', 'Niche 5'), ('ExtFig5g', 'Niche 8'), ('ExtFig5h', 'Niche 9')]:
    sdf = top_terms[top_terms.cluster == cluster]

    split_bar_gsea(sdf,
                   'Term',
                   'Combined score',
                   'state',
                   'up',
                   spacing=15,
                   cutoff=45,
                   color_up=colors_dict[cluster]['color_up'],
                   color_down=colors_dict[cluster]['color_down'],
                   figsize=(12, 6),
                   title=f'Top 10 GO Terms in each Condition {cluster}',
                   path=figure_path,
                   filename=f'{figname}_SplitBar{cluster}.svg')

# </editor-fold>


# <editor-fold desc="Extended Figure 4f. Dotplot MP in niches">
# Generate an AnnData of the cell type abundance
df = aging.obsm['c2l'].copy()
adata = ad.AnnData(X=df.values, obs=list(df.index), var=list(df.columns))
adata.obs_names = adata.obs[0]
adata.var_names = adata.var[0]
adata.obs = aging.obs.copy()
# Remove duplicates (celltype is in obs and var_names)
for col in adata.var_names:
    del adata.obs[col]

adata.obs['cluster+Cond'] = pd.Categorical(adata.obs['clusters'].astype(str) + '_' + adata.obs['condition'].astype(str))

axs = sc.pl.dotplot(adata, groupby='cluster+Cond', swap_axes=True, var_names=['MP', 'Ccr2+MP'],
                    expression_cutoff=0.01,
                    categories_order=['Niche 0_Young', 'Niche 0_Old', 'Niche 1_Young', 'Niche 1_Old',
                                      'Niche 2_Young', 'Niche 2_Old', 'Niche 3_Young', 'Niche 3_Old',
                                      'Niche 4_Young', 'Niche 4_Old', 'Niche 5_Young', 'Niche 5_Old',
                                      'Niche 6_Young', 'Niche 6_Old', 'Niche 7_Young', 'Niche 7_Old',
                                      'Niche 8_Young', 'Niche 8_Old','Niche 9_Young', 'Niche 9_Old',
                                      'Niche 10_Young', 'Niche 10_Old'],
                    show=False, cmap='Reds', colorbar_title='Mean prop\n in group',
                    size_title='Fraction of spots\nin group(%)', figsize=(7.8, 2.4))
# Layout
ax = axs['mainplot_ax']
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
ax.set_xticklabels([txt.get_text().split('_')[-1] for txt in ax.get_xticklabels()], fontweight='bold')
axs['color_legend_ax'].grid(False)
# Add brackets on top with the categories
fig = plt.gcf()
# Add a subplot of top to add the brackets
pos = ax.get_position()
top_ax = fig.add_axes([pos.x0, pos.y1, pos.width, 0.1])  # Adjust height as needed

# Set the categories
labels = ['Niche 0', 'Niche 1', 'Niche 2', 'Niche 3', 'Niche 4', 'Niche 5',
          'Niche 6', 'Niche 7', 'Niche 8', 'Niche 9', 'Niche 10']
bracket_positions = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
                     (12, 14), (14, 16), (16, 18), (18, 20), (20, 22)]  # Tuples of (start, end) indices of clusters to group

# Add brackets on top for the groups
for (i, (x_start, x_end)) in enumerate(bracket_positions):
    path = davidrUtility.create_bracket(x_start, x_end)
    patch = PathPatch(path, lw=2, fill=False)
    top_ax.add_patch(patch)
    label_position = (x_start + x_end) / 2
    top_ax.text(label_position+0.5, 1.5, labels[i], ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=45)

# Set limits and remove spines
top_ax.set_xlim(-.5, 22.5)  # Adjust based on total number of var_names
top_ax.set_ylim(0, 2)
top_ax.set_xticks([])  # Remove the default ticks
top_ax.set_yticks([])
top_ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
plt.savefig(os.path.join(figure_path, 'ExtFig4f_Dotplot_MPCcr2MP_Niches.svg'), bbox_inches='tight')

# Test Significance
for clust in adata.obs.clusters.unique():
    tdata = adata[adata.obs.clusters == clust]
    sc.tl.rank_genes_groups(tdata, groupby='condition', method='wilcoxon', tie_correct=True)
    tdf = sc.get.rank_genes_groups_df(tdata, group='Old', pval_cutoff=0.05)
    tdf = tdf[tdf.names.isin(['MP', 'Ccr2+MP'])]
    print (clust, tdf, '\n\n')
# </editor-fold>


# <editor-fold desc="Extended Figure 5a. Dotplot genes associated to microglia GO terms">
genes = ['Grn', 'Il33', 'Trem2', 'Tyrobp']
ax = davidrScanpy.dotplot(ref, groupby='annotation', var_names=genes, show=False, figsize=(2.85, 5))
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), fontweight='bold', rotation=45, ha='right', va='top', fontsize=13)
ax['color_legend_ax'].grid(False)
plt.savefig(os.path.join(figure_path, 'ExtFig5a_MicrogliaGenes.svg'), bbox_inches='tight')
#</editor-fold>


# <editor-fold desc="Extended Figure 5b. MatrixPlot UpGenes in MP/Ccr2+MP">
vidal = ref[ref.obs.Experiment == 'JulianAging'].copy()
# DGE Young Vs Old in MP
vidal_mp = vidal[vidal.obs.annotation.isin(['MP'])]
sc.tl.rank_genes_groups(vidal_mp, groupby='age', method='wilcoxon', tie_correct=True)
table = sc.get.rank_genes_groups_df(vidal_mp, group='Old', pval_cutoff=0.05, log2fc_min=.25)

# DGE Young Vs Old in Niche 7
niche7 = aging[aging.obs.clusters == 'Niche 7']
sc.tl.rank_genes_groups(niche7, groupby='condition', method='wilcoxon', tie_correct=True, layer='SCT_norm')
table7 = sc.get.rank_genes_groups_df(niche7, group='Old', pval_cutoff=0.05, log2fc_min=.25)

# Intersect of both
set(table7.names) & set(table.names) #  Genes Upregulated in Both

# Manually explore genes associated to M1/M2 phenotype
table.to_excel(os.path.join(table_path, 'DGE_MyeloidYoungOld_Vidal.xlsx'), index=False)


genes = ['Fgr', 'Irf7', 'Bst2', 'Cd209g', 'Cd209f', 'Ccl8', 'Ccl6', 'Timp2', 'Vsig4']

ax = sc.pl.matrixplot(aging, groupby='clusters', var_names=genes, standard_scale='var', cmap='Reds',show=False,
                      colorbar_title='Scaled Mean\nExpression in group')
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['color_legend_ax'].grid(False)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'ExtFig5b_M1M2MP_UpOldMP_UpOldNiche7.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 6c and 6d (lineplots)">
# Compute distribution
cellInteraction = pd.DataFrame()  # Bc --> Proportion of Ct in Intra/Hexa/Ext space
for batch in tqdm(aging.obs['sample'].unique()):
    sdata = davidrUtility.select_slide(aging, batch)  # Subset by sample
    for idx, bc in enumerate(sdata.obs_names):
        # Get proportion in intra space
        intra_c2l = sdata.obsm['c2l_prop'].loc[bc, :]
        intra_c2l.index = intra_c2l.index + '_IntraSpace'

        # Get Proportions in Hexa Space - Spots that are 100 um away
        neigh_BCs = list(davidrUtility.get_surrounding(sdata, idx, radius=100, get_bcs=True))
        neigh_BCs.remove(bc)  # Exclude the intra spot
        norm_hexa = sdata.obsm['c2l'].loc[neigh_BCs,:].sum().sum()
        hexa_c2l = sdata.obsm['c2l'].loc[neigh_BCs,:].sum(axis=0) / norm_hexa  # Compute Proportion
        hexa_c2l.index = hexa_c2l.index + '_HexaSpace'

        # Get Proportion in Extended Space - Spots that are 200 um away
        extend_BCs = list(davidrUtility.get_surrounding(sdata, idx, radius=200, get_bcs=True))
        extend_BCs.remove(bc)  # exclude the intra spot
        extend_BCs = [val for val in extend_BCs if val not in neigh_BCs]  # exclude the hexa spots
        norm_extend = sdata.obsm['c2l'].loc[extend_BCs,:].sum().sum()
        extend_c2l = sdata.obsm['c2l'].loc[extend_BCs,:].sum(axis=0) / norm_extend  # compute proportion
        extend_c2l.index = extend_c2l.index + '_ExtendedSpace'

        new_row = pd.DataFrame(pd.concat([intra_c2l, hexa_c2l, extend_c2l]), columns=[bc]).T
        new_row['Condition'] = batch.split('_')[0]

        cellInteraction = pd.concat([cellInteraction, new_row])

# Convert to wide format
data_wide = cellInteraction.reset_index().melt(id_vars=['Condition', 'index'], var_name='Group', value_name='Proportion')
data_wide['space'] = data_wide['Group'].str.split('_').str[-1]
data_wide['CellTypes'] = data_wide['Group'].str.split('_IntraSpace').str[0].str.split('_HexaSpace').str[0].str.split('_ExtendedSpace').str[0]
# Duplicate df to have the same but in reverse to generate a simetric plot
tmp = data_wide.copy()
tmp['space'] = '-' + tmp['space']
tmp =tmp[tmp.space !='-Intra']  # Exclude the intra spot, only once
data_wide = pd.concat([data_wide, tmp]).reset_index()
data_wide['space'] = pd.Categorical(data_wide.space, categories=['-ExtendedSpace', '-HexaSpace', 'IntraSpace', 'HexaSpace', 'ExtendedSpace'], ordered=True)


# CM distribution compared to rest of cell types
cm_all = data_wide[data_wide['CellTypes'].isin(['Fibroblasts','CapEC', 'SMC', 'Ccr2+MP', 'CM'])].copy()
# Get the BCs with the top 8 % of CM proportion
tmp = cm_all[cm_all['Group'] == 'CM_IntraSpace']
tmp = tmp.loc[tmp['Proportion'] > np.percentile(tmp['Proportion'], 92), 'index']
cm_all = cm_all[cm_all['index'].isin(tmp)].copy()

# Extended Figure 6c
fig, ax1 = plt.subplots(1, 2, figsize=(6, 3))
fig.subplots_adjust(wspace=0.5)
sns.lineplot(cm_all[cm_all.CellTypes.isin(['CM'])], x='space', y='Proportion', hue='CellTypes', estimator='mean',
             markers=True, dashes=True, palette=ct_zip, ax=ax1[0])
sns.lineplot(cm_all[cm_all.CellTypes.isin(['Fibroblasts', 'CapEC', 'SMC', 'Ccr2+MP'])],
             x='space', y='Proportion', hue='CellTypes', estimator='mean',
             markers=True, dashes=True, palette=ct_zip, ax=ax1[1])
for idx in range(2):
    ax1[idx].set_xlabel('')
    ax1[idx].set_ylabel('Mean Proportion')
    ax1[idx].set_xticklabels(['Extended', 'Hexa', 'SameSpot', 'Hexa', 'Extended'], rotation=75, ha='right', va='top')

handles, labels = ax1[0].get_legend_handles_labels()
handles2, labels2 = ax1[1].get_legend_handles_labels()
handles.extend(handles2)
labels.extend(labels2)

# Remove duplicates from the labels (keep unique ones)
unique_labels = list(dict.fromkeys(labels))  # Removes duplicates while preserving order
unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Create new handles list
legend = dict(zip(unique_labels, unique_handles))
legend_order = ['CM', 'CapEC', 'Fibroblasts', 'Ccr2+MP', 'SMC']
ax1[0].legend().set_visible(False)
# Update the legend with the unique handles and labels
ax1[1].legend([legend[val] for val in legend_order], legend_order, title="CellTypes",
              bbox_to_anchor=(1.55, 0.5), loc='center right', ncol=1, frameon=False,
              title_fontproperties={'weight': 'bold'})
plt.savefig(os.path.join(figure_path, 'ExtFig6c_lineplot_CM_vs_Rest.svg'), bbox_inches='tight')


# Extended Figure 6d - lineplot part
# B cells and Ccr2+Mp
mp_b = data_wide[data_wide['CellTypes'].isin(['B_cells', 'Ccr2+MP'])]
# Get the BCs with the top 8 % of B_cells proportion
tmp = mp_b[mp_b['Group'] == 'B_cells_IntraSpace']
tmp = tmp.loc[tmp['Proportion'] > np.percentile(tmp['Proportion'], 92), 'index']
mp_b = mp_b[mp_b['index'].isin(tmp)].copy()

lp = sns.lineplot(mp_b, x='space', y='Proportion', style='Condition', hue='CellTypes',  estimator='mean', markers=True, dashes=True, palette=ct_zip)
lp.set_xlabel('')
lp.set_ylabel('Mean Cell Proportion')
lp.set_xticklabels(['Extended', 'Hexa', 'SameSpot', 'Hexa', 'Extended'], rotation=75, ha='right', va='top')
sns.move_legend(lp, loc='upper center',ncols=2,frameon=False, bbox_to_anchor=(0.5, 1.35))
plt.savefig(os.path.join(figure_path, 'ExtFig6d_lineplot.svg'), bbox_inches='tight')
#</editor-fold>


# <editor-fold desc="Extended Figure 6d. Spatial distribution of B cells and Ccr2+MP">
old5 = davidrUtility.select_slide(aging, 'Old_5')
young2 = davidrUtility.select_slide(aging, 'Young_2')

fig, axs = plt.subplots(2,2, figsize=(8, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young2, color=['Ccr2+MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[0], title='Young_2', colorbar_loc=None)
sc.pl.spatial(young2, color=['B_cells'], size=1.5, bw=True, vmax='p99.2', ax=axs[2], title='', colorbar_loc=None)
sc.pl.spatial(old5, color=['Ccr2+MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[1], title='Old_5', colorbar_loc=None)
sc.pl.spatial(old5, color=['B_cells'], size=1.5, bw=True, vmax='p99.2', ax=axs[3], title='', colorbar_loc=None)
for ax in axs:
    davidrUtility.axis_format(ax, 'SP')
fig.text(0.03, 0.75, 'Ccr2+MP', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'B_cells', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.05, .08, 0.2, 0.015])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('Cell Proportions', fontweight='bold', loc='center', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min', 'Max'], fontweight='bold', fontsize=12)

plt.savefig(os.path.join(figure_path, 'ExtFig6d_ST_BcellsCcr2MP.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 6e. C3 and C3ar1 in snRNA">
# We check on Vidal et al. dataset which is Aging
vidal = ref[ref.obs.Experiment =='JulianAging']
vidal_mp = vidal[vidal.obs.annotation =='MP']
vidal_fb = vidal[vidal.obs.annotation =='Fibroblasts']

davidrPlotting.pl_umap(vidal, ['C3', 'C3ar1'], layer='logcounts', size=5, path=figure_path,
                       filename='ExtFig6e_UMAP_C3C3ar1.svg', ncols=1, figsize=(6, 12))

davidrPlotting.barplot_nUMI(vidal_mp, 'C3ar1', 'age', 'logcounts',
                            palette={'Young':'sandybrown', 'Old':'royalblue'},
                            order=['Young', 'Old'], figsize=(3.2, 5), ctrl_cond='Young',
                            groups_cond=['Old'], path=figure_path,
                            filename='ExtFig6e_Barplot_C3ar1_VidalMP.svg')

davidrPlotting.barplot_nUMI(vidal_fb, 'C3', 'age', 'logcounts',
                            palette={'Young':'sandybrown', 'Old':'royalblue'},
                            order=['Young', 'Old'], figsize=(3.2, 5), ctrl_cond='Young',
                            groups_cond=['Old'], path=figure_path,
                            filename='ExtFig6e_Barplot_C3_VidalFB.svg')
# </editor-fold>


# <editor-fold desc="Extended Figure 7a. Vessel annotation">
davidrPlotting.plot_slides(aging, 'vessels', ncols=5, bw=True,
                           order=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                  'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'],
                           title_fontweight='bold',
                           fig_path=figure_path,
                           filename='ExtFig7a_SpatialPlot_ST_Vessels_AllSamples.svg')
# </editor-fold>


# <editor-fold desc="Extended Figure 7b. Senescence score">
vmax = np.percentile(aging.obs.senescence, 99.2)

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.flatten()
fig.subplots_adjust(left=0.1)
for idx, batch in enumerate(['Young_2', 'Young_3', 'Young_4', 'Young_5', 'Old_1','Old_2','Old_3','Old_4']):
    sdata = davidrUtility.select_slide(aging, batch)
    ax = sc.pl.spatial(sdata, color='senescence', size=1.5,
                        cmap='RdBu_r', show=False, vmax=vmax,
                        ax=axs[idx], colorbar_loc=None)[0]
    davidrUtility.axis_format(ax, 'SP')
    ax.set_title(batch)

# Add Colorbar Manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.98, .15, 0.03, 0.13])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=-2.5, vmax=2.5)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Senescence\nScore', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels(['Min', 'Max'], fontweight='bold')
cbar.ax.grid(False)
plt.savefig(os.path.join(figure_path, 'ExtFig7b_Spatial_Senescence.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 7c. Boxplot Senescence">
df = aging.obs[['senescence_gradient', 'condition', 'clusters', 'sample', 'senescence']].copy()
fig, axs = plt.subplots(1, 1, figsize=(5, 6))
bx = sns.boxplot(df, x='senescence_gradient', y='senescence', hue='condition', order=['Hspot', 'dist100', 'dist200','dist300', 'dist400', 'dist500', 'rest'],
                 hue_order=['Young', 'Old'], palette={'Young':'darkorange', 'Old':'royalblue'}, ax=axs)
bx.set_ylabel('Senescence Score')
bx.set_xlabel('')
bx.set_xticklabels(bx.get_xticklabels(), rotation=45, ha='right', va='top', fontweight='bold')
sns.move_legend(bx, loc='upper center', ncols=2, frameon=False, title='Condition', bbox_to_anchor=(.5, 1), title_fontproperties={'weight':'bold', 'size':14}, fontsize=12)
plt.savefig(os.path.join(figure_path, 'ExtFig7c_BoxPlot_SenescenceGradient.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 7d. Proportion of cts in Hspots">
df = aging.obsm['c2l'].copy()
df[['gradient', 'condition']] = aging.obs[['senescence_gradient', 'condition']]
df = df.groupby(['gradient', 'condition']).agg('sum')
df['total'] = df.groupby(['gradient', 'condition']).agg('sum').sum(axis=1).reindex(index=df.index)
df = df.div(df['total'], axis=0).iloc[:,:-1]  # Compute proportions

# Stacked barplot
df_hspots = df.head(2)  # Hspots
df_hspots.index = ['Old', 'Young']
df_hspots = df_hspots.iloc[-2:, :].reindex(index=['Young', 'Old'], columns=['CM', 'SMC', 'Fibro_activ', 'CapEC',
                                                                            'Fibroblasts', 'ArtEC', 'EndoEC', 'T_cells',
                                                                            'MP', 'Ccr2+MP', 'Pericytes', 'Adip', 'LymphEC',
                                                                            'VeinEC', 'Epi_cells', 'B_cells'])

fig, axs = plt.subplots(1, 1, figsize=(3, 5))
ax = df_hspots.plot.bar(stacked=True, color=ct_zip, ax=axs)
sns.move_legend(ax, loc='center right', frameon=False, bbox_to_anchor=(1.3,.5), fontsize=8, title='CellType',
                title_fontproperties={'weight':'bold'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontweight='bold', ha='right')
ax.set_ylabel('Proportion')
plt.savefig(os.path.join(figure_path, 'ExtFig7d_StackedBar_ProportionCellTypes_Hspots.svg'), bbox_inches='tight')
#</editor-fold>


# <editor-fold desc="Extended Figure 7e. Lineplot showing the proportion of cts along the gradient">
df = aging.obsm['c2l'].copy()
df[['gradient', 'condition']] = aging.obs[['senescence_gradient', 'condition']]
df = df.groupby(['gradient', 'condition']).agg('sum')
df['total'] = df.groupby(['gradient', 'condition']).agg('sum').sum(axis=1).reindex(index=df.index)
df = df.div(df['total'], axis=0).iloc[:,:-1]  # Calculate proportions

df_plot = df.reset_index()
df_plot = df_plot.melt(id_vars=['gradient', 'condition'])
# Duplicate to make the plot symetric
tmp = df_plot.copy()
tmp['gradient'] = '-' + tmp['gradient'].astype(str)
tmp =tmp[tmp.gradient !='-Hspot']
df_plot = pd.concat([df_plot, tmp]).reset_index()
df_plot['gradient'] = pd.Categorical(df_plot.gradient, categories=['-rest', '-dist500', '-dist400', '-dist300', '-dist200',
                                                                   '-dist100', 'Hspot', 'dist100', 'dist200', 'dist300',
                                                                   'dist400', 'dist500', 'rest'], ordered=True)
# Select some celltypes
df_plot = df_plot[df_plot['variable'].isin(['SMC', 'Fibroblasts', 'Fibro_activ', 'Ccr2+MP','ArtEC',
                                            'T_cells', 'Pericytes', 'B_cells', 'VeinEC', 'LymphEC'])]

fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.75))
fig.subplots_adjust(wspace=0.05)
# Plot the first line (on the first y-axis)
lp = sns.lineplot(df_plot[df_plot.condition == 'Young'], x='gradient', y='value', hue='variable', style='condition',
                  estimator='mean', palette=ct_zip, markersize=5, markers=True, dashes=True, ax=axs[0])
lp.legend().set_visible(False)
lp.set_ylim(0, 0.11)
lp.set_ylabel('Cell Proportion')
lp.set_xlabel('Young Condition')
lp.set_xticklabels(['Rest', '', '400$\mu$m', '', '200$\mu$m', '', 'Hspot', '', '200$\mu$m', '', '400$\mu$m', '', 'Rest'],
                   fontsize=10, rotation=45, ha='right', va='top')

lp.set_yticklabels(lp.get_yticklabels(), fontsize=10)
lp = sns.lineplot(df_plot[df_plot.condition == 'Old'], x='gradient', y='value', hue='variable', style='condition',
                  estimator='mean',palette=ct_zip, markersize=5, markers=True, dashes=True, ax=axs[1])
lp.set_ylabel('')
lp.set_ylim(0, 0.11)
lp.set_xticklabels(['Rest', '', '400$\mu$m', '', '200$\mu$m', '', 'Hspot', '', '200$\mu$m', '', '400$\mu$m', '', 'Rest'],
                   fontsize=10, rotation=45, ha='right', va='top')
lp.set_yticklabels(lp.get_yticklabels(), fontsize=10)

lp.spines['left'].set_visible(False)
lp.tick_params(axis='y', which='both', left=False)
lp.set_yticklabels('')
lp.set_xlabel('Old Condition')

# Retrieve the handles and labels from the  plot
handles, labels = lp.get_legend_handles_labels()

# Remove duplicates from the labels (keep unique ones)
legend = dict(zip(labels, handles))
legend['CellTypes'] = legend['variable']
legend['Condition'] = legend['condition']

legend_order = ['CellTypes', 'B_cells', 'Fibroblasts', 'Fibro_activ', 'LymphEC', 'Ccr2+MP', 'Pericytes', 'SMC', 'T_cells', 'VeinEC']
lp.legend().set_visible(False)

# Update the legend with the unique handles and labels
lp.legend([legend[val] for val in legend_order], legend_order, bbox_to_anchor=(1, .8), loc='upper left', ncol=1, frameon=False,
           title_fontproperties={'weight':'bold'}, fontsize=10)
plt.savefig(os.path.join(figure_path, 'ExtFig7e_LinePlot_CellProportion_Gradient.svg'), bbox_inches='tight')
# </editor-fold>


# <editor-fold desc="Extended Figure 7f. Proportion gradient across vessels">
df = aging.obs.value_counts(['senescence_gradient', 'condition', 'vessels']).reset_index()
total = df[['condition', 'vessels', 'count']].groupby(['condition', 'vessels']).agg('sum')
df['norm'] = [row['count'] / total.loc[(row.condition, row.vessels)].values[0] for idx, row in df.iterrows()]

df_young = df[df['condition'] == 'Young']
df_old = df[df['condition'] == 'Old']

del df_young['condition'], df_old['condition']

df_young = df_young.pivot(index=['vessels'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
    index=['Arteries', 'Veins', 'Lymphatics', 'nonVasc'])
df_old = df_old.pivot(index=['vessels'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
    index=['Arteries', 'Veins', 'Lymphatics', 'nonVasc'])

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = df_young.plot.bar(stacked=True, color={'Hspot': 'firebrick', 'dist100': 'tomato', 'dist200': 'lightsalmon',
                                             'dist300': 'royalblue', 'dist400': 'cornflowerblue', 'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'}, ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'ExtFig7f_StackedBar_Proportion_VesselsGradient_Young.svg'), bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = df_old.plot.bar(stacked=True, color={'Hspot': 'firebrick', 'dist100': 'tomato', 'dist200': 'lightsalmon',
                                             'dist300': 'royalblue', 'dist400': 'cornflowerblue', 'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'}, ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'ExtFig7f_StackedBar_Proportion_VesselsGradient_Old.svg'), bbox_inches='tight')
# </editor-fold>


# Extended Figure 7g --> staining


# <editor-fold desc="Extended Figure 7h. Dotplot expression of immune inhibitory genes">
genes = ['Cd47', 'Sirpa', 'Cd24a', 'Pilra', 'Clec4a1', 'Clec12a',
         'Ccr2', 'Ccl2', 'Ccl4', 'Ccl5', 'Ccr5', 'Vcam1', 'Icam1']
# Only consider Vidal et al.
vidal = ref[ref.obs.Experiment == 'JulianAging']
vidal.obs['tmp'] = vidal.obs.annotation.astype(str).copy()
# Group cells in myeloid and non-myeloid cells
vidal.obs['tmp'] = vidal.obs['tmp'].map({'CapEC': 'Non_Myeloid', 'Pericytes': 'Non_Myeloid',
                                         'Fibroblasts': 'Non_Myeloid', 'Fibro_activ': 'Non_Myeloid',
                                         'ArtEC': 'Non_Myeloid', 'CM': 'Non_Myeloid',
                                         'MP': 'Myeloid', 'EndoEC': 'Non_Myeloid',
                                         'LymphEC': 'Non_Myeloid', 'Epi_cells': 'Non_Myeloid',
                                         'Ccr2+MP': 'Myeloid', 'T_cells': 'Non_Myeloid',
                                         'B_cells': 'Non_Myeloid', 'SMC': 'Non_Myeloid',
                                         'VeinEC': 'Non_Myeloid', 'Adip': 'Non_Myeloid'})

ax = sc.pl.dotplot(vidal, groupby=['tmp', 'age'], var_names=genes, show=False, figsize=(7, 2))
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), rotation=45, fontweight='bold', ha='right', va='top')
plt.savefig(os.path.join(figure_path, 'ExtFig7h_Dotplot_MyeloidNonMyeloid_ImmuneInhibitoryGenes.svg'), bbox_inches='tight')
# </editor-fold>

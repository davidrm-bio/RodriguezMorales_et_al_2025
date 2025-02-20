#!/usr/bin/env python

"""
Description: Processing of snRNA-seq for the generation of a reference
to use for ST deconvolution

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""

#<editor-fold desc="Sep-Up">
import os
import itertools

import scanpy as sc
import bbknn as bkn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import davidrUtility
import davidrPlotting
from davidrScRNA import concatenate_anndata, automatic_annotation
from davidrPlotting import split_umap

np.random.seed(13)

main_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results'
object_path = os.path.join(main_path, 'Objects/Scanpy/')
figure_path = os.path.join(main_path, 'Figures/1_QualityControl/snRNA/')
table_path = os.path.join(main_path, 'Tables')
#</editor-fold>

#<editor-fold desc="QC - snRNA Julian (Vidal et al.)">
snRNA_processed = sc.read_h5ad(os.path.join(object_path, 'Cell2location/Individual_Reference/', 'Julian-Aging.h5ad'))

# Filter cells and genes
snRNA_processed.X = snRNA_processed.layers['counts'].copy()
sc.pp.filter_cells(snRNA_processed, min_genes=250)
sc.pp.filter_genes(snRNA_processed, min_cells=3)

# Filter low quality cells
snRNA_processed.var['mt'] = snRNA_processed.var_names.str.startswith('Mt-')  # Annotate mitochondria genes
sc.pp.calculate_qc_metrics(snRNA_processed, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)
snRNA_processed = snRNA_processed[snRNA_processed.obs.pct_counts_mt < 5, :]

# Filter doublets
sc.pp.scrublet(snRNA_processed, batch_key='orig.ident')  # doublet detection
snRNA_processed = snRNA_processed[snRNA_processed.obs['predicted_doublet'] == False]

# Normalise the expression
snRNA_processed.layers['counts'] = snRNA_processed.X.copy()
sc.pp.normalize_total(snRNA_processed, target_sum=10_000)
sc.pp.log1p(snRNA_processed)
snRNA_processed.layers['logcounts'] = snRNA_processed.X.copy()
#</editor-fold>

#<editor-fold desc="QC - snRNA Julian (Senolytic)">
snRNA_senolytic = sc.read_h5ad(
    os.path.join(object_path, 'Cell2location/Individual_Reference/Senolytics-autoAnnot.h5ad'))
snRNA_senolytic.X = snRNA_senolytic.layers['counts'].copy()

# Filter cells and genes
sc.pp.filter_cells(snRNA_senolytic, min_genes=250)
sc.pp.filter_genes(snRNA_senolytic, min_cells=3)

# Filter low quality cells
snRNA_senolytic.var['mt'] = snRNA_senolytic.var_names.str.startswith('Mt-')  # Annotate mitochondria genes
sc.pp.calculate_qc_metrics(snRNA_senolytic, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)
snRNA_senolytic = snRNA_senolytic[snRNA_senolytic.obs.pct_counts_mt < 5, :]

# Filter doublets
sc.pp.scrublet(snRNA_senolytic, batch_key='sample')  # doublet detection
snRNA_senolytic = snRNA_senolytic[snRNA_senolytic.obs['predicted_doublet'] == False]

# Normalise the data
snRNA_senolytic.layers['counts'] = snRNA_senolytic.X.copy()
sc.pp.normalize_total(snRNA_senolytic, target_sum=10000)
sc.pp.log1p(snRNA_senolytic)
snRNA_senolytic.layers['logcounts'] = snRNA_senolytic.X.copy()
#</editor-fold>

#<editor-fold desc="Concatenate objects and clean AnnDatas">
snRNA_processed.obs['sample'] = snRNA_processed.obs['orig.ident']
snRNA_senolytic.obs['condition'] = snRNA_senolytic.obs['condition'].astype(str) + '_' + snRNA_senolytic.obs['age'].astype(str)
snrna = concatenate_anndata({'JulianAging': snRNA_processed, 'JulianSenolytic': snRNA_senolytic}, sample_label='Experiment')
keep_obs = ['sample', 'condition', 'Experiment']
for obs in snrna.obs.columns:
    if obs not in keep_obs:
        del snrna.obs[obs]

del snrna.obsm['X_pca'], snrna.obsm['X_umap']
del snrna.layers['scaledata']


rename_id = {'104383-014-001': 'Old_B1', '104383-014-002': 'Old_B2',
             '104383-014-003': 'Old_B3', '104383-014-004': 'Old_B4',
             '104383-014-005': 'Old_B5', '104383-014-006': 'Young_B1',
             '104383-014-007': 'Old_B6', 'Old1': 'Old_B7',
             'Old2': 'Old_B8', 'Old3': 'Old_B9', 'Young1': 'Young_B2',
             'Young2': 'Young_B3', 'Young3': 'Young_B4', }
snrna.obs['batch'] = snrna.obs['sample'].map(rename_id)

snrna.obs['age'] = snrna.obs['condition'].copy()
snrna.obs['age'] = pd.Categorical(snrna.obs['age'].replace({'Placebo_Old': 'Old', 'Placebo_Young': 'Young',
                                                            'Senolytic_Old': 'Old'}).astype(str))
snrna.uns['age_colors'] = ['royalblue', 'darkorange']  # Old, Young - Fix Colors

#</editor-fold>

#<editor-fold desc="Integration using scVI">
# Pre-scVI
sc.pp.highly_variable_genes(snrna, batch_key='sample')
snrna.var['mt'] = snrna.var_names.str.startswith('Mt-')  # Annotate mitochondria genes
snrna.var['ribo'] = snrna.var_names.str.startswith(('Rpl', 'Rps'))  # Annotate ribosomal genes
sc.pp.calculate_qc_metrics(snrna, qc_vars=['mt', 'ribo'], percent_top=None, log1p=True, inplace=True)

adata_hvg = snrna[:, snrna.var.highly_variable].copy()
adata_hvg.X = adata_hvg.layers['counts'].copy()
adata_hvg.write(os.path.join(object_path, 'Cell2location/Individual_Reference/snrna_RefAging_hvg.h5ad'))

# Run Integration on GPU Server --> See Jupyter Notebook

# Post-scVI
# adata_hvg = sc.read_h5ad(os.path.join(object_path, 'Cell2location/Individual_Reference/snrna_RefAging_hvg.h5ad'))
adata_hvg = sc.read_h5ad('/mnt/davidr/scStorage/DavidR/snrna_RefAging_hvg.h5ad')
snrna.obsm['X_scVI'] = adata_hvg.obsm['X_scVI']

bkn.bbknn(snrna, batch_key='sample', use_rep='X_scVI', neighbors_within_batch=8)
sc.tl.umap(snrna, spread=1.2, min_dist=.3)
sc.tl.leiden(snrna, resolution=1.5, key_added='leiden1_5')
#</editor-fold>

#<editor-fold desc="Check Integration and clustering">
split_umap(snrna, 'batch', ncol=5, figsize=(18, 9),
           path=os.path.join(figure_path), filename='SplitUMAP_snRNARef_Sample_240524.svg')


fig, axs = plt.subplots(1, 1, figsize=(12, 8))
sc.pl.umap(snrna, color='leiden1_5', legend_loc='on data', legend_fontweight=750,
           legend_fontsize=12, legend_fontoutline=2, ax=axs, size=20)
davidrUtility.axis_format(axs)
axs.set_title('Leiden Res = 1.5', fontsize=20, fontweight='bold')
plt.savefig(os.path.join(figure_path, 'UMAP_snRNARef_leiden_240524.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Annotation">
# Use a high resolution for the automatic annotation
sc.tl.leiden(snrna, resolution=5, key_added='leiden5', flavor='igraph', directed=False, n_iterations=2)
snrna.X = snrna.layers['logcounts'].copy()
snrna = automatic_annotation(snrna, 'leiden5', update_label=False)
sc.pl.umap(snrna, color=['autoAnnot'])

update_ct_labels = {'Adip1': 'Adip',
                    'B': 'B_cells',
                    'CD16+Mo': 'Ccr2+MP',
                    'CD4+T_naive': 'T_cells',
                    'EC1_cap': 'CapEC',
                    'EC5_art': 'ArtEC',
                    'EC7_endocardial': 'EndoEC',
                    'EC6_ven': 'VeinEC',
                    'EC8_ln': 'LymphEC',
                    'FB1': 'Fibroblasts',
                    'FB4_activated': 'Fibro_activ',
                    'FB5': 'Fibroblasts',
                    'LYVE1+IGF1+MP': 'MP',
                    'LYVE1+MP_cycling': 'MP',
                    'Meso': 'Epi_cells',
                    'NC2_glial_NGF+': 'Neural',
                    'PC1_vent': 'Pericytes',
                    'SMC1_basic': 'SMC',
                    'vCM1': 'CM',
                    'vCM4': 'CM',
                    'MP':'MP'
                    }
snrna.obs['annotation'] = [update_ct_labels[ct] for ct in snrna.obs['autoAnnot']]

# Visualise annotation
fig, axs = plt.subplots(1, 1, figsize=(12, 8))
sc.pl.umap(snrna, color='annotation', ax=axs, size=20)
axs.set_title('Annotation', fontsize=20, fontweight='bold')
davidrUtility.axis_format(axs)
plt.savefig(os.path.join(figure_path,'UMAP_snRNARef_annotation_240524.svg'),bbox_inches='tight')

# Check markers
sc.pl.umap(snrna, color=['Ttn', 'Tnni3', # CM
                         'Acta2', 'Tagln', # SMC
                         'Rgs5', 'Abcc9', # Pericytes
                         'Gsn', 'Abca9', # Fibroblasts
                         'Ptprc', # Immune cells
                         'Cd74',  # B cells
                         'Cd3e', # T cells
                         'Ccr2',  # Bone marrow
                         'Lyz2', # Macrophages
                         'Cadm2', 'Tubb3', 'Tubb3',  # Neural
                         'Vwf', 'Pecam1', # Endothelial cells
                         'Plin1',# Adipocytes
                         'Wwc1', # Epicardial cells
                         ],
           ncols=5)
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_Markers.svg'), bbox_inches='tight')

# Remove Neural cells --> Does not express any marker and might not be useful for spatial
snrna = snrna[snrna.obs['annotation'] != 'Neural']
#</editor-fold>

#<editor-fold desc="Sub-Clustering EC and Myeloid">
sc.pl.umap(snrna, color='leiden1_5', legend_loc='on data')

# Re-clustering of Endothelials - Select base on leiden 1.5
ec = snrna[snrna.obs.leiden1_5.isin(['8', '4', '10'])]
bkn.bbknn(ec, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(ec)
sc.tl.leiden(ec, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)
ec = automatic_annotation(ec, 'leiden')

sc.pl.umap(ec, color='leiden', legend_fontsize=10, legend_fontoutline=2, legend_loc='on data')
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_SubClusterEC_leiden.svg'), bbox_inches='tight')


# Re-clustering of Myeloid cells - Select base on autoAnnot
md = snrna[snrna.obs['annotation'].isin(['MP', 'Monocytes'])].copy()
bkn.bbknn(md, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(md)
sc.tl.leiden(md, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)
sc.pl.umap(md, color=['leiden', 'Ccr2'])
md.obs['autoAnnot'] = md.obs.leiden.replace({'0': 'MP', '1': 'MP', '2': 'MP', '3': 'MP', '4': 'MP',
                                             '5': 'MP', '6': 'MP', '7': 'MP', '8': 'MP', '9': 'MP',
                                             '10': 'Ccr2+MP', '11': 'MP', '12': 'Ccr2+MP', '13': 'MP', '14': 'MP',
                                             '15': 'MP', '16': 'Ccr2+MP', '17': 'MP', })

sc.pl.umap(md, color=['leiden','annotation', 'autoAnnot', 'Ccr2'], legend_fontsize=10, legend_fontoutline=2, legend_loc='on data')
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_SubClusterMyeloid_leiden.svg'), bbox_inches='tight')

# ec.write(os.path.join(object_path, 'Cell2location/RefAging_SubCluster_EC.h5ad'))
# md.write(os.path.join(object_path, 'Cell2location/RefAging_SubCluster_Myeloid.h5ad'))

# Transfer annotation
labels = []
for bc in snrna.obs_names:
    if bc in ec.obs_names:
        labels.append(ec.obs.loc[bc, 'autoAnnot'])
    elif bc in md.obs_names:
        labels.append(md.obs.loc[bc, 'autoAnnot'])
    else:
        labels.append(snrna.obs.loc[bc, 'autoAnnot'])

snrna.obs['annotation'] = pd.Categorical([update_ct_labels[ct] for ct in labels],
                                         categories=['Adip', 'ArtEC', 'B_cells', 'CM', 'CapEC', 'Ccr2+MP', 'EndoEC',
                                                     'Fibroblasts', 'Fibro_activ', 'LymphEC', 'MP', 'Epi_cells',
                                                     'Pericytes',
                                                     'SMC', 'T_cells', 'VeinEC'], ordered=True)

# Update UMAP with annotation saved
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
sc.pl.umap(snrna, color='annotation', ax=axs, size=20)
axs.set_title('Annotation', fontsize=20, fontweight='bold')
davidrUtility.axis_format(axs)
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_Ref_annotation_final_240524.svg'),bbox_inches='tight')

snrna.X = snrna.layers['counts'].copy()
snrna.write(os.path.join(object_path, 'Cell2location/snRNA_RefAging_Manuscript.h5ad'))
#</editor-fold>

#<editor-fold desc="DGE snRNA">
ref = sc.read_h5ad(os.path.join(object_path, 'Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()
sc.tl.rank_genes_groups(ref, groupby='annotation', method='wilcoxon', tie_correct=True)
table = sc.get.rank_genes_groups_df(ref, group=None)

with pd.ExcelWriter(os.path.join(table_path, 'DGE', '241205_DGE_snRNA_Aging_Annotation_ExtendedTable.xlsx')) as writer:
    table.to_excel(writer, sheet_name='AllGenes', index=False)  # Sheet Name with all the genes
    for cluster in table.group.unique():
        # For each cluster save only significant genes (Padj < 0.05)
        table_subset = table[table.group == cluster]
        table_subset = table_subset[table_subset['pvals_adj'] < 0.05].sort_values('logfoldchanges', ascending=False)
        table_subset.to_excel(writer, sheet_name=f'SigGenes_{cluster}', index=False)
#</editor-fold>

#<editor-fold desc="Quality Control Integration & Clustering for Manuscript">
figure_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/0_Manuscript/'

ref = sc.read_h5ad(os.path.join(object_path, 'Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()
random_indices = np.random.permutation(list(range(ref.shape[0])))  # Sort barcodes randomly

# UMAP showing the age
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'age',
                            size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(ref.obs.age.cat.categories[::-1]), ref.uns['age_colors'][::-1])],
          loc='center right', frameon=False, edgecolor='black', title='Condition',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'UMAP_QC_Age_snRNA.svg'), bbox_inches='tight', dpi=300)


# UMAP showing the Samples (Batches)
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'batch', size=12, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('')

batch_colors = [('Young_B1', '#8c564b'), ('Young_B2', '#ffbb78'), ('Young_B3', '#98df8a'),
                ('Young_B4', '#ff9896'), ('Old_B1', '#1f77b4'), ('Old_B2', '#ff7f0e'),
                ('Old_B3', '#279e68'), ('Old_B4', '#d62728'), ('Old_B5', '#aa40fc'),
                ('Old_B6', '#e377c2'), ('Old_B7', '#b5bd61'), ('Old_B8', '#17becf'),
                ('Old_B9', '#aec7e8'),]
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in batch_colors],
          loc='center right', frameon=False, edgecolor='black', title='Sample',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'UMAP_Batches_snRNA.svg'), bbox_inches='tight', dpi=300)

# Here we show the clustering
ax = davidrPlotting.pl_umap(ref[random_indices, :], 'leiden5', size=12,
                            figsize=(5, 6), show=False, alpha=.9)
ax.set_title('')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(ref.obs.leiden5.cat.categories), ref.uns['leiden5_colors'])],
          loc='center right', frameon=False, edgecolor='black', title='Clusters',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13, ncols=2)
plt.savefig(os.path.join(figure_path, f'UMAP_leiden5_snRNA.svg'), bbox_inches='tight', dpi=300)
#</editor-fold>

#<editor-fold desc="ViolinPlots QC Metrics snRNA-seq">
# Extract the data we are going to plot - snRNA-seq (Number of UMIs; Number of Genes;)

data = ref.obs[['log1p_total_counts', 'log1p_n_genes_by_counts']].copy()
data['log(nUMIs)'] = data['log1p_total_counts']
data['log(nGenes)'] = data['log1p_n_genes_by_counts']
data['batch'] = pd.Categorical(ref.obs['batch'].copy(),
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
plt.savefig(os.path.join(figure_path, f'QC_Metrics_snRNA.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Heatmap Markers snRNA-seq">
# Get colors for each ct
ct_catgs = list(ref.obs.annotation.cat.categories)
ct_colors = ref.uns['annotation_colors']
# Reverse the marker dict
ct_zip = dict(zip(ct_catgs, ct_colors))

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

genes = list(itertools.chain(*markers.values()))
table = davidrUtility.AverageExpression(ref, group_by='annotation', feature=genes, out_format='wide')
table = table.reindex(index=genes, columns=markers.keys())

# Reverse the marker dict
reversed_markers = {
    gene: [ct for ct, genes in markers.items() if gene in genes][0]
    for gene in {gene for genes in markers.values() for gene in genes}}

row_colors = [ct_zip[reversed_markers[gene]] for gene in table.index]
col_colors = [ct_zip[col] for col in markers.keys()]

# Plot: Heatmap showing marker genes
cm = sns.clustermap(table, cmap='Reds',
                    yticklabels=1, xticklabels=1,  # Show all ticks
                    row_colors=row_colors,  # Indicate what celltype the marker is from
                    col_cluster=False, row_cluster=False,  # Do not cluster
                    z_score=0, robust=True,  # Z-score over genes
                    figsize=(9, 12),
                    #cbar_pos=None,
                    square=True,
                    cbar_pos =([0.85, .15, 0.03, 0.15]))

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

# Adjustment of how the plot looks like
heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')

plt.savefig(os.path.join(figure_path, f'HeatMap_MarkerGenes_RefAging.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Heatmap v2 (wide format) Markers snRNA-seq">
table = table.T

cm = sns.clustermap(table, cmap='Reds',
                    yticklabels=1, xticklabels=1,  # Show all ticks
                    col_colors=row_colors,  # Indicate what celltype the marker is from
                    col_cluster=False, row_cluster=False,  # Do not cluster
                    z_score=0, robust=True,  # Z-score over genes
                    cbar_kws={'orientation':'horizontal'},
                    figsize=(12, 7), cbar_pos=[0.19, .13, 0.15, 0.03], square=True)

# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
colorbar_ax = cm.cax

# Set black border around the outer edge of the heatmap
heatmap_ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)

# Replace xticks with circles colored by the annotation
xtick_pos, xtick_lab = heatmap_ax.get_xticks(), heatmap_ax.get_xticklabels()
ytick_pos, ytick_lab = heatmap_ax.get_yticks(), heatmap_ax.get_yticklabels()
heatmap_ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
for y, color in zip(ytick_pos, col_colors):
    heatmap_ax.plot(table.shape[1] + 0.5, y, 'o', color=color, markersize=8, clip_on=False)  # Version 3

# Adjust Xticks and Yticks to correct positions
heatmap_ax.xaxis.tick_top()
heatmap_ax.set_xticks(np.array([tk - 0.35 for tk in xtick_pos]), xtick_lab,  rotation=75, ha='left', va='bottom',)
heatmap_ax.tick_params(axis='x', pad=12)
heatmap_ax.tick_params(axis='y', pad=15)

# Set Colorbar Ticks
colorbar_ax.set_title('Z score', fontweight='bold', loc='center', fontsize=10)
colorbar_ax.set_xticks([-0.7, np.percentile(cm.data, 99.83)])
colorbar_ax.set_xticklabels(['Min', 'Max'], fontweight='bold')

heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')

plt.savefig(os.path.join(figure_path, f'wide_HeatMap_MarkerGenes_RefAging.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="UMAP of the subclustering of Endothelial cells snRNA-seq">
ec = ref[ref.obs.annotation.isin(['ArtEC', 'VeinEC', 'CapEC', 'EndoEC'])].copy()
bkn.bbknn(ec, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(ec)
sc.tl.leiden(ec, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)


ax = davidrPlotting.pl_umap(ec, ['annotation', 'leiden', 'Rgcc',
                                     'Stmn2', 'Vwf', 'Npr3'],
                            size=20, ncols=3, figsize=(10, 8), show=False,
                            common_legend=True, legend_loc='on data', legend_fontsize=12, legend_fontweight=750,
                            legend_fontoutline=1.5)
ax[0].set_title('Annotation')
ax[1].set_title('Leiden Clusters')
plt.savefig(os.path.join(figure_path, f'UMAP_RefAging_SubClusteringEC_snRNA.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="UMAP of the subclustering of myeloid snRNA-seq">
md = ref[ref.obs.annotation.isin(['MP', 'Ccr2+MP'])].copy()
bkn.bbknn(md, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(md)
sc.tl.leiden(md, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)


ax = davidrPlotting.pl_umap(md, ['annotation', 'leiden',
                                     'Lyve1', 'Ccr2'], size=40, ncols=2, figsize=(10, 8), show=False, common_legend=True,
                            legend_loc='on data', legend_fontsize=12, legend_fontweight=750, legend_fontoutline=1.5)
ax[0].set_title('Annotation')
ax[1].set_title('Leiden Clusters')
plt.savefig(os.path.join(figure_path, f'UMAP_RefAging_SubClusteringMyeloid_snRNA.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="UMAP Annotation snRNA-seq">
ax = davidrPlotting.pl_umap(ref, 'annotation', size=12, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in zip(ct_catgs, ct_colors)],
          loc='center right', frameon=False, edgecolor='black', title='CellTypes',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13, ncols=1)

plt.savefig(os.path.join(figure_path, f'UMAP_MainFig_RefAging_snRNA_Annotation.svg'), bbox_inches='tight')
#</editor-fold>

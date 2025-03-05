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

import scanpy as sc
import bbknn as bkn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import davidrUtility
from davidrScRNA import concatenate_anndata, automatic_annotation
from davidrPlotting import split_umap

np.random.seed(13)

main_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results'
object_path = os.path.join(main_path, 'Objects/Scanpy/')
figure_path = os.path.join(main_path, 'Figures/1_QualityControl/snRNA/')
table_path = os.path.join(main_path, 'Tables')
#</editor-fold>


########################################################################################################################
# - Quality Control, Integration and Annotation of the snRNAseq
########################################################################################################################


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
snRNA_processed.obs['sample'] = snRNA_processed.obs['orig.ident']  # Set a common column for sample information
snRNA_senolytic.obs['condition'] = snRNA_senolytic.obs['condition'].astype(str) + '_' + snRNA_senolytic.obs['age'].astype(str)  # set the age metadata
snrna = concatenate_anndata({'JulianAging': snRNA_processed, 'JulianSenolytic': snRNA_senolytic}, sample_label='Experiment')  # set the dataset informartion
keep_obs = ['sample', 'condition', 'Experiment']  # .obs columns to keep
for obs in snrna.obs.columns:
    if obs not in keep_obs:
        del snrna.obs[obs]

# Remove previous analysis
del snrna.obsm['X_pca'], snrna.obsm['X_umap']
del snrna.layers['scaledata']

# Rename Sample ID to have a common format
rename_id = {'104383-014-001': 'Old_B1', '104383-014-002': 'Old_B2', '104383-014-003': 'Old_B3', '104383-014-004': 'Old_B4',
             '104383-014-005': 'Old_B5', '104383-014-006': 'Young_B1', '104383-014-007': 'Old_B6', 'Old1': 'Old_B7',
             'Old2': 'Old_B8', 'Old3': 'Old_B9', 'Young1': 'Young_B2', 'Young2': 'Young_B3', 'Young3': 'Young_B4', }
snrna.obs['batch'] = snrna.obs['sample'].map(rename_id)

# Set the Age Metadata
snrna.obs['age'] = snrna.obs['condition'].copy()
snrna.obs['age'] = pd.Categorical(snrna.obs['age'].replace({'Placebo_Old': 'Old', 'Placebo_Young': 'Young',
                                                            'Senolytic_Old': 'Old'}).astype(str))
snrna.uns['age_colors'] = ['royalblue', 'darkorange']  # Old, Young - Fix Colors
#</editor-fold>


#<editor-fold desc="Integration using scVI">
# Pre-scVI
sc.pp.highly_variable_genes(snrna, batch_key='sample')  # Identify HVG
snrna.var['mt'] = snrna.var_names.str.startswith('Mt-')  # Annotate mitochondria genes
snrna.var['ribo'] = snrna.var_names.str.startswith(('Rpl', 'Rps'))  # Annotate ribosomal genes
sc.pp.calculate_qc_metrics(snrna, qc_vars=['mt', 'ribo'], percent_top=None, log1p=True, inplace=True)  # Compute Metrics

adata_hvg = snrna[:, snrna.var.highly_variable].copy()  # Select only HVG for Integration
adata_hvg.X = adata_hvg.layers['counts'].copy()  # Set raw counts for scVI
adata_hvg.write(os.path.join(object_path, 'Cell2location/Individual_Reference/snrna_RefAging_hvg.h5ad'))

# Run Integration on GPU Server --> See Jupyter Notebook

# Post-scVI
adata_hvg = sc.read_h5ad('/mnt/davidr/scStorage/DavidR/snrna_RefAging_hvg.h5ad')
snrna.obsm['X_scVI'] = adata_hvg.obsm['X_scVI']  # Copy the latent representation from autoencoder

bkn.bbknn(snrna, batch_key='sample', use_rep='X_scVI', neighbors_within_batch=8) # Find neighbors
sc.tl.umap(snrna, spread=1.2, min_dist=.3)  # Do UMAP embedding
sc.tl.leiden(snrna, resolution=1.5, key_added='leiden1_5')  # Clustering with high resolution
#</editor-fold>


#<editor-fold desc="Check Integration and clustering  - UMAPs">
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
snrna.X = snrna.layers['logcounts'].copy()  # Make sure we use the normalised expression values
snrna = automatic_annotation(snrna, 'leiden5', update_label=False)  # Run celltypist
sc.pl.umap(snrna, color=['autoAnnot'])

update_ct_labels = {'Adip1': 'Adip', 'B': 'B_cells', 'CD16+Mo': 'Ccr2+MP', 'CD4+T_naive': 'T_cells',
                    'EC1_cap': 'CapEC', 'EC5_art': 'ArtEC', 'EC7_endocardial': 'EndoEC',
                    'EC6_ven': 'VeinEC', 'EC8_ln': 'LymphEC', 'FB1': 'Fibroblasts', 'FB4_activated': 'Fibro_activ',
                    'FB5': 'Fibroblasts', 'LYVE1+IGF1+MP': 'MP', 'LYVE1+MP_cycling': 'MP',
                    'Meso': 'Epi_cells', 'NC2_glial_NGF+': 'Neural', 'PC1_vent': 'Pericytes',
                    'SMC1_basic': 'SMC', 'vCM1': 'CM', 'vCM4': 'CM', 'MP':'MP'}
snrna.obs['annotation'] = [update_ct_labels[ct] for ct in snrna.obs['autoAnnot']]  # Update celltype annotation


# Check annotation
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
                         ], ncols=5)
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_Markers.svg'), bbox_inches='tight')

# Remove Neural cells --> Does not express any marker and might not be useful for spatial
snrna = snrna[snrna.obs['annotation'] != 'Neural']
#</editor-fold>


#<editor-fold desc="Sub-Clustering EC and Myeloid">

# Re-clustering of Endothelials - Select base on leiden 1.5
ec = snrna[snrna.obs.leiden1_5.isin(['8', '4', '10'])]
bkn.bbknn(ec, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(ec)
sc.tl.leiden(ec, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)
ec = automatic_annotation(ec, 'leiden')

# Re-clustering of Myeloid cells - Select base on autoAnnot
md = snrna[snrna.obs['leiden1_5'].isin(['1', '19', '15'])].copy()
bkn.bbknn(md, use_rep='X_scVI', batch_key='sample')
sc.tl.umap(md)
sc.tl.leiden(md, resolution=1.5, n_iterations=2, flavor='igraph', directed=False)
sc.pl.umap(md, color=['leiden', 'Ccr2'])
md.obs['autoAnnot'] = md.obs.leiden.replace({'0': 'MP', '1': 'MP', '2': 'MP', '3': 'MP', '4': 'MP',
                                             '5': 'MP', '6': 'MP', '7': 'MP', '8': 'MP', '9': 'MP',
                                             '10': 'Ccr2+MP', '11': 'MP', '12': 'Ccr2+MP', '13': 'MP', '14': 'MP',
                                             '15': 'MP', '16': 'Ccr2+MP', '17': 'MP', })

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
                                         categories=['Adip', 'ArtEC', 'B_cells',  'CM', 'CapEC', 'Ccr2+MP', 'EndoEC',
                                                     'Fibroblasts', 'Fibro_activ', 'LymphEC', 'MP', 'Epi_cells',
                                                     'Pericytes', 'SMC', 'T_cells', 'VeinEC'], ordered=True)

# Update UMAP with annotation saved
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
sc.pl.umap(snrna, color='annotation', ax=axs, size=20)
axs.set_title('Annotation', fontsize=20, fontweight='bold')
davidrUtility.axis_format(axs)
plt.savefig(os.path.join(figure_path, 'UMAP_snRNA_Ref_annotation_final_240524.svg'),bbox_inches='tight')

snrna.X = snrna.layers['counts'].copy()  # Set Counts for cell2location
snrna.write(os.path.join(object_path, 'Cell2location/snRNA_RefAging_Manuscript.h5ad'))
#</editor-fold>


#<editor-fold desc="DGE snRNA - CellType Vs Rest">
# Check the annotation
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

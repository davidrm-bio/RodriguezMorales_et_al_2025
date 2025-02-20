#!/usr/bin/env python

"""
Description: Process the ST data

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""
#<editor-fold desc="Set-Up">
import os
from tqdm import  tqdm
from datetime import date
from typing import Union

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import decoupler as dc
import gseapy as gp
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats
from scipy.cluster.hierarchy import  linkage, dendrogram
import scipy.sparse

import davidrUtility
import davidrSpatial
from davidrSpatial import spatial_integration
from davidrScRNA import SCTransform
from davidrPlotting import split_umap, plot_slides, pl_umap
import davidrPlotting
import davidrExperimental

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns

today = date.today().strftime('%y%m%d')

main_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/'
input_path = os.path.join(main_path, 'Raw_Data/SpaceRanger/')
result_path = os.path.join(main_path, 'Results/')
object_path = os.path.join(result_path, 'Objects/Scanpy/')
table_path = os.path.join(result_path, 'Tables/')

def standard_quality_control_visium(anndata: ad.AnnData,
                                    min_count: int = 500,
                                    min_genes: int = 300,
                                    mouse: bool = True,
                                    target_sum: int = 10_000,
                                    hvg_n: int = 3000,
                                    ) -> ad.AnnData:

    anndata = anndata.copy()  # Copy to not modify input anndata
    anndata.var_names_make_unique()

    # Correction of the type from Spatial
    anndata.obsm['spatial'] = np.array(anndata.obsm['spatial'], dtype=float)
    anndata.obs['array_row'] = anndata.obs['array_row'].astype(int)
    anndata.obs['array_col'] = anndata.obs['array_col'].astype(int)
    anndata.var['GeneSymbol'] = anndata.var_names  # Add Gene Symbols to var

    # Identify Mitochondrial Genes
    mt = 'MT-'  # Assume Human Data
    if mouse:
        mt = 'mt-'  # Change to mouse format
    anndata.var['mt'] = anndata.var['GeneSymbol'].str.startswith(mt)

    # Filter base on counts
    sc.pp.filter_cells(anndata, min_counts=min_count)
    sc.pp.filter_cells(anndata, min_genes=min_genes)  # Minimum number of genes for a spot

    # Normalise the data
    anndata.layers['counts'] = anndata.X.copy()
    sc.pp.normalize_total(anndata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(anndata)
    sc.pp.highly_variable_genes(anndata, flavor='seurat', n_top_genes=hvg_n)
    anndata.layers['logcounts'] = anndata.X.copy()

    return anndata
#</editor-fold>

#<editor-fold desc="Pre-Process">
slides = {name: sc.read_h5ad(os.path.join(input_path, name)) for name in os.listdir(input_path)}

# Remove Spots not capturing tissue (Manually selected using LoupeBrowser)
in_tissue = {name: pd.read_csv(os.path.join(table_path, 'LoupeBrowser', name, f'{name}_in_tissue.csv')) for name in
             slides}
for name, adata in slides.items():
    BCs = in_tissue[name].dropna()['Barcode'].tolist()
    slides[name] = adata[adata.obs_names.isin(BCs)]
#</editor-fold>

#<editor-fold desc="Quality Control">
for name, adata in slides.items():
    adata = standard_quality_control_visium(adata,  # AnnData object
                                            min_count=500,  # Minimum number of counts to include a spot
                                            min_genes=300,  # Minimum number of genes expressed to include a spot
                                            mouse=True,  # The data is Mouse
                                            target_sum=10_000,  # Normalise to 10,000 Reads per spot
                                            )
    slides[name] = adata
#</editor-fold>

#<editor-fold desc="Concatenate adatas">
slide_concat = ad.concat(slides.values(), join='outer',
                         label='sample', keys=slides.keys(),
                         uns_merge='unique', index_unique='-')

slide_concat.var_names_make_unique()  # Make Sure we have unique gene names
slide_concat.X = slide_concat.layers['counts'].copy()  # Make Sure we have the raw counts in .X for cell2location
#</editor-fold>

#<editor-fold desc="Add Metadata and Save ">
slides_id = {k: list(v.uns['spatial'].keys())[0] for k, v in slides.items()}
slides_id_inv = {v: k for k, v in slides_id.items()}
slide_concat.obs['library_id'] = [slides_id[lid] for lid in slide_concat.obs['sample']]

# Update .uns name for spatial slide; sample name instead library ID
tmp_spatial = slide_concat.uns['spatial']
new_spatial = {}
for key, val in tmp_spatial.items():
    new_spatial[slides_id_inv[key]] = val

slide_concat.uns['spatial'] = new_spatial
slide_concat.obs['condition'] = slide_concat.obs['sample'].str.split('_').str[0]

slide_concat.write(os.path.join(object_path, 'Visium_MouseHeart_OldYoungAMI.h5ad'))
#</editor-fold>

#<editor-fold desc="Integration">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Integration/')

# Load Objects
visium_dimmeler = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_MouseHeart_OldYoungAMI.h5ad'))
visium_yamada = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_MouseHeart_MI_Yamada.h5ad'))

visium = ad.concat([visium_dimmeler, visium_yamada],
                   join='outer',  # To keep all the genes
                   label='lab',
                   keys=['Dimmeler', 'Yamada'])

# Transfer the spatial information lost
sp_tmp_dict = {}
for dataset in [visium_dimmeler, visium_yamada]:
    for sample in dataset.obs['sample']:
        # Correct Scale Factor of Spatial Data in Yamada
        if 'd1' in sample or 'd7' in sample or 'd14' in sample:
            dataset.uns['spatial'][sample]['scalefactors']['tissue_hires_scalef'] = 1  # Correction->match image & spots
        if sample not in sp_tmp_dict:
            sp_tmp_dict[sample] = dataset.uns['spatial'][sample]

# Rename samples to be consistent
rename_samples = {'Old_1': 'Old_1', 'Old_2': 'Old_2', 'Old_3': 'Old_3', 'Old_4': 'Old_4', 'Old_5': 'Old_5',
                  'Young_1': 'Young_1', 'Young_2': 'Young_2', 'Young_3': 'Young_3', 'Young_4': 'Young_4',
                  'Young_5': 'Young_5', 'AMI_1': 'MI_d28_1', 'AMI_2': 'MI_d28_2', 'AMI_3': 'MI_d28_3',
                  'AMI_4': 'MI_d28_4', 'AMI_5': 'MI_d28_5', 'MI_d7_1': 'MI_d7_1', 'MI_d7_2': 'MI_d7_2',
                  'MI_d7_3': 'MI_d7_3', 'MI_d1_1': 'MI_d1_1', 'MI_d1_2': 'MI_d1_2', 'MI_d1_3': 'MI_d1_3',
                  'MI_d14_1': 'MI_d14_1', 'MI_d14_2': 'MI_d14_2', 'MI_d14_3': 'MI_d14_3',
                  }
visium.obs['sample'] = [rename_samples[sample] for sample in visium.obs['sample']]
spatial_dict = {rename_samples[sample]: sp_tmp_dict[sample] for sample in sp_tmp_dict}
visium.uns['spatial'] = spatial_dict

# Remove Redundant / Not Complete/shared / Unnecesary .obs and .obsm from previous analysis excluded
remove_obs = ['n_counts', 'n_genes', 'tile_path', 'stLearn_louvain', 'log1p_total_counts', 'pct_counts_in_top_50_genes',
              'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes',
              'log1p_total_counts_mt', 'log1p_total_counts_ribo',
              'imagecol', 'imagerow', 'condition', 'library_id']
remove_obsm = ['X_morphology', 'X_pca', 'X_tile_feature', 'X_umap']

for col in remove_obs:
    del visium.obs[col]
for col in remove_obsm:
    del visium.obsm[col]

# Add Metadata --> Condition; TimePoint; Replicate
visium.obs['condition'] = visium.obs['sample'].str.split('_').str[0]
visium.obs['replicate'] = visium.obs['sample'].str.split('_').str[-1]

timepoints = []
for sample in visium.obs['sample']:
    if 'Young' in sample:
        timepoints.append('3_month')
    elif 'Old' in sample:
        timepoints.append('18_month')
    else:
        timepoints.append(sample.split('_')[1])
visium.obs['timepoint'] = timepoints

# Re-Normalise, Identify HVG & Filter out lowely expressed genes
sc.pp.filter_genes(visium, min_cells=3)  # 32,325 --> 21,663

visium.X = visium.layers['counts'].copy()
sc.pp.normalize_total(visium, target_sum=10_000)
sc.pp.log1p(visium)
sc.pp.highly_variable_genes(visium, batch_key='sample')
visium.layers['logcounts'] = visium.X.copy()

# Remove Hemoglobin genes & mitochondria genes  --> Represent Contamination / Not Informative in Spatial Data
hb = ['Hba-a1', 'Hba-a2', 'Hbb-bs', 'Hbb-bt']
visium.var['Hb'] = [True if gene in hb else False for gene in visium.var_names]
visium.obsm['Hb'] = visium[:, visium.var['Hb'].values].X.toarray()
visium = visium[:, ~visium.var['Hb'].values]  # 21,663 --> 21,659
visium.var['mt'] = [gene.startswith('mt-') for gene in visium.var_names]
visium.obsm['mt'] = visium[:, visium.var['mt'].values].X.toarray()
visium = visium[:, ~visium.var['mt'].values]  # 21,659 --> 21,646

# SCTransform Normalisation
visium.X = visium.layers['counts'].copy()
visium = SCTransform(visium, '/media/Storage/DavidR/tmp/')
visium.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AMId1-28_11May.h5ad'))


# Integrate Old and Young --> Aging Project
aging = visium[visium.obs['condition'].isin(['Young', 'Old'])].copy()
aging = spatial_integration(aging,
                            'sample',  # Column with batch information
                            margin=2.5,  # Larger values to stronger batch correction
                            radius=125,  # Leads to 5-6 neighbors in Visium
                            ngs=5,  # Number of Neighbors for Distance Matrix
                            res=.8,  # Resolution for Leiden clustering
                            h_dim=[450, 30],  # AutoEncoder Dimension
                            knn_ng=100,  # Neighbors for Spatial Graph
                            n_hvg=8500,  # Number of HVG to calculate; High to ensure the inner join keeps enough
                            min_dist=0.2,
                            spread=1,
                            )
aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_11May.h5ad'))
#</editor-fold>

#<editor-fold desc="Plots Aging">
plot_slides(aging, 'leiden_STAligner', common_legend=True, minimal_title=True, ncols=5,
            order=sorted(aging.obs['sample'].unique()), fig_path=figure_path,
            filename=f'Spatial_Aging_leiden_STAligner.svg')
split_umap(aging, 'condition', path=figure_path, filename=f'UMAP_Aging_SplitbyCondition.svg',
           ncol=2, figsize=(6, 4))
split_umap(aging, 'AnatomicRegion', path=figure_path, filename=f'UMAP_Aging_SplitbyAnatomicRegion.svg',
           ncol=3, figsize=(8, 5))
split_umap(aging, 'sample', path=figure_path, filename=f'UMAP_Aging_Splitbysample.svg',
           ncol=5, figsize=(12, 8))
pl_umap(aging, 'leiden_STAligner', path=figure_path, filename=f'UMAP_Aging_leidenSTAligner.svg', figsize=(6, 4))

# figure_path = os.path.join(main_path, 'Figures/2_Annotation/Niches/')
# for group in aging.obs['leiden_STAligner'].unique():
#     plot_slides(aging, 'leiden_STAligner', common_legend=True, minimal_title=True, ncols=5,
#             order=sorted(aging.obs['sample'].unique()), fig_path=figure_path, groups=[group],
#             filename=f'Spatial_Aging_leiden_STAligner_{group}.svg')
#</editor-fold>

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#<editor-fold desc="Post c2l">
model_path = os.path.join(main_path, 'Models/')
figure_path = os.path.join(main_path, 'Figures/')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_11May.h5ad'))
young_c2l = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/Visium_Young_200_240524_Spatrained.h5ad'))
old_c2l = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/Visium_Old_20_240524_Spatrained.h5ad'))

# Export Cell Abundance
young_c2l.obs[young_c2l.uns['mod']['factor_names']] = young_c2l.obsm['q05_cell_abundance_w_sf']
old_c2l.obs[old_c2l.uns['mod']['factor_names']] = old_c2l.obsm['q05_cell_abundance_w_sf']


# Transfer Cell Abundance to aging
meta_young = young_c2l.obsm['q05_cell_abundance_w_sf']
meta_old = old_c2l.obsm['q05_cell_abundance_w_sf']
c2l = pd.concat([meta_old, meta_young])
c2l.columns = old_c2l.uns['mod']['factor_names']
c2l = c2l.reindex(index=aging.obs_names, columns=sorted(c2l.columns))
aging.obsm['c2l'] = c2l

spot_abundance = c2l.sum(axis=1).to_frame('Abundance')  # Total abundance per spot
spot_abundance['slide'] = spot_abundance.index.str.split('-').str[-1]

# Plot the distribution
fig, axs = plt.subplots(1, 1, figsize=(18, 9))
g = sns.violinplot(spot_abundance, y='Abundance', x='slide',
                   ax=axs,
                   order=sorted(spot_abundance['slide'].unique()),
                   palette='tab20')
sns.despine()
g.set_title('Distribution of Cell Abundance per Spot', fontsize=20, fontweight='bold')
g.set_xlabel('Samples', fontsize=18)
g.grid(False)
g.set_ylabel('Abundance', fontsize=20)
plt.savefig(os.path.join(figure_path, '2_Cell2location/ViolinPlot_TotalAbundance_PerSpot.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Calculate c2l proportions">
aging_copy = aging.copy()
del aging_copy.obsm['c2l']['Neural']  # Exclude (unsure annotation, also removed in snRNA-seq)

# Re-name cell types with the final annotation of the snRNA-seq
update_cts = {'Adip': 'Adip', 'Artery_EC':'ArtEC', 'B':'B_cells', 'CM':'CM',
              'Capillary_EC':'CapEC', 'Monocytes':'Ccr2+MP', 'Endocardial_EC':'EndoEC',
              'FB':'Fibroblasts', 'FBa':'Fibro_activ', 'Lymphatic_EC':'LymphEC',
              'MP':'MP', 'Meso':'Epi_cells', 'PC':'Pericytes', 'SMC':'SMC',
              'T_cells':'T_cells', 'Vein_EC':'VeinEC', 'Neural':'Neural'}
aging_copy.obsm['c2l'].columns = [update_cts[name] for name in aging_copy.obsm['c2l'].columns]

for key in update_cts.keys():  # Remove from obs
    del aging_copy.obs[key]

# Compute proportions
abundance = aging_copy.obsm['c2l'].copy()
abundance['nCells'] = abundance.sum(axis=1)

c2l_props = abundance.iloc[:,:-1].div(abundance['nCells'], axis=0)  # Compute proportions Abundance / total nCells
c2l = abundance.iloc[:,:-1].copy()

aging_copy.obsm['c2l'] = c2l.copy()
aging_copy.obsm['c2l_prop'] =c2l_props.copy()
aging_copy.obs[aging_copy.obsm['c2l_prop'].columns] = aging_copy.obsm['c2l_prop'].values
aging_copy.obs['nCells'] = abundance['nCells'].copy()
aging_copy.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
#</editor-fold>

#<editor-fold desc="QC Plots">
figure_path = os.path.join(main_path, 'Figures/1_QualityControl')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()

davidrUtility.iOff()
for batch in aging.obs['sample'].unique():
    slide = davidrUtility.select_slide(aging, batch)
    davidrPlotting.generate_violin_plot(slide, title=f'{batch} QC Metrics',path=figure_path, filename=f"{today}_ViolinPlot_{batch}_QCMetrics.svg")
    sc.pl.spatial(slide, color= ['total_counts','n_genes_by_counts','pct_counts_mt'], frameon=False, cmap='inferno', bw=True, size=1.5,)
    plt.savefig(os.path.join(figure_path, f'{today}_Spatial_{batch}_QCMetrics.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Add Anatomic Regions">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Add_AnatomicRegions')
aging = sc.read_h5ad(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
files = [f for f in os.listdir(os.path.join(table_path, 'LoupeBrowser')) if 'Anatomic' in f and 'AMI' not in f]

anat_tmp_dict = {}
for file in files:
    samplename = file.split('_Anatomic')[0]
    df = pd.read_csv(os.path.join(table_path, 'LoupeBrowser', file))
    df['Barcode'] = df['Barcode'] + f'-{samplename}'  # Correction to match barcode of anndata
    df.set_index('Barcode', inplace=True)
    anat_tmp_dict[samplename] = df

meta = [anat_tmp_dict[sample].loc[bc][0] for bc, sample in aging.obs['sample'].items()]
aging.obs['AnatomicRegion'] = meta

davidrPlotting.pl_umap(aging, color='AnatomicRegion', figsize=(5,5), path=figure_path,
                       filename=f'UMAP_ST_AnatomicRegion.svg')
davidrPlotting.plot_slides(aging, 'AnatomicRegion',
                           order=sorted(aging.obs['sample'].unique()), ncols=5,
                           common_legend=True, minimal_title=True, fig_path=figure_path,
                           filename=f'SpatialPlot_ST_AnatomicRegion.svg')

aging.write(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

df = pd.read_csv(os.path.join(table_path, 'LoupeBrowser', 'Young_3_AnatomicRegion_Corrected_100624.csv'))
df['Barcode'] = df['Barcode'] + '-Young_3'
df.set_index('Barcode', inplace=True)

for bc in aging.obs_names:
    if bc in df.index:
        aging.obs.loc[bc, 'AnatomicRegion'] = df.loc[bc, 'Anatomic']

# davidrPlotting.plot_slides(aging, 'AnatomicRegion',  order=sorted(aging.obs['sample'].unique()), ncols=5,
#                           common_legend=True, minimal_title=True, fig_path=figure_path,
#                           filename=f'SpatialPlot_ST_AnatomicRegion.svg')

aging.write(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
#</editor-fold>

#<editor-fold desc="Niches">
figure_path = os.path.join(main_path, 'Figures/', '3_Annotation/Niches/')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()
ref= sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()


aging.obs['clusters'] = 'Niche ' + aging.obs.leiden_STAligner.astype(str)
# Convert in Categorical and force to have the same order to keep color scheme
aging.obs['clusters'] = pd.Categorical(aging.obs.clusters).reorder_categories(
    ['Niche ' + clust for clust in aging.obs.leiden_STAligner.cat.categories])
aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

davidrPlotting.plot_slides(aging, 'clusters', ncols=2, bw=True, common_legend=True,
                           minimal_title=True, select_samples=['Young_3', 'Old_5'], order=['Young_3', 'Old_5'],
                           title_fontweight='bold', title_fontsize=25,
                           fig_path=figure_path, filename='SpatialPlot_ST_Old5Young3_clusters.svg')

davidrPlotting.pl_umap(aging, ['clusters'], size=15, figsize=(4, 4),
                path=figure_path, filename='UMAP_ST_clusters.svg')

davidrPlotting.plot_slides(aging, 'clusters', bw=True, common_legend=True, minimal_title=True,
                           select_samples=['Young_1', 'Young_2', 'Young_4', 'Young_5',
                                           'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           order=['Young_1', 'Young_2', 'Young_4', 'Young_5',
                                           'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           title_fontweight='bold', fig_path=figure_path,
                           filename='SpatialPlot_ST_Clusters_Old1-4Young1245.svg')

sc.tl.rank_genes_groups(aging, groupby='clusters', method='wilcoxon', tie_correct=True, pts=True, logcounts=True)
# Save DEGs as ExcelSheet
dge_leiden = sc.get.rank_genes_groups_df(aging, group=None)
with pd.ExcelWriter(
        os.path.join(table_path, 'DGE', 'DGE_ST_Aging_Niches.xlsx')) as writer:
    dge_leiden.to_excel(writer, sheet_name='AllGenes', index=False)  # Sheet Name with all the genes
    for cluster in dge_leiden.group.unique():
        # For each cluster save only significant genes (Padj < 0.05) & rank by LFC
        dge_subset = dge_leiden[dge_leiden.group == cluster]
        dge_subset = dge_subset[dge_subset['pvals_adj'] < 0.05].sort_values('logfoldchanges', ascending=False)
        dge_subset.to_excel(writer, sheet_name=f'SigGenes_Clust{cluster}', index=False)
#</editor-fold>

#<editor-fold desc="PseudoBulk of Niches">
aging_copy = aging[:, aging.var.highly_variable].copy()

pdata = dc.get_pseudobulk(
    aging_copy,
    sample_col='sample',
    groups_col='clusters',
    layer='counts',
    mode='sum',
    min_cells=10,
    min_counts=1000
)

dc.plot_psbulk_samples(pdata, groupby=['sample', 'clusters'], figsize=(12, 4))

pdata.layers['counts'] = pdata.X.copy()

# Normalise, scale and pca
sc.pp.normalize_total(pdata, target_sum=10_000)
sc.pp.log1p(pdata)
sc.pp.scale(pdata, max_value=10)
sc.tl.pca(pdata)

dc.swap_layer(pdata, 'counts', X_layer_key=None, inplace=True)

sc.pl.pca(pdata, color=['sample', 'clusters'])
sc.pl.pca_variance_ratio(pdata)


dc.get_metadata_associations(
    pdata,
    obs_keys = ['sample', 'condition', 'clusters', 'psbulk_n_cells', 'psbulk_counts'],  # Metadata columns to associate to PCs
    obsm_key='X_pca',  # Where the PCs are stored
    uns_key='pca_anova',  # Where the results are stored
    inplace=True,
)

dc.plot_associations(
    pdata,
    uns_key='pca_anova',  # Summary statistics from the anova tests
    obsm_key='X_pca',  # where the PCs are stored
    stat_col='p_adj',  # Which summary statistic to plot
    obs_annotation_cols = ['sample', 'condition', 'clusters'], # which sample annotations to plot
    titles=['Principle component scores', 'Adjusted p-values from ANOVA'],
    figsize=(7, 5),
    n_factors=10,
    cmap_cats='tab20c',
)
plt.savefig(os.path.join(figure_path, 'PCA_PseudoBulk_ST_PCAssociations.svg'), bbox_inches='tight')


pca = pdata.obsm['X_pca'].copy()
meta = pdata.obs[['clusters', 'condition']]
niche_color = {pdata.obs['clusters'].cat.categories[i]: pdata.uns['clusters_colors'][i] for i in range(len(pdata.obs.clusters.unique()))}
cond_color = {'Young': 'o', 'Old':'x'}


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i, bc in enumerate(pdata.obs_names):
    niche = pdata.obs.loc[bc, 'clusters']
    cond = pdata.obs.loc[bc, 'condition']
    ax.scatter(pca[i, 0], pca[i, 1], color=niche_color[niche],
               marker=cond_color[cond], s=100)

ax.grid(False)
ax.set_xlabel('PC1 (33.29 %)')
ax.set_ylabel('PC2 (9.57 %)')
ax.set_xticks([])
ax.set_yticks([])
ax.legend()

color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=niche_color[n], markersize=10, label=n)
                 for n in niche_color]

marker_handles = [plt.Line2D([0], [0], marker=m, color='k', linestyle='None', markersize=10, label=c)
                  for c, m in cond_color.items()]

color_legend = ax.legend(handles=color_handles, title='Clusters', loc='upper left', bbox_to_anchor=(1, .5), title_fontproperties={'weight':'bold', 'size':13})
ax.add_artist(color_legend)  # To avoid replacing the first legend
ax.legend(handles=marker_handles, title='Conditions', loc='upper left', bbox_to_anchor=(1, 0.9), title_fontproperties={'weight':'bold', 'size':13})
ax.set_title('PCA Pseudobulk', fontsize=20)
plt.savefig(os.path.join(figure_path, 'PCA_ST_Pseudobulk.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Changes in the abundance of the niches upon aging">
df = aging.obs[['clusters', 'sample']].copy()
df_cond = df.value_counts(['sample', 'clusters']).reset_index()

# Normalise by sample
df_groups = df_cond[['sample', 'count']].groupby('sample').agg('sum')
df_cond['norm'] = [cell['count'] / df_groups.loc[cell['sample'], 'count'] for idx, cell in df_cond.iterrows()]
df_cond['condition'] = df_cond['sample'].str.split('_').str[0]

fig, axs = plt.subplots(1, 1, figsize=(25, 9))
bp = sns.barplot(df_cond, x='clusters', y='norm', capsize=.1, palette={'Young':'darkorange', 'Old':'royalblue'},
                     hue_order=['Young', 'Old'], hue='condition', ax=axs)
bp.set_xlabel('')
bp.set_ylabel('Mean Proportion', fontsize=18)
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold', fontsize=15)
sns.move_legend(bp, loc='upper right', ncols=2, frameon=False, title='Condition',
                title_fontproperties={'weight':'bold', 'size':15})
bp.grid(False)
plt.savefig(os.path.join(figure_path, f'Barplot_clusters_ProportionYoungOld.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Add large vessels">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Add_LargeVessels')

# Load data
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()
snrna = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad'))
snrna.X = snrna.layers['logcounts'].copy()

hexamers_prop = pd.DataFrame([])
for slide in tqdm(aging.obs['sample'].unique(), desc='Slide'):
    # Per slide, Otherwise the distance is not correctly computed
    adata = davidrUtility.select_slide(aging, slide)
    for idx, bc in enumerate(adata.obs_names):
        hexa_idx = davidrUtility.get_surrounding(adata, idx, 100)  # Immediate Neighbors (Radius = 100)
        hexa_bcs = adata.obs_names[hexa_idx]
        # Proportion of a spot is the Hexamer proportion -> Prop = Abundance Ct / Total Abundance
        hexa_abund = adata.obsm['c2l'].loc[hexa_bcs, :]
        hexa_prop = round(hexa_abund.sum(axis=0).div(hexa_abund.sum(axis=0).sum(), axis=0), 4)
        # Proportion Cutoff to have at least 1 cell
        hexa_cutoff = round(1 / hexa_abund.sum(axis=0).sum(), 4)
        newbc = '_'.join([str(val) for val in hexa_idx]) + '_' + slide  # Idx1_Idx2_[...]_SlideName
        hexa_prop = pd.DataFrame(hexa_prop.values, columns=[bc], index=hexa_prop.index).T
        hexa_prop['dynamic_cutoff'] = hexa_cutoff
        hexamers_prop['NewBc'] = newbc
        hexamers_prop = pd.concat([hexamers_prop, hexa_prop])

# Binarise the Matrix ; At least 1 cell is present
hexamers_1_0 = pd.DataFrame([])
for idx in range(hexamers_prop.shape[0]):
    row = hexamers_prop.iloc[idx, :].drop(['NewBc', 'dynamic_cutoff'])
    row_cutoff = hexamers_prop.iloc[idx, -1]
    row_0_1 = row.apply(lambda x: 1 if x > row_cutoff else -1)
    row_0_1 = pd.DataFrame(row_0_1.values, index=row_0_1.index, columns=[hexamers_prop.index[idx]]).T
    hexamers_1_0 = pd.concat([hexamers_1_0, row_0_1])

# Generate annotation
HexaVessels, HexaFB, HexaFBa, EndoPeri = [], [], [], []
for bc in aging.obs_names:
    bc_cts = hexamers_1_0.loc[bc, :]
    bc_cts = list(bc_cts[bc_cts == 1].index)
    # Annotate Vessels
    if ('ArtEC' in bc_cts and 'SMC' in bc_cts) and ('VeinEC' in bc_cts):
        HexaVessels.append('MixVasc')
    elif ('ArtEC' in bc_cts and 'SMC' in bc_cts) and 'LymphEC' in bc_cts:
        HexaVessels.append('Art_Lymph')
    elif ('VeinEC' in bc_cts) and 'LymphEC' in bc_cts:
        HexaVessels.append('Vein_Lymph')
    elif 'ArtEC' in bc_cts and 'SMC' in bc_cts:
        HexaVessels.append('Arteries')
    elif 'VeinEC' in bc_cts:
        HexaVessels.append('Veins')
    elif 'LymphEC' in bc_cts:
        HexaVessels.append('Lymphatics')
    else:
        HexaVessels.append('nVasc')

    # Fibroblasts Spots - At least 1 cell for FB
    if (('ArtEC' in bc_cts and 'SMC' in bc_cts) or
       ('VeinEC' in bc_cts)) and ('Fibroblasts' in bc_cts):
        HexaFB.append('FB_PeriVasc')
    elif 'Fibroblasts' in bc_cts:
        HexaFB.append('FB_Inters')
    else:
        HexaFB.append('Tissue')

    # Fibroblasts Activated Spots
    if (('ArtEC' in bc_cts and 'SMC' in bc_cts) or
       ('VeinEC' in bc_cts)) and ('Fibro_activ' in bc_cts):
        HexaFBa.append('FBa_PeriVasc')
    elif 'Fibro_activ' in bc_cts:
        HexaFBa.append('FBa_Inters')
    else:
        HexaFBa.append('Tissue')

    # Endo / Epi region
    if 'EndoEC' in bc_cts :
        EndoPeri.append('Endocardium')
    elif 'Epi_cells' in bc_cts:
        EndoPeri.append('Epicardium')
    else:
        EndoPeri.append('Tissue')

# Add annotation
aging.obs['vessels'] = pd.Categorical(HexaVessels)
aging.obs['FB_Hexa'] = pd.Categorical(HexaFB)
aging.obs['FBa_Hexa'] = pd.Categorical(HexaFBa)
aging.obs['EndoEpi_Hexa'] = pd.Categorical(EndoPeri)

aging.obs['Hexa_cutoff'] = hexamers_prop['dynamic_cutoff']
aging.obsm['Hexa_prop'] = hexamers_prop.iloc[:,1:-1]

# Set colors for categories
colors = ['gold', 'tomato', 'forestgreen', 'darkorchid', '#ffe4b5', 'royalblue', 'white']
aging.uns['vessels_colors'] = colors
colors = ['moccasin', 'tomato','white']
aging.uns['FB_Hexa_colors'] = colors
colors = ['gold', 'tomato','white']
aging.uns['FBa_Hexa_colors'] = colors

aging.obs['clusters'] = 'Niche ' + aging.obs.leiden_STAligner.astype(str)
aging.obs['clusters'] = pd.Categorical(aging.obs.clusters).reorder_categories(
    ['Niche ' + clust for clust in aging.obs.leiden_STAligner.cat.categories])

aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

# Vessels
davidrPlotting.plot_slides(aging, 'vessels', common_legend=True, minimal_title=True,
                           ncols=5, order=sorted(aging.obs['sample'].unique()),
                           alpha_img=.75, bw=True, figsize=(15, 8),
                           fig_path=figure_path,
                           filename='SpatialPlot_ST_HexaVessels.svg',
                           palette={'Art_Lymph': 'gold',
                                    'Arteries': 'tomato',
                                    'Lymphatics': 'forestgreen',
                                    'MixVasc': 'darkorchid',
                                    'Vein_Lymph': '#ffe4b5',
                                    'Veins': 'royalblue',
                                    'nVasc': '#FF000000'},
                           title_fontweight='bold')

df = aging.obsm['c2l'].copy()
df[['condition', 'type_vessel', 'sample']] = aging.obs[['condition', 'vessels', 'sample']].copy()

# Select LargeVessels
arteries = df[df['type_vessel'].isin(['Arteries', 'MixVasc', 'Art_Lymph'])]
veins = df[df['type_vessel'].isin(['Veins', 'MixVasc', 'Vein_Lymph'])]
lymphatics = df[df['type_vessel'].isin(['Lymphatics', 'Vein_Lymph', 'Art_Lymph'])]

# Clean
del arteries['type_vessel'], veins['type_vessel'], lymphatics['type_vessel']

# Groupby condition and calculate Proportions
arteries = arteries.groupby(['condition', 'sample']).agg('sum').T
veins = veins.groupby(['condition', 'sample']).agg('sum').T
lymphatics = lymphatics.groupby(['condition', 'sample']).agg('sum').T


# Compute proportions
arteries = arteries / arteries.sum()
veins = veins / veins.sum()
lymphatics = lymphatics / lymphatics.sum()


# Select the CellType we want
arteries = arteries.loc['ArtEC', :].reset_index().dropna()
veins = veins.loc['VeinEC', :].reset_index().dropna()
lymphatics = lymphatics.loc['LymphEC', :].reset_index().dropna()

# Merge Subpopulations
arteries['Type'] = 'Arteries'
veins['Type'] = 'Veins'
lymphatics['Type'] = 'Lymphatics'

arteries.columns = ['condition', 'sample', 'Proportions', 'Type']
veins.columns = ['condition', 'sample', 'Proportions', 'Type']
lymphatics.columns = ['condition', 'sample', 'Proportions', 'Type']
VesselsSpots = pd.concat([arteries, veins, lymphatics], ignore_index=True)

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
bp = sns.barplot(VesselsSpots, x='Type', y='Proportions',
                 hue='condition', estimator='mean',
                 palette={'Old': 'royalblue', 'Young': 'darkorange'},
                 hue_order=['Young', 'Old'],
                 capsize=0.1, errorbar='ci', gap=.1, ax=axs)

sns.stripplot(VesselsSpots, x='Type', y='Proportions',
              hue='condition', hue_order=['Young', 'Old'],
              ax=bp, color='black', dodge=True, alpha=.6)

sns.despine()
bp.grid(False)
bp.set_xlabel('Vessels Subtype', fontsize=15, fontweight='bold')
bp.set_ylabel('Proportion', fontsize=15, fontweight='bold')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
for bar in bp.patches:
    bar.set_zorder(3)
# remove extra legend handles
handles, labels = bp.get_legend_handles_labels()
bp.legend(handles[:2], labels[:2], title='Condition',
          bbox_to_anchor=(.5, 1.1), loc='upper center',
          frameon=False, ncols=2, title_fontproperties={'weight': 'bold'}
          )
plt.savefig(os.path.join(figure_path, 'Barplot_ST_Distribution_Vessels_Aging.svg'),
            bbox_inches='tight')


# Test if there is a significant difference
from scipy.stats import mannwhitneyu, shapiro
for vessel in VesselsSpots['Type'].unique():
    sdata = VesselsSpots[VesselsSpots['Type'] == vessel]
    s, p = mannwhitneyu(sdata[sdata['condition'] == 'Old']['Proportions'],
                        sdata[sdata['condition'] == 'Young']['Proportions'],
                        use_continuity=True)
    print(vessel, p)
#</editor-fold>

#<editor-fold desc="Differential cell proportion on vessels">
figure_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/0_Manuscript/'
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

df = aging.obsm['c2l_prop'].copy()
adata = ad.AnnData(X=df.values, obs=list(df.index), var=list(df.columns))
adata.obs_names = adata.obs[0]
adata.var_names = adata.var[0]
adata.obs = aging.obs.copy()

sc.tl.rank_genes_groups(adata, groupby='vessels', method='wilcoxon', tie_correct=True, pts=True)
for col in adata.var_names:
    del adata.obs[col]
tmp = sc.get.rank_genes_groups_df(adata, group=None, pval_cutoff=0.05, log2fc_min=1)
tmp.to_excel(os.path.join(table_path, 'Differential_Cell_Proportion_0_05Pval_1LFC.xlsx'))

ax = sc.pl.dotplot(adata,
                   groupby='vessels',
                   cmap=davidrUtility.generate_cmap('white', '#ff4d2e'),
                   expression_cutoff=.01,
                   var_names=[ 'SMC', 'ArtEC', 'VeinEC', 'LymphEC'],
                   categories_order=['Arteries', 'Art_Lymph', 'MixVasc',
                                     'Veins', 'Vein_Lymph', 'Lymphatics',
                                     'nVasc'],
                   vmax=.1,
                   show=False)
ax['color_legend_ax'].set_title('Mean cell prop.\nin group', fontsize=10)
ax['color_legend_ax'].grid(False)


ax['size_legend_ax'].set_title('\n\n\n'+ax['size_legend_ax'].get_title(), fontsize=12)
main_ax = ax['mainplot_ax']
main_ax.set_xticklabels(main_ax.get_xticklabels(), fontweight='bold', rotation=70, ha='center', va='top')
main_ax.set_yticklabels(main_ax.get_yticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, '241002_DCP_Vessels.svg'), bbox_inches='tight')
#</editor-fold>

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# <editor-fold desc="Progeny">
figure_path = os.path.join(main_path, 'Figures/4_FunctionalAnalysis/Progeny')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()

# Pathway Activity Inference with Progeny
progeny = dc.get_progeny(organism='human', top=500)  # Mouse gives problems
progeny['target'] = progeny['target'].str.capitalize()  # Convert in Mouse format

dict_adatas = {}
for s in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, s)
    sdata.X = sdata.layers['SCT_norm'].copy()
    dc.run_mlm(
        mat=sdata,
        net=progeny,
        source='source',
        target='target',
        weight='weight',
        verbose=True,
        use_raw=False
    )

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
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
mid_scaled = pd.DataFrame((np.zeros(14) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)),
                          index=adata_concat.var_names)
X_shifted = X + np.abs(X.min(axis=0))
mid_shifted = pd.DataFrame(0 + np.abs(X.min(axis=0)), index=adata_concat.var_names)

adata_concat.layers['X_scaled'] = X_scaled
adata_concat.layers['X_shifted'] = X_shifted

# In .X --> Positive means that it is activated and Negative that is inactive

table = pd.DataFrame()
for clust in adata_concat.obs.clusters.unique():
    sdata = adata_concat[adata_concat.obs.clusters == clust]
    sc.tl.rank_genes_groups(sdata, groupby='condition', method='wilcoxon', tie_correct=True, layer='X_shifted', logcounts=True)
    df = sc.get.rank_genes_groups_df(sdata, group='Old', pval_cutoff=0.05)
    df['cluster'] = clust
    table = pd.concat([table, df])

data = davidrUtility.ExtractExpression(adata_concat, adata_concat.var_names, groups=['condition', 'clusters'])
data_summary = data.groupby(['clusters', 'condition', 'genes']).agg('mean').reset_index().pivot(
    index=['clusters', 'genes'], columns='condition', values='expr')

table['OldMean'] = [data_summary.loc[(row['cluster'], row['names']), 'Old'] for idx, row in table.iterrows()]
table['YoungMean'] = [data_summary.loc[(row['cluster'], row['names']), 'Young'] for idx, row in table.iterrows()]

table_filt = table[~((table['OldMean'] < 0) & (table['YoungMean'] < 0))].sort_values('scores', ascending=False)
table_filt = table_filt[~table_filt.names.isin(['Androgen', 'Estrogen'])]
tmp = table_filt.pivot(index=['cluster'], columns='names', values='logfoldchanges')
tmp[tmp.isna()] =0
hm_lfc = tmp.reindex(index=['Niche 9','Niche 4', 'Niche 5','Niche 3','Niche 2',
                            'Niche 6','Niche 7','Niche 1','Niche 10','Niche 0',
                            'Niche 8'], columns=['EGFR', 'TNFa', 'WNT', 'p53',
                                                 'Hypoxia', 'TGFb'])

#  Clustermap showing Up and Down Pathways
cm = sns.clustermap(hm_lfc, cmap='RdBu_r',center=0, row_cluster=False, col_cluster=False, robust=True,
                    cbar_pos=[0.2, .86, .15, .03], cbar_kws={'orientation':'horizontal'}, square=True,
                    figsize=(4, 6), linewidth=0.1)
# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
colorbar_ax = cm.cax

heatmap_ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)

heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontweight='bold', fontsize=12, rotation=75, ha='right')

# Modify colorbar
colorbar_ax.set_title('Log2FC', fontweight='bold', loc='center', fontsize=12)
colorbar_ax.set_xticklabels(colorbar_ax.get_xticklabels(), fontweight='bold', fontsize=12)

plt.savefig(os.path.join(figure_path, '241018_Clustermap_SigUpAging_Niches.svg'), bbox_inches='tight')
#</editor-fold>

# <editor-fold desc="ORA on Heart">
msigdb = dc.get_resource('MSigDB')
msigdb['genesymbol'] = msigdb['genesymbol'].str.capitalize()
msigdb = msigdb[msigdb['collection'] == 'go_biological_process']
msigdb = msigdb[~msigdb.duplicated(['geneset', 'genesymbol'])]
# msigdb.loc[:, 'geneset'] = [name.split('GOBP_')[1] for name in msigdb['geneset']]

dict_adatas = {}
for sample in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, sample)

    dc.run_ora(
        mat=sdata,
        net=msigdb,
        source='geneset',
        target='genesymbol',
        verbose=True,
        use_raw=False)

    # Store in a different key
    sdata.obsm['msigdb_ora_estimate'] = sdata.obsm['ora_estimate'].copy()
    sdata.obsm['msigdb_ora_pvals'] = sdata.obsm['ora_pvals'].copy()

    # adata of activities
    acts = dc.get_acts(sdata, obsm_key='msigdb_ora_estimate')

    # We need to remove inf and set them to the maximum value observed
    acts_v = acts.X.ravel()
    max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
    acts.X[~np.isfinite(acts.X)] = max_e
    acts.X[acts.X < -np.log10(0.05)] = 0  # If not significant put value to 0

    dict_adatas[sample] = acts

adata_concat = ad.concat(dict_adatas.values(),join='outer', uns_merge='unique')
max_e = np.nanmax(adata_concat.X.ravel()[np.isfinite(adata_concat.X.ravel())])
adata_concat.X[~np.isfinite(adata_concat.X)] = max_e  # Infinite replace with the highest value
adata_concat.X[adata_concat.X < -np.log10(0.05)] = 0  # Not significant spots replace with 0

# Identify top terms per niche considering both old and young
sc.tl.rank_genes_groups(adata_concat, groupby='clusters', method='wilcoxon', tie_correct=True, pts=True)

table = sc.get.rank_genes_groups_df(adata_concat, group=None)
table.to_excel(os.path.join(table_path, '241021_Wilcox_GOBP_Niches_VsRest.xlsx'), index=False)
# </editor-fold>

# <editor-fold desc="Analysis on pseudobulk">
# DGE per cluster for condition
pdata = dc.get_pseudobulk(aging, sample_col='sample', groups_col='clusters',
                          layer='counts', mode='sum', min_cells=10, min_counts=1000)
pdata.layers['counts'] = pdata.X.copy()

# Normalize, scale and compute pca
sc.pp.normalize_total(pdata, target_sum=1e4)
sc.pp.log1p(pdata)
sc.pp.scale(pdata, max_value=10)
sc.tl.pca(pdata)

# Return raw counts to X
dc.swap_layer(pdata, 'counts', X_layer_key=None, inplace=True)

dc_matrix = pd.DataFrame([])
dc_summary = pd.DataFrame([])
for clust in pdata.obs.clusters.unique():
    # Subset
    cluster = pdata[pdata.obs['clusters'] == clust].copy()

    # Filter genes
    genes = dc.filter_by_expr(cluster, group='condition', min_count=10, min_total_count=15)
    cluster = cluster[:, genes].copy()

    # DESeq2 Analysis
    inference = DefaultInference(1, n_cpus=8)
    dds = DeseqDataSet(adata=cluster, design_factors='condition',
                       ref_level=['condition', 'Young'], refit_cooks=True,
                       inference=inference, )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", 'Old', 'Young'], inference=inference, )
    stat_res.summary()

    results_df = stat_res.results_df
    results_df['cluster'] = clust
    mat = results_df[['stat']].T.rename(index={'stat': f'{clust}'})
    dc_matrix = pd.concat([dc_matrix, mat])
    dc_summary = pd.concat([dc_summary, results_df])

dc_matrix[dc_matrix.isna()] = 0
progeny = dc.get_progeny(top=500)
progeny.target = progeny.target.str.capitalize()
pathway_acts, pathway_pvals = dc.run_mlm(mat=dc_matrix, net=progeny)
dc.run_mlm(mat=pdata, net=progeny, use_raw=False)
pathway_acts_counts = pdata.obsm['mlm_estimate'].copy()
pathway_acts_counts['cond'] = pathway_acts_counts.index.str.split('_').str[0]
pathway_acts_counts['clust'] = pathway_acts_counts.index.str.split('_').str[-1]
pathway_acts_counts = pathway_acts_counts.melt(id_vars=['cond', 'clust']).groupby(['cond', 'clust', 'variable']).agg(
    'mean').reset_index().pivot(index=['clust', 'cond'], columns=['variable'], values='value')

# MSIGDB
msigdb = dc.get_resource('MSigDB')
msigdb.genesymbol = msigdb.genesymbol.str.capitalize()
msigdb = msigdb[msigdb['collection'] == 'go_biological_process']
msigdb = msigdb[~msigdb.duplicated(['geneset', 'genesymbol'])]
msigdb.loc[:, 'geneset'] = [name.split('GOBP_')[1] for name in msigdb['geneset']]

dc_pathway = pd.DataFrame([])
for clust in dc_summary.cluster.unique():
    up_genes = dc_summary[(dc_summary['padj'] < 0.05) &
                          (dc_summary['cluster'] == clust) &
                          (dc_summary['log2FoldChange'] > 0.25)]

    dw_genes = dc_summary[(dc_summary['padj'] < 0.05) &
                          (dc_summary['cluster'] == clust) &
                          (dc_summary['log2FoldChange'] < -0.25)]

    enr_pvals_up = dc.get_ora_df(df=up_genes, net=msigdb, source='geneset', target='genesymbol')

    enr_pvals_dw = dc.get_ora_df(df=dw_genes, net=msigdb, source='geneset',target='genesymbol')

    enr_pvals_up['state'] = 'up'
    enr_pvals_dw['state'] = 'dw'
    enr_pvals = pd.concat([enr_pvals_up, enr_pvals_dw])
    enr_pvals['cluster'] = clust
    dc_pathway = pd.concat([dc_pathway, enr_pvals])

dc_pathway.to_excel(os.path.join(table_path, '241017_DecoupleR_GOBP_ORA_PseudoBulk_Young_Vs_Old_Niches.xlsx'),
                    index=False)
dc_pathway_sig = dc_pathway[dc_pathway['FDR p-value'] < 0.05]
dc_summary.to_excel(os.path.join(table_path, '241017_DecoupleR_PseudoBulk_Metrics_Yizbf_Vs_Old_Niches.xlsx'))

# Plot GO Niches in barplots
df = pd.read_excel(
    '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Tables/241017_DecoupleR_GOBP_ORA_PseudoBulk_Young_Vs_Old_Niches.xlsx')

# Take the top 20 per cluster and condition sorting by combined score
# Additionally, we filter by the FDR < 0.05
top_terms = pd.DataFrame([])
for cluster in df['cluster'].unique():
    for state in df['state'].unique():
        sdf = df[(df['cluster'] == cluster) & (df['state'] == state)]
        sdf_filt = sdf[sdf['FDR p-value'] < 0.05].sort_values('Combined score', ascending=False)
        sdf_filt['Term'] = sdf_filt['Term'].str.replace('_', ' ')
        top_terms = pd.concat([top_terms, sdf_filt.head(10)])

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
                   color_up = [],
                   color_down = [],
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

# palegreen --> 'Immune process'
# tomato --> Muscle related
# sandybrown --> ECM and cell juction
"""
Niche 0
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 1
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 2
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 3
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 4
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 5
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 6
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 7
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
Niche 8
color_up =['sandybrown', 'lightgray', 'tomato', 'sandybrown', 'lightgray',
                           'lightgray', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['lightgray', 'tomato', 'sandybrown', 'sandybrown', 'sandybrown',
                           'lightgray', 'tomato', 'lightgray', 'sandybrown', 'sandybrown']
Niche 9
color_up = ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                           'lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray']
color_down = ['sandybrown', 'lightgray', 'tomato', 'lightgray', 'sandybrown',
                           'sandybrown', 'lightgray', 'lightgray', 'lightgray', 'sandybrown']
Niche 10
color_up = ['palegreen', 'palegreen', 'lightgray', 'palegreen', 'lightgray',
            'tomato', 'lightgray', 'lightgray', 'sandybrown', 'lightgray']
color_down = ['palegreen', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
              'lightgray', 'lightgray', 'lightgray', 'tomato', 'lightgray']
"""

clust = 'Niche 0'
sdf = top_terms[top_terms.cluster == clust]

split_bar_gsea(sdf,
               'Term',
               'Combined score',
               'state',
               'up',
               spacing=15,
               cutoff=45,
               color_up=['lightgray', 'palegreen', 'palegreen', 'lightgray', 'palegreen',
                         'lightgray', 'palegreen', 'lightgray', 'lightgray', 'sandybrown'],

               color_down=['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray',
                           'lightgray', 'lightgray', 'tomato', 'lightgray', 'lightgray'],
               figsize=(12, 6),
               title=f'Top 10 GO Terms in each Condition {clust}',
               path=figure_path,
               filename=f'241021_SplitBar_GOBP_YoungOld_DecoupleR_PseudoBulk_{clust}_sortBy_CombinedScores.svg')
#</editor-fold>

#<editor-fold desc="MistyR">
in_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis'

aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['logcounts'].copy()
ref = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()

for batch in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, batch)
    sdata.write(f'/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis/Objects/{batch}.h5ad')

# postMistyR analysis with R2 filtering"

# Load views
df = pd.read_csv(os.path.join(in_path, 'importances_samples_c2l_v3.csv'))
df_contrib = pd.read_csv(os.path.join(in_path, 'importances_samples_c2l_r2vals_v3.csv'))

#  Remove importances if R2 is < 5
cutoff = 5
df_contrib_intra = df_contrib[df_contrib.measure == 'intra.R2']
df_contrib_intra = df_contrib_intra[df_contrib_intra['mean'] < cutoff]
df_intra = df[df['view'] == 'intra']

ndf_intra = pd.DataFrame()
for batch  in tqdm(df_intra['sample'].unique()):
    rm_target = df_contrib_intra[df_contrib_intra['sample'] == batch]['target'].tolist()
    sdf_intra = df_intra[df_intra['sample'] == batch]
    sdf_intra = sdf_intra[~sdf_intra.Target.isin(rm_target)]
    ndf_intra = pd.concat([ndf_intra, sdf_intra])

# Juxta View uses multi.R2
df_contrib_juxta = df_contrib[df_contrib.measure == 'multi.R2']
df_contrib_juxta = df_contrib_juxta[df_contrib_juxta['mean'] < cutoff]
df_juxta = df[df['view'].str.startswith('juxta')]

ndf_juxta = pd.DataFrame()
for batch  in tqdm(df_juxta['sample'].unique()):
    rm_target = df_contrib_juxta[df_contrib_juxta['sample'] == batch]['target'].tolist()
    sdf_juxta = df_juxta[df_juxta['sample'] == batch]
    sdf_juxta = sdf_juxta[~sdf_juxta.Target.isin(rm_target)]
    ndf_juxta = pd.concat([ndf_juxta, sdf_juxta])

# para View uses multi.R2
df_contrib_para = df_contrib[df_contrib.measure == 'multi.R2']
df_contrib_para = df_contrib_para[df_contrib_para['mean'] < cutoff]
df_para = df[df['view'].str.startswith('para')]

ndf_para = pd.DataFrame()
for batch  in tqdm(df_para['sample'].unique()):
    rm_target = df_contrib_para[df_contrib_para['sample'] == batch]['target'].tolist()
    sdf_para = df_para[df_para['sample'] == batch]
    sdf_para = sdf_para[~sdf_para.Target.isin(rm_target)]
    ndf_para = pd.concat([ndf_para, sdf_para])

# Clustermap on IntraView
ndf_intra.loc[ndf_intra.Importance < 0, 'Importance'] = 0
ndf_intra.loc[ndf_intra.Importance.isna(), 'Importance']= 0
ndf_intra['Predictor'] = ndf_intra['Predictor'].str.replace('Ccr2_MP','Ccr2+MP')
ndf_intra['Target'] = ndf_intra['Target'].str.replace('Ccr2_MP','Ccr2+MP')

cm_intra = ndf_intra.loc[:,['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg('median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_intra[cm_intra < 0] = 0
cm_intra[cm_intra.isna()] =0

Z = linkage(cm_intra, method='complete')  # Using Ward's method
dg = dendrogram(Z, no_plot=True)
new_idx = cm_intra.index[dg['leaves']]

Z = linkage(cm_intra.T, method='complete')  # Using Ward's method
dg = dendrogram(Z, no_plot=True)
new_cols = cm_intra.columns[dg['leaves']]
cm_intra = cm_intra.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_intra, cmap='Reds',
               col_cluster=False, row_cluster=False, robust=True,
               square=True, vmin=.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation':'horizontal'},
               figsize=(6, 6), xticklabels=1, yticklabels=1,linewidths=.1, vmax=3)
main_ax = cm.ax_heatmap
cb_ax = cm.ax_cbar

main_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)

cb_ax.set_xticks([0.25, 1.5, 3], ['>0.25', '1.5', '3'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(in_path, '241028_ClusterMap_Importance_IntraView_CellAbundance_R2Filt.svg'), bbox_inches='tight')

# Clustermap on JuxtaView
ndf_juxta.loc[ndf_juxta.Importance < 0, 'Importance'] = 0
ndf_juxta.loc[ndf_juxta.Importance.isna(), 'Importance']= 0
ndf_juxta['Predictor'] = ndf_juxta['Predictor'].str.replace('Ccr2_MP','Ccr2+MP')
ndf_juxta['Target'] = ndf_juxta['Target'].str.replace('Ccr2_MP','Ccr2+MP')

cm_juxta = ndf_juxta.loc[:,['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg('median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_juxta[cm_juxta < 0] = 0
cm_juxta[cm_juxta.isna()] =0
cm_juxta = cm_juxta.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_juxta, cmap='Reds',
               col_cluster=False, row_cluster=False, robust=True,
               square=True, vmin=0.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation':'horizontal'},
               figsize=(6, 6), xticklabels=1, yticklabels=1,linewidths=.1, vmax=1.5)
main_ax = cm.ax_heatmap
cb_ax = cm.ax_cbar

main_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
cb_ax.set_xticks([0.25,  0.75, 1.5], ['>0.25',  '0.75', '1.5'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(in_path, '241028_ClusterMap_Importance_JuxtaView_CellAbundance_R2Filt.svg'), bbox_inches='tight')

# Clustermap on ParaView
ndf_para.loc[ndf_para.Importance < 0, 'Importance'] = 0
ndf_para.loc[ndf_para.Importance.isna(), 'Importance']= 0
ndf_para['Predictor'] = ndf_para['Predictor'].str.replace('Ccr2_MP','Ccr2+MP')
ndf_para['Target'] = ndf_para['Target'].str.replace('Ccr2_MP','Ccr2+MP')

cm_para = ndf_para.loc[:,['Predictor', 'Target', 'Importance']].groupby(['Predictor', 'Target']).agg('median').reset_index().pivot(index='Predictor', columns='Target', values='Importance')
cm_para[cm_para < 0] = 0
cm_para[cm_para.isna()] =0
cm_para = cm_para.reindex(index=new_idx, columns=new_cols).T

cm = sns.clustermap(cm_para, cmap='Reds',
               col_cluster=False, row_cluster=False, robust=True,
               square=True, vmin=0.25, cbar_pos=[0.2, .9, .15, .03], cbar_kws={'orientation':'horizontal'},
               figsize=(6, 6), xticklabels=1, yticklabels=1,linewidths=.1, vmax=1.5 )
main_ax = cm.ax_heatmap
cb_ax = cm.ax_cbar

main_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
cb_ax.set_xticks([0.25,.75,  1.5], ['>0.25', '0.75' ,'1.5'], fontsize=10, fontweight='bold')
cb_ax.set_title('Median Importance', fontweight='bold', fontsize=10)
plt.savefig(os.path.join(in_path, '241028_ClusterMap_Importance_ParaView_CellAbundance_R2Filt.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Spatialplot in co-localisation">
old5 = davidrUtility.select_slide(aging, 'Old_5')
old4 = davidrUtility.select_slide(aging, 'Old_4')

young3 = davidrUtility.select_slide(aging, 'Young_3')
young2 = davidrUtility.select_slide(aging, 'Young_2')
young1 = davidrUtility.select_slide(aging, 'Young_1')

fig, axs = plt.subplots(2,2, figsize=(12, 8))
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
plt.savefig(os.path.join(in_path, 'Spatial_Old4Young3_SMCFibroActiv.svg'), bbox_inches='tight')


fig, axs = plt.subplots(2,2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young1, color=['MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[0], title='Young_2')
sc.pl.spatial(young1, color=['Fibroblasts'], size=1.5, bw=True, vmax='p99.2', ax=axs[2], title='')
sc.pl.spatial(old5, color=['MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[1], title='Old_5')
sc.pl.spatial(old5, color=['Fibroblasts'], size=1.5, bw=True, vmax='p99.2', ax=axs[3], title='')

for ax in axs:
    davidrUtility.axis_format(ax, 'SP')

fig.text(0.03, 0.75, 'MP', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'Fibroblasts', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
plt.savefig(os.path.join(in_path, 'Spatial_Old5Young1_MPFibroblasts.svg'), bbox_inches='tight')

fig, axs = plt.subplots(2,2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.05, wspace=.05, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young2, color=['Ccr2+MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[0], title='Young_2')
sc.pl.spatial(young2, color=['B_cells'], size=1.5, bw=True, vmax='p99.2', ax=axs[2], title='')
sc.pl.spatial(old5, color=['Ccr2+MP'], size=1.5, bw=True, vmax='p99.2', ax=axs[1], title='Old_5')
sc.pl.spatial(old5, color=['B_cells'], size=1.5, bw=True, vmax='p99.2', ax=axs[3], title='')

for ax in axs:
    davidrUtility.axis_format(ax, 'SP')

fig.text(0.03, 0.75, 'Ccr2+MP', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'B_cells', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
plt.savefig(os.path.join(in_path, 'Spatial_Old5Young2_Ccr2MPBcells.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Lineplots">
cellInteraction = pd.DataFrame()
for batch in tqdm(aging.obs['sample'].unique()):
    sdata = davidrUtility.select_slide(aging, batch)
    for idx, bc in enumerate(sdata.obs_names):
        # Get proportion in intra space
        intra_c2l = sdata.obsm['c2l_prop'].loc[bc, :]
        intra_c2l.index = intra_c2l.index + '_IntraSpace'

        # Get Proportions in Hexa Space
        neigh_BCs = list(davidrUtility.get_surrounding(sdata, idx, radius=100, get_bcs=True))
        neigh_BCs.remove(bc)
        norm_hexa = sdata.obsm['c2l'].loc[neigh_BCs,:].sum().sum()
        hexa_c2l = sdata.obsm['c2l'].loc[neigh_BCs,:].sum(axis=0) / norm_hexa
        hexa_c2l.index = hexa_c2l.index + '_HexaSpace'

        # Get Proportion in Extended Space
        extend_BCs = list(davidrUtility.get_surrounding(sdata, idx, radius=200, get_bcs=True))
        extend_BCs.remove(bc)
        extend_BCs = [val for val in extend_BCs if val not in neigh_BCs]
        norm_extend = sdata.obsm['c2l'].loc[extend_BCs,:].sum().sum()
        extend_c2l = sdata.obsm['c2l'].loc[extend_BCs,:].sum(axis=0) / norm_extend
        extend_c2l.index = extend_c2l.index + '_ExtendedSpace'

        new_row = pd.DataFrame(pd.concat([intra_c2l, hexa_c2l, extend_c2l]), columns=[bc]).T
        new_row['Condition'] = batch.split('_')[0]

        cellInteraction = pd.concat([cellInteraction, new_row])


ct_colors = dict(zip(ref.obs.annotation.cat.categories, ref.uns['annotation_colors']))
data_wide = cellInteraction.reset_index().melt(id_vars=['Condition', 'index'], var_name='Group', value_name='Proportion')

data_wide['space'] = data_wide['Group'].str.split('_').str[-1]
data_wide['CellTypes'] = data_wide['Group'].str.split('_IntraSpace').str[0].str.split('_HexaSpace').str[0].str.split('_ExtendedSpace').str[0]
tmp = data_wide.copy()
tmp['space'] = '-' + tmp['space']
tmp =tmp[tmp.space !='-Intra']
data_wide = pd.concat([data_wide, tmp]).reset_index()
data_wide['space'] = pd.Categorical(data_wide.space, categories=['-ExtendedSpace', '-HexaSpace', 'IntraSpace', 'HexaSpace', 'ExtendedSpace'], ordered=True)



# CM distribution compared to rest of cell types
cm_all = data_wide[data_wide['CellTypes'].isin(['Fibroblasts','CapEC', 'SMC', 'Ccr2+MP', 'CM'])].copy()
tmp = cm_all[cm_all['Group'] == 'CM_IntraSpace']
tmp = tmp.loc[tmp['Proportion'] > np.percentile(tmp['Proportion'], 92), 'index']
cm_all = cm_all[cm_all['index'].isin(tmp)].copy()


# Create the figure and axes
fig, ax1 = plt.subplots(1, 2, figsize=(6, 3))
fig.subplots_adjust(wspace=0.5)
# Plot the first line (on the first y-axis)
sns.lineplot(cm_all[cm_all.CellTypes.isin(['CM'])], x='space', y='Proportion', hue='CellTypes',  estimator='mean',
             markers=True, dashes=True, palette=ct_colors, ax=ax1[0])

sns.lineplot(cm_all[cm_all.CellTypes.isin(['Fibroblasts', 'CapEC', 'SMC', 'Ccr2+MP'])],
             x='space', y='Proportion', hue='CellTypes', estimator='mean',
             markers=True, dashes=True, palette=ct_colors, ax=ax1[1])

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
           bbox_to_anchor=(1.45, 0.5), loc='center right', ncol=1, frameon=False)

plt.savefig(os.path.join(in_path, 'LinePlot_CMAll_AlongSpace_CMPredictor.svg'), bbox_inches='tight')


# B cells and Ccr2+Mp
mp_b = data_wide[data_wide['CellTypes'].isin(['B_cells', 'Ccr2+MP'])]
tmp = mp_b[mp_b['Group'] == 'Ccr2+MP_IntraSpace']
tmp = tmp.loc[tmp['Proportion'] > np.percentile(tmp['Proportion'], 92), 'index']
mp_b = mp_b[mp_b['index'].isin(tmp)].copy()


lp = sns.lineplot(mp_b,
             x='space', y='Proportion', style='Condition', hue='CellTypes',  estimator='mean',
             markers=True, dashes=True, palette=ct_colors)
lp.set_xlabel('')
lp.set_ylabel('Mean Cell Proportion')
lp.set_xticklabels(['Extended', 'Hexa', 'SameSpot', 'Hexa', 'Extended'], rotation=75, ha='right', va='top')
sns.move_legend(lp, loc='upper center',
                ncols=2,
                frameon=False, bbox_to_anchor=(0.5, 1.35))
plt.savefig(os.path.join(in_path, 'LinePlot_BCellCcr2MP_AlongSpace_BcellsPredictor.svg'), bbox_inches='tight')
#</editor-fold>

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#<editor-fold desc="Senescence">
figure_path = os.path.join(main_path, 'Figures/4_FunctionalAnalysis/Senescence/')

# Load Data
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['logcounts'].copy()
ref = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()


# CellAge
cellage = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/CellAge3.tsv', sep='\t')
cellage['GeneMouse'] = cellage['Gene symbol'].str.capitalize()
cellage['dataset'] = 'CellAge'
cellage = cellage[cellage['Senescence Effect'] == 'Induces']  # Correct to remove antisenescence genes
# CellAge --> 370 genes


# AgingAtlas
tmp_files = [f for f in os.listdir('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/') if 'AgingAtlas' in f]
agingAtlas = pd.DataFrame()
for f in tmp_files:
    tmp = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/' + f)
    agingAtlas = pd.concat([agingAtlas, tmp])
agingAtlas['GeneMouse'] = agingAtlas['Symbol'].copy()
agingAtlas['dataset'] = 'AgingAtlas'  # AgingAtlas --> 391 genes

# SenMayo
senmayo = pd.read_excel('/mnt/davidr/scStorage/DavidR/BioData/SenMayo_Genes.xlsx', sheet_name='mouse')
senmayo['GeneMouse'] = senmayo['Gene(murine)']
senmayo['dataset'] = 'SenMayo'  # SenMayo --> 118 genes

# Cellular Senescence
msig = gp.Msigdb('2023.1.Mm')
gmt = msig.get_gmt(category='msigdb', dbver="2023.1.Mm")
cellularSenescence = pd.DataFrame(gmt['GOBP_CELLULAR_SENESCENCE'], columns=['GeneMouse'])
cellularSenescence['dataset'] = 'GOSenescence'  # CellularSenescence --> 141 genes

# Fridman & Tainsky
fridTain = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/Fridman&Tainsky.csv')
fridTain['GeneMouse'] = fridTain['SYMBOL'].str.capitalize()
fridTain['dataset'] = 'Fridman&Tainsky' # Friedman & Tainsky --> 101 genes

# Combine every dataset
cols = ['GeneMouse', 'dataset']
score = pd.concat([cellage[cols], agingAtlas[cols], senmayo[cols], cellularSenescence[cols], fridTain[cols]])  # 1246
score = score.groupby('GeneMouse')['dataset'].apply(lambda x: '; '.join(x)).reset_index()  # 929

# Exclude genes not present in our object
score = score[score.GeneMouse.isin(aging.var_names)]  # 824

# DGE Young Vs Old
sc.tl.rank_genes_groups(aging, groupby='condition', method='wilcoxon', tie_correct=True, logcounts=True)
dge = sc.get.rank_genes_groups_df(aging, group='Old', pval_cutoff=0.05)  # 10775
dge_set = set(dge.names) # 10775
score_set = set(score.GeneMouse) # 824

ssg = dge_set & score_set  # 593

df_ssg = pd.DataFrame(list(ssg), columns=['GeneMouse'])
df_ssg.to_excel(os.path.join(table_path, 'DataFrame_GenesUsedFor_SenescenceScore.xlsx'), index=False)

sc.tl.score_genes(aging, gene_list=ssg, score_name='senescence')

def get_surrounding(adata: ad.AnnData,
                    in_spot: int = None,
                    bc_spot: str = None,
                    radius: float = 100,
                    get_bcs=True) -> list:
    """Find the index positions that surrounds a position (index)
    :param adata: anndata object
    :param in_spot: index of the barcode
    :param bc_spot: barcode to check
    :param get_bcs: return barcodes instead of index position
    :param radius: radius. Minimum of 100 for Visium
    :return: a list of surrounding indices
    """
    if in_spot is None and bc_spot is None:
        assert 'Specify only in_spot or bc_spot'

    if in_spot is not None:
        spot = adata.obsm['spatial'][in_spot]
    if bc_spot is not None:
        spot = adata[adata.obs_names == bc_spot,:].obsm['spatial'][0]


    surrounding = []
    for i, sp in enumerate(adata.obsm['spatial']):
        distance = ((spot[0] - sp[0]) ** 2 + (spot[1] - sp[1]) ** 2) ** .5
        if distance <= radius:
            surrounding.append(i)
    if get_bcs:
        return list(adata.obs_names[surrounding])
    else:
        return surrounding

top5 =  aging.obs.senescence.copy()
top5 =  list(top5[top5 > np.percentile(top5, 95)].index)

data = pd.DataFrame()
for batch in tqdm(aging.obs['sample'].unique()):
    sdata = aging[aging.obs['sample'] == batch]
    annotation = {bc:[10, 10,  10, 10, 10, 10, 10] for bc in sdata.obs_names}
    for bc in top5:
        if bc not in sdata.obs_names:
            continue
        # SSS > Hexamer > Extended Tissue > Greater Space > Others
        annotation[bc][0] = 0

        # Calculate BCs in Hexamer, Extended & Greater Space
        hexamer = get_surrounding(sdata, bc_spot=bc, radius=100)

        extended = get_surrounding(sdata, bc_spot=bc, radius=200)
        extended = [val for val in extended if val not in hexamer]

        greater = get_surrounding(sdata, bc_spot=bc, radius=300)
        greater = [val for val in greater if val not in hexamer and val not in extended]

        major = get_surrounding(sdata, bc_spot=bc, radius=400)
        major = [val for val in major if val not in hexamer and val not in extended and val not in greater]

        external = get_surrounding(sdata, bc_spot=bc, radius=500)
        external = [val for val in external if val not in hexamer and val not in major and val not in greater and val not in major]

        hexamer.remove(bc)

        cont = 1
        for case, label in [(hexamer, 1), (extended, 2), (greater, 3), (major, 4), (external, 5)]:
            for bc_inner in case:
                annotation[bc_inner][cont] = label
            cont +=1

    # sss -> 0; hexamer -->1; extened -->2; greater --> 3; others --> 10
    annotation  = pd.DataFrame.from_dict(annotation).T
    annotation = pd.DataFrame(annotation.min(axis=1), columns=['code'])
    annotation['annotation'] = annotation.code.replace({0:'Hspot',
                                                        1:'dist100',
                                                        2:'dist200',
                                                        3:'dist300',
                                                        4:'dist400',
                                                        5:'dist500',
                                                        10:'rest'})

    data = pd.concat([data, annotation['annotation']])

data = data.reindex(index=aging.obs_names)
aging.obs['senescence_gradient'] = pd.Categorical(data.annotation)
#</editor-fold>

#<editor-fold desc="Visualisation Senescence">
davidrPlotting.plot_slides(aging, 'senescence', cmap='RdBu_r', ncols=5,
                           order=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                  'Old_1','Old_2','Old_3','Old_4','Old_5'],
                           title_fontweight='bold', fig_path=figure_path, filename='SpatialPlot_ST_Senescence.svg')
davidrPlotting.plot_slides(aging, 'senescence', cmap='RdBu_r', ncols=5,
                           select_samples=['Young_1', 'Old_5'],
                           order=['Young_1', 'Old_5'], title_fontweight='bold',
                           fig_path=figure_path, filename='SpatialPlot_ST_Senescence_Young1Old5.svg')

# # # # # # # # #
young1 = davidrUtility.select_slide(aging, 'Young_1')
old5 = davidrUtility.select_slide(aging, 'Old_5')
vmax = pd.concat([young1.obs.senescence, old5.obs.senescence])

# Young and Old representative for manuscript
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
fig.subplots_adjust(left=0.1)
ax1 = sc.pl.spatial(young1, color='senescence', size=1.5, cmap='RdBu_r', show=False, vmax=np.percentile(vmax, 99.2),
                    ax=axs[0], colorbar_loc=None, title='Young_1')[0]
ax2 = sc.pl.spatial(old5, color='senescence', size=1.5, cmap='RdBu_r', show=False,  vmax=np.percentile(vmax, 99.2),
                    ax=axs[1], colorbar_loc=None, title='Old_5')[0]
davidrUtility.axis_format(ax1, 'SP')
davidrUtility.axis_format(ax2, 'SP')
fig = plt.gcf()
cbar_ax = fig.add_axes([0.6, .15, 0.03, 0.13])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=-2.5, vmax=2.5)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Senescence\nScore', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels(['Min', 'Max'], fontweight='bold')
cbar.ax.grid(False)
plt.subplots_adjust(right=0.8)
plt.savefig(os.path.join(figure_path, 'Spatial_Senescence_Young1Old5_ForManuscript.svg'), bbox_inches='tight')

# Density Plots
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.subplots_adjust(hspace=.3, wspace=.2, left=.05)
axs = axs.flatten()
for idx, batch in enumerate(['Young_1', 'Old_5']):
    sdata = davidrUtility.select_slide(aging, batch)

    # Extract SS Spots
    sss = sdata.obs.senescence_gradient.replace({'Hspot': 12, 'dist100': 10, 'dist200': 8,
                                          'dist300': 6, 'dist400': 4, 'dist500': 2,
                                          'rest': 0})

    coords = pd.DataFrame(sdata.obsm['spatial']) * sdata.uns['spatial'][batch]['scalefactors']['tissue_hires_scalef']
    max_x, min_x = coords[0].max(), coords[0].min()
    max_y, min_y = coords[1].max(), coords[1].min()
    ax=sc.pl.spatial(sdata, color=None, bw=True, size=1.5, show=False, ax=axs[idx])[0]
    ax=sc.pl.spatial(sdata, color=None, bw=True, size=1.5, show=False, ax=ax, alpha_img=0)[0]

    sns.kdeplot(coords, x=0, y=1,
                weights=sss.values, fill=True, cmap='Reds', alpha=0.5, ax=ax,
                bw_adjust=.75, levels=8,
                #clip=((coords[0].min(), coords[0].max()),
                #      (coords[1].min(), coords[1].max())),
                cut=0)

    sc.pl.spatial(sdata, basis='spatial', color='senescence_gradient',
                  palette={'Hspot': 'firebrick','dist100': (1, 1, 1, 0), 'dist200': (1, 1, 1, 0),
                           'dist300': (1, 1, 1, 0), 'dist400': (1, 1, 1, 0), 'dist500': (1, 1, 1, 0),
                           'rest': (1, 1, 1, 0)},
                  size=1.5, show=False, ax=axs[idx],alpha_img=0, legend_loc=None)

    axs[idx].set_title(batch, fontsize=15)
    davidrUtility.axis_format(axs[idx], 'SP')
fig.supylabel('SSS Density', fontsize=23, fontweight='bold')
plt.savefig(os.path.join(figure_path, '241126_Spatial_SSSDensity_Old5Young1.svg'), bbox_inches='tight', dpi=250)

# Boxplot of senescence score across locations
df = aging.obs[['senescence_gradient', 'condition', 'clusters', 'sample', 'senescence']].copy()

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
bx = sns.boxplot(df, x='senescence_gradient', y='senescence', hue='condition', order=['Hspot', 'dist100', 'dist200','dist300', 'dist400', 'dist500', 'rest'],
                 hue_order=['Young', 'Old'], palette={'Young':'darkorange',
                                                      'Old':'royalblue'},
                 ax=axs)
bx.set_ylabel('Senescence Score')
bx.set_xlabel('')
bx.set_xticklabels(bx.get_xticklabels(), rotation=45, ha='right', va='top', fontweight='bold')
sns.move_legend(bx, loc='upper center', ncols=2, frameon=False, title='Condition', bbox_to_anchor=(.5, 1), title_fontproperties={'weight':'bold', 'size':14}, fontsize=12)
plt.savefig(os.path.join(figure_path, 'BoxPlot_SSSGradient_SensitiveScore.svg'), bbox_inches='tight')


# CellTypes Enriched in Senescence Spots
df = aging.obsm['c2l_prop'].copy()
df[['gradient']] = aging.obs[['senescence_gradient']]

cm = sns.clustermap(df.groupby('gradient').agg('median').T.reindex(columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400',
                                          'dist500', 'rest']),
                    z_score=0,
                    cmap='RdBu_r', center=0,
                    cbar_pos=(0.2, 0.9, 0.18, 0.05),
                    col_cluster=False,
                    yticklabels=1, xticklabels=1,
                    figsize=(5.2, 6.3),
                    cbar_kws={'orientation': 'horizontal'},
                    square=True, robust=True, linewidth=0.1,)
cm.ax_cbar.set_title('Scaled Median \nCell Prop', fontsize=8)  # Adjust font size to 8 or desired value
cm.ax_heatmap.set_xlabel('')
cm.ax_heatmap.grid(False)
cm.ax_cbar.grid(False)
cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), rotation=45, ha='right', va='top', fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Heatmap_MedianScaled_c2l_prop_SSSGradient.svg'), bbox_inches='tight')


# Lineplot showing enriched celltypes
df = aging.obsm['c2l'].copy()
df[['gradient', 'condition']] = aging.obs[['senescence_gradient', 'condition']]

df = df.groupby(['gradient', 'condition']).agg('sum')
df['total'] = df.groupby(['gradient', 'condition']).agg('sum').sum(axis=1).reindex(index=df.index)
df = df.div(df['total'], axis=0).iloc[:,:-1]

df_plot = df.reset_index()
df_plot = df_plot.melt(id_vars=['gradient', 'condition'])

tmp = df_plot.copy()
tmp['gradient'] = '-' + tmp['gradient'].astype(str)
tmp =tmp[tmp.gradient !='-Hspot']
df_plot = pd.concat([df_plot, tmp]).reset_index()
df_plot['gradient'] = pd.Categorical(df_plot.gradient, categories=['-rest', '-dist500', '-dist400', '-dist300', '-dist200', '-dist100',
                                                             'Hspot',
                                                            'dist100', 'dist200', 'dist300', 'dist400', 'dist500','rest'], ordered=True)

df_plot = df_plot[df_plot['variable'].isin(['SMC', 'Fibroblasts', 'Fibro_activ',
                                            'Ccr2+MP','ArtEC',
                                            'T_cells', 'Pericytes', 'B_cells',
                                            'VeinEC', 'LymphEC'])]

ct_colors = dict(zip(ref.obs.annotation.cat.categories, ref.uns['annotation_colors']))
# Create the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.75))
fig.subplots_adjust(wspace=0.05)
# Plot the first line (on the first y-axis)
lp = sns.lineplot(df_plot[df_plot.condition == 'Young'],
                  x='gradient', y='value', hue='variable', style='condition',
                  estimator='mean',
                  palette=ct_colors,
                  markersize=5,
                  markers=True, dashes=True, ax=axs[0])
lp.legend().set_visible(False)
lp.set_ylim(0, 0.11)
lp.set_ylabel('Cell Proportion')
lp.set_xlabel('Young Condition')
lp.set_xticklabels(['Rest', '', '400$\mu$m', '', '200$\mu$m', '',
                    'Hspot', '', '200$\mu$m', '', '400$\mu$m', '', 'Rest'],
                   fontsize=10,
                   rotation=45, ha='right', va='top')

lp.set_yticklabels(lp.get_yticklabels(), fontsize=10)
lp = sns.lineplot(df_plot[df_plot.condition == 'Old'],
                  x='gradient', y='value', hue='variable', style='condition',
                  estimator='mean',
                  palette=ct_colors,
                  markersize=5,
                  markers=True, dashes=True, ax=axs[1])
lp.set_ylabel('')
lp.set_ylim(0, 0.11)
lp.set_xticklabels(['Rest', '', '400$\mu$m', '', '200$\mu$m', '',
                    'Hspot', '', '200$\mu$m', '', '400$\mu$m', '', 'Rest'],
                   fontsize=10,
                   rotation=45, ha='right', va='top')
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

legend_order = ['CellTypes', 'B_cells', 'Fibroblasts', 'Fibro_activ',
                'LymphEC', 'Ccr2+MP', 'Pericytes', 'SMC', 'T_cells', 'VeinEC']
lp.legend().set_visible(False)

# Update the legend with the unique handles and labels
lp.legend([legend[val] for val in legend_order], legend_order,
           bbox_to_anchor=(1, .8), loc='upper left', ncol=1, frameon=False,
           title_fontproperties={'weight':'bold'}, fontsize=10)

plt.savefig(os.path.join(figure_path, 'LinePlot_CellProportion_SSGradient.svg'), bbox_inches='tight')

# Stacked BarPlot showing proportions
tmp = aging.obs.value_counts(['senescence_gradient', 'condition', 'clusters']).reset_index()
tmp = tmp[tmp.sss_gradient == 'sss'].iloc[:,1:]
total = tmp[['condition', 'clusters', 'count']].groupby(['condition', 'clusters']).agg('sum')

norm = []
for idx, row in tmp.iterrows():
    norm.append(row['count'] / total.loc[(row.condition, row.clusters)].values[0])
tmp['norm'] = norm

tmp_young = tmp[tmp['condition'] =='Young']
tmp_old = tmp[tmp['condition'] =='Old']

del tmp_young['condition'], tmp_old['condition']

tmp_young = tmp_young.pivot(index=['clusters'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
    index=['Niche 7', 'Niche 0', 'Niche 1', 'Niche 2', 'Niche 8', 'Niche 5',
           'Niche 6', 'Niche 10', 'Niche 3', 'Niche 9', 'Niche 4'])
tmp_old = tmp_old.pivot(index=['clusters'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
    index=['Niche 7', 'Niche 0',
           'Niche 1', 'Niche 2',
           'Niche 8', 'Niche 5',
           'Niche 6', 'Niche 10',
           'Niche 3', 'Niche 9',
           'Niche 4'])
fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = tmp_young.plot.bar(stacked=True, color={'Hspot': 'firebrick',
                                             'dist100': 'tomato',
                                             'dist200': 'lightsalmon',
                                             'dist300': 'royalblue',
                                             'dist400': 'cornflowerblue',
                                             'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'},
                        ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'StackedBar_Proportion_NichesSSSGradient_Young.svg'), bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = tmp_old.plot.bar(stacked=True, color={'Hspot': 'firebrick',
                                             'dist100': 'tomato',
                                             'dist200': 'lightsalmon',
                                             'dist300': 'royalblue',
                                             'dist400': 'cornflowerblue',
                                             'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'},
                        ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'StackedBar_Proportion_NichesSSSGradient_Old.svg'), bbox_inches='tight')


# Same as above but for vessel annotation
tmp = aging.obs.value_counts(['senescence_gradient', 'condition', 'vessels']).reset_index()
total = tmp[['condition', 'vessels', 'count']].groupby(['condition', 'vessels']).agg('sum')

norm = []
for idx, row in tmp.iterrows():
    norm.append(row['count'] / total.loc[(row.condition, row.vessels)].values[0])
tmp['norm'] = norm

tmp_young = tmp[tmp['condition'] == 'Young']
tmp_old = tmp[tmp['condition'] == 'Old']

del tmp_young['condition'], tmp_old['condition']

tmp_young = tmp_young.pivot(index=['vessels'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
    index=['Arteries', 'Veins', 'Lymphatics', 'nVasc'])
tmp_old = tmp_old.pivot(index=['vessels'], columns='senescence_gradient', values='norm').reindex(
    columns=['Hspot', 'dist100', 'dist200', 'dist300', 'dist400', 'dist500', 'rest'],
index=['Arteries', 'Veins', 'Lymphatics', 'nVasc'])

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = tmp_young.plot.bar(stacked=True, color={'Hspot': 'firebrick',
                                             'dist100': 'tomato',
                                             'dist200': 'lightsalmon',
                                             'dist300': 'royalblue',
                                             'dist400': 'cornflowerblue',
                                             'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'},
                        ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'StackedBar_Proportion_VesselsSSSGradient_Young.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
ax = tmp_old.plot.bar(stacked=True, color={'Hspot': 'firebrick',
                                             'dist100': 'tomato',
                                             'dist200': 'lightsalmon',
                                             'dist300': 'royalblue',
                                             'dist400': 'cornflowerblue',
                                             'dist500': 'lightsteelblue',
                                             'rest': 'sandybrown'},
                        ax=axs, width=.9)
sns.move_legend(ax, loc='center right',frameon=False, title='Senescence\nGradient', title_fontproperties={'weight':'bold'},
                bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'StackedBar_Proportion_VesselsSSSGradient_Old.svg'), bbox_inches='tight')



df = aging.obsm['c2l'].copy()
df[['gradient', 'condition']] = aging.obs[['senescence_gradient', 'condition']]

df = df.groupby(['gradient', 'condition']).agg('sum')
df['total'] = df.groupby(['gradient', 'condition']).agg('sum').sum(axis=1).reindex(index=df.index)
df = df.div(df['total'], axis=0).iloc[:,:-1]
# Stacked barplot
df_sss = df.head(2)  # Hspots
df_sss.index = ['Old', 'Young']
df_sss = df_sss.iloc[-2:, :].reindex(index=['Young', 'Old'],
                                     columns=['CM', 'SMC', 'Fibro_activ', 'CapEC',
                                              'Fibroblasts', 'ArtEC', 'EndoEC', 'T_cells',
                                              'MP', 'Ccr2+MP', 'Pericytes', 'Adip', 'LymphEC',
                                              'VeinEC', 'Epi_cells', 'B_cells'])

ct_colors = dict(zip(ref.obs.annotation.cat.categories, ref.uns['annotation_colors']))

fig, axs = plt.subplots(1, 1, figsize=(3, 5))
ax = df_sss.plot.bar(stacked=True, color=ct_colors, ax=axs)
sns.move_legend(ax, loc='center right', frameon=False, bbox_to_anchor=(1.3,.5), fontsize=8, title='CellType',
                title_fontproperties={'weight':'bold'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontweight='bold', ha='right')
ax.set_ylabel('Proportion')
plt.savefig(os.path.join(figure_path, 'StackedBar_ProportionCellTypes_SSS.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="HoloNet">
def holonet_CCC(adata: ad.AnnData,
                df_lr: pd.DataFrame,
                pickle_path: str,
                pickle_filename: str,
                min_dist: int = 25,
                max_dist: int = 100,
                organism: str = 'mouse',
                ) -> ad.AnnData:
    """
    Perform Spatial Communication using `HoloNet <https://holonet-doc.readthedocs.io/en/latest/index.html>`_
    and generate an anndata object of LR pairs

    :param adata: anndata object
    :param df_lr: dataframe with LR signaling pathways
    :param pickle_path: path to save pickle file of Interaction strength
    :param pickle_filename: name of the pickle file
    :param min_dist: minimum distance to consider communication
    :param max_dist: maximum distance to consider communication
    :param organism: organism (moouse or human)
    :return: anndata object
    """

    import HoloNet as hn
    import pickle
    assert 'predicted_cell_type' in adata.obsm.keys(), '"predicted_cell_type" missing in adata.obsm'
    assert 'max' in adata.obsm[
        'predicted_cell_type'], '"max" column in adata.obsm["predicted_cell_type"] missing (Max Props per Barcode)'

    interc_db, cofact_db, cplx_db = hn.pp.load_lr_df(human_or_mouse=organism)
    w_best = hn.tl.default_w_visium(adata, min_cell_distance=min_dist,
                                    cover_distance=max_dist)
    df_expr_dict = hn.tl.elements_expr_df_calculate(df_lr, cplx_db, cofact_db, adata)
    ce_tensor = hn.tl.compute_ce_tensor(df_lr, w_best, df_expr_dict, adata)
    filt_ce_tensor = hn.tl.filter_ce_tensor(ce_tensor, adata, df_lr, df_expr_dict, w_best)

    # filt_ce_tensor --> lr x spotS x spotR

    # Get Centralities lr x spots
    eigv = hn.tl.compute_ce_network_eigenvector_centrality(filt_ce_tensor, diff_thres=0.05, tol=1e-4)
    H_s = hn.tl.compute_ce_network_degree_centrality(filt_ce_tensor, consider_cell_role='sender')
    H_r = hn.tl.compute_ce_network_degree_centrality(filt_ce_tensor, consider_cell_role='receiver')
    H_sr = hn.tl.compute_ce_network_degree_centrality(filt_ce_tensor, consider_cell_role='sender_receiver')

    # Create anndata
    adata_CCC = ad.AnnData(scipy.sparse.csr_matrix(eigv.T))
    adata_CCC.obs_names = adata.obs_names
    adata_CCC.var_names = df_lr['LR_Pair']
    adata_CCC.obs = adata.obs
    adata_CCC.obsm['spatial'] = adata.obsm['spatial']
    adata_CCC.uns['spatial'] = adata.uns['spatial']
    adata_CCC.layers['centrality_sender'] = scipy.sparse.csr_matrix(H_s.T)
    adata_CCC.layers['centrality_receiver'] = scipy.sparse.csr_matrix(H_r.T)
    adata_CCC.layers['centrality_sender_receiver'] = scipy.sparse.csr_matrix(H_sr.T)
    adata_CCC.layers['centrality_eigenvector'] = scipy.sparse.csr_matrix(eigv.T)

    with open(os.path.join(pickle_path, pickle_filename), 'wb') as f:
        pickle.dump(filt_ce_tensor, f)
    return adata_CCC


DB_common = davidrSpatial.holonet_common_lr(aging)

dict_CCC = {}
for sample in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, sample, 'sample')

    sdata.obsm['predicted_cell_type'] = sdata.obsm['c2l_prop'].copy()
    sdata.obsm['predicted_cell_type']['max'] = sdata.obsm['predicted_cell_type'].max(axis=1)
    sdata.obs['max_cell_type'] = sdata.obsm['predicted_cell_type'].idxmax(axis=1)

    ndata = holonet_CCC(sdata, DB_common, '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Objects/Pickle/HoloNet',
                        f'{sample}_CCC_tensor.pkl')
    dict_CCC[sample] = ndata

merged_CCC = ad.concat(dict_CCC.values(), keys=dict_CCC.keys(), label='sample',
                       join='outer', uns_merge='unique')

merged_CCC.write(os.path.join(object_path, 'Scanpy/HoloNet/Visium_YoungOld_HoloNet_CCC_230624.h5ad'))
#</editor-fold>

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#<editor-fold desc="C3:C3ar1">
adata = sc.read_h5ad(
    '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Objects/Scanpy/HoloNet/Visium_YoungOld_HoloNet_CCC_230624.h5ad')
adata.X = adata.layers['centrality_eigenvector'].copy()
adata.obs = aging.obs.reindex(adata.obs_names)
ref = sc.read_h5ad(
    '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Objects/Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad')

# Spatial distribution of CE
davidrPlotting.plot_slides(adata, 'C3 :C3ar1',
                           select_samples=['Young_1', 'Old_5'],
                           ncols=1, title_fontweight='bold',
                           bw=True,
                           order=['Young_1', 'Old_5'], fig_path=figure_path,
                           filename='Spatial_Young1Old5_C3C3ar1.svg')

# Matrixplots CE, Expr and Ct prop
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
fig.subplots_adjust(wspace=0.2)
ax1 = sc.pl.matrixplot(adata, groupby='clusters',
                       var_names=['C3 :C3ar1'],  colorbar_title='Scale Mean CE\nActivity in group',
                       standard_scale='var', cmap='Reds', ax=axs[0], show=False, figsize=(2, 5),
                       title='CE\nActivity')
ax2 = sc.pl.matrixplot(aging, groupby='clusters', var_names=['C3', 'C3ar1'], standard_scale='var',
                       cmap='Reds', ax=axs[1], show=False, colorbar_title='Scale Mean\nExpression in group',
                       figsize=(3, 5), title='Ligand-Receptor\nExpression')
ax3 =  sc.pl.matrixplot(aging, groupby='clusters', var_names=[ 'MP', 'Fibroblasts', 'Ccr2+MP', 'Fibro_activ'], standard_scale='var',
                       cmap='Reds', ax=axs[2], show=False, colorbar_title='Scale Mean\nProp in group',
                       figsize=(6, 5), title='CellType\nProportion')
ax1['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax2['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax3['mainplot_ax'].spines[['top', 'right']].set_visible(True)

for idx, ax in enumerate([ax1['mainplot_ax'], ax2['mainplot_ax'], ax3['mainplot_ax']]):
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', rotation=45, ha='right', va='top',  fontsize=15)
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
cbar.ax.set_title('CE Actv', fontweight='bold', loc='left', fontsize=12)

# Add Colorbar for each pannel
fig = plt.gcf()
cbar_ax = fig.add_axes([0.60, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Expr', fontweight='bold', loc='left', fontsize=12)

# Add Colorbar for each pannel
fig = plt.gcf()
cbar_ax = fig.add_axes([0.92, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop', fontweight='bold', loc='left', fontsize=12)
plt.savefig(os.path.join(figure_path, 'MatrixPlot_C3C3ar1MPFB.svg'), bbox_inches='tight')

sc.tl.rank_genes_groups(adata, groupby='condition', method='wilcoxon', tie_correct=True)
table = sc.get.rank_genes_groups_df(adata, group='Old')
table = table[table.names == 'C3 :C3ar1']

df = davidrUtility.ExtractExpression(adata, 'C3 :C3ar1', ['condition'])
fig, axs = plt.subplots(1, 1, figsize=(3, 5))
bp = sns.barplot(df, x='condition', y='expr', palette={'Young': 'sandybrown', 'Old': 'royalblue'},
                 ax=axs, order=['Young', 'Old'], estimator='median', capsize=.1)
davidrExperimental.plot_stats_adata(bp, adata, 'condition', 'C3 :C3ar1',
                                    'Young', ['Old'], list(table.pvals_adj), text_offset=4.5e-7)
bp.set_ylim(0, 4.5e-5)
bp.set_xlabel('')
bp.set_ylabel('Median CE strength')
bp.set_title('C3:C3ar1')
bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Barplot_MedianCE_C3C3ar1.svg'), bbox_inches='tight')

# Analysis on snRNA
davidrPlotting.pl_umap(ref, ['C3', 'C3ar1'], layer='logcounts', size=5,
                       path=figure_path, filename='UMAP_C3_C3ar1_Expr_snRNA.svg', ncols=1,
                       figsize=(6, 12))

# DGE on the Vidal which is young and old
vidal = ref[ref.obs.Experiment =='JulianAging']
vidal_mp = vidal[vidal.obs.annotation =='MP']
vidal_fb = vidal[vidal.obs.annotation =='Fibroblasts']

vidal = ref[ref.obs.Experiment == 'JulianAging'].copy()
vidal_mp = vidal[vidal.obs.annotation == 'MP']
vidal_fb = vidal[vidal.obs.annotation == 'Fibroblasts']

davidrPlotting.barplot_nUMI(vidal_mp, 'C3ar1', 'age', 'logcounts',
                            palette={'Young':'sandybrown', 'Old':'royalblue'},
                            order=['Young', 'Old'], figsize=(3.2, 5), ctrl_cond='Young',
                            groups_cond=['Old'], path=figure_path,
                            filename='Barplot_C3ar1_VidalMP.svg')

davidrPlotting.barplot_nUMI(vidal_fb, 'C3', 'age', 'logcounts',
                            palette={'Young':'sandybrown', 'Old':'royalblue'},
                            order=['Young', 'Old'], figsize=(3.2, 5), ctrl_cond='Young',
                            groups_cond=['Old'], path=figure_path,
                            filename='Barplot_C3_VidalFB.svg')
#</editor-fold>

#<editor-fold desc="Immune inhibitory pathway">
genes = ['Cd47', 'Sirpa', 'Cd24a', 'Pilra', 'Clec4a1', 'Clec12a', # Do not eat me signals
         'Ccr2', 'Ccl2', 'Ccl4', 'Ccl5', 'Ccr5', 'Vcam1', 'Icam1']

df = davidrUtility.AverageExpression(aging, group_by=['senescence_gradient'], feature=genes, out_format='wide')
df = df.reindex(index=genes, columns=['Hspot', 'dist100', 'dist200', 'dist300',
                                      'dist400', 'dist500','rest'])
ndf = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1), axis=0)

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
hm1 = sns.heatmap(ndf, cmap='Reds', square=True,
                  linewidths=.1, ax=axs, cbar=False, yticklabels=1, xticklabels=1)
# Set Axis Label
hm1.set_xlabel('')
hm1.set_ylabel('')
# Set Ticks
hm1.set_yticklabels(hm1.get_yticklabels(), fontsize=15)
hm1.set_xticklabels([txt.get_text().split('_')[0] for txt in hm1.get_xticklabels()], fontsize=15, fontweight='bold', rotation=45, ha='right', va='top')
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
plt.savefig(os.path.join(figure_path, 'Heatmap_EatMeNotEatMeGenes_SenescenceGradient.svg'), bbox_inches='tight')

# Check the snRNA Vidal
vidal = ref[ref.obs.Experiment == 'JulianAging']
vidal.obs['tmp'] = ref.obs.annotation.astype(str).copy()
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
plt.savefig(os.path.join(figure_path, 'Dotplot_MyeloidNonMyeloid_EatMeNotMean.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="M1/M2 MP phenotype genes">
# DGE Young Vs Old in MP
vidal_mp = vidal[vidal.obs.annotation.isin(['MP'])]
sc.tl.rank_genes_groups(vidal_mp, groupby='age', method='wilcoxon', tie_correct=True)
table = sc.get.rank_genes_groups_df(vidal_mp, group='Old', pval_cutoff=0.05, log2fc_min=.25)

# DGE Young Vs Old in Niche 7
niche7 = aging[aging.obs.clusters == 'Niche 7']
sc.tl.rank_genes_groups(niche7, groupby='condition', method='wilcoxon', tie_correct=True, layer='SCT_norm')
table7 = sc.get.rank_genes_groups_df(niche7, group='Old', pval_cutoff=0.05, log2fc_min=.25)

# Intersect of both
set(table7.names) & set(table.names)

# M1/M2 genes
genes = ['Fgr', 'Irf7', 'Bst2', 'Cd209g', 'Cd209f', 'Ccl8', 'Ccl6', 'Timp2', 'Vsig4']

ax = sc.pl.matrixplot(aging, groupby='clusters', var_names=genes, logcounts=True, standard_scale='var', cmap='Reds',
                      show=False)
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['color_legend_ax'].grid(False)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'MatrixPlot_M1M2MP_UpOldMP_UpOldNiche7.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Microglia associated genes expression">
genes = ['Grn', 'Il33', 'Trem2', 'Tyrobp']

ax = sc.pl.dotplot(ref, groupby='annotation', var_names=genes, layer='logcounts', logcounts=True, show=False)
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), fontweight='bold')
ax['color_legend_ax'].grid(False)
plt.savefig(os.path.join(figure_path, 'Dotplot_MicrogliaRelatedGenes_snRNA.svg'), bbox_inches='tight')
#</editor-fold>

#<editor-fold desc="Vessels and aging">
df = aging.obsm['c2l'].copy()
adata = ad.AnnData(X=df.values, obs=list(df.index), var=list(df.columns))
adata.obs_names = adata.obs[0]
adata.var_names = adata.var[0]
adata.obs = aging.obs.copy()

for col in adata.var_names:
    del adata.obs[col]

cmap_per =davidrUtility.generate_cmap('white', '#ff4d2e')

adata.obs['cluster+Cond'] = adata.obs['clusters'].astype(str) + '_' + adata.obs['condition'].astype(str)

axs = sc.pl.dotplot(adata, groupby='cluster+Cond',
                    swap_axes=True,
                    var_names=['MP', 'Ccr2+MP'],
                    expression_cutoff=0.01, standard_scale='var',
                    categories_order=['Niche 0_Young', 'Niche 0_Old',
                                      'Niche 1_Young', 'Niche 1_Old',
                                      'Niche 2_Young', 'Niche 2_Old',
                                      'Niche 3_Young', 'Niche 3_Old',
                                      'Niche 4_Young', 'Niche 4_Old',
                                      'Niche 5_Young', 'Niche 5_Old',
                                      'Niche 6_Young', 'Niche 6_Old',
                                      'Niche 7_Young', 'Niche 7_Old',
                                      'Niche 8_Young', 'Niche 8_Old',
                                      'Niche 9_Young', 'Niche 9_Old',
                                      'Niche 10_Young', 'Niche 10_Old',],
                    show=False,
                    cmap='Reds',
                    colorbar_title='Scaled Mean prop\n in group',
                    size_title='Fraction of spots\nin group(%)',
                    figsize=(7.8, 2.4),
                    logcounts=False,
                    )
axs['mainplot_ax'].spines[['top', 'right', 'left', 'bottom']].set_visible(True)
axs['color_legend_ax'].grid(False)
ax = axs['mainplot_ax']
# Replace xticks of the mainplot
axs['mainplot_ax'].set_xticklabels([txt.get_text().split('_')[-1] for txt in ax.get_xticklabels()], fontweight='bold')
axs['mainplot_ax'].set_yticklabels(ax.get_yticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'DotPlot_ScaledMeanProp_CellType_Aging_Clusters_wide.svg'), bbox_inches='tight')

for clust in adata.obs.clusters.unique():
    tdata = adata[adata.obs.clusters == clust]
    sc.tl.rank_genes_groups(tdata, groupby='condition', method='wilcoxon', tie_correct=True)
    tdf = sc.get.rank_genes_groups_df(tdata, group='Old', pval_cutoff=0.05)
    tdf = tdf[tdf.names.isin(['MP', 'Ccr2+MP'])]
    print (clust, tdf, '\n\n')

adata.obs['Vessel+Cond'] = adata.obs['vessels'].astype(str) + '_' + adata.obs['condition'].astype(str)
sdata = adata[adata.obs.vessels.isin(['Arteries', 'Veins', 'Lymphatics', 'nonVasc'])]
sdata.obs['Vessel+Cond'] = sdata.obs['vessels'].astype(str) + '_' + sdata.obs['condition'].astype(str)

# Generate the Dotplot
axs = sc.pl.dotplot(sdata, groupby='Vessel+Cond',
                    swap_axes=True,
                    var_names=['MP', 'Ccr2+MP', 'B_cells', 'T_cells', 'Fibroblasts', 'Fibro_activ', 'Adip'],
                    expression_cutoff=0.01, standard_scale='var',
                    categories_order=['Arteries_Young', 'Arteries_Old',
                                      'Veins_Young', 'Veins_Old',
                                      'Lymphatics_Young', 'Lymphatics_Old',
                                      'nonVasc_Young', 'nonVasc_Old'],
                    show=False,
                    cmap=cmap_per,
                    colorbar_title='Scaled Mean prop\n in group',
                    figsize=(5.8, 3.2),
                    )
axs['mainplot_ax'].spines[['top', 'right', 'left', 'bottom']].set_visible(True)
axs['color_legend_ax'].grid(False)
# Get Current Figure
fig = plt.gcf()
ax = axs['mainplot_ax']  # Get Main axs
# Add a subplot of top to add the brackers
pos = ax.get_position()
top_ax = fig.add_axes([pos.x0, pos.y1, pos.width, 0.1])  # Adjust height as needed

# Generate the brackets
def create_bracket(x_start, x_end, y_bottom=0, y_top=1, stem_length=0.2):
    from matplotlib.path import Path
    verts = [
        (x_start, y_bottom),            # Start of the bracket (bottom-left)
        (x_start, y_top),               # Vertical stem up
        (x_end - stem_length, y_top),   # Horizontal part
        (x_end - stem_length, y_bottom) # Down to bottom-right
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    return Path(verts, codes)

labels =['Arteries', 'Veins', 'Lymphatics', 'nonVasc']
bracket_positions = [(0, 2), (2, 4), (4, 6), (6, 8)]  # Tuples of (start, end) indices of clusters to group

for (i, (x_start, x_end)) in enumerate(bracket_positions):
    path = create_bracket(x_start, x_end)
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
ax.set_xticklabels([txt.get_text().split('_')[-1] for txt in ax.get_xticklabels()], fontweight='bold', rotation=75, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
#plt.savefig(os.path.join(figure_path, 'DotPlot_ScaledMeanProp_CellType_Aging_Vessels.svg'), bbox_inches='tight')
plt.savefig(os.path.join(figure_path, 'DotPlot_ScaledMeanProp_CellType_Aging_Vessels_WithoutMix.svg'), bbox_inches='tight')

#</editor-fold>

#<editor-fold desc="Figures QC/Visualisation for Manuscript not generated before">
figure_path = os.path.join(main_path, 'Figures/', '0_Manuscript/')


# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()
ref = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Cell2location/snRNA_RefAging_Manuscript.h5ad'))
ref.X = ref.layers['logcounts'].copy()
ct_zip = dict(zip(list(ref.obs.annotation.cat.categories), ref.uns['annotation_colors']))

# SpatialPlot of the niches in ST
davidrPlotting.plot_slides(aging, 'clusters', order=['Young_1', 'Young_2', 'Young_4', 'Young_5',
                                                     'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           select_samples=['Young_1', 'Young_2', 'Young_4', 'Young_5',
                                                     'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           common_legend=True, minimal_title=True, bw=True,
                           title_fontweight='bold',
                           fig_path=figure_path,
                           filename=f'SpatialPlot_ST_Niches_ExcludingOld5Young3.svg')

# Dotplot showing the expression of marker genes">
ec_markers = ['Tagln', 'Myh11',  # SMC
              'Stmn2', 'Fbln5',  # ArtEC
              'Vwf', 'Vcam1',  # VeinEC
              'Lyve1', 'Mmrn1']  #LymphEC
cmap_per =davidrUtility.generate_cmap('white', '#ff4d2e')
aging.obs['vessels'] = aging.obs['vessels'].str.replace('nVasc', 'nonVasc')

ax = sc.pl.dotplot(aging, groupby='vessels', var_names=ec_markers,
                   dot_max=.75, cmap=cmap_per,
                   categories_order=['Arteries', 'Art_Lymph', 'MixVasc',
                                     'Veins', 'Vein_Lymph', 'Lymphatics',
                                     'nonVasc'],
                   standard_scale='var', swap_axes=True,
                   size_title='Fraction of spots\nin group (%)',
                   show=False, colorbar_title='Scaled expression\nin group',
                   figsize=(4, 3))
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['color_legend_ax'].set_title('Scaled expression\nin group', fontsize=10)
ax['color_legend_ax'].grid(False)

ax['size_legend_ax'].set_title('\n\n\n'+ax['size_legend_ax'].get_title(), fontsize=10)
main_ax = ax['mainplot_ax']
main_ax.set_xticklabels(main_ax.get_xticklabels(), fontweight='bold')
main_ax.set_yticklabels(main_ax.get_yticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Dotplot_ST_Vessels_MarkersECs.svg'), bbox_inches='tight')

# SpatialPlot showing distribution of Cts in Old_5 and Young_3">
old5 = davidrUtility.select_slide(aging, 'Old_5')
young3 = davidrUtility.select_slide(aging, 'Young_3')

fig, axs = plt.subplots(2, 5, figsize=(15, 10))
plt.subplots_adjust(hspace=0, wspace=.08, left=.05)  # Spacing between subplots
for idx, color in enumerate(['SMC', 'EndoEC','Epi_cells', 'Fibroblasts', 'MP']):
    sc.pl.spatial(young3, color=color, bw=True, size=1.5, cmap='Reds',
                  vmax='p99.2', ncols=4,
                  ax=axs[0, idx], colorbar_loc=None, show=False)
    sc.pl.spatial(old5, color=color, bw=True, size=1.5, cmap='Reds',
                  vmax='p99.2', ncols=4,
                  ax=axs[1, idx], colorbar_loc=None, show=False)

    davidrUtility.axis_format(axs[0, idx], 'SP')
    davidrUtility.axis_format(axs[1, idx], 'SP')

    axs[0, idx].set_title(color, fontsize=18, fontweight='bold')
    axs[1, idx].set_title(None, fontsize=18, fontweight='bold')

fig.text(0.03, 0.75, 'Young_3', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.34, 'Old_5', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.05, .12, 0.1, 0.015])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('Cell Proportions', fontweight='bold', loc='center', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min','Max'], fontweight='bold', fontsize=12)
plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_Young3Old5_CellDistribution.svg'), bbox_inches='tight')

# SpatialPlot showing Co-localisation of FB and MP in Old_5
fig, axs = plt.subplots(2,1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.05, wspace=0, left=.05)  # Spacing between subplots
sc.pl.spatial(old5, color=['Fibroblasts'], vmax='p99.2', size=1.5, ncols=1, ax=axs[0],colorbar_loc=None, bw=True)
sc.pl.spatial(old5, color=['MP'], vmax='p99.2', size=1.5, ncols=1, ax=axs[1], colorbar_loc=None, bw=True)
davidrUtility.axis_format(axs[0], 'SP')
davidrUtility.axis_format(axs[1], 'SP')
axs[0].set_title('Old_5')
axs[1].set_title(None)
fig.text(0.35, 0.75, 'Fibroblasts', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.35, 0.34, 'MP', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.65, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop.', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min','Max'], fontweight='bold', fontsize=12)
plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_Old5_Distribution_FBMP.svg'), bbox_inches='tight')

# Same for young1
young1 = davidrUtility.select_slide(aging, 'Young_1')

fig, axs = plt.subplots(2,1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.05, wspace=0, left=.05)  # Spacing between subplots
sc.pl.spatial(young1, color=['Fibroblasts'], vmax='p99.2', size=1.5, ncols=1, ax=axs[0],colorbar_loc=None, bw=True)
sc.pl.spatial(young1, color=['MP'], vmax='p99.2', size=1.5, ncols=1, ax=axs[1], colorbar_loc=None, bw=True)
davidrUtility.axis_format(axs[0], 'SP')
davidrUtility.axis_format(axs[1], 'SP')
axs[0].set_title('Old_1')
axs[1].set_title(None)
fig.text(0.35, 0.75, 'Fibroblasts', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.35, 0.34, 'MP', va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.65, .15, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop.', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min','Max'], fontweight='bold', fontsize=12)

plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_Young1_Distribution_FBMP.svg'), bbox_inches='tight')

# SpatialPlot showing Co-localisation of Ccr2+MP and Bcells in Old_5
fig, axs = plt.subplots(1,2, figsize=(15, 10))
plt.subplots_adjust(hspace=0, wspace=0.05, left=.05)  # Spacing between subplots
sc.pl.spatial(old5, color=['Ccr2+MP'], vmax='p99.2', size=1.5, ncols=1, ax=axs[0],colorbar_loc=None,)
sc.pl.spatial(old5, color=['B_cells'], vmax='p99.2', size=1.5, ncols=1, ax=axs[1], colorbar_loc=None,)
davidrUtility.axis_format(axs[0], 'SP')
davidrUtility.axis_format(axs[1], 'SP')
axs[0].set_title('Ccr2+MP')
axs[1].set_title('B_cells')
fig.text(0.03, 0.55, 'Old_5', va='center', ha='center', rotation='vertical', fontsize=35, fontweight='bold')

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.98, .28, 0.015, 0.15])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=0.15)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title('Cell Prop.', fontweight='bold', loc='left', fontsize=12)
cbar.set_ticks([0, 0.15])
cbar.set_ticklabels(['Min','Max'], fontweight='bold', fontsize=12)

plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_Old5_Distribution_Ccr2+MPB.svg'), bbox_inches='tight')

# UMAPs for QC of the Integration of ST
np.random.seed(13)
random_indices = np.random.permutation(list(range(aging.shape[0])))  # Sort barcodes randomly

# UMAP showing the Age Conditions
ax = davidrPlotting.pl_umap(aging[random_indices, :], 'condition',
                            size=12, figsize=(5, 6), show=False, alpha=.9)
ax.set_title('')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(aging.obs.condition.cat.categories[::-1]), aging.uns['condition_colors'][::-1])],
          loc='center right', frameon=False, edgecolor='black', title='Condition',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'UMAP_ST_Condition.svg'), bbox_inches='tight', dpi=300)

# UMAP Showing the samples
ax = davidrPlotting.pl_umap(aging[random_indices, :], 'sample', size=15, figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('')

sp_batch_colors = [('Young_1', '#8c564b'), ('Young_2', '#e377c2'), ('Young_3', '#7f7f7f'),
                   ('Young_4', '#bcbd22'), ('Young_5', '#17becf'), ('Old_1', '#1f77b4'),
                   ('Old_2', '#ff7f0e'), ('Old_3', '#2ca02c'), ('Old_4', '#d62728'),
                   ('Old_5', '#9467bd'),]
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in sp_batch_colors],
          loc='center right', frameon=False, edgecolor='black', title='Sample',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13)
plt.savefig(os.path.join(figure_path, f'UMAP_ST_Batches.svg'), bbox_inches='tight', dpi=300)


# UMAP showing the clustering
ax = davidrPlotting.pl_umap(aging[random_indices, :], 'clusters', size=12,
                            figsize=(5, 6), show=False, alpha=.9, )
ax.set_title('')
ax.legend(handles=[mlines.Line2D([0], [0],
                                 marker=".", color=c,
                                 lw=0, label=lab,
                                 markerfacecolor=c, markeredgecolor=None,
                                 markersize=18) for lab, c in
                   zip(list(aging.obs.clusters.cat.categories), aging.uns['clusters_colors'])],
          loc='center right', frameon=False, edgecolor='black', title='Clusters',
          title_fontproperties={'size': 16, 'weight': 'bold'},
          bbox_to_anchor=(1.35, .5), fontsize=13, ncols=1)
plt.savefig(os.path.join(figure_path, f'UMAP_ST_Niches.svg'), bbox_inches='tight', dpi=300)

# Select two representatives slides to visualise the niches spatially
young3 = davidrUtility.select_slide(aging, 'Young_3')
old5 = davidrUtility.select_slide(aging, 'Old_5')

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
sc.pl.spatial(young3, color='clusters', legend_loc=None, bw=True, size=1.5,
              crop_coord=(1100, 6000, 1000, 5800), ax=axs[0])
sc.pl.spatial(old5, color='clusters', bw=True, size=1.5,
              crop_coord= (400, 5300, 600, 5400), ax=axs[1])
for batch, ax in [('Young_3', axs[0]), ('Old_5', axs[1])]:
    davidrUtility.axis_format(ax, 'SP')
    ax.set_title(batch, fontsize=20)
plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_Niches_Young3Old5.svg'), bbox_inches='tight', dpi=300)

# ViolinPlots & Spatial QC Metrics ST
# For ST --> Number of UMIs; Number of Genes; Also show the distribution of counts in slides
aging_copy = aging.copy()
sc.pp.calculate_qc_metrics(aging_copy, inplace=True)

data = aging_copy.obs[['log1p_total_counts', 'log1p_n_genes_by_counts']]
data['log(nUMIs)'] = data['log1p_total_counts']
data['log(nGenes)'] = data['log1p_n_genes_by_counts']
data['batch'] = pd.Categorical(aging.obs['sample'], categories=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                                                'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'], ordered=True)
data = data.sort_values('batch')
sp_batch_colors = {key:aging.uns['sample_colors'][idx] for idx, key in enumerate(aging.obs['sample'].cat.categories)}

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
plt.savefig(os.path.join(figure_path, f'ViolinPlot_ST_QC_Metrics.svg'), bbox_inches='tight')


# Spatial Plots showing the Metrics
coord_sample = {'Old_1': (700.0, 6000.0, 1200.0, 6100.0),
                'Old_2': (500, 5000.0, 1400, 5800.0),
                'Old_3': (200.0, 6100.0, 1000.0, 6200.0),
                'Old_4': (900.0, 6800.0, 850.0, 6050.0),
                'Old_5': (400, 5300.0, 800.0, 5100.0),
                'Young_1': (600.0, 6500.0, 600.0, 5800.0),
                'Young_2': (600.0, 6500.0, 800.0, 6000.0),
                'Young_3': (600.0, 6500.0, 800.0, 6000.0),
                'Young_4': (1100, 5400, 1300, 6000),
                'Young_5':  (1100, 4600, 1000, 4500),
                }

# QC Histology Young Samples
fig, axs = plt.subplots(2, 5, figsize=(15, 8))
plt.subplots_adjust(hspace=.05, wspace=.15, left=.05)  # Spacing between subplots
axs = axs.flatten()
cont = 0
for sample in ['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5']:
    slide = davidrUtility.select_slide(aging_copy, sample)
    sc.pl.spatial(slide, color=None, ax=axs[cont], colorbar_loc=None, crop_coord=coord_sample[sample])
    sc.pl.spatial(slide, color='log1p_total_counts', cmap='inferno', size=1.5, ax=axs[cont + 5],
                  vmax=np.max(aging_copy.obs['log1p_total_counts']),
                  vmin= np.min(aging_copy.obs['log1p_total_counts']),
                  colorbar_loc=None, crop_coord=coord_sample[sample])
    for ax in [axs[cont], axs[cont+5]]:
        davidrUtility.axis_format(ax, txt='SP')
    axs[cont].set_title(sample, fontsize=18, fontweight='bold')
    cont += 1

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.43, .075, 0.15, 0.05])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=aging_copy.obs.log1p_total_counts.min(), vmax=aging_copy.obs.log1p_total_counts.max())
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('log1p(nUMI)', fontweight='bold', loc='center')
cbar.set_ticks([aging_copy.obs.log1p_total_counts.min(),
                aging_copy.obs.log1p_total_counts.median(),
                aging_copy.obs.log1p_total_counts.max()])
cbar.set_ticklabels(['{:.2f}'.format(aging_copy.obs.log1p_total_counts.min()),
                     '{:.2f}'.format(aging_copy.obs.log1p_total_counts.median()),
                     '{:.2f}'.format(aging_copy.obs.log1p_total_counts.max())], fontweight='bold')
cbar_ax.grid(False)

plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_YoungSamples_QC_Metrics.svg'),
            bbox_inches='tight')


# QC Histology Old Samples
fig, axs = plt.subplots(2, 5, figsize=(15, 8))
plt.subplots_adjust(hspace=.05, wspace=.15, left=.05)  # Spacing between subplots
axs = axs.flatten()
cont = 0
for sample in ['Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5']:
    slide = davidrUtility.select_slide(aging_copy, sample)
    sc.pl.spatial(slide, color=None, ax=axs[cont], colorbar_loc=None, crop_coord=coord_sample[sample])
    sc.pl.spatial(slide, color='log1p_total_counts', cmap='inferno', size=1.5, ax=axs[cont + 5],
                  vmax=np.max(aging_copy.obs['log1p_total_counts']),
                  vmin= np.min(aging_copy.obs['log1p_total_counts']),
                  colorbar_loc=None, crop_coord=coord_sample[sample])
    for ax in [axs[cont], axs[cont+5]]:
        davidrUtility.axis_format(ax, txt='SP')
    axs[cont].set_title(sample, fontsize=18, fontweight='bold')
    cont += 1

# Adjust color bar manually
fig = plt.gcf()
cbar_ax = fig.add_axes([0.43, .075, 0.15, 0.05])  # Position [left, bottom, width, height]
norm = plt.Normalize(vmin=aging_copy.obs.log1p_total_counts.min(), vmax=aging_copy.obs.log1p_total_counts.max())
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('log1p(nUMI)', fontweight='bold', loc='center')
cbar.set_ticks([aging_copy.obs.log1p_total_counts.min(),
                aging_copy.obs.log1p_total_counts.median(),
                aging_copy.obs.log1p_total_counts.max()])
cbar.set_ticklabels(['{:.2f}'.format(aging_copy.obs.log1p_total_counts.min()),
                     '{:.2f}'.format(aging_copy.obs.log1p_total_counts.median()),
                     '{:.2f}'.format(aging_copy.obs.log1p_total_counts.max())], fontweight='bold')
cbar_ax.grid(False)

plt.savefig(os.path.join(figure_path, f'SpatialPlot_ST_OldSamples_QC_Metrics.svg'), bbox_inches='tight')

# SpatialPlot showing the manual annotation of the regions
labels_color = {'LVi':'lightsteelblue', 'LVm':'cornflowerblue', 'LVo':'royalblue',
                'RV':'#f6b278', 'SEP':'salmon', 'BG':'whitesmoke'}

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
plt.savefig(os.path.join(figure_path, 'SpatialPlot_ST_AnatomicRegions.svg'), bbox_inches='tight')


# Proportion of each niche per slide and in aging
# Df --> Clusters per sample
df = aging.obs.value_counts(['clusters','sample']).sort_index().reset_index()
df['prop'] = df['count'] / df.groupby('sample')['count'].transform('sum')
df_pivot = df.pivot(index='sample', columns='clusters', values='prop')

# Df --> Clusters per condition
df_v2 = aging.obs.value_counts(['clusters','condition']).sort_index().reset_index()
df_v2['prop'] = df_v2['count'] / df_v2.groupby('condition')['count'].transform('sum')
df_pivot_v2 = df_v2.pivot(index='condition', columns='clusters', values='prop')

# Combine dfs
df_pivot = pd.concat([df_pivot, df_pivot_v2], axis=0)
df_pivot = df_pivot.reindex(index=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                   'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5',
                                   'Young', 'Old'])

cmap = {key:ref.uns['annotation_colors'][idx]
        for idx, key in enumerate(ref.obs['annotation'].cat.categories)}

fig = plt.figure( figsize=(6, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, .5], height_ratios=[1], wspace=.1)
ax1 = fig.add_subplot(gs[0])
ax1 = df_pivot.iloc[:-2].plot(kind='bar', stacked=True, grid=False, rot =75, ax=ax1,
                              width=0.9)
sns.despine()
ax1.set_ylabel('Proportions', fontsize=18, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15, fontweight='bold')
ax1.legend().set_visible(False)

ax2 = fig.add_subplot(gs[1])
ax2 = df_pivot.iloc[-2:].plot(kind='bar', stacked=True, grid=False, rot =75, ax=ax2, figsize=(8, 6),
                              width=0.9)
ax2.spines[['left', 'top', 'right']].set_visible(False)
ax2.set_ylabel('')
ax2.set_yticklabels('')
ax2.tick_params(axis='y', which='major', length=0)  # Remove ticks
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=15, fontweight='bold')

sns.move_legend(ax2, loc='center right', frameon=False, title='Clusters', title_fontproperties={'weight':'bold', 'size':15},
                ncols=1, bbox_to_anchor=(1.6, .5))
plt.savefig(os.path.join(figure_path, 'Barplot_ST_Proportion_NichesSamples_Condition.svg'), bbox_inches='tight')


df_v2 = aging.obs.value_counts(['clusters','condition']).sort_index().reset_index()
df_v2['prop'] = df_v2['count'] / df_v2.groupby('condition')['count'].transform('sum')
df_pivot_v2 = df_v2.pivot(index='condition', columns='clusters', values='prop')
cmap = {key:ref.uns['annotation_colors'][idx] for idx, key in enumerate(ref.obs['annotation'].cat.categories)}
df_pivot_v2 = df_pivot_v2.sort_index(ascending=False)


sbp = df_pivot_v2.plot(kind='bar', stacked=True, grid=False, rot =75, width=0.9, figsize=(3, 6))
sbp.set_ylabel('Proportions', fontsize=18, fontweight='bold')
sbp.set_xlabel('', fontsize=18, fontweight='bold')
sbp.set_xticklabels(sbp.get_xticklabels(), fontsize=15, fontweight='bold')
sns.move_legend(sbp, loc='center right', frameon=False, title='Clusters', title_fontproperties={'weight':'bold', 'size':15},
                ncols=1, bbox_to_anchor=(1.4, .5))
plt.savefig(os.path.join(figure_path, 'Barplot_ST_Proportion_NichesCondition.svg'), bbox_inches='tight')

# Clustermap showing distribution of cell types in niches
df = aging.obsm['c2l'].copy()
df['clusters'] = aging.obs['clusters'].copy()
df = df.groupby('clusters').agg('sum')
df['total'] = df.sum(axis=1)
df_prop = df.iloc[:,:-1].div(df['total'], axis=0)
df_prop = df_prop.reindex(columns=df_prop.sum().sort_values(ascending=False).index)

df_prop = df_prop.reindex(columns = ['ArtEC', 'VeinEC', 'LymphEC', 'CapEC', 'EndoEC',
                           'MP', 'Ccr2+MP', 'B_cells', 'T_cells', 'Pericytes', 'SMC',
                           'Fibroblasts', 'Fibro_activ', 'CM', 'Adip', 'Epi_cells'])

cm = sns.clustermap(df_prop.T, cmap='RdBu_r', xticklabels=1, yticklabels=1, z_score='col',
               row_cluster=False, center=0, vmax=2.5, vmin=-2.5, cbar_pos=None, square=False)

# Get axis we want to modify
heatmap_ax = cm.ax_heatmap
colorbar_ax = cm.cax

# Set black border around the outer edge of the heatmap
heatmap_ax.axhline(y=0, color='k', linewidth=1.5)
heatmap_ax.axhline(y=df_prop.T.shape[0], color='k', linewidth=1.5)
heatmap_ax.axvline(x=0, color='k', linewidth=1.5)
heatmap_ax.axvline(x=df_prop.T.shape[1], color='k', linewidth=1.5)

# Replace xticks with circles colored by the annotation
xtick_pos, xtick_lab = heatmap_ax.get_xticks(), heatmap_ax.get_xticklabels()
ytick_pos, ytick_lab = heatmap_ax.get_yticks(), heatmap_ax.get_yticklabels()
heatmap_ax.tick_params(axis='y', which='both', length=0)  # Remove ticks
# Get colors for each ct
ct_catgs = list(ref.obs.annotation.cat.categories)
ct_colors = ref.uns['annotation_colors']
ct_zip = dict(zip(ct_catgs, ct_colors))
col_colors = [ct_zip[col] for col in df_prop.T.index]

for y, color in zip(ytick_pos, col_colors):
    heatmap_ax.plot(-.2, y, 'o', color=color, markersize=8, clip_on=False)

# Adjust Xticks and Yticks to correct positions
heatmap_ax.set_xticks(np.array([tk + 0 for tk in xtick_pos]))
heatmap_ax.set_xticklabels(xtick_lab, rotation=75, ha='right', va='top', fontweight='bold')
heatmap_ax.set_yticks(ytick_pos)
heatmap_ax.set_yticklabels(ytick_lab, rotation=0, ha='right')
heatmap_ax.tick_params(axis='y', pad=14, labelright=False, labelleft=True)

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
heatmap_ax.grid(False)
heatmap_ax.set_xlabel('')
heatmap_ax.set_ylabel('')
plt.savefig(os.path.join(figure_path, 'Heatmap_ST_Celltype_Niches_Overrepresentated.svg'), bbox_inches='tight')

# In representative samples look at the distribution of vessels
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0, wspace=0, left=.05)  # Spacing between subplots
axs = axs.flatten()
sc.pl.spatial(young3, color=None, ax=axs[0], size=1.5, )
sc.pl.spatial(young3, color='vessels', ax=axs[1], size=1.5, )
sc.pl.spatial(old5, color=None, ax=axs[2], size=1.5, )
sc.pl.spatial(old5, color='vessels', ax=axs[3], size=1.5, )

davidrUtility.axis_format(axs[2], 'SP')
axs[0].spines[['bottom']].set_visible(False)
axs[1].spines[['left', 'bottom']].set_visible(False)
axs[3].spines[['left']].set_visible(False)

for idx in [0, 1, 3]:
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('')
for idx in range(4):
    axs[idx].set_title(None)
fig.text(0.10, 0.75, 'Young_3', va='center', ha='center', rotation='vertical', fontsize=25, fontweight='bold')
fig.text(0.10, 0.34, 'Old_5', va='center', ha='center', rotation='vertical', fontsize=25, fontweight='bold')
plt.savefig(os.path.join(figure_path, 'SpatialPlot_ST_Vessels_Old5Young3.svg'), bbox_inches='tight')

davidrPlotting.plot_slides(aging, 'vessels', ncols=5, bw=True,
                           order=['Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5',
                                  'Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5'],
                           title_fontweight='bold',
                           fig_path=figure_path,
                           filename='SpatialPlot_ST_Vessels_AllSamples.svg')

# Barplot showing the proportion cell types in ST
df = aging.obsm['c2l'].copy()
df = df.melt().groupby('variable').agg('sum') / df.melt().groupby('variable').agg('sum').sum()
df = df.sort_values('value',ascending=False)

# Create the barplot
ct_colors = dict(zip(ref.obs.annotation.cat.categories, ref.uns['annotation_colors']))

plt.figure(figsize=(8, 5))
ax = sns.barplot(df, x='variable', y='value', palette=ct_colors)

# Add annotations
for i, value in enumerate(df['value']):
    ax.text(i, value + 0.025, f'{value:.3f}', ha='center', va='top', fontsize=10, fontweight='bold')

ax.set_xlabel('')
ax.set_ylabel('Proportion')
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right', va='top', fontweight='bold')
plt.savefig(os.path.join(figure_path, 'BarPlot_ST_Proportion_OverallCts.svg'), bbox_inches='tight')


# Check where the Niches are enriched in AnatomicRegions">
df = aging.obs['clusters'].to_frame().copy()
df['region'] = aging.obs['AnatomicRegion'].copy()
df = df.value_counts(['region', 'clusters']).reset_index()
df_pivot = df.pivot(index='region', columns='clusters', values='count')
df_pivot['total'] = df_pivot.sum(axis=1)
df_pivot = df_pivot.iloc[1:, :]  # Exclude BG

df_prop = df_pivot.iloc[:,:-1].div(df_pivot['total'], axis=0)
df_prop = df_prop.reindex(columns=['Niche 0', 'Niche 1', 'Niche 2', 'Niche 3',
                                   'Niche 4', 'Niche 5', 'Niche 6', 'Niche 7',
                                   'Niche 8', 'Niche 9', 'Niche 10'])

cmap = {key:aging.uns['clusters_colors'][idx] for idx, key in enumerate(aging.obs['clusters'].cat.categories)}
colors = [cmap[col] for col in df_prop.columns]

fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
ax1 = df_prop.plot(kind='bar', stacked=True, grid=False, rot =75, ax=ax1, color=colors,
                   width=.95)
sns.despine()
ax1.set_ylabel('Proportions', fontsize=18, fontweight='bold')
ax1.set_xlabel('')
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15, fontweight='bold')
sns.move_legend(ax1, loc='center right', frameon=False, title='Clusters', title_fontproperties={'weight':'bold', 'size':15},
                ncols=1, bbox_to_anchor=(1.3, .5))
plt.savefig(os.path.join(figure_path, 'Barplot_ST_Proportion_NichesAnatomicRegions.svg'), bbox_inches='tight')


# Markers associated to microglia activation expression across celltypes
genes = ['Grn', 'Trem2', 'Tyrobp']
ax = sc.pl.dotplot(ref, groupby='annotation', var_names=genes, show=False, figsize=(3.5, 5))
ax['mainplot_ax'].spines[['top', 'right']].set_visible(True)
ax['mainplot_ax'].set_xticklabels(ax['mainplot_ax'].get_xticklabels(), fontweight='bold')
ax['color_legend_ax'].grid(False)
plt.savefig(os.path.join(figure_path, 'Dotplot_MicrogliaAssociatedGenes_snRNARef.svg'), bbox_inches='tight')
#</editor-fold>


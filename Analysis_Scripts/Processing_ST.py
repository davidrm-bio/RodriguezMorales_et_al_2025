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

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import decoupler as dc
import gseapy as gp
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats
import scipy.sparse

import davidrUtility
import davidrSpatial
from davidrSpatial import spatial_integration
from davidrScRNA import SCTransform
import davidrPlotting

import matplotlib.pyplot as plt
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
    anndata.var_names_make_unique()  # Make sure var_names are unique

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
    sc.pp.calculate_qc_metrics(anndata, qc_vars=['mt'], inplace=True, log1p=True)

    # Normalise the data
    anndata.layers['counts'] = anndata.X.copy()
    sc.pp.normalize_total(anndata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(anndata)
    sc.pp.highly_variable_genes(anndata, flavor='seurat', n_top_genes=hvg_n)
    anndata.layers['logcounts'] = anndata.X.copy()

    return anndata
#</editor-fold>


########################################################################################################################
# - Quality Control and Integration of the Spatial Transcriptomics
########################################################################################################################


#<editor-fold desc="Pre-Process of ST per sample  - Select spots containing tissue">
# Create a dictionary of AnnDatas of each sample
slides = {name: sc.read_h5ad(os.path.join(input_path, name)) for name in os.listdir(input_path)}

# Remove Spots not capturing tissue (Manually selected using LoupeBrowser)
in_tissue = {name: pd.read_csv(os.path.join(table_path, 'LoupeBrowser', name, f'{name}_in_tissue.csv')) for name in
             slides}
for name, adata in slides.items():
    BCs = in_tissue[name].dropna()['Barcode'].tolist()
    slides[name] = adata[adata.obs_names.isin(BCs)]
#</editor-fold>


#<editor-fold desc="Quality Control per sample">
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
slide_concat = ad.concat(slides.values(), join='outer', label='sample', keys=slides.keys(), uns_merge='unique', index_unique='-')
slide_concat.var_names_make_unique()  # Make Sure we have unique gene names
slide_concat.X = slide_concat.layers['counts'].copy()  # Make Sure we have the raw counts in .X for cell2location
#</editor-fold>


#<editor-fold desc="Add Metadata and Save">
# Extract slide ID from .uns['spatial']
slides_id = {k: list(v.uns['spatial'].keys())[0] for k, v in slides.items()}
slides_id_inv = {v: k for k, v in slides_id.items()}
slide_concat.obs['library_id'] = [slides_id[lid] for lid in slide_concat.obs['sample']]

# Update .uns name for spatial slide; sample name instead library ID --> Make the access easier
tmp_spatial = slide_concat.uns['spatial']
new_spatial = {slides_id_inv[key]: val for key, val in tmp_spatial.items()}
slide_concat.uns['spatial'] = new_spatial
slide_concat.obs['condition'] = slide_concat.obs['sample'].str.split('_').str[0]
slide_concat.write(os.path.join(object_path, 'Visium_MouseHeart_OldYoungAMI.h5ad'))
#</editor-fold>


#<editor-fold desc="Integration of ST - Spatial aware manner">

# Load Objects  --> Generate an AnnData Object of all the ST data
visium_dimmeler = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_MouseHeart_OldYoungAMI.h5ad'))
visium_yamada = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_MouseHeart_MI_Yamada.h5ad'))

visium = ad.concat([visium_dimmeler, visium_yamada], join='outer',  # To keep all the genes
                   label='lab', keys=['Dimmeler', 'Yamada'])

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
visium.obsm['Hb'] = visium[:, visium.var['Hb'].values].X.toarray()  # We keep the data in adata.obsm
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


#<editor-fold desc="QC Integration Plots">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Integration/')

# Plot the Clustering
davidrPlotting.plot_slides(aging, 'leiden_STAligner', common_legend=True, minimal_title=True, ncols=5,
                           order=sorted(aging.obs['sample'].unique()), fig_path=figure_path,
                           filename=f'Spatial_Aging_leiden_STAligner.svg')
# Plot SplittingBy Condition
davidrPlotting.split_umap(aging, 'condition', path=figure_path, filename=f'UMAP_Aging_SplitbyCondition.svg',
                          ncol=2, figsize=(6, 4))
# Plot SplittingBy AnatomicRegion
davidrPlotting.split_umap(aging, 'AnatomicRegion', path=figure_path, filename=f'UMAP_Aging_SplitbyAnatomicRegion.svg',
                          ncol=3, figsize=(8, 5))
# Plot SplittingBy Sample
davidrPlotting.split_umap(aging, 'sample', path=figure_path, filename=f'UMAP_Aging_Splitbysample.svg',
                          ncol=5, figsize=(12, 8))
# Plot Clustering in UMAP
davidrPlotting.pl_umap(aging, 'leiden_STAligner', path=figure_path, filename=f'UMAP_Aging_leidenSTAligner.svg',
                       figsize=(6, 4))
# </editor-fold>


########################################################################################################################
# - Deconvolution Analysis, Map celltypes to spatial locations
########################################################################################################################

# Pre c2l --> See Run_c2l.py (Run on GPU Server)

#<editor-fold desc="Post c2l - Extract celltype abundances">
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

# Plot the total abundance per sample in ViolinPlot
fig, axs = plt.subplots(1, 1, figsize=(18, 9))
g = sns.violinplot(spot_abundance, y='Abundance', x='slide', ax=axs,
                   order=sorted(spot_abundance['slide'].unique()), palette='tab20')
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
update_cts = {'Adip': 'Adip', 'Artery_EC':'ArtEC', 'B':'B_cells', 'CM':'CM', 'Capillary_EC':'CapEC',
              'Monocytes':'Ccr2+MP', 'Endocardial_EC':'EndoEC', 'FB':'Fibroblasts', 'FBa':'Fibro_activ',
              'Lymphatic_EC':'LymphEC', 'MP':'MP', 'Meso':'Epi_cells', 'PC':'Pericytes', 'SMC':'SMC',
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
aging_copy.obsm['c2l_prop'] = c2l_props.copy()
aging_copy.obs[aging_copy.obsm['c2l_prop'].columns] = aging_copy.obsm['c2l_prop'].values
aging_copy.obs['nCells'] = abundance['nCells'].copy()
aging_copy.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
#</editor-fold>


########################################################################################################################
# - Analysis to define AnatomicRegions, Niches and LargeVessel
########################################################################################################################

#<editor-fold desc="Add Anatomic Regions">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Add_AnatomicRegions')
aging = sc.read_h5ad(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
files = [f for f in os.listdir(os.path.join(table_path, 'LoupeBrowser')) if 'Anatomic' in f and 'AMI' not in f]

# Create a dictionary: sample : df with BC as index and first column is Anatomic Region
anat_tmp_dict = {}
for file in files:
    samplename = file.split('_Anatomic')[0]
    df = pd.read_csv(os.path.join(table_path, 'LoupeBrowser', file))
    df['Barcode'] = df['Barcode'] + f'-{samplename}'  # Correction to match barcode of anndata
    df.set_index('Barcode', inplace=True)
    anat_tmp_dict[samplename] = df

meta = [anat_tmp_dict[sample].loc[bc][0] for bc, sample in aging.obs['sample'].items()]
aging.obs['AnatomicRegion'] = meta

# Plot the Anatomic Region in UMAP
davidrPlotting.pl_umap(aging, color='AnatomicRegion', figsize=(5,5), path=figure_path, filename=f'UMAP_ST_AnatomicRegion.svg')
# Show the spatial distribution of the Anatomic Regions
davidrPlotting.plot_slides(aging, 'AnatomicRegion', order=sorted(aging.obs['sample'].unique()), ncols=5,
                           common_legend=True, minimal_title=True, fig_path=figure_path, filename=f'SpatialPlot_ST_AnatomicRegion.svg')
# Save Object
aging.write(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

# Correction of the Anatomic Region from Sample Young_3
df = pd.read_csv(os.path.join(table_path, 'LoupeBrowser', 'Young_3_AnatomicRegion_Corrected_100624.csv'))
df['Barcode'] = df['Barcode'] + '-Young_3'
df.set_index('Barcode', inplace=True)

for bc in aging.obs_names:
    if bc in df.index:
        aging.obs.loc[bc, 'AnatomicRegion'] = df.loc[bc, 'Anatomic']

# Save Corrected Object
aging.write(os.path.join(object_path, 'Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
#</editor-fold>


#<editor-fold desc="Niches - Visualitation and DGE">
figure_path = os.path.join(main_path, 'Figures/', '3_Annotation/Niches/')

# Load Objects
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()

# Rename clusters to Niche + Number
aging.obs['clusters'] = 'Niche ' + aging.obs.leiden_STAligner.astype(str)
# Convert in Categorical and force to have the same order to keep color scheme
aging.obs['clusters'] = pd.Categorical(aging.obs.clusters).reorder_categories(['Niche ' + clust for clust in
                                                                               aging.obs.leiden_STAligner.cat.categories])
aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))


# Some Plots to See the Distribution
davidrPlotting.plot_slides(aging, 'clusters', ncols=2, bw=True, common_legend=True, minimal_title=True,
                           select_samples=['Young_3', 'Old_5'], order=['Young_3', 'Old_5'], title_fontweight='bold', title_fontsize=25,
                           fig_path=figure_path, filename='SpatialPlot_ST_Old5Young3_clusters.svg')

davidrPlotting.pl_umap(aging, ['clusters'], size=15, figsize=(4, 4), path=figure_path, filename='UMAP_ST_clusters.svg')

davidrPlotting.plot_slides(aging, 'clusters', bw=True, common_legend=True, minimal_title=True,
                           select_samples=['Young_1', 'Young_2', 'Young_4', 'Young_5', 'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           order=['Young_1', 'Young_2', 'Young_4', 'Young_5', 'Old_1', 'Old_2', 'Old_3', 'Old_4'],
                           title_fontweight='bold', fig_path=figure_path,
                           filename='SpatialPlot_ST_Clusters_Old1-4Young1245.svg')

# DGE Cluster Vs Rest
sc.tl.rank_genes_groups(aging, groupby='clusters', method='wilcoxon', tie_correct=True, pts=True)
# Save DEGs as ExcelSheet
dge_leiden = sc.get.rank_genes_groups_df(aging, group=None)
with pd.ExcelWriter( os.path.join(table_path, 'DGE', 'DGE_ST_Aging_Niches.xlsx')) as writer:
    dge_leiden.to_excel(writer, sheet_name='AllGenes', index=False)  # Sheet Name with all the genes
    for cluster in dge_leiden.group.unique():
        # For each cluster save only significant genes (Padj < 0.05) & rank by LFC
        dge_subset = dge_leiden[dge_leiden.group == cluster]
        dge_subset = dge_subset[dge_subset['pvals_adj'] < 0.05].sort_values('logfoldchanges', ascending=False)
        dge_subset.to_excel(writer, sheet_name=f'SigGenes_Clust{cluster}', index=False)
#</editor-fold>


#<editor-fold desc="Add large vessels">
figure_path = os.path.join(main_path, 'Figures/3_Annotation/Add_LargeVessels')

# Load data
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()

# Identify and annotate vessels using the hexamer approach
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
        HexaVessels.append('nonVasc')

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
aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))

# Visualisation the vessels
davidrPlotting.plot_slides(aging, 'vessels', common_legend=True, minimal_title=True, ncols=5,
                           order=sorted(aging.obs['sample'].unique()), alpha_img=.75, bw=True, figsize=(15, 8),
                           fig_path=figure_path, filename='SpatialPlot_ST_HexaVessels.svg',
                           palette={'Art_Lymph': 'gold', 'Arteries': 'tomato', 'Lymphatics': 'forestgreen',
                                    'MixVasc': 'darkorchid', 'Vein_Lymph': '#ffe4b5', 'Veins': 'royalblue',
                                    'nonVasc': '#FF000000'},
                           title_fontweight='bold')
#</editor-fold>


########################################################################################################################
# - Functional Analysis of Spatial transcriptomics: ORA (Overrepresentation Analysis), MistyR (co-localisation),
#   Study Senescence and Inference of Communication Event
########################################################################################################################


# <editor-fold desc="ORA on Heart - Supplementary Table 1 ">
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['SCT_norm'].copy()

msigdb = dc.get_resource('MSigDB')
msigdb['genesymbol'] = msigdb['genesymbol'].str.capitalize()
msigdb = msigdb[msigdb['collection'] == 'go_biological_process']
msigdb = msigdb[~msigdb.duplicated(['geneset', 'genesymbol'])]

dict_adatas = {}
for sample in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, sample)

    dc.run_ora(mat=sdata, net=msigdb, source='geneset', target='genesymbol', verbose=True, use_raw=False)

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


# <editor-fold desc="ORA Young Vs Old - PseudoBulk - Supplementary Table 2">
pdata = dc.get_pseudobulk(aging, sample_col='sample', groups_col='clusters', layer='counts', mode='sum',
                          min_cells=10, min_counts=1000)
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
    dds = DeseqDataSet(adata=cluster, design_factors='condition', ref_level=['condition', 'Young'], refit_cooks=True,
                       inference=inference, )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", 'Old', 'Young'], inference=inference, )
    stat_res.summary()

    results_df = stat_res.results_df
    results_df['cluster'] = clust
    mat = results_df[['stat']].T.rename(index={'stat': f'{clust}'})
    dc_matrix = pd.concat([dc_matrix, mat])
    dc_summary = pd.concat([dc_summary, results_df])

dc_matrix[dc_matrix.isna()] = 0  # Replace NaN with 0

# MSIGDB - For ORA Analysis
msigdb = dc.get_resource('MSigDB')
msigdb.genesymbol = msigdb.genesymbol.str.capitalize()
msigdb = msigdb[msigdb['collection'] == 'go_biological_process']
msigdb = msigdb[~msigdb.duplicated(['geneset', 'genesymbol'])]
msigdb.loc[:, 'geneset'] = [name.split('GOBP_')[1] for name in msigdb['geneset']]

dc_pathway = pd.DataFrame([])
for clust in dc_summary.cluster.unique():
    # Upregulated genes
    up_genes = dc_summary[(dc_summary['padj'] < 0.05) &  # Select significant genes
                          (dc_summary['cluster'] == clust) &
                          (dc_summary['log2FoldChange'] > 0.25)]  # Select genes with at least 0.25 Log2FC
    # Downregulated genes
    dw_genes = dc_summary[(dc_summary['padj'] < 0.05) &
                          (dc_summary['cluster'] == clust) &
                          (dc_summary['log2FoldChange'] < -0.25)]

    # Do ORA
    enr_pvals_up = dc.get_ora_df(df=up_genes, net=msigdb, source='geneset', target='genesymbol')
    enr_pvals_dw = dc.get_ora_df(df=dw_genes, net=msigdb, source='geneset',target='genesymbol')

    # Add some Metadata information
    enr_pvals_up['state'] = 'up'
    enr_pvals_dw['state'] = 'dw'
    enr_pvals = pd.concat([enr_pvals_up, enr_pvals_dw])
    enr_pvals['cluster'] = clust
    dc_pathway = pd.concat([dc_pathway, enr_pvals])

# Save Supplementary Table 2
dc_pathway.to_excel(os.path.join(table_path, '241017_DecoupleR_GOBP_ORA_PseudoBulk_Young_Vs_Old_Niches.xlsx'), index=False)

# Metrics from DESeq2
dc_pathway_sig = dc_pathway[dc_pathway['FDR p-value'] < 0.05]
dc_summary.to_excel(os.path.join(table_path, '241017_DecoupleR_PseudoBulk_Metrics_Young_Vs_Old_Niches.xlsx'))
#</editor-fold>


#<editor-fold desc="MistyR - preparation">
in_path = '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis'


aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['logcounts'].copy()

# Save each ST sample separately for MistyR
for batch in aging.obs['sample'].unique():
    sdata = davidrUtility.select_slide(aging, batch)
    sdata.write(f'/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis/Objects/{batch}.h5ad')

# MistyR Analysis in MistyR.R file

#</editor-fold>


#<editor-fold desc="Senescence  - Identification of Hotspots of senescence - Supplementary Table 4">
figure_path = os.path.join(main_path, 'Figures/4_FunctionalAnalysis/Senescence/')

# Load Data
aging = sc.read_h5ad(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
aging.X = aging.layers['logcounts'].copy()


# CellAge
cellage = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/CellAge3.tsv', sep='\t')
cellage['GeneMouse'] = cellage['Gene symbol'].str.capitalize()
cellage['dataset'] = 'CellAge'
cellage = cellage[cellage['Senescence Effect'] == 'Induces']  # Remove anti-senescence genes
# CellAge --> 370 unique genes

# AgingAtlas
tmp_files = [f for f in os.listdir('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/') if 'AgingAtlas' in f]
agingAtlas = pd.DataFrame()
for f in tmp_files:
    tmp = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/' + f)
    agingAtlas = pd.concat([agingAtlas, tmp])
agingAtlas['GeneMouse'] = agingAtlas['Symbol'].copy()
agingAtlas['dataset'] = 'AgingAtlas'  # AgingAtlas --> 391 unique genes

# SenMayo
senmayo = pd.read_excel('/mnt/davidr/scStorage/DavidR/BioData/SenMayo_Genes.xlsx', sheet_name='mouse')
senmayo['GeneMouse'] = senmayo['Gene(murine)']
senmayo['dataset'] = 'SenMayo'  # SenMayo --> 118 unique genes

# Cellular Senescence
msig = gp.Msigdb('2023.1.Mm')
gmt = msig.get_gmt(category='msigdb', dbver="2023.1.Mm")
cellularSenescence = pd.DataFrame(gmt['GOBP_CELLULAR_SENESCENCE'], columns=['GeneMouse'])
cellularSenescence['dataset'] = 'GOSenescence'  # CellularSenescence --> 141 unique genes

# Fridman & Tainsky
fridTain = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/Fridman&Tainsky.csv')
fridTain['GeneMouse'] = fridTain['SYMBOL'].str.capitalize()
fridTain['dataset'] = 'Fridman&Tainsky' # Friedman & Tainsky --> 101 unique genes

# Combine every dataset
cols = ['GeneMouse', 'dataset']
score = pd.concat([cellage[cols], agingAtlas[cols], senmayo[cols], cellularSenescence[cols], fridTain[cols]])  # 1246 genes
score = score.groupby('GeneMouse')['dataset'].apply(lambda x: '; '.join(x)).reset_index()  # 929 unique genes

# Exclude genes not present in our object
score = score[score.GeneMouse.isin(aging.var_names)]  # 824 genes

# DGE Young Vs Old of ST
sc.tl.rank_genes_groups(aging, groupby='condition', method='wilcoxon', tie_correct=True, logcounts=True)
dge = sc.get.rank_genes_groups_df(aging, group='Old', pval_cutoff=0.05)  # 10775 --> All Significant genes

dge_set = set(dge.names) # 10775 genes
score_set = set(score.GeneMouse) # 824 genes
ssg = dge_set & score_set  # 593 senesence sensitive genes

# Save Genes Used to estimate the Score
df_ssg = pd.DataFrame(list(ssg), columns=['GeneMouse'])
df_ssg.to_excel(os.path.join(table_path, 'DataFrame_GenesUsedFor_SenescenceScore.xlsx'), index=False)  # Supplementary Table 3

sc.tl.score_genes(aging, gene_list=ssg, score_name='senescence') # Compute the score

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

# Take the spots with the top 5 % highest senescence score --> Hotspots of senescence
top5 =  aging.obs.senescence.copy()
top5 =  list(top5[top5 > np.percentile(top5, 95)].index)

data = pd.DataFrame()
for batch in tqdm(aging.obs['sample'].unique()):
    sdata = aging[aging.obs['sample'] == batch]
    annotation = {bc:[10, 10,  10, 10, 10, 10, 10] for bc in sdata.obs_names}
    for bc in top5:
        if bc not in sdata.obs_names:
            continue

        # Hspot > dist100 > dist200 > dist300 > dist400 > dist500 > rest
        annotation[bc][0] = 0

        # Calculate BCs in the gradient space
        hexamer = get_surrounding(sdata, bc_spot=bc, radius=100)

        extended = get_surrounding(sdata, bc_spot=bc, radius=200)
        extended = [val for val in extended if val not in hexamer]  # Remove BCs in  previous locations

        greater = get_surrounding(sdata, bc_spot=bc, radius=300)
        greater = [val for val in greater if val not in hexamer and val not in extended] # Remoce BCs in previous locations

        major = get_surrounding(sdata, bc_spot=bc, radius=400)
        major = [val for val in major if val not in hexamer and val not in extended and val not in greater]  # Remoce BCs in previous locations

        external = get_surrounding(sdata, bc_spot=bc, radius=500)
        external = [val for val in external if val not in hexamer and val not in major and val not in greater and val not in major]  # Remoce BCs in previous locations

        hexamer.remove(bc) # Remove Hspot BCs

        # Add a score for each location
        cont = 1
        for case, label in [(hexamer, 1), (extended, 2), (greater, 3), (major, 4), (external, 5)]:
            for bc_inner in case:
                annotation[bc_inner][cont] = label
            cont +=1

    # Hspot -> 0; dist100 -->1; dist200 -->2; dist300 --> 3; dist400 --> 4; dist500 --> 5; rest --> 10
    annotation  = pd.DataFrame.from_dict(annotation).T
    annotation = pd.DataFrame(annotation.min(axis=1), columns=['code'])
    annotation['annotation'] = annotation.code.replace({0:'Hspot', 1:'dist100', 2:'dist200', 3:'dist300',
                                                        4:'dist400', 5:'dist500', 10:'rest'})
    data = pd.concat([data, annotation['annotation']])

data = data.reindex(index=aging.obs_names)
aging.obs['senescence_gradient'] = pd.Categorical(data.annotation)
aging.write(os.path.join(object_path, 'Scanpy/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad'))
#</editor-fold>


#<editor-fold desc="HoloNet - Inference of Communication Event">
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

    interc_db, cofact_db, cplx_db = hn.pp.load_lr_df(human_or_mouse=organism)  # Load LR DB from the specified organism
    w_best = hn.tl.default_w_visium(adata, min_cell_distance=min_dist, cover_distance=max_dist)  # Specify the maximum distance of the Communication Event
    df_expr_dict = hn.tl.elements_expr_df_calculate(df_lr, cplx_db, cofact_db, adata)  # Calculate the expression
    ce_tensor = hn.tl.compute_ce_tensor(df_lr, w_best, df_expr_dict, adata)  # Calculate CE matrix
    filt_ce_tensor = hn.tl.filter_ce_tensor(ce_tensor, adata, df_lr, df_expr_dict, w_best)  # Filter CE  --> lr x spotS x spotR

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


DB_common = davidrSpatial.holonet_common_lr(aging)  # Compute the CE that are common across samples

dict_CCC = {}
for sample in aging.obs['sample'].unique():
    # Inference of CE per sample
    sdata = davidrUtility.select_slide(aging, sample, 'sample')

    # Prepare the AnnData Object
    sdata.obsm['predicted_cell_type'] = sdata.obsm['c2l_prop'].copy()
    sdata.obsm['predicted_cell_type']['max'] = sdata.obsm['predicted_cell_type'].max(axis=1)
    sdata.obs['max_cell_type'] = sdata.obsm['predicted_cell_type'].idxmax(axis=1)
    # Inference
    ndata = holonet_CCC(sdata, DB_common, '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Objects/Pickle/HoloNet',
                        f'{sample}_CCC_tensor.pkl')
    # Save the AnnData of CE for each sample in a dict of anndata
    dict_CCC[sample] = ndata

# Concatenate the dict of AnnDatas
merged_CCC = ad.concat(dict_CCC.values(), keys=dict_CCC.keys(), label='sample', join='outer', uns_merge='unique')
# Save the Object
merged_CCC.write(os.path.join(object_path, 'Scanpy/HoloNet/Visium_YoungOld_HoloNet_CCC_230624.h5ad'))
#</editor-fold>



########################################################################################################################
# - Processing of Human Left Ventricle ST
########################################################################################################################

#<editor-fold desc="Integration">
import STAligner
import scanpy.external as sce

human_lv = sc.read_h5ad(os.path.join(object_path, 'visium-OCT_LV_lognormalised.h5ad'))

batch_list, adj_list, sections_ids = [], [], []
# Calculate Spatial Network
for sample in human_lv.obs['sangerID'].unique():
    adata = davidrUtility.select_slide(human_lv, sample, 'sangerID')
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=375)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=9000)
    adata = adata[:, adata.var.highly_variable].copy()
    adj_list.append(adata.uns['adj'])
    batch_list.append(adata)
    sections_ids.append(sample)

# Concatenate Samples
ad_concat = ad.concat(batch_list, label='slice_name', keys=sections_ids)
ad_concat.obs['batch_name'] = ad_concat.obs['slice_name'].astype('category')

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(sections_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

ad_concat.uns['edgeList'] = np.nonzero(adj_concat)

ad_concat = STAligner.train_STAligner(ad_concat, verbose=True, hidden_dims=[250, 30],
                                      knn_neigh=50, margin=2.5)

# Trasfer Integrated matrix
human_lv.obsm['STAligner'] = ad_concat.obsm['STAligner']

sce.pp.bbknn(human_lv, batch_key='sangerID', use_rep='STAligner', neighbors_within_batch=3, metric='manhattan')
sc.tl.leiden(human_lv, random_state=666, key_added="leiden_STAligner", resolution=0.5)
sc.tl.umap(human_lv, random_state=666, min_dist=0.05, spread=2)

sc.pl.umap(human_lv, color=['sangerID', 'age', 'leiden_STAligner'])
human_lv.write('/media/Storage/DavidR/Objects_Submission/HumanLV_Integrated.h5ad')
#</editor-fold>


#<editor-fold desc="Senescence Scoring">
human_lv = sc.read_h5ad('/media/Storage/DavidR/Objects_Submission/HumanLV_Integrated.h5ad')
human_lv.obs['clusters'] = 'Niche h' + human_lv.obs.leiden_STAligner.astype(str)


# CellAge
cellage = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/CellAge3.tsv', sep='\t')
cellage['GeneHuman'] = cellage['Gene symbol']
cellage['dataset'] = 'CellAge'
cellage = cellage[cellage['Senescence Effect'] == 'Induces']

# AgingAtlas
tmp_files = [f for f in os.listdir('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/') if 'AgingAtlas_Human' in f]
agingAtlas = pd.DataFrame()
for f in tmp_files:
    tmp = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/' + f)
    agingAtlas = pd.concat([agingAtlas, tmp])
agingAtlas['GeneHuman'] = agingAtlas['Symbol'].copy()
agingAtlas['dataset'] = 'AgingAtlas'

# SenMayo
senmayo = pd.read_excel('/mnt/davidr/scStorage/DavidR/BioData/SenMayo_Genes.xlsx', sheet_name='mouse')
senmayo['GeneHuman'] = senmayo['Gene(murine)'].str.upper()
senmayo['dataset'] = 'SenMayo'

# Cellular Senescence
msig = gp.Msigdb('2023.1.Hs')
gmt = msig.get_gmt(category='msigdb', dbver="2023.1.Hs")
cellularSenescence = pd.DataFrame(gmt['GOBP_CELLULAR_SENESCENCE'], columns=['GeneHuman'])
cellularSenescence['dataset'] = 'GOSenescence'  # CellularSenescence --> 108 unique genes

# Fridman & Tainsky
fridTain = pd.read_csv('/mnt/davidr/scStorage/DavidR/BioData/SenescenceScore/Fridman&Tainsky.csv')
fridTain['GeneHuman'] = fridTain['SYMBOL']
fridTain['dataset'] = 'Fridman&Tainsky'

# Combine every dataset
cols = ['GeneHuman', 'dataset']
score = pd.concat(
    [cellage[cols], agingAtlas[cols], senmayo[cols], cellularSenescence[cols], fridTain[cols]])
score = score.groupby('GeneHuman')['dataset'].apply(lambda x: '; '.join(x)).reset_index()

score = score[score.GeneHuman.isin(human_lv.var_names)]
human_lv = human_lv[human_lv.obs.sangerID != 'HCAHeartST8795933']
sc.tl.rank_genes_groups(human_lv, groupby='age', method='wilcoxon', tie_correct=True)
dge = sc.get.rank_genes_groups_df(human_lv, group='65-70', pval_cutoff=0.05)

dge_set = set(dge.names)  # 6621 genes
score_set = set(score.GeneHuman)  # 892 genes
ssg = dge_set & score_set  # 463 senesence sensitive genes
sc.tl.score_genes(human_lv, list(ssg), score_name='senescence')



top5 = human_lv.obs.senescence.copy()
top5 = list(top5[top5 > np.percentile(top5, 95)].index)

import anndata as ad
from tqdm import tqdm

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
        spot = adata[adata.obs_names == bc_spot, :].obsm['spatial'][0]

    surrounding = []
    for i, sp in enumerate(adata.obsm['spatial']):
        distance = ((spot[0] - sp[0]) ** 2 + (spot[1] - sp[1]) ** 2) ** .5
        if distance <= radius:
            surrounding.append(i)
    if get_bcs:
        return list(adata.obs_names[surrounding])
    else:
        return surrounding


data = pd.DataFrame()
for batch in tqdm(human_lv.obs['sangerID'].unique()):
    sdata = human_lv[human_lv.obs['sangerID'] == batch]
    annotation = {bc: [10, 10, 10, 10, 10, 10, 10] for bc in sdata.obs_names}
    for bc in top5:
        if bc not in sdata.obs_names:
            continue

        # Hspot > dist100 > dist200 > dist300 > dist400 > dist500 > rest
        annotation[bc][0] = 0

        # Calculate BCs in the gradient space
        hexamer = get_surrounding(sdata, bc_spot=bc, radius=300)

        extended = get_surrounding(sdata, bc_spot=bc, radius=450)
        extended = [val for val in extended if val not in hexamer]  # Remove BCs in  previous locations

        greater = get_surrounding(sdata, bc_spot=bc, radius=750)
        greater = [val for val in greater if
                   val not in hexamer and val not in extended]  # Remoce BCs in previous locations

        major = get_surrounding(sdata, bc_spot=bc, radius=850)
        major = [val for val in major if
                 val not in hexamer and val not in extended and val not in greater]  # Remoce BCs in previous locations

        external = get_surrounding(sdata, bc_spot=bc, radius=1050)
        external = [val for val in external if
                    val not in hexamer and val not in major and val not in greater and val not in major]  # Remoce BCs in previous locations

        hexamer.remove(bc)  # Remove Hspot BCs

        # Add a score for each location
        cont = 1
        for case, label in [(hexamer, 1), (extended, 2), (greater, 3), (major, 4), (external, 5)]:
            for bc_inner in case:
                annotation[bc_inner][cont] = label
            cont += 1

    # Hspot -> 0; dist100 -->1; dist200 -->2; dist300 --> 3; dist400 --> 4; dist500 --> 5; rest --> 10
    annotation = pd.DataFrame.from_dict(annotation).T
    annotation = pd.DataFrame(annotation.min(axis=1), columns=['code'])
    annotation['annotation'] = annotation.code.replace({0: 'Hspot', 1: 'dist300', 2: 'dist450',
                                                        3: 'dist650',
                                                        4: 'dist750', 5: 'dist800', 10: 'rest'})
    data = pd.concat([data, annotation['annotation']])

data = data.reindex(index=human_lv.obs_names)
human_lv.obs['senescence_gradient'] = pd.Categorical(data.annotation)

human_lv.write('/media/Storage/DavidR/Objects_Submission/HumanLV_Integrated.h5ad')
#</editor-fold>





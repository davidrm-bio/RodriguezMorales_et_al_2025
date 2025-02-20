#!/usr/bin/env python

"""
Description: Module for Pre-Processing of scRNA or snRNA including
quality control, normalisation, integration & annotation

Author: David Rodriguez Morales
Date Created: 15-11-2023
Date Modified: 20-08-2024
Version: 2.0
Python Version: 3.11.8
"""

# Libraries
import os
import warnings
import subprocess
from datetime import datetime, date
from typing import Union
from tqdm import tqdm

import numpy as np
import polars
import scanpy as sc
import scvelo as scv
import scanpy.external as sce
import anndata as ad
import bbknn as bkn
import celltypist

import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import seaborn as sns

from davidrPlotting import generate_violin_plot
from davidrUtility import iOn, config_verbose, set_plt_theme

# Global Configuration
warnings.filterwarnings('ignore')  # Hide Warnings created by Scanpy
set_plt_theme()
iOn()

logger = config_verbose(True)

# Pre-Processing (i.e., Quality Control)
def standard_filter(adata: ad.AnnData,
                    fig_path: str,
                    samplename: str,
                    n_genes: int = 250,
                    n_cells: int = 3,
                    cut_mt: int = 10,
                    doublet_rate: float = 0.06,
                    metric_path: str = None,
                    is_mouse: bool = True,
                    include_Rbs: bool = False,
                    ) -> ad.AnnData:
    """ **Standard Quality Control of sc/snRNA-seq in Scanpy.**

    The input is an unprocessed anndata object of a single sample. Generated for example, after using
    ``scanpy.read()``. Among the quality control steps applied by the function we have:

    * Remove low quality / dying cells
    * Remove lowly expressed or undetected genes
    * Remove cells with high percentage of mitochondrial genes
    * Optionally - Compute Ribosomal genes metrics
    * Remove doublets

    It is recommended to provide a *metric_path*. This will generate a file which records the number of cells
    and genes remove in each step of the processing.

    :param adata: anndata object of a single sample
    :param fig_path: path to save plots generated during QC.
    :param samplename: name of the sample or a key that will be used in the filenames generated and saved. Plots like
                       the histogram from scrublet will be saved including this key.
    :param n_genes: minimum number of genes per cell. (Default value **250**)
    :param n_cells: minimum number of cells expressing a gene. (Default value **3**)
    :param cut_mt: maximum percentage of mitochondrial genes per cell. It is recommended to use a lower
                   cutoff for snRNA. (Default value **10 %**)
    :param doublet_rate: expected doublet rate of the experiment. (Default value **0.06**)
    :param metric_path: path to save the metrics file. An Excel Sheet will be generated
    :param is_mouse: input data is mouse. It is important to modify this to identify correctly the mitochondrial genes
                  (Default is **True**)
    :param include_Rbs: compute metrics for the total number of ribosomal genes per cell. (Default is **False**)
    :return: filtered anndata object
    """

    # Set logger settings
    today = date.today().strftime("%y%m%d")

    # Copy anndata object to not modify the input anndata
    adata = adata.copy()

    # Create metrics file
    metric_file = f'{today}_Metrics_{samplename}.xlsx'  # Assume Excel Sheet
    # Create a df to save metrics
    df_metrics = pd.DataFrame([], columns=['QC_Step', 'Cells', 'Genes'])  # Record data
    df_metrics.loc[0] = ['Input_Shape', adata.shape[0], adata.shape[1]]

    # Step 1 - Remove low-quality cells
    logger.info('Remove low-quality cells')
    sc.pp.filter_cells(adata, min_genes=n_genes, inplace=True)
    df_metrics.loc[1] = ['Rm_poor_Cells', adata.shape[0], adata.shape[1]]

    # Step 2 - Remove genes with low expression or not detected
    logger.info('Remove low-expressed genes')
    sc.pp.filter_genes(adata, min_cells=n_cells, inplace=True)
    df_metrics.loc[2] = ['Rm_low_Genes', adata.shape[0], adata.shape[1]]

    # Step 3 - Remove cells with high Mt-genes (low quality / dying cells)
    logger.info('Remove cells with high MT- content')
    # Assume human data
    mt_gene, ribo_gene = 'MT-', ('RBL', 'RBS')  # human genes are uppercase
    if is_mouse:
        mt_gene, ribo_gene = 'mt-', ('Rbs', 'Rbl')  # mouse genes are lowercase

    qc_metrics = ['mt']
    if include_Rbs:
        qc_metrics = ['mt', 'ribo']

    adata.var['mt'] = adata.var_names.str.startswith(mt_gene)  # Annotate mitochondria genes
    adata.var['ribo'] = adata.var_names.str.startswith(ribo_gene)  # Annotate mitochondria genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_metrics, percent_top=None, log1p=True, inplace=True, use_raw=False)
    adata = adata[adata.obs.pct_counts_mt < cut_mt, :]
    df_metrics.loc[3] = ['Rm_cell_high_MT', adata.shape[0], adata.shape[1]]

    # Step 4 - Remove doublets
    logger.info('Remove doublets')
    sc.pp.scrublet(adata, expected_doublet_rate=doublet_rate, verbose=True, log_transform=True)
    adata = adata[adata.obs['predicted_doublet'] == False]
    df_metrics.loc[4] = ['Scrublet_doublet', adata.shape[0], adata.shape[1]]

    # Step 5  - Remove cells with the highest counts (top 5 %)
    logger.info('Remove the top 5 % of cells with highest total counts')
    perc95 = np.percentile(adata.obs['total_counts'], 95)
    adata = adata[adata.obs.total_counts < perc95, :]
    df_metrics.loc[5] = ['Rm_Cell_high_count', adata.shape[0], adata.shape[1]]

    if metric_path is not None:
        df_plot = df_metrics.melt(id_vars='QC_Step')
        fig, axs = plt.subplots(1, 1, figsize=(18, 12))  # initializes figure and plots
        bp = sns.barplot(df_plot, y='QC_Step', x='value', hue='variable', hue_order=['Cells', 'Genes'],
                         palette={'Cells': 'royalblue', 'Genes': 'tomato'}, ax=axs)
        for container in bp.containers:
            bp.bar_label(container)
        bp.set_title('')
        bp.set_xlabel('Quality Control Step', fontsize=18)
        bp.set_ylabel('Counts', fontsize=18)
        bp.legend(title='Feature', fontsize=12, frameon=False, title_fontproperties={'weight': 'bold', 'size': 15})
        bp.spines[['top', 'right']].set_visible(False)
        bp.grid(False)
        plt.savefig(os.path.join(fig_path, f'{today}_QC_Metrics{samplename}.svg'), bbox_inches='tight')

        # Save Metric File
        df_metrics.to_excel(os.path.join(metric_path, metric_file), index=False)
    return adata


def NormaliseData(adata: ad.AnnData,
                  n_reads: int = 10_000,
                  max_val: float = None):
    """**Data Normalisation**

    The input is an unnormalise anndata object. The data in .X will be log-normalise to 10,000 reads
    per cell.
    The returned anndata object will contain 3 layers:

    * counts: contains the raw unnormalised counts
    * logcounts: contains the log-normalise counts
    * scaled: contained the log-normalise counts scaled

    Additionally, the log-normalise counts will also be save under the .X attribute.

    :param adata: annData object
    :param n_reads: target number of reads per cell to normalize to. (Default  is **10,000**)
    :param max_val: maximum expression value after scaling. (Default is **None**)
    :return: log-normalise anndata object
    """
    adata = adata.copy()  # Do not modify input
    adata.layers['counts'] = adata.X.copy()  # Save raw counts
    # Normalise
    sc.pp.normalize_total(adata, target_sum=n_reads)
    sc.pp.log1p(adata)
    adata.layers['logcounts'] = adata.X.copy()
    # Scale
    sc.pp.scale(adata, zero_center=True, max_value=max_val)
    adata.layers['scaled'] = adata.X.copy()
    # Save logcounts in .X
    adata.X = adata.layers['logcounts'].copy()
    return adata


def SCTransform(anndata: ad.AnnData,
                tmp_path: str,
                rscript: str = '/mnt/davidr/scStorage/DavidR/SCTransform_from_scanpy.R',
                layer=None) -> ad.AnnData:
    """**Normalisation base on SCTransform**

    This function performs an alternative normalisation base on the SCTransform wrapper from Seurat.

    :param anndata: anndata object
    :param tmp_path: tmp path where intermediate files are saved when transfering data from scanpy to seurat
    :param rscript: rscript for the SCTransform
    :param layer: layer to use
    :return: anndata object
    """
    print('Preparing to transfer to R...')
    adata = anndata.copy()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    del adata.uns
    del adata.obsm
    adata.write(os.path.join(tmp_path, 'adata_to_seurat_tmp.h5ad'))

    print('SCTransform in R...')
    subprocess.call(['Rscript', rscript, '--input=' + tmp_path, '--out=' + tmp_path])

    print('Transfering to anndata...')
    raw_counts = polars.read_csv(os.path.join(tmp_path, 'SCTransform_raw.csv'), infer_schema_length=0)
    raw_counts = raw_counts.to_pandas().astype(float)
    raw_counts = raw_counts.set_index(adata.obs_names)

    norm_counts = polars.read_csv(os.path.join(tmp_path, 'SCTransform_norm.csv'), infer_schema_length=0)
    norm_counts = norm_counts.to_pandas().astype(float)
    norm_counts = norm_counts.set_index(adata.obs_names)

    # raw_counts = pd.read_csv(os.path.join(tmp_path, 'SCTransform_raw.csv'), index_col='Unnamed: 0')
    # norm_counts = pd.read_csv(os.path.join(tmp_path, 'SCTransform_norm.csv'), index_col='Unnamed: 0')

    # Transfer genes not kept during normalisation to .obsm
    excluded_genes = [gene for gene in anndata.var_names if gene not in norm_counts.columns]
    anndata.var['SCT_rm'] = [True if gene in excluded_genes else False for gene in anndata.var_names]
    anndata.obsm['SCT_rm'] = anndata[:, anndata.var['SCT_rm'].values].X.toarray()
    anndata = anndata[:, ~anndata.var['SCT_rm'].values]

    # Make sure we have the same order or barcodes and features
    norm_counts = norm_counts.reindex(index=anndata.obs_names, columns=anndata.var_names)
    raw_counts = raw_counts.reindex(index=anndata.obs_names, columns=anndata.var_names)

    anndata.layers['SCT_norm'] = sparse.csr_matrix(norm_counts.values)
    anndata.layers['SCT_counts'] = sparse.csr_matrix(raw_counts.values)
    return anndata


def standard_quality_control(adata: ad.AnnData,
                             samplename: str,
                             fig_path: str,
                             n_genes: int = 250,
                             n_cells: int = 3,
                             cut_mt: float = 10,
                             max_val: float = None,
                             doublet_rate: float = 0.06,
                             n_reads: int = 10_000,
                             metric_path: str = None,
                             cellcycle: bool = False,
                             Include_Rbs: bool = False,
                             is_mouse: bool = True,
                             obj_path: str = None,
                             obj_name: str = None,
                             genesS_path: str = '/mnt/davidr/scStorage/DavidR/BioData/genes-cellcycle-s.txt',
                             genesG2M_path: str = '/mnt/davidr/scStorage/DavidR/BioData/gene-cellcycle-g2m.txt'
                             ) -> ad.AnnData:
    """**Standard Quality Control of sc/snRNA base on Scanpy (davidrPackage)**

    The input is an unprocessed anndata object of a single sample. Generated for example, after using
    ``scanpy.read()``. Among the quality control steps applied by the function we have:

    * Remove low quality / dying cells
    * Remove lowly expressed or undetected genes
    * Remove cells with high percentage of mitochondrial genes
    * Optionally - Compute Ribosomal genes metrics
    * Remove doublets
    * Log-Normalisation of the Data
    * Optionally - Compute Cell Cycle Score

    It is recommended to provide a *metric_path*. This will generate a file which records the number of cells
    and genes remove in each step of the processing.

    :param adata: anndata object of a single sample unprocessed. The .X attribute will be used for the analysis
    :param samplename: name of the sample or a key that will be used in the filenames generated and saved. Plots like
                       the histogram from scrublet will be saved including this key.
    :param fig_path: path to save plots generated during QC.
    :param n_genes: minimum number of genes per cell. (Default value **250**)
    :param n_cells: minimum number of cells expressing a gene. (Default value **3**)
    :param cut_mt: maximum percentage of mitochondrial genes per cell. It is recommended to use a lower
                   cutoff for snRNA. (Default value **10 %**)
    :param max_val: cutoff for scaling the data. Values with an expression greater than this cutoff will be truncated.
    :param doublet_rate: expected doublet rate of the experiment. (Default value **0.06**)
    :param n_reads: target number of reads per cell to normalize to. (Default  is **10,000**)
    :param metric_path: path to save the metrics file. An Excel Sheet will be generated
    :param cellcycle: compute Cell-Cycle scores. (Default is **False**)
    :param Include_Rbs: compute metrics for the ribosomal genes (e.g., percentage of ribosomal genes per cell).
                        (Default is **False**)
    :param is_mouse: input data is mouse. It is important to modify this to identify correctly the mitochondrial genes
                  (Default is **True**)
    :param Include_Rbs: compute metrics for the total number of ribosomal genes per cell. (Default is **False**)
    :param obj_path: path to save the anndata object generated.
    :param obj_name: name of the H5AD file
    :param genesS_path: path to a txt file containing genes involve in the S phase (One gene per line). Needed if
                        cellcycle score is True
    :param genesG2M_path: path to a txt file contaning genes involve in the G2M phase (One gene per line). Needed if
                          cellcycle score is True
    :return: processed anndata object
    """

    # Set logger settings
    today = date.today().strftime("%y%m%d")

    # Step 1 - Set-up: Save QC Plots in a SubFolder within the Figure Path
    logger.info('Standard Quality Control in Scanpy')
    logger.info(f'#, {datetime.now()} - Set-up...')
    adata = adata.copy()  # Avoid modifying the input

    try:
        os.makedirs(os.path.join(fig_path, 'QC_Plots'))
    except FileExistsError:
        pass
    fig_path = os.path.join(fig_path, 'QC_Plots')  # Update variable

    # Step 2 - Filter the data
    logger.info(f'#, {datetime.now()} -Filtering data...')
    adata = standard_filter(adata,
                            fig_path=fig_path,
                            samplename=samplename,
                            n_genes=n_genes,
                            n_cells=n_cells,
                            cut_mt=cut_mt,
                            doublet_rate=doublet_rate,
                            metric_path=metric_path,
                            is_mouse=is_mouse,
                            include_Rbs=Include_Rbs,
                            )

    # Step 3 - Normalise counts
    logger.info(f'#, {datetime.now()} - Normalise counts...')
    adata = NormaliseData(adata, n_reads=n_reads, max_val=max_val)

    # Optional Step - Add Cell Cycle information / Optional - Data need to be log-normalise
    if cellcycle:
        logger.info(f'#, {datetime.now()} - Get Cell Cycle stats...')
        try:
            s_genes = pd.read_csv(genesS_path)
            g2m_genes = pd.read_csv(genesG2M_path)
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        except FileNotFoundError:
            print('Files with CellCycle Genes not found, moving to the next step...')

    # Step 4 - Generate ViolinPlots PostQC
    logger.info(f'#, {datetime.now()} - ViolinPlots post-QC...')
    generate_violin_plot(adata,
                         title='{}-PostQC'.format(samplename),
                         path=fig_path,
                         filename=f'{today}_ViolinPlot-PostQC-{samplename}.svg')

    # Step 5 - Save Processed object
    if obj_path is not None:
        if obj_name is None:
            obj_name = samplename + '.h5ad'
        logger.info(f'#, {datetime.now()} -Saving object...')
        adata.write(os.path.join(obj_path, obj_name))
    return adata


# Integration
def concatenate_anndata(adata_dict: dict,
                        sample_label: str = 'sample',
                        typejoin: str = 'outer',
                        **kwargs) -> ad.AnnData:
    """ **Concatenate a dictionary of anndata objects (davidrPackage)**

    The input is a dictionary of anndata objects. The keys of the dictionary will be considered as batch
    information, and they will be included in the .obs attribute. There are two ways of concatenating
    the anndata objects:

    * outer join: keep all the genes. If a gene is not present in a sample the counts (expression) will be
                  set to 0. Note this might lead to false DGEs, this should not be an issue with many samples.
    * inner join: keep genes present in all samples. Might be problematic if many genes are not shared between
                  samples.

    :param adata_dict: dictionary of anndata objects.
    :param sample_label: name of the column where keys of the dictionary are going to be saved. (Default is **sample**)
    :param typejoin: type of concatenation (inner or outer). (Default is **outer**)
    :param kwargs: additional arguments passed to ``anndata.concat()``
    :return: concatenated anndata object
    """
    logger.info('Concatenate a dictionary of anndata objects')
    adata_concat = ad.concat(adata_dict.values(),
                             label=sample_label,
                             keys=adata_dict.keys(),
                             join=typejoin,
                             index_unique='-',
                             fill_value=0, **kwargs)
    adata_concat.var['GeneSymbol'] = adata_concat.var_names  # Save Gene Symbols
    return adata_concat


def integration_plus_clustering(anndata: ad.AnnData,
                                batch: str = 'sample',
                                resolution: float = 0.3,
                                clustering_key: str = 'leiden',
                                scanorama: bool = False,
                                combat: bool = False,
                                harmony: bool = True,
                                bbknn: bool = False) -> ad.AnnData:
    """**Integration of anndata based on scanpy (davidrPackage)**

    The input is an anndata object that should contain several batches/samples. Dimensionality reduction
    will be performed employing Principal Component Analysis. This function supports 4 type of integration
    methods:

    * scanorama
    * combat
    * harmony
    * BBKNN --> employs the workflow from Park et al., 2020.

    The distance matrix is computed using BBKNN replacing the standard ``sc.pp.neighbors()`` function
    from scanpy. This function will also perfome unsupervised clustering base on the leiden algorithm.


    :param anndata: anndata object of concatenated samples
    :param batch: .obs column label of batch information. (Default is **sample**)
    :param resolution: resolution for clustering with leiden algorithm. (Default is **0.3**)
    :param clustering_key: .obs column name for the clustering. (Default **leiden**)
    :param scanorama: Perform Scanorama integration.(Default is **False**)
    :param combat: Perform Combat integration. (Default is **False**)
    :param harmony: harmony integration. (Default is **True**)
    :param bbknn: BBKNN integration. (Default is **False**)
    :return: integrated anndata object
    """
    logger.info('Integration and Clustering based on Scanpy')

    anndata_copy = anndata.copy()

    # Step 1 - Compute highly variable genes to be used in PCA computation
    logger.info(f'# {datetime.now()} - Compute HVGs')
    sc.pp.highly_variable_genes(anndata_copy, batch_key=batch, min_mean=0.0125, max_mean=3, min_disp=0.5, )

    # Step 2 - PCA (automatically takes highly variable genes)
    logger.info(f'# {datetime.now()} - Compute PCA')
    if 'scaled' in anndata_copy.layers.keys():
        anndata_copy.X = anndata_copy.layers['scaled'].copy()
    else:
        anndata_copy.X = anndata_copy.layers['logcounts'].copy()
        sc.pp.scale(anndata_copy, zero_center=True)
    sc.pp.pca(anndata_copy, mask_var=anndata_copy.var.highly_variable)  # Use highly variable genes
    anndata_copy.X = anndata_copy.layers['logcounts'].copy()

    # Step 3 - Integration with BBKNN on PCA or Scanorama
    dim_reduc = 'X_pca'

    if scanorama:
        print('# ', datetime.now(), ' - Step 3: Scanorama Integration...')
        sce.pp.scanorama_integrate(anndata_copy, key=batch)
        dim_reduc = 'X_scanorama'
    if harmony:
        print('# ', datetime.now(), ' - Step 3: Harmony Integration...')
        sce.pp.harmony_integrate(anndata_copy, key=batch)
        dim_reduc = 'X_pca_harmony'
    if combat:
        print('# ', datetime.now(), ' - Step 3: Combat Integration...')
        print('Not Implemented yet, moving on to BBKNN')  # TODO
        pass

    print('# ', datetime.now(), ' - Step 3: Integrating using BBKNN...')
    neighbors_within_batch = 25 if anndata_copy.n_obs > 100000 else 3  # Community recommendations
    bkn.bbknn(anndata_copy, batch_key=batch, neighbors_within_batch=neighbors_within_batch, use_rep=dim_reduc)

    if bbknn:
        print('# ', datetime.now(), ' - Step 3: BBKNN Integration...')
        sc.tl.leiden(anndata_copy, resolution=resolution, key_added=clustering_key)
        bkn.ridge_regression(anndata_copy, batch_key=[batch], confounder_key=[clustering_key])
        sc.tl.pca(anndata_copy)
        bkn.bbknn(anndata_copy, batch_key=batch, neighbors_within_batch=neighbors_within_batch, use_rep=dim_reduc)

    # Step 4 - Compute UMAP
    print('# ', datetime.now(), ' - Step 4: Calculating UMAP...')
    sc.tl.umap(anndata_copy)

    # Step 5 - Clustering with leiden algorithm
    print('# ', datetime.now(), ' - Step 5: Clustering the cells...')
    sc.tl.leiden(anndata_copy, resolution=resolution, key_added=clustering_key, flavor='igraph', n_iterations=2, directed=False)
    print('# ', datetime.now(), ' - Finished!\n\n')
    return anndata_copy


# Annotation
CellMarkers_Mm = {
    'Art_EC': ['Rbp7', 'Ly6a', 'Id1', 'Stmn2', 'Fbln5', 'Glul', 'Cxcl12', 'Sox17', 'Hey1', 'Mgll', 'Dusp1', 'Alpl',
                  'Btg2', 'Klf4', 'Crip1'],  # Refined with Kalucka et al., Cell, 2022
    'CapEC': ['Cd36', 'Fabp4', 'Aqp1', 'Rgcc', 'Gpihbp1', 'Aplnr', 'Lpl', 'Sparcl1', 'Car4', 'Sparc', 'Tcf15',
                     'Sgk1', 'Kdr', 'Cav1', 'Vwa1'],  # Refined with Kalucka et al., Cell, 2022
    'VeinEC': ['Mgp', 'Cfh', 'Apoe', 'Cpe', 'Bgn', 'Vwf', 'Fabp5', 'Vcam1', 'H19', 'Tmem108', 'F2r', 'Ptgs1', 'Il6st',
                'Vim', 'Comp'],  # Refined with Kalucka et al., Cell, 2022
    'LymphEC': ['Prox1', 'Lyve1', 'Pdpn', 'Ccl21a', 'Fgl2', 'Mmrn1', 'Lcn2', 'Nts', 'Cp', 'Reln', 'Cd63', 'Maf',
                     'Lmo2', 'Ntn1', 'Anxa1'],  # Refined with Kalucka et al., Cell, 2022
    'EndoEC': ['Nfatc', 'Npr3', 'Nrg1', 'Pecam1', 'Cdh5', 'Etv2', 'Flk1'],  # ZMM_shared JW
    'SMC': ['Myh11', 'Itga8', 'Acta2', 'Tagln', 'Carmn', 'Kcnab1', 'Ntrk3', 'Rcan2'],  # Refined
    'PC': ['Rgs5', 'Abcc9', 'Gucy1a2', 'Egflam', 'Dlc1', 'Pdgfrb', 'Des', 'Cd248', 'Mcam'],  # Refined
    'FB': ['Dcn', 'Abca9', 'Mgp', 'Lama2', 'Abca6', 'Gsn', 'Pdgfra', 'Vim', 'Fap', 'Pdgfrb'],  # Refined
    'FBa': ['Postn'],  # Refined JW ZMM_shared
    'Neurons': ['Nrxn1', 'Cadm2', 'Chl1', 'Kirrel3', 'Sorcs1', 'Ncam2', 'Pax3'],  # Refined
    'CM': ['Ryr2', 'Mlip', 'Ttn', 'Fhl2', 'Rbm20', 'Ankrd1', 'Tecrl', 'Mybpc3', 'Tnni3', 'Myh7', 'Mybpc3',
                      'Irx4'],  # Refined
    'B_cells': ['Igkc', 'Ighm', 'Aff3', 'Cd74', 'Bank1', 'Ms4a1', 'Cd79a', 'Cd69'],  # Refined
    'T_cells': ['Il7r', 'Themis', 'Skap1', 'Cd247', 'Itk', 'Ptprc', 'Camk4', 'Cd3e', 'Cd3d', 'Cd4', 'Cd8a', 'Cd8b'],
    # Refined
    'Myeloid': ['F13a1', 'Rbpj', 'Cd163', 'Rbm47', 'Mrc1', 'Fmn1', 'Msr1', 'Frmd4b', 'Mertk',
                           'Lyz2'],  # Refined
    'Mesothelial': ['Pdzrn4', 'Slc39a8', 'Gfpt2', 'C3', 'Wwc1', 'Kcnt2', 'Wt1', 'Dpp4', 'Ano1'],  # Refined
    'Adip': ['Gpam', 'Adipoq', 'Acacb', 'Ghr', 'Pde3b', 'Fasn', 'Prkar2b', 'Plin1', 'Pparg'],  # Refined
    'Mast': ['Il18r1', 'Kit', 'Slc24a3', 'Ntm', 'Cpa3', 'Slc8a3', 'Cdk15', 'Hpgds', 'Slc38a11', 'Rab27b']  # Refined
}  # Cell Type Markers in Mouse Format

CellMarkers_Hs = {cell: [gene.upper() for gene in CellMarkers_Mm[cell]]
                  for cell in CellMarkers_Mm}  # Cell Type Markers in Human Format

DictUpdateCellLabels = {
    'PC1_vent': 'PC',
    'SMC1_basic': 'SMC',  # SMC1_basic has transcripts that indicate immaturity
    'SMC2_art': 'SMC_art',
    'CD16+Mo': 'Ccr2+MP',
    'LYVE1+IGF1+MP': 'MP',
    'B': 'B_cells',
    'CD4+T_naive': 'T_cells',
    'EC1_cap': 'CapEC',
    'EC3_cap': 'CapECpI',
    'EC5_art': 'ArtEC',
    'EC6_ven': 'VeinEC',
    'EC7_endocardial': 'EndoEC',
    'EC8_ln': 'LymphEC',
    'FB3': 'FB',  # FB3 is less abundant in the left ventricule, more in the atria
    'FB4_activated': 'FBa',  # More abundant in LV
    'FB5': 'FB',  # Less abundant in the right atrium
    'Meso': 'Meso',
    'vCM1': 'CM',  # Mainly in the left ventricule
    'Adip1': 'Adip',  # Genes for PPAR pathway, metabolism of lipids and lipoproteins and lipolysis
    'NC1_glial': 'NC_glial',  # Neural cells with a gene program required for glia development and axon myelination
    'LYVE1+TIMD4+MP': 'MP',  # Positive for TIMD4 predicted to act on apoptotic cell clearance
    'MoMP': 'Monocyte_Macrophages',
    'DC': 'DC',
    'Mast': 'Mast',
    'FB1': 'FB',
    'CD8+T_trans': 'T_cells',
    'vCM4': 'CM',
    'NC2_glial_NGF+': 'NC_glial_NGF+',
}


def update_cell_labels(anndata: ad.AnnData,
                       cell_col: str,
                       key_added: str = 'annotation') -> ad.AnnData:
    """**Re-name cell type labels generated by celltypist**

    This function will rename the cell type labels returned by celltypist
    when using the Heart Model.
    :param anndata: anndata object previously analysed by Celltypist
    :param cell_col: .obs column with cell type labels
    :param key_added: .obs column where new labels are saved
    :return: anndata object
    """
    anndata.obs[key_added] = [DictUpdateCellLabels[cell] for cell in anndata.obs[cell_col]]
    return anndata


def automatic_annotation(anndata: ad.AnnData,
                         leiden_col: str,
                         model: str = 'Healthy_Adult_Heart.pkl',
                         key_added: str = 'autoAnnot',
                         majority: bool = True,
                         convert: bool = True,
                         update_label: bool = False,
                         key_updated: str = 'annotation',
                         verbose: bool = False) -> ad.AnnData:
    """**Automatic Annotation base on CellTypist Package**

    This function takes an anndata object and automatically annotate the clusters
    employing a model available for celltypist.

    :param anndata: anndata object
    :param leiden_col: .obs column with leiden / louvain clusters
    :param model: model to use for the prediction. Default Healthy Adult Heart (Human)
    :param key_added: .obs column name where to save the predicted cell types
    :param majority: majority voting for predictions (See CellTypist documentation). Default True
    :param convert: convert the gene format of the model. If a Human  model is used, then gene in mouse format
                    will be use and viceverse.
    :param update_label: add a .obs column with cell type labels updated to standard names. Default False
    :param key_updated: .obs column name where updated cell type labels are saved. To be used when update_labels is set
                        to True. Default False
    :param verbose: show information of the analysis steps
    :return: andata
    """
    logger.info('Automatic Annotation with CellTypist\n')

    # Make a copy // Heart Model is Human; Need to Uppercase Gene Names
    anndata_copy = anndata.copy()  # In case Mouse Data is Provided and Human Model

    steps = ['Setting-up', 'Predicting', 'Saving predictions', 'Updating labels']
    total_steps = len(steps) if update_label else len(steps) - 1

    with tqdm(total=total_steps, desc='Progress', disable=not verbose, colour='tomato') as pbar:
        # Get model
        pbar.set_description(steps.pop(0))
        model_loaded = celltypist.models.Model.load(model=model)
        if convert:
            model_loaded.convert()
        anndata_copy.X = anndata_copy.X.toarray()  # Leads to high memory usage
        pbar.update(1)

        # Do the prediction
        pbar.set_description(steps.pop(0))
        predictions_cells = celltypist.annotate(
            anndata_copy, model=model_loaded, majority_voting=majority,
            over_clustering=leiden_col)
        pbar.update(1)

        # Save predictions
        pbar.set_description(steps.pop(0))
        predictions_cells_adata = predictions_cells.to_adata()
        anndata_copy.obs['cell_type'] = predictions_cells_adata.obs.loc[
            anndata_copy.obs.index, "majority_voting"]
        anndata.obs[key_added] = anndata_copy.obs['cell_type']  # Transfer to original object
        pbar.update(1)

        if update_label:
            # Update labels
            pbar.set_description(steps.pop(0))
            update_cell_labels(anndata, key_added, key_updated)
            pbar.update(1)

    return anndata

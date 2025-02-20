#!/usr/bin/env python

"""
Description: Module with functions for running cell2location
and analysis spatial transcriptomics (Visium) data

Author: David Rodriguez Morales
Date Created: 31-01-2024
Date Modified: 01-08-24
Version: 2.0
Python Version: 3.11.8
"""

# Modules
import os
import warnings
from datetime import datetime

import numpy as np
import scanpy as sc
import pandas as pd
import polars
import anndata as ad
import bbknn
from scipy.sparse import block_diag, csr_matrix
from scipy import sparse
import scipy

import matplotlib.pyplot as plt
import matplotlib as mpl

from davidrScRNA import SCTransform
from davidrUtility import select_slide, iOn, set_plt_theme, config_verbose
from davidrPlotting import generate_violin_plot

# Global configuration
warnings.filterwarnings('ignore')  # silence scanpy that prints a lot of warnings
logger = config_verbose(True)
set_plt_theme()
iOn()


# Quality-Control
def ref_gene_selection(anndata: ad.AnnData,
                       cell_count: int = 5,
                       cell_fraction: float = 0.03,
                       expr_saving: float = 1.12) -> ad.AnnData:
    """
    Perform a permissive gene selection proposed and recommended by cell2location on the reference
    sc/snRNAseq data. Keeps markers of rare genes while removing most of the uninformative genes.
    Filter genes base on three parameters: 1) remove genes present in low number of cells, 2) remove
    genes only present in specific fraction of cells, and 3) include genes whose average expression is
    above the cutoff for the cells that fall in the previous two categories.

    :param anndata: reference anndata object
    :param cell_count: minimum number of cells that have to contain the gene
    :param cell_fraction: minimum fraction of cells that must contain the gene
    :param expr_saving: average expression cutoff for 'saving' the genes excluded previously
    :return: filtered anndata object
    """
    from cell2location.utils.filtering import filter_genes
    selected = filter_genes(anndata, cell_count_cutoff=cell_count, cell_percentage_cutoff2=cell_fraction,
                            nonz_mean_cutoff=expr_saving)
    return anndata[:, selected].copy()


def extract_signature_anndata(anndata: ad.AnnData) -> pd.DataFrame:
    """
    Extract reference cell types signatures as a pandas dataframe. Needed for the step 2
    of the cell2location pipeline (Mapping from sc/snRNA to Spatial Transcriptomics). This
    function takes the trained anndata object directly. As an alternative you can use
    ``extract_signature_load`` providing the path and filename to the trained anndata object
    :param anndata: trained anndata object
    :return: dataframe with signatures
    """
    # Export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in anndata.varm.keys():
        inf_aver = anndata.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                            for i in anndata.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = anndata.var[[f'means_per_cluster_mu_fg_{i}'
                                for i in anndata.uns['mod']['factor_names']]].copy()
    inf_aver.columns = anndata.uns['mod']['factor_names']
    return inf_aver


def extract_signature_load(path: str, filename: str) -> pd.DataFrame:
    """
    Extract reference cell types signatures as a pandas dataframe. Needed for the step 2
    of the cell2location pipeline (Mapping from sc/snRNA to Spatial Transcriptomics).
    :param path: path to folder with anndata object
    :param filename: name of the anndata object
    :return: dataframe with signatures
    """
    anndata = sc.read_h5ad(os.path.join(path, filename))
    # Export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in anndata.varm.keys():
        inf_aver = anndata.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                            for i in anndata.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = anndata.var[[f'means_per_cluster_mu_fg_{i}'
                                for i in anndata.uns['mod']['factor_names']]].copy()
    inf_aver.columns = anndata.uns['mod']['factor_names']
    return inf_aver


def estimate_signatures(anndata: ad.AnnData,
                        batch_col: str,
                        celltype_col: str,
                        fig_path: str,
                        object_path: str,
                        model_path: str,
                        filename_prefix: str,
                        epochs: int = 400,
                        gpu: bool = False,
                        categ_covar: list = None,
                        cont_covar: list = None) -> tuple:
    """**Step 1 of cell2location pipeline.**

    Estimate the reference cell type signatures. Unnormalised raw expression counts need to be provided.
    Several paramaters need to be specified for the model  including batch information and celltype annotation.
    Additionally, categorial and or continuous covariates can be specified (E.g., correct for using different Methods,
    technologies (scRNA, snRNA). Should not be biologically-relevant factors). Specify output folders for the QC plots,
    anndata object and model as well a filename_prefix

    :param anndata: reference anndata object
    :param batch_col: obs column with batch information (E.g., Sample)
    :param celltype_col: obs column with CellType annotation
    :param fig_path: figure output folder
    :param object_path: anndata folder
    :param model_path: output folder to save the model
    :param filename_prefix: prefix use for all the Plots, anndata, model output
    :param epochs: Default 400
    :param gpu: use GPU. Default False
    :param categ_covar: categorical covariates (Optional)
    :param cont_covar: continuous covariates (Optional)
    :return: tranined anndata, model
    """
    import cell2location
    from cell2location.models import RegressionModel

    # Prepare the anndata object for the regression model
    cell2location.models.RegressionModel.setup_anndata(adata=anndata,
                                                       batch_key=batch_col,
                                                       labels_key=celltype_col,
                                                       # Technical effects
                                                       categorical_covariate_keys=categ_covar,
                                                       continuous_covariate_keys=cont_covar
                                                       )
    # Create the regression model
    model = RegressionModel(anndata)
    model.view_anndata_setup()  # Visualise the model

    model.train(max_epochs=epochs, use_gpu=gpu, batch_size=10000)  # Train the model

    # Export the estimated cell abundance
    anndata = model.export_posterior(anndata, sample_kwargs={
        'num_samples': 1000,
        'batch_size': 10000,  # Modify this if too big for the GPU
        'use_gpu': gpu,
    })

    # Save the model
    model.save(os.path.join(model_path, filename_prefix + '_RefModel'), overwrite=True)

    # Save the anndata trained
    anndata.write(os.path.join(object_path, filename_prefix + '_Reftrained.h5ad'))

    # Save QC Plots in a Sub-Folder
    try:
        os.makedirs(os.path.join(fig_path, 'RefModel_QC'))
    except FileExistsError:
        pass
    plt.figure()
    model.plot_history(50)  # Plot the ELBO loss removing first 50 values
    plt.savefig(os.path.join(fig_path, 'RefModel_QC', filename_prefix + '_ModelHistory.svg'), bbox_inches='tight')
    plt.figure()
    model.plot_QC()
    plt.savefig(os.path.join(fig_path, 'RefModel_QC', filename_prefix + '_ReconstructionAccuracy.svg'),
                bbox_inches='tight')
    return anndata, model


def spatial_mapping(anndata: ad.AnnData,
                    df_signature: pd.DataFrame,
                    n_cells: int,
                    spatial_batch_key: str,
                    fig_path: str,
                    object_path: str,
                    model_path: str,
                    filename_prefix: str,
                    alpha: int = 20,
                    epochs: int = 30_000,
                    gpu: bool = True):
    """**Step 2 of cell2location pipeline.**

    Estimate the abundance of reference cell type in spatial data. Un-normalised spatial data, which needs to
    be pre-processed to remove mitochondrial genes must be provided. Additionally, a dataframe with the cell
    signatures (run ``estimate_signatures`` and ``extract_signature_anndata``). Two hyperparameters have to be
    provided:

    * expected number of cells in each spot (can be estimated from paired histology images),
    * detection alpha to improve accuracy and sensitivity on datasets with large technical variability in RNA sensitivity
      within the slide batch (Recommended 20 for high technical variability or 200 for low technical effects).

    This function will do a pre-process to keep only shared genes between the anndata and the signature table
    Indicate figure path, object path and model path as well as filename prefix to save the output of the training.
    The recommendation for training the model is 30,000 epochs. Less epoch might lead to an incomplate training

    :param anndata: anndata object with spatial transcriptomics data
    :param df_signature: reference cell type signatures
    :param n_cells: number of cells per location
    :param spatial_batch_key: obs column with sample information on the anndata
    :param alpha: regularisation of per-location normalisation. Default 20 (High technical variability)
    :param fig_path: figure path
    :param object_path: object path
    :param model_path: mode path
    :param filename_prefix: prefix for the files generated
    :param epochs: Default 30,000 (Should not be lower)
    :param gpu: use gpu for the training. Default False
    :return: trained anndata, model
    """
    import cell2location

    # Find shared genes for the spatial mapping
    intersect = np.intersect1d(anndata.var_names, df_signature.index)
    anndata = anndata[:, intersect].copy()
    df_signature = df_signature.loc[intersect, :].copy()

    # Create and train the model
    cell2location.models.Cell2location.setup_anndata(adata=anndata, batch_key=spatial_batch_key)

    model = cell2location.models.Cell2location(
        anndata, cell_state_df=df_signature,
        N_cells_per_location=n_cells,  # hyper-prior which can be estimated from paired histology
        detection_alpha=alpha,  # control normalisation of within-experiment variation in RNA detection
        # A_factors_per_location=4.0,  # expected number of cells per location
        # B_groups_per_location=2.0  # expected number of co-located cell type groups per location
    )

    model.view_anndata_setup()  # Visualise the model

    model.train(max_epochs=epochs,
                batch_size=None,  # Use the full data for training
                train_size=1,  # Use all data points
                use_gpu=gpu)  # Train the model

    # Export the estimated cell abundance
    anndata = model.export_posterior(anndata, sample_kwargs={
        'num_samples': 1000,
        'batch_size': model.adata.n_obs,  # Modify this if too big for the GPU
        'use_gpu': gpu,
    })

    # Save the model
    model.save(os.path.join(model_path, filename_prefix + '_SpaModel'), overwrite=True)

    # Save the anndata trained
    anndata.write(os.path.join(object_path, filename_prefix + '_Spatrained.h5ad'))

    # Save QC Plots in a Sub-Folder
    try:
        os.makedirs(os.path.join(fig_path, 'SpaModel_QC'))
    except FileExistsError:
        pass
    plt.figure()
    model.plot_history(1000)  # Plot the ELBO loss removing first 50 values
    plt.savefig(os.path.join(fig_path, 'SpaModel_QC', filename_prefix + '_ModelHistory.svg'), bbox_inches='tight')
    plt.figure()
    model.plot_QC()
    plt.savefig(os.path.join(fig_path, 'SpaModel_QC', filename_prefix + '_ReconstructionAccuracy.svg'),
                bbox_inches='tight')
    return anndata, model


def standard_quality_control_visium(anndata: ad.AnnData,
                                    sample_id: str,
                                    fig_path: str,
                                    obj_path: str = None,
                                    min_count: int = 2000,
                                    min_cell: int = 3,
                                    mouse: bool = True,
                                    target_sum: int = 10000,
                                    hvg_n: int = 3000,
                                    size_spa: float = 1.5,
                                    SCT: bool = True):
    """**Standard Quality control for Visium Data.** 
    
    The input is an unprocessed anndata object of a single sample. Generated for example, after using
    ``scanpy.read()``. Among the quality control steps applied by the function we have:

    * Remove low quality spots (Permisive filtering is recommended)
    * Remove lowly expressed or undetected genes
    * Remove Mitochondrial genes
    * Log-Normalisation of the Data
    
    The returned object will have 2 layers that include the raw counts and the log-normalised counts. 
    The latter will also be save in the .X attribute. 
    
    :param anndata: anndata containing visium data
    :param sample_id: ID for the Sample. Used for Title and filenames saved
    :param fig_path: path folder where figures are saved
    :param obj_path: path folder where the object is saved
    :param min_count: minimum number of counts per spot. Default 2000
    :param min_cell: minimum number of spots expressing a gene. Default 3
    :param mouse: whether the input data is mouse or not
    :param target_sum: normalise counts to target sum. If None is normalise to median 
    :param hvg_n: number of highly variable genes to extract. Default 3000
    :param size_spa: size of spots for plotting. Default 1.5
    :param SCT: perform SCT normalisation using R
    :return: anndata filtered
    """""

    print('\n# ', datetime.now(), ' - Standard Quality Control for Visium Data')
    print('\n# ', datetime.now(), ' - Setting up...')

    anndata = anndata.copy()  # Copy to not modify input anndata
    # Create a SubFolder for the QC Plots
    if os.path.isdir(os.path.join(fig_path, 'QualityControl')) is False:
        os.makedirs(os.path.join(fig_path, 'QualityControl'))
    qc_path = os.path.join(fig_path, 'QualityControl')

    # Correction for sc.read_visium()
    print('\n# ', datetime.now(), ' - Adjusting the data...')
    anndata.obsm['spatial'] = np.array(anndata.obsm['spatial'], dtype=float)
    anndata.obs['array_row'] = anndata.obs['array_row'].astype(int)
    anndata.obs['array_col'] = anndata.obs['array_col'].astype(int)
    anndata.var['GeneSymbol'] = anndata.var_names  # Add Gene Symbols to var
    anndata.var_names_make_unique()  # Make Var Names Unique

    # Remove Mitochondrial genes
    print('\n# ', datetime.now(), ' Removing Mitochondrial genes...')
    mt = 'MT-'  # Assume Human Data
    if mouse:
        mt = 'mt-'  # Change to mouse format
    print('\n# ', datetime.now(), ' - Computing metrics...')
    anndata.var['mt'] = anndata.var['GeneSymbol'].str.startswith(mt)
    sc.pp.calculate_qc_metrics(anndata, qc_vars=['mt'], inplace=True)

    anndata.var['mt'] = [gene.startswith(mt) for gene in anndata.var_names]
    # remove MT genes for spatial mapping (keeping their counts in the object)
    anndata.obsm['mt'] = anndata[:, anndata.var['mt'].values].X.toarray()
    anndata = anndata[:, ~anndata.var['mt'].values].copy()

    # Filter base on counts
    print('\n# ', datetime.now(), ' - Filtering spots...')
    sc.pp.filter_cells(anndata, min_counts=min_count)
    # sc.pp.filter_cells(anndata, max_counts=max_count)
    sc.pp.filter_genes(anndata, min_cells=min_cell)

    # Scanpy Normalisation
    print('\n# ', datetime.now(), ' - Normalising the data...')
    anndata.layers['counts'] = anndata.X.copy()
    sc.pp.normalize_total(anndata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(anndata)
    sc.pp.highly_variable_genes(anndata, flavor='seurat', n_top_genes=hvg_n)
    anndata.layers['logcounts'] = anndata.X.copy()

    if SCT:
        anndata.X = anndata.layers['counts'].copy()
        anndata = SCTransform(anndata, '/media/Storage/DavidR/tmp/')
        anndata.X = anndata.layers['SCT_norm'].copy()
    generate_violin_plot(anndata, f'Violin Plots - {sample_id} - PostQC',
                         qc_path, f'QC_ViolinPlots_{sample_id}_PostQC.svg')

    sc.pp.scale(anndata)
    sc.pp.pca(anndata, use_highly_variable=True)
    anndata.X = anndata.layers['logcounts'].copy()
    sc.pp.neighbors(anndata)
    sc.tl.umap(anndata)

    # UMAP - Metrics
    sc.pl.umap(anndata, cmap='viridis', color=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], show=False)
    plt.savefig(os.path.join(qc_path, f'UMAP_QC_{sample_id}.svg'), bbox_inches='tight')
    # Spatial - Metrics
    sc.pl.spatial(anndata, cmap='viridis', img_key='hires',
                  color=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], show=False, bw=True, size=size_spa)
    plt.savefig(os.path.join(qc_path, f'Spatial_QC_{sample_id}.svg'), bbox_inches='tight')
    if obj_path is not None:
        anndata.write(os.path.join(obj_path, f'Visium_{sample_id}.h5ad'))
    return anndata


def spatial_integration(visium: ad.AnnData,
                        batch_col: str,
                        radius: int = 120,
                        margin: float = 2.5,
                        ngs: int = 3,
                        res: float = 0.8,
                        h_dim: tuple = (512, 50),
                        knn_ng: int = 50,
                        n_hvg: int = 8000,
                        min_dist: float = 0.05,
                        spread: float = 2,
                        dist: int = 'manhattan'
                        ) -> ad.AnnData:
    """**Integrate Spatial Anndata Object**

    This function integrate an anndata object base on spatial transcriptomics data. The integration
    is base on the package `STAligner <https://staligner.readthedocs.io/en/latest/>`_

    :param visium: anndata object
    :param batch_col: .obs column with batch information
    :param radius: distance to consider arround a spot to compute the neighbors. For Visium 100 lead to 5-6 neighbors
    :param margin: how strong the batch integration should be. The grater the value the stronger it will be
    :param ngs:  number of neighbors to consider for the distance matrix
    :param res: resolution for the clustering
    :param h_dim: dimension of the hidden layer of the autoencoder from STALigner
    :param knn_ng: number of neighbors for the spatial graph
    :param n_hvg: number of highly variable genes to include per batch. A value around 8,000 is recommended because
                  individual batches will be concatenated base on the shared highly variable genes
    :param min_dist: minimum distance for the UMAP embeddings. See `Scanpy <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html>`_
    :param spread: spread parameter for the UMAP embeddings. See `Scanpy <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html>`_
    :param dist: distance metric use for the distance matrix generated with BBKNN
    :return: anndata object
    """
    import STAligner  # Requires a different environment
    # TODO Correct the fact that the count layer might not be present. allow the user to specify layer or use .X

    visium = visium.copy()
    batch_list, adj_list, sections_ids = [], [], []

    # Calculate Spatial Network
    for sample in visium.obs[batch_col].unique():
        anndata = select_slide(visium, sample, batch_col)
        anndata.X = anndata.layers['counts'].copy()

        STAligner.Cal_Spatial_Net(anndata, rad_cutoff=radius)

        # Normalise
        anndata = SCTransform(anndata, '/media/Storage/DavidR/tmp/')
        anndata.X = anndata.layers['SCT_norm'].copy()
        # sc.pp.normalize_total(anndata, target_sum=10_000)
        # sc.pp.log1p(anndata)

        sc.pp.highly_variable_genes(anndata, flavor='seurat_v3',
                                    n_top_genes=n_hvg)
        anndata = anndata[:, anndata.var.highly_variable].copy()
        adj_list.append(anndata.uns['adj'])
        batch_list.append(anndata)
        sections_ids.append(sample)

    # Concatenate Samples
    ad_concat = ad.concat(batch_list, label='slice_name', keys=sections_ids)
    ad_concat.obs['batch_name'] = ad_concat.obs['slice_name'].astype('category')

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(sections_ids)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

    ad_concat.uns['edgeList'] = np.nonzero(adj_concat)

    ad_concat = STAligner.train_STAligner(ad_concat,
                                          verbose=True,
                                          hidden_dims=h_dim,
                                          knn_neigh=knn_ng,
                                          margin=margin,  # Larger values more agressive correction
                                          )

    # Trasfer Integrated matrix
    visium.obsm['STAligner'] = ad_concat.obsm['STAligner']
    # visium.layers['STAGATE_ReX'] = ad_concat.obsm['STAGATE_ReX']

    bbknn.bbknn(visium, batch_key=batch_col, use_rep='STAligner',
                neighbors_within_batch=ngs, metric=dist)
    sc.tl.leiden(visium, random_state=666,
                 key_added="leiden_STAligner", resolution=res)
    sc.tl.umap(visium, random_state=666, min_dist=min_dist, spread=spread)
    return visium


# Spatial Communication
def holonet_common_lr(adata: ad.AnnData,
                      min_spot=0.05,
                      organism: str = 'mouse',
                      batch_col: str = 'sample') -> pd.DataFrame:
    """
    Filter a database of ligand-receptor pairs to keep only pairs common in
    multiple slides.
    :param adata: anndata object
    :param min_spot: minimum percentage of cell expressing the LR
    :param organism: organism (mouse or human)
    :param batch_col: obs column with batch information
    :return:
    """
    import HoloNet as hn
    from tqdm import tqdm
    interc_db, cofact_db, cplx_db = hn.pp.load_lr_df(human_or_mouse=organism)
    database = interc_db.copy()
    for sample in tqdm(adata.obs['sample'].unique(), desc='Slide '):
        sdata = select_slide(adata, sample, batch_col)
        df_lr = hn.pp.get_expressed_lr_df(interc_db, cplx_db, sdata,
                                          expressed_prop=min_spot)
        database = pd.merge(database, df_lr, how='inner')
    return database
#!/usr/bin/env python

"""
Description:  Module for Stats calculations and utility tools
for sc/snRNA and Spatial Visium Data

Author: David Rodriguez Morales
Date Created: 31-01-24
Date Modified: 02-08-24
Version: 2.0
Python Version: 3.11.8
"""

import os
import sys
import shutil
import urllib.request
import warnings
import itertools

from tqdm import tqdm
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import scanpy as sc
import scvelo as scv
import anndata as ad
import gseapy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind

# Global configuration
warnings.filterwarnings("ignore")  # Scanpy creates many warnings


# SetUp functions
def setup_paths(main_path: str, models: bool = False) -> tuple:
    """ **Get paths for the working directory**

    Run ``setup_wd`` before. This function returns the paths for the following working
    directory structure:

    * main_path
    * main_path/Raw_data
    * main_path/Results
    * main_path/Results/Objects
    * main_path/Results/Figures
    * main_path/Results/Tables
    * main_path/Results/Models (Optional)

    **The paths returned will be in the same order as stated above**

    :param main_path: path to folder environment.
    :param models: return path to model path (Optional)
    :return: input_path, result_path, object_path, figure_path,  table_path
    """
    input_path = os.path.join(main_path, 'Raw_Data/')
    result_path = os.path.join(main_path, 'Results/')
    figure_path = os.path.join(result_path, 'Figures/')
    object_path = os.path.join(result_path, 'Objects/')
    table_path = os.path.join(result_path, 'Tables/')

    if models:
        model_path = os.path.join(result_path, 'Models/')
        return input_path, result_path, object_path, figure_path, table_path, model_path

    return input_path, result_path, object_path, figure_path, table_path


def setup_wd(main_path, models: bool = False) -> None:
    """**Set up the working directory**

    The following structure will be implemented:
    * main_path
    * main_path/Raw_data
    * main_path/Results
    * main_path/Results/Objects
    * main_path/Results/Figures
    * main_path/Results/Tables
    * main_path/Results/Models (Optional)

    :param main_path: absolute path to a directory to save results
    :param models: create a folder called Models
    :return: None
    """
    os.makedirs(os.path.join(main_path, 'Raw_Data/'))
    os.makedirs(os.path.join(main_path, 'Results/Objects/'))
    os.makedirs(os.path.join(main_path, 'Results/Figures/'))
    os.makedirs(os.path.join(main_path, 'Results/Tables/'))
    if models:
        os.makedirs(main_path, 'Results/Models/')
    return


def free_memory():
    import gc
    import ctypes
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    return


def config_verbose(complete: bool, debug_mode: bool = False):
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    _logger = logging.getLogger(__name__)

    lvl = 3 if complete else 0
    lvl = 4 if debug_mode else lvl

    sc.settings.verbosity = lvl  # Deactivate scanpy verbosity
    scv.settings.verbosity = lvl
    _logger.setLevel(logging.INFO)
    logging.getLogger('celltypist').setLevel(logging.CRITICAL)  # for celltypist
    logging.getLogger('scvelo').setLevel(
        logging.CRITICAL)  # for scvelo  # TODO scvelo has a bug ? cannot deactivate warnings

    return _logger


logger = config_verbose(True)


# Utility functions
def generate_results_short(anndata: ad.AnnData,
                           key: str = 'rank_genes_groups') -> pd.DataFrame:
    """**Extract DGE results from anndata object**

    This function extract the DGE analysis results from the .uns attribute of an anndata object and
    returns a dataframe contaning 3 columns: Gene names, Adjusted P value and Logfoldchanges.

    :param anndata: anndata object
    :param key: .uns key with DGE results. Default rank_genes_groups
    :return: dataframe with DGE results
    """
    result = anndata.uns[key]
    groups = result['names'].dtype.names
    df_results = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group] for group in groups
         for key in ['names', 'pvals_adj', 'logfoldchanges']})
    return df_results


def AverageExpression(anndata: ad.AnnData,
                      group_by: str,
                      feature: list or str = None,
                      out_format: str = 'long',
                      layer: str = None) -> pd.DataFrame:
    """**Calculate Average Expression in AnnData Objects**

    This function calculates the average expression of a set of features grouping by one
    or several categories.

    :param anndata: anndata object
    :param group_by: .obs column name or list of names to group by
    :param feature: list of features of .var to use. (Default is **all genes**)
    :param out_format: format of the dataframe returned. This can be wide or long format. (Default  is **long**)
    :param layer: layer of the anndata to use. (Default uses **.X**)
    :return: panda DataFrame in long (or wide) format with average expression
    """
    import scipy.sparse
    # Set-up configuration
    if feature is not None:
        anndata = anndata[:, feature]  # Retain only the specified features
    if layer is not None:
        anndata.X = anndata.layers[layer].copy()  # Select the specified layer

    data = anndata.copy()
    data.X = scipy.sparse.csr_matrix.expm1(data.X) # Undo the log transformation for the mean

    # Check out_format specified
    assert out_format == 'wide' or out_format == 'long', f'{out_format} not recognize, try "long" or "wide"'

    # If a string is provided convert to list
    if isinstance(group_by, str):
        group_by = [group_by]

    # Group data by the specified values
    group_obs = anndata.obs.groupby(group_by, as_index=False)

    # Compute AverageExpression
    main_df = pd.DataFrame([])
    for group_name, df in group_obs:
        df_tmp = np.log1p(pd.DataFrame(data[df.index].X.mean(axis=0).T, columns=['expr']))  # Mean expr per gene in groupN
        df_tmp['gene'] = anndata[df.index].var_names  # Update with Gene names
        if type(group_name) is str:  # If only grouping by one category
            group_name = [group_name]
        for idx, name in enumerate(group_name):
            df_tmp['group' + str(idx)] = name.replace('-', '_')  # Update with metadata
        main_df = pd.concat([main_df, df_tmp], axis=0)
    main_df['expr'] = pd.to_numeric(main_df['expr'])  # Convert to numeric values

    # Move expr column to last position
    expr_col = main_df.pop('expr')
    main_df['expr'] = expr_col

    # Change to wide format
    if out_format == 'wide':
        main_df = pd.pivot_table(main_df, index='gene',
                                 columns=list(main_df.columns[main_df.columns.str.startswith('group')]),
                                 values='expr')
        if len(group_by) > 1:
            main_df.columns = main_df.columns.map('_'.join)

    return main_df


def ExtractExpression(anndata: ad.AnnData,
                      features: str,
                      groups: str = None,
                      out_format: str = 'long',
                      layer: str = None) -> pd.DataFrame:
    """ **Extract the expression from an anndata object**

    This function extract the expression from an anndata object and returns a dataframe. If layer
    is not specified the expression in .X will be extracted. Additionally, metadata from .obs can be added
    to the dataframe.

    :param anndata: anndta object
    :param groups: .obs metadata column to include in the dataframe
    :param features: var_names to include
    :param out_format: format of the dataframe (wide or long)
    :param layer: layer in the anndata object to extract the expression from
    :return: dataframe with expression values
    """
    # Set-up configuration
    if features is not None:
        anndata = anndata[:, features]  # Retain only the specified features
    if layer is not None:
        anndata.X = anndata.layers[layer].copy()  # Select the specified layer

    # Check out_format specified
    assert out_format == 'wide' or out_format == 'long', f'{out_format} not recognize, try "long" or "wide"'

    # Extract expression
    if isinstance(features, str):  # If just one feature is provided convert to list
        features = [features]

    table_expr = pd.DataFrame(anndata[:, features].X.toarray(),  # densify the matrix (Replace .A)
                              index=anndata.obs_names,
                              columns=features)
    # Add Metadata
    if groups is not None:
        if isinstance(groups, str):
            table_expr[groups] = anndata.obs[groups]  # One column
        else:
            for group in groups:  # Multiple columns
                table_expr[group] = anndata.obs[group]

    if out_format == 'long':
        table_expr = pd.melt(table_expr, id_vars=groups, var_name='genes', value_name='expr')

    return table_expr


def add_feature_metadata(adata: ad.AnnData,
                         database: Union[pd.DataFrame, str] = 'Uniprot',
                         path_to_pickle: str = None,
                         organism: str = 'mouse',
                         ) -> ad.AnnData:
    """
    Add metadata to the features using uniprot database

    :param adata: anndata object to annotate
    :param database: dataframe with the annotation. Each row contains information for one gene and gene names should
                    be set to be the index. If no pandas dataframe is provided a self-made will be used
    :param path_to_pickle: path to pickle file with the information
    :param organism: it accepts mouse or human
    :return: anndata with annotation added in the var attribute
    """

    adata = adata.copy()  # Not modify input

    filename = None
    if (path_to_pickle is None) and (database == 'Uniprot'):
        logger.info(f'Path to database not provided, downloading {organism} database...')

        if not os.path.isdir('.db'):
            os.makedirs('.db/')

        if organism == 'mouse':
            filename, header = urllib.request.urlretrieve(
                'https://github.com/davidrm-bio/BioData/raw/refs/heads/main/Uniprot_DataBase_Mice_September2024.pickle',
                f"./.db/uniprot_{organism}.pkl")
        elif organism == 'human':
            raise AssertionError('Not Implemented')
        else:
            raise AssertionError('Not Implemented')

    # Either read the pickle provided; or the downloaded one or take the input db
    path = filename if path_to_pickle is None else path_to_pickle
    db = database if path is None else pd.read_pickle(filename)

    if os.path.isdir('.db/'):
        logger.info(f'Removing tmp files generated...')
        shutil.rmtree('.db/')  # Remove the tmp dir

    # Extract the current features
    logger.info(f'Adding metadata to the features...')
    features = adata.var
    db_filt = db.iloc[db.index.isin(list(features.index)), :]  # Only take common genes

    # Set Common Column
    features['GeneName'] = features.index
    db_filt['GeneName'] = db_filt.index

    # Merge by GeneName
    new_features = pd.merge(features, db_filt, how='outer', on='GeneName')
    new_features.index = new_features['GeneName']
    new_features = new_features.reindex(index=adata.var_names)
    new_features[new_features[db_filt.columns].isna()] = 'No Info'  # Correct for missing data
    adata.var = new_features
    return adata


def generate_results(anndata: ad.AnnData,
                     key: str = 'rank_genes_groups',
                     add_uniprot_metadata: bool = False,
                     organism='mouse') -> pd.DataFrame:
    """**Extract DGE results from anndata object**

    This function extract the DGE analysis results from the .uns attribute of an anndata object and
    returns a dataframe contaning 3 columns: Gene names, Adjusted P value and Logfoldchanges.

    :param anndata: anndata object
    :param key: .uns key with DGE results. Default rank_genes_groups
    :param add_uniprot_metadata: add information on the genes from Uniprot
    :param organism: organism from where to extract information (mouse or human)
    :return: dataframe with DGE results
    """

    update_columns = {'names': 'GeneName',
                      'scores': 'wilcox_score',
                      # U1 from formula, higher absolute indicate lower p-value; High score indicate high expression
                      'pvals': 'pvals',
                      'group': 'group',
                      'logfoldchanges': 'log2FC',
                      'pvals_adj': 'padj',
                      'pct_nz_group': 'pts_group',
                      'pct_nz_reference': 'pts_ref'
                      }

    df_results = sc.get.rank_genes_groups_df(anndata, group=None, key=key)
    df_results.columns = [update_columns[col] for col in df_results.columns]

    if 'pts_ref' not in df_results.columns:
        result = anndata.uns[key]
        ref = result['params']['reference']
        pts_ref = result['pts'][ref]
        if 'group' in df_results and len(df_results.group.unique()) > 1:
            df_results['pts_ref'] = df_results['GeneName'].map(pts_ref)
        else:
            df_results['pts_ref'] = pts_ref.reindex(index=df_results.GeneName).tolist()

    if add_uniprot_metadata:
        anndata_copy = add_feature_metadata(anndata, organism=organism)
        meta_cols = ['GeneName', 'Pathway', 'Function', 'Tissue specificity',
                     'Subcellular location', 'Nucleus', 'Cytoplasm',
                     'Membrane', 'Secreted']
        meta_cols = anndata_copy.var[meta_cols]
        for col in meta_cols:
            df_results[col] = df_results['GeneName'].map(meta_cols[col])
    return df_results



def add_uniprot_annotation(dge: pd.DataFrame,
                           gene_col: str = 'names',
                           database_path: str = '/mnt/davidr/scStorage/DavidR/BioData/241203_Uniprot_SubLoc.xlsx',
                           is_mouse: bool = True):
    """**Add Uniprot Annotation to DGE Table**

    Add annotation about the subcellular localisation from `Uniprot database <https://www.uniprot.org/>`.
    Only annotation from proteins with a review status will be added. Unreview status will be marked.

    Nine columns will be added: the first 7 indicate if the protein is localised to specific regions (cell membrane,
    membrane, nucleus, cytoplasm, mitochondria, cell projection and secreted). The 8th column will indicate if the protein
    has other localisations or has an unreview status and the last column includes the annotation extracted
    from the uniprot database

    :param dge: pandas dataframe with DGEs
    :param gene_col: column with gene names (GeneSymbol)
    :param database_path: path to file containing the annotation status for each gene
    :param is_mouse: whether the input is mouse or human
    :return: dataframe with the annotation
    """

    assert isinstance(dge, pd.DataFrame), 'The input data needs to be a pandas dataframe'

    sheet_name = 'Mouse' if is_mouse else 'Human'
    db = pd.read_excel(database_path, sheet_name=sheet_name)

    dge['GeneName'] = dge[gene_col]

    # Correction ! --> Some genes might not be in Uniprot
    tmp = pd.DataFrame(dge[~dge[gene_col].isin(db['GeneName'].tolist())][gene_col].tolist(), columns=['GeneName'])
    tmp[['CellMembrane', 'Membrane', 'Nucleus',
         'Cytoplasm', 'Mitochondria', 'CellProjection',
         'Secreted', 'Alternative', 'UniprotAnnotation']] = ['-', '-', '-', '-', '-', '-', '-', 'No Information (Unreview status Uniprot)', '-']
    db = pd.concat([db, tmp])
    db = db[db['GeneName'].isin(dge['GeneName'])].drop_duplicates()

    assert len(db['GeneName'].unique())  == len(dge['GeneName'].unique()), 'Error, different number of genes'
    ndge = pd.merge(dge, db, on='GeneName')
    return ndge


# Statistics
def grouped_ttest(anndata: ad.AnnData,
                  cell_col: str = 'annotation',
                  cond_col: str = 'condition',
                  sample_col: str = 'sample',
                  key_added: str = 'grouped_ttest',
                  filtering: bool = True,
                  sig_cut: str = 0.05,
                  layer: str = None) -> ad.AnnData:
    """**Calculate grouped t-test**

    This function calculate a grouped t-test for all the genes in each group in *cell_col*. For each gene,
    the average expression per sample is employed for the test. If more than two conditions are available,
    the test will be applied to all possible combinations (for instance, for cond A, B and C; the grouped
    t-test will be computed for A-Vs-B; A-Vs-C and B-Vs-C). Results are saved as a dataframe in the
    .uns attribute.

    :param anndata: anndata object
    :param cell_col: .obs column name with the cell type annotation. Default 'annotation'
    :param cond_col: .obs column name with the conditions. Default 'condition'
    :param sample_col: .obs column name with the sample IDs. Default 'sample'
    :param key_added: key to use in .uns. Default 'grouped_ttest'
    :param filtering: whether to keep only significant genes or not. Default True
    :param sig_cut: significance threshold. Default 0.05
    :param layer: layer of the anndata object to use. Default None
    :return: anndata object with results in .uns attribute
    """
    if layer is not None:
        anndata.X = anndata.layers[layer].copy()  # Select the specified layer

    main_df = pd.DataFrame([])
    for cell in anndata.obs[cell_col].unique():
        subset = anndata[anndata.obs[cell_col] == cell]  # Select a cell type
        df_expr = AverageExpression(subset, [cell_col, cond_col, sample_col], layer=layer)  # Compute average expression

        cond_comb = [comb for comb in
                     itertools.combinations(anndata.obs[cond_col].unique(), 2)]  # Get all conditions combinations

        # Compute ttest for all possible combinations
        for comb in cond_comb:
            df_a = df_expr[df_expr['group1'] == comb[0]]
            df_b = df_expr[df_expr['group1'] == comb[1]]

            df_a_wide = df_a.pivot(index='gene', values='expr', columns='group2')
            df_b_wide = df_b.pivot(index='gene', values='expr', columns='group2')

            p_values = pd.DataFrame(df_a_wide.index, columns=['gene'])
            p_values['annotation'] = cell
            p_values['condition'] = '-Vs-'.join(comb)
            p_values['pval'] = pd.DataFrame(ttest_ind(df_a_wide, df_b_wide, axis=1)[1])

        main_df = pd.concat([main_df, p_values], axis=0)

    if filtering:
        main_df = main_df[main_df['pval'] < sig_cut]  # Select only significant genes

    anndata.uns[key_added] = main_df
    return anndata


def dge_conditions(anndata: ad.AnnData,
                   ref: str,
                   select_case: str or list = None,
                   annot_col: str = 'annotation',
                   test_col: str = 'condition',
                   clusternames: str or list = 'All',
                   method: str = 'wilcoxon',
                   pval_sig: float = 0.05,
                   lfc_cut: float = 0.25,
                   path: str = None,
                   preffix: str = '',
                   layer: str = None,
                   add_uniprot_metadata: bool = False,
                   organism: str = 'mouse'):
    """**Perform DGE Analysis on an Anndata Object**

    This function does a differential gene expression analysis comparing 2 conditions in each cell type (default).
    Specify the reference condition using the *ref* argument. In case more than 2 conditions are available, the referece
    condition will be compared to every alternative condition in each cell type. A single cell type or a group of cell
    types can be selected using the *clusternames* argument:

    * ``$ ... clustername='celltype' ...``  # Specify one celltype;
    * ``$ ... clustername=['celltype1', 'celltype2']...`` # Specify two celltypes;

    The wilcoxon rank sum test is employed (default) with the Benjamini-Hochberg correction for multiple testing. Other
    available methods include: 't-test', 't-test_overestim_var' and 'logreg'. Check ``scanpy.tl.rank_gene_groups``
    documentation for more information
    (`Scanpy <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html>`_)

    If a path is provided an Excel sheet will be generated. Three sheets are included:

    * List of all the genes, their p-adjusted value and LogFoldChange (ref Vs Conditions)
    * List of upregulated genes (same format as first sheet)
    * List of downsregulated genes (same format as first sheet)

    To filter up- and downregulated genes a significance threshold of 0.05 and a LogFoldChange cutoff of 0.25
    is used (default).

    :param anndata: anndata object
    :param ref: reference condition
    :param select_case: alternative condition. Provide one alternative condition only. Default None
    :param annot_col: .obs column with cell type annotations. The test will be done comparing conditions for All or the
                       specified cell types. Default 'annotation'
    :param test_col: .obs column with conditions. Default 'condition'
    :param clusternames: name of clusters in annot_col. Provide one or a list. Default 'All' (analysis on all celltypes)
    :param method: method for the DGE test. Default 'wilcoxon'
    :param pval_sig: significance cutoff for the p-adjusted. Default 0.05
    :param lfc_cut: LogFoldChange cutoff. Default 0.25
    :param path: path to save DGE Excel files. Default None
    :param preffix: preffix for the Excell file (Optional). Format DGE-Preffix-CellType-CondA_Vs_CondB.xlsx
    :param layer: layer of the anndata object to use. Default None
    :param add_uniprot_metadata: add metadata for genes extracted from Uniprot
    :param organism: organism (mouse or human)
    :return: anndata object with DGE analysis in .uns
    """

    def run_analysis(case,
                     ref,
                     adata,
                     annot_col,
                     clusternames,
                     path,
                     method,
                     filename_id):

        for cluster in clusternames:
            subset = adata[adata.obs[annot_col] == cluster]  # Subset to do the analysis in each cell type

            sc.tl.rank_genes_groups(subset, groupby=test_col, method=method,
                                    reference=ref, groups=[case], tie_correct=True,
                                    pts=True,
                                    key_added=f'rank_{cluster}_{ref}_Vs_{case}')
            adata.uns[f'rank_{cluster}_{ref}_Vs_{case}'] = subset.uns[f'rank_{cluster}_{ref}_Vs_{case}']

            if path is not None:
                tmpdf = generate_results(subset, f'rank_{cluster}_{ref}_Vs_{case}',
                                         add_uniprot_metadata=add_uniprot_metadata,
                                         organism=organism)  # Extract DGE analysis
                tmpdf_up = tmpdf[(tmpdf[case + '_p'] < pval_sig) & (tmpdf[case + '_l'] > lfc_cut)]
                tmpdf_down = tmpdf[(tmpdf[case + '_p'] < pval_sig) & (tmpdf[case + '_l'] < -lfc_cut)]

                with pd.ExcelWriter(os.path.join(path, f'DGE-{filename_id}-{cluster}-{ref}-Vs-{case}.xlsx')) as writer:
                    tmpdf.to_excel(writer, sheet_name='All-Genes', index=False)
                    tmpdf_up.to_excel(writer, sheet_name=f'Upreg_in_{case}', index=False)
                    tmpdf_down.to_excel(writer, sheet_name=f'Downreg_in_{case}', index=False)
        return adata

    # Global Set-Up
    anndata = anndata.copy()  # Not modify input
    if layer is not None:  # select the layer we want to use
        anndata.X = anndata.layers[layer].copy()

    # Select cell types to test
    clusternames = list(anndata.obs[annot_col].unique()) if clusternames == 'All' else clusternames

    # If a single cluster is provided, convert to list (for loop is used later)
    if isinstance(clusternames, str):  # clusternames can be 'example' or ['example1', 'example2']
        clusternames = [clusternames]

    if select_case is not None:
        # User can provide one or more conditions (In case of several)
        if not isinstance(select_case, list):
            groups_to_test = [select_case]
    else:
        # If nothing is provided, test against the others minus reference
        tmp_list = list(anndata.obs[test_col].unique())
        tmp_list.remove(ref)
        groups_to_test = tmp_list

    for case in groups_to_test:
        run_analysis(case=case,
                     ref=ref,
                     adata=anndata,
                     annot_col=annot_col,
                     clusternames=clusternames,
                     path=path,
                     method=method,
                     filename_id=preffix)

    return anndata


"""
def dge_conditions(anndata: ad.AnnData,
                   ref: str,
                   select_case: str or list = None,
                   annot_col: str = 'annotation',
                   test_col: str = 'condition',
                   clusternames: str or list = 'All',
                   method: str = 'wilcoxon',
                   pval_sig: float = 0.05,
                   lfc_cut: float = 0.25,
                   path: str = None,
                   preffix: str = '',
                   layer: str = None):
    

    def run_analysis(case, adata=anndata, annot_col=annot_col, clusternames=clusternames, path=path, method=method,
                     ref=ref, preffix=preffix):
        for cluster in clusternames:
            subset = adata[adata.obs[annot_col] == cluster]  # Subset to do the analysis in each cell type
            sc.tl.rank_genes_groups(subset, groupby=test_col, method=method,
                                    reference=ref, groups=[case], tie_correct=True,
                                    key_added=f'rank_{cluster}_{ref}_Vs_{case}')

            tmpdf = generate_results(subset, f'rank_{cluster}_{ref}_Vs_{case}')  # Extract DGE analysis
            tmpdf_up = tmpdf[(tmpdf[case + '_p'] < pval_sig) & (tmpdf[case + '_l'] > lfc_cut)]
            tmpdf_down = tmpdf[(tmpdf[case + '_p'] < pval_sig) & (tmpdf[case + '_l'] < -lfc_cut)]

            adata.uns[f'rank_{cluster}_{ref}_Vs_{case}'] = subset.uns[f'rank_{cluster}_{ref}_Vs_{case}']

            if path is not None:
                with pd.ExcelWriter(os.path.join(path, f'DGE-{preffix}-{cluster}-{ref}-Vs-{case}.xlsx')) as writer:
                    tmpdf.to_excel(writer, sheet_name='All-Genes', index=False)
                    tmpdf_up.to_excel(writer, sheet_name=f'Upreg_in_{case}', index=False)
                    tmpdf_down.to_excel(writer, sheet_name=f'Downreg_in_{case}', index=False)
        return adata

    anndata = anndata.copy()

    if layer is not None:  # select the layer we want to use
        anndata.X = anndata.layers[layer].copy()

    if clusternames == 'All':
        clusternames = list(anndata.obs[annot_col].unique())

    # If a single cluster is provided, convert to list (for loop is used later)
    if isinstance(clusternames, str):  # clusternames can be 'example' or ['example1', 'example2']
        clusternames = [clusternames]

    if select_case is not None:
        # In case we have more than one alternative condition
        # do DGE comparing ref to all alternative conditions
        tmp_list = list(anndata.obs[test_col].unique())
        tmp_list.remove(ref)

        for case in tmp_list:  # In case of more than 1 condition
            run_analysis(case, anndata, annot_col, clusternames, path, method, ref, preffix=preffix)
    else:
        run_analysis(select_case, anndata, annot_col, clusternames, path, method, ref, preffix=preffix)

    return anndata
"""


def custom_estimator(values):
    """
    Perform the mean over logcount data by undoing the log
    doing the mean and then logging the data again
    :param values: numpy array with log data
    :return: mean value
    """
    return np.log1p(np.mean(np.expm1(values)))



# Spatial Utility cell2location
def select_slide_old_version(adata: ad.AnnData,
                             s: str,
                             s_col: str = 'sample') -> ad.AnnData:
    """ **Subset a Spatial AnnData object**
    This function selects the data for one slide from the spatial anndata
    object. (Taken from cell2location tutorial)
    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in obs listing experiment name for each location
    """
    slid = adata[adata.obs[s_col].isin([s]), :]
    s_keys = list(slid.uns['spatial'].keys())
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]
    slid.uns['spatial'] = {s_spatial: slid.uns['spatial'][s_spatial]}
    return slid


def select_slide(adata: ad.AnnData,
                 s: str,
                 s_col: str = 'sample') -> ad.AnnData:
    """ **Subset a Spatial AnnData object**
    This function selects the data for one slide from the spatial anndata
    object.
    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in obs listing experiment name for each location
    """
    slid = adata[adata.obs[s_col].isin([s]), :].copy()
    s_keys = list(slid.uns['spatial'].keys())
    s_keys.remove(s)
    for val in s_keys:
        del slid.uns['spatial'][val]
    return slid


def get_surrounding(adata: ad.AnnData,
                    in_spot: int,
                    radius: float,
                    get_bcs=False) -> list:
    """Find the index positions that surrounds a position (index)
    :param adata: anndata object
    :param in_spot: index of the barcode
    :param get_bcs: return barcodes instead of index position
    :param radius: radius. Minimum of 100 for Visium
    :return: a list of surrounding indices
    """
    spot = adata.obsm['spatial'][in_spot]
    surrounding = []

    for i, sp in enumerate(adata.obsm['spatial']):
        distance = ((spot[0] - sp[0]) ** 2 + (spot[1] - sp[1]) ** 2) ** .5
        if distance <= radius:
            surrounding.append(i)
    if get_bcs:
        return adata.obs_names[surrounding]
    else:
        return surrounding


def add_smooth_kernel(adata: ad.AnnData,
                      layer_name: str = 'smooth_X',
                      bandwidth: int = 100,
                      multiple: bool = True) -> ad.AnnData:
    """
    Compute a smooth kernel, i.e, expression matrix is smooth
    :param adata: anndata object
    :param layer_name: name of the layer with smooth expression matrix
    :param bandwidth: radius (the greater, the more neighbors are considered)
    :param multiple: AnnData Object Contains Multiple Sample
    :return: anndata object with new layer
    """
    import liana

    adata = adata.copy()

    if multiple:
        smooth_x = pd.DataFrame([])
        for sample in tqdm(adata.obs['sample'].unique(), desc='Analysed samples :'):
            slid = select_slide(adata, sample, 'sample')
            liana.ut.spatial_neighbors(slid,
                                       bandwidth=bandwidth, cutoff=0.1,
                                       kernel='gaussian', set_diag=True,
                                       standardize=True)
            slid.X = slid.obsp['spatial_connectivities'].toarray().dot(slid.X.toarray())
            current_x = ad.AnnData.to_df(slid)
            smooth_x = pd.concat([smooth_x, current_x])
    else:
        liana.ut.spatial_neighbors(adata,
                                   bandwidth=bandwidth, cutoff=0.1,
                                   kernel='gaussian', set_diag=True,
                                   standardize=True)
        adata.X = adata.obsp['spatial_connectivities'].A.dot(adata.X.toarray())
        smooth_x = ad.AnnData.to_df(adata)

    smooth_x = smooth_x.reindex(index=adata.obs_names, columns=adata.var_names)
    adata.layers[layer_name] = csr_matrix(smooth_x.values, dtype=np.float32)
    return adata


# Utility Functions for Plotting
def generate_cmap(*args) -> LinearSegmentedColormap:
    """ **Generate a colormap**

    This functions returns a color map. Specify colors to set a gradient in the specified order.
    (1, 1, 1, 0) to set transparent
    :param args: colors, RGB or HexaCodes
    :return: custom cmap
    """
    colors = [col for col in args]
    return LinearSegmentedColormap.from_list('Custom', colors, N=256)


def get_subplot_shape(n_samples: int,
                      ncols: int) -> tuple:
    """
    Compute the number of rows and columns to use for defining the figure
    base on a desired number of samples and columns
    :param n_samples: number of samples to plot
    :param ncols: number of columns to plot
    :return: nrows, ncols, extras (extra subplots that should be hidden)
    """
    if n_samples < ncols:  # Correction
        ncols = n_samples  # Adjust plot if more cols than samples are specified
    nrows = int(np.ceil(n_samples / ncols))
    extras = nrows * ncols - n_samples  # For hiding empty subplots
    return nrows, ncols, extras


def axis_format(axis: plt.axis, txt: str = 'UMAP') -> None:
    """
    Formatting the axis for Embeddings
    :param axis: axis object
    :param txt: type of embedding
    :return:
    """
    axis.spines[['right', 'top']].set_visible(False)
    axis.set_xlabel(txt + '1', loc='left', fontsize=8, fontweight='bold')
    axis.set_ylabel(txt + '2', loc='bottom', fontsize=8, fontweight='bold')
    return


def remove_extra(extras: int, nrows: int, ncols: int, axs: plt.Axes) -> None:
    """
    Hide the last "extras" subplots
    :param extras: number of subplots to remove
    :param nrows: number of rows of the plot
    :param ncols: number of columns of the plot
    :param axs: axis object
    :return:
    """
    if extras == 0:
        return
    else:
        for check in range(nrows * ncols - extras, nrows * ncols):
            axs[check].set_visible(False)
        return

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


def set_plt_theme():
    # Scanpy Settings
    sc.settings.set_figure_params(dpi=200, dpi_save=150,
                                  facecolor='white', color_map='Reds',
                                  frameon=True, transparent=False)

    # Set global font sizes and styles
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'

    # Set title font size and style
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'

    # Hide top and right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Remove grid
    plt.rcParams['axes.grid'] = False

    # Set Font family
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
    plt.rcParams['text.usetex'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    #plt.rcParams['pdf.fonttype'] = 'none'

    return


set_plt_theme()


def get_hex_colormaps(colormap: str):
    """
    Get a list with Hexa IDs for a colormap
    :param colormap:
    :return:
    """
    cmap = plt.get_cmap(colormap)
    return [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]


def extended_tab20c(n_shades: int = 6) -> list:
    """
    Extends the colormap tab20c to more than 4 shades for a color
    :param n_shades:  number of shades
    :return: list of colors
    """
    # Base colors from the 'tab20' colormap
    base_colors = plt.cm.tab20c.colors
    extended_colors = []

    # Generate 6 shades per color
    for i in range(0, len(base_colors), 4):  # Go by pairs, as 'tab20' has pairs of each color
        main_color = base_colors[i]
        secondary_color = base_colors[i + 3]

        # Interpolate between main and secondary color
        for j in range(n_shades):
            # Linear interpolation between the main and secondary color
            interp = j / (n_shades - 1)
            color = [
                main_color[k] * (1 - interp) + secondary_color[k] * interp
                for k in range(3)
            ]
            extended_colors.append(color)
    return extended_colors



def extended_tab20(n_shades: int = 6) -> list:
    """
    Extends the colormap tab20c to more than 4 shades for a color
    :param n_shades:  number of shades
    :return: list of colors
    """
    # Base colors from the 'tab20' colormap
    base_colors = plt.cm.tab20.colors
    extended_colors = []

    # Generate 6 shades per color
    for i in range(0, len(base_colors), 2):  # Go by pairs, as 'tab20' has pairs of each color
        main_color = base_colors[i]
        secondary_color = base_colors[i + 1]

        # Interpolate between main and secondary color
        for j in range(n_shades):
            # Linear interpolation between the main and secondary color
            interp = j / (n_shades - 1)
            color = [
                main_color[k] * (1 - interp) + secondary_color[k] * interp
                for k in range(3)
            ]
            extended_colors.append(color)
    return extended_colors

def iOn():
    """Activate Interactive plotting (tkagg backed)"""
    plt.ion()
    mpl.use('TkAgg')
    return


def iOff():
    """Deactivate Interactive plotting (agg backed)"""
    plt.ioff()
    mpl.use('Agg')
    return


##### Generate the db from Uniprot
"""
import pandas as pd
from tqdm import  tqdm
import numpy as np
uniprot_mice = pd.read_excel('/mnt/davidr/scStorage/DavidR/Uniprot_Metadata_Mice2024_09_26.xlsx')

select_columns = ['Gene Names', 'Protein names', 'Organism', 'Gene Names (synonym)',
                  'Pathway', 'Function [CC]', 'Tissue specificity', 'Gene Ontology (biological process)',
                  'Subcellular location [CC]']

uniprot_mice = uniprot_mice[select_columns]



db = {}
for idx, row in tqdm(uniprot_mice.iterrows(), total = len(uniprot_mice)):
    genes = str(row['Gene Names']).split(' ')
    if len(genes) == 1 and genes[0] == 'nan':
        continue
    for gene in genes:
        db[gene] = {'Pathway': '', 'Function': '',
                    'Tissue specificity': '', 'Subcellular location': '',
                    'Nucleus': 'no', 'Cytoplasm': 'no',
                    'Membrane':'no', 'Secreted':'no',
                    }

        db[gene]['Pathway'] = str(row['Pathway']).replace('nan', 'No Information')
        db[gene]['Function'] = str(row['Function [CC]']).replace('nan', 'No Information')
        db[gene]['Tissue specificity'] = str(row['Tissue specificity']).replace('nan', 'No Information')
        db[gene]['Subcellular location'] = str(row['Subcellular location [CC]']).replace('nan', 'No Information')

        if 'nucleus' in db[gene]['Subcellular location'].lower():
            db[gene]['Nucleus'] = 'yes'
        if 'cytoplasm' in db[gene]['Subcellular location'].lower():
            db[gene]['Cytoplasm'] = 'yes'
        if 'membrane' in db[gene]['Subcellular location'].lower():
            db[gene]['Membrane'] = 'yes'
        if 'secreted' in db[gene]['Subcellular location'].lower():
            db[gene]['Secreted'] = 'yes'

df = pd.DataFrame(db).T

df.to_csv('/mnt/davidr/scStorage/DavidR/Uniprot_DataBase_Mice_September2024.csv')




"""

#!/usr/bin/env python

"""
Description:  

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""

#<editor-fold desc="Deconvolution of ST">
import warnings

import scanpy as sc
import cell2location
from cell2location.models import RegressionModel
from davidrSpatial import ref_gene_selection, extract_signature_anndata, spatial_mapping

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sc.settings.set_figure_params(dpi=150, facecolor='white', dpi_save=150, frameon=True, color_map='viridis')

main_path = '/home/drodriguez/Analysis_24_05_24/'
adata_ref = sc.read_h5ad('./snRNA_RefAging_240524.h5ad')
#</editor-fold>


#<editor-fold desc="Train snRNAseq">
adata_ref = ref_gene_selection(adata_ref, 3, 0.03, 1.12)

# Prepare the anndata object for the regression model
cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                                                   batch_key='sample',
                                                   labels_key='annotation',
                                                   # Technical effects
                                                   categorical_covariate_keys=['Experiment'], # Correct for this
                                                   layer='counts'
                                                   )
# Create the regression model
model = RegressionModel(adata_ref)
model.view_anndata_setup()  # Visualise the model

model.train(max_epochs=450, use_gpu=True, batch_size=2500)  # Train the model

plt.figure()
model.plot_history(50)  # Plot the ELBO loss removing first 50 values
model.save('RefModel_snRNA_Aging_24May/')

# Export the estimated cell abundance
adata_ref = model.export_posterior(adata_ref, sample_kwargs={
    'num_samples': 1000,
    'batch_size': 2500,  # Modify this if too big for the GPU
    'use_gpu': True,
})

model.plot_QC()
adata_ref.write('./Ref_snRNA_Aging_24May_trained.h5ad')


#<editor-fold desc="Integration of snRNA and  ST">
visium = sc.read_h5ad('/home/drodriguez/Spatial_v2/Mapping_020524/Visium_MouseHeart_OldYoungAMI_SCT_norm.h5ad')

# Remove Mt-genes (Not Informative)
# visium.var['mt'] = [gene.startswith('mt-') for gene in visium.var_names]
# visium.obsm['mt'] = visium[:, visium.var['mt'].values].X.toarray()
# visium = visium[:, ~visium.var['mt'].values].copy()

visium.X = visium.layers['SCT_counts'].copy()

young = visium[visium.obs['condition'].isin(['Young'])].copy()
old = visium[visium.obs['condition'].isin(['Old'])].copy()

df_aging = extract_signature_anndata(adata_ref)

old, mod_old = spatial_mapping(old,  # Visium Old Slide
                                  df_aging,  # Reference signature
                                  8,  # Number of cells per spot
                                  'sample',  # Obs col with sample ID
                                  './',  # Path to save QC plots
                                  './',  # Path to save H5AD file
                                  './',  # Path to save the model
                                  'Visium_Old_20_240524',  # Unique ID for filenames
                                  20,  # alpha hyperparameter
                                  30000,  # epochs
                                  True)  # use GPU

young, mod_young = spatial_mapping(young,  # Visium Slide
                                  df_aging,  # Reference signature
                                  8,  # Number of cells per spot
                                  'sample',  # Obs col with sample ID
                                  './',  # Path to save QC plots
                                  './',  # Path to save H5AD file
                                  './',  # Path to save the model
                                  'Visium_Young_200_240524',  # Unique ID for filenames
                                  200,  # alpha hyperparameter
                                  30000,  # epochs
                                  True)  # use GPU
#</editor-fold>

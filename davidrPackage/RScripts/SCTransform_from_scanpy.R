############################################
# Description:  Normalisation using SCT
#
# Author: David Rodriguez Morales
# Date Created: 27-05-2024
# Date Modified: 02-08-2024
# Version: 1.0
# R Version: 4.3.2
############################################

suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(data.table)))

option_list = list(
  make_option(c("--input"), type="character", default=NULL, 
              help="Absolute path to Anndata object (H5AD File)", metavar="character"),
  make_option(c("--out"), type="character", default=NULL, 
              help="Absolute path to the directory where the output files will be saved", 
              metavar="character")
)

opt_parser = OptionParser(usage = "usage: %prog [options] SCtransform", 
                          option_list=option_list)

opt = parse_args(opt_parser)




# Read h5ad as ScE
print('Reading anndata into R environment...')
adata <- readH5AD(paste0(opt$input, 'adata_to_seurat_tmp.h5ad'))

print('Converting anndata to seurat...')
Seu <- as.Seurat(adata, counts = "counts", data = 'counts')

print('SCTransform the counts...')
Seu <- Seurat::SCTransform(Seu, vst.flavor="v2", assay = 'originalexp', return.only.var.genes = F, min_cells=3)

print('Preparing to export normalise counts to python...')
norm_expr <- GetAssayData(Seu, assay = 'SCT', slot = 'data')
norm_expr <- as.data.frame(norm_expr)
norm_expr <- t(norm_expr)
print(norm_expr[1:3, 1:3])

print('Preparing to export corrected raw counts to python...')
raw_expr <- GetAssayData(Seu, assay = 'SCT', slot = 'counts')
raw_expr <- as.data.frame(raw_expr)
raw_expr <- t(raw_expr)
print(raw_expr[1:3, 1:3])

fwrite(norm_expr, paste0(opt$out, 'SCTransform_norm.csv'))
fwrite(raw_expr, paste0(opt$out, 'SCTransform_raw.csv'))

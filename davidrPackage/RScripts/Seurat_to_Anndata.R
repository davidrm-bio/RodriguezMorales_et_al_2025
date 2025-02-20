############################################
# Description:  Convert Seurat to Anndata
#
# Author: David Rodriguez Morales
# Date Created: 22-02-2024
# Date Modified: 02-08-2024
# Version: 1.0
# R Version: 4.3.2
############################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(anndata)))

option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="Absolute path to Seurta Object (RDS File)", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="./", 
              help="Absolute path to the directory where the Anndata object will be saved", 
              metavar="character"),
  make_option(c("-n", "--name"), type="character", default="./", 
              help="Unique name to add to the output files", 
              metavar="character")
)

opt_parser = OptionParser(usage = "usage: %prog [options]
Convert Seurat object to anndata object. 
    Please specify the absolute path to the Seurat object, the output directory and a unique name 
    which will be added to the output files. In total 3 files are generated: the anndata object, a 
    CSV with the PCA Embeddings and a CSV with the UMAP Embeddings", option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("Please provide the specified arguments", call.=FALSE)
} else if (length(args)< 3) {
  print_help(opt_parser)
  stop("Arguments missing")
}


# Set-up Seurat Object
Seu.obj <- readRDS(opt$file)
DefaultAssay(object=Seu.obj) <- 'RNA'


# Get UMAP embeddings & PCA embeddings
UMAPEmbeddings <- Seu.obj@reductions$umap
UMAPEmbeddings <- UMAPEmbeddings@cell.embeddings
write.csv(UMAPEmbeddings, file=paste0(opt$out, 'UMAPEmbeddings_', opt$name, '.csv'), row.names = T)

PCAEmbeddings <- Seu.obj@reductions$pca
PCAEmbeddings <- PCAEmbeddings@cell.embeddings
write.csv(PCAEmbeddings,file=paste0(opt$out, 'PCAEmbeddings_', opt$name, '.csv'), row.names = T)

# From Seurat to ScE 
tmp.seu <- Seurat::as.SingleCellExperiment(Seu.obj)

# From ScE to h5ad
zellkonverter::writeH5AD(tmp.seu, 
                         file = file=paste0(opt$out, 'Seurat_to_Anndata_', opt$name, '.csv'), 
                         X_name='logcounts')

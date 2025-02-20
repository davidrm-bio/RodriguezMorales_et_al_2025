# davidrPackage - Python, R and Bash scripts to analyse sc/snRNA and Spatial Transcriptomics

The modules of this package are:

* davidrScRNA
* davidrSpatial
* davidrPlotting
* davidrUtility
* davidrExperimental
* davidrScanpy

## davidrScRNA
This module contains functions to perform basic quality control and pre-processing of sc/snRNA transcriptomics.

## davidrSpatial
This module contains functions to analysis of  Spatial transcriptomics (Visium 10X).

## davidrPlotting
This module contains functions to generate different visualisation plots, both for scRNA and ST. The funcstions
build on scanpy functions.

## davidrUtility
This module contains functions to perform statistical tests or general functions and supplementary functions.

## davidrExperimental
This module contain some classes for adding statistical information to barplots

## davidrScanpy
This module contains funcstion from scanpy that have been modified.

## Supplementary Modules

Besides the main modules. Several RScripts can also be found, which are employed by
several functions in the python modules to use implementations from Seurat.

### Notes
To use this package run conda develop `/path/to/folder/davidrPackage`
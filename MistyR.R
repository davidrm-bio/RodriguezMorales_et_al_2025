library(zellkonverter)
library(mistyR)
library(tidyverse)
library(SingleCellExperiment)
library(distances)

path <- '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis/Objects/'

files <- grep('h5ad', list.files(path, full.names = T), value = T)
file.names <- c('Old_1', 'Old_2', 'Old_3', 'Old_4', 'Old_5', 'Young_1', 'Young_2', 'Young_3', 'Young_4', 'Young_5')

counter = 1
importances <- list()
for (file in files) {
  print(file.names[counter])
  # Load AnnData Object
  sce <- readH5AD(file)

  # Get Spatial Coordinates
  coords <- as_tibble(reducedDim(sce, 'spatial'))
  colnames(coords) <- c('x', 'y')

  # Calculate radius
  geom_dist <- as.matrix(distances(as.data.frame(coords)))
  dist_nn <- apply(geom_dist, 1, function(x) (sort(x)[2]))
  juxta_radius <- ceiling(mean(dist_nn + sd(dist_nn)))

  dist_en <- apply(geom_dist, 1, function(x) (sort(x)[8]))
  paraview_radius <- ceiling(mean(dist_en + sd(dist_en)))

  # Get c2l abundances
  c2l <- as_tibble(reducedDim(sce, 'c2l'))

  # Correct the colnames
  colnames(c2l) <- c('Adip', 'ArtEC', 'B_cells', 'CM', 'CapEC', 'EndoEC', 'Fibroblasts', 'Fibro_activ', 'LymphEC', 'MP', 'Epi_cells', 'Ccr2_MP', 'Pericytes', 'SMC', 'T_cells', 'VeinEC')

  # Create misty views
  heart_views <- create_initial_view(c2l) %>%
    add_juxtaview(coords, neighbor.thr = juxta_radius, verbose = T) %>%   # Juxta View is the Hexamers
    add_paraview(coords, l = paraview_radius, verbose = T, zoi = juxta_radius)  # Para View - Juxtaview

  # Run Misty
  #new.path <- paste0(path,'tmp_misty_v2/', '_', file.names[counter] )
  new.path <- paste0(path, 'tmp_misty_v3/', '_', file.names[counter])
  dir.create(new.path)

  run_misty(heart_views, new.path)
  misty_results <- collect_results(new.path)

  df <- misty_results$importances.aggregated
  df['sample'] <- file.names[counter]
  importances[[counter]] <- df
  counter = counter + 1
}

df <- do.call(rbind, importances)
write_csv(df, '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis/importances_samples_c2l_v3.csv')


# Get improvements stats
counter <- 1
r2.values <- list()
for (file in file.names) {
  print(file)
  #new.path <- paste0(path,'tmp_misty_v2/', '_',file )
  new.path <- paste0(path, 'tmp_misty_v3/', '_', file)

  misty_results <- collect_results(new.path)

  df.r2 <- misty_results$improvements.stats
  df.r2['sample'] <- file
  r2.values[[counter]] <- df.r2
  counter <- counter + 1
}

df.r2 <- do.call(rbind, r2.values)
write_csv(df.r2, '/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Figures/4_FunctionalAnalysis/mistyR_Analysis/importances_samples_c2l_r2vals_v3.csv')


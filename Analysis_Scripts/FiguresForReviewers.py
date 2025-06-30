#!/usr/bin/env python

"""
Description:  

Author: David Rodriguez Morales
Date Created: 
Python Version: 3.11.8
"""
# <editor-fold desc="Sep-Up">
import davidrUtility
import davidrPlotting
import davidrScRNA
import davidrScanpy
import davidrConfiguration

from pathlib import Path
import os
import scanpy as sc

from matplotlib.patches import PathPatch
import seaborn as sns
import matplotlib.pyplot as plt
import davidrScanpy
import gseapy as gp
import pandas as pd

results_path = Path('/mnt/davidr/scStorage/DavidR/Spatial/Visium/Results/Revision/')
object_path = results_path / 'Objects'
figure_path = results_path / 'FiguresForReviewers' 
table_path = results_path / 'Tables'
#local_path = '/Volumes/scStorage/DavidR/Spatial/Visium/Results/Revision/Objects/Vidal_plus_TabulaMuris_Harmony.h5ad'
#adata = sc.read_h5ad(local_path)

adata = sc.read_h5ad(object_path / 'Vidal_plus_TabulaMuris_Harmony.h5ad')
aging = sc.read_h5ad('/media/Storage/DavidR/Objects_Submission/Visium_YoungOld_AgingProject_28August_Cleaned.h5ad')
aging.obs['condition'] = aging.obs['condition'].map({'Young': '3m', 'Old': '18m'})

# </editor-fold>

########################################################################################################################
# - Figure for Reviewer - Sex
########################################################################################################################

# <editor-fold desc="FigB. Umap of Integrated object">
davidrPlotting.pl_umap(adata, 'annotation', labels='annotation',
                       figsize=(10, 6.67), ncols=1, title='Annotation',
                       labels_fontproporties={'size': 15}, legend_fontsize=15,
                       path=figure_path, filename='FigureReviewerSex_UMAP_Annotation.svg')
# </editor-fold>


# <editor-fold desc="FigD. Barplots - Circadian Genes">
colors= {'3m':'sandybrown', '18m':'royalblue'}

adata = adata[adata.obs['age'].isin(['3m', '18m'])]
genes = ['Per1', 'Usp2']
df = davidrUtility.ExtractExpression(adata, groups=['sex', 'age'], features=genes)
for gene in genes:
    tdf = df[df.genes == gene]
    fig, axs = plt.subplots(1, 1, figsize=(3.6, 4.8))
    bp = sns.barplot(tdf, x='sex', hue='age', y='expr', palette=colors, capsize=.1)
    bp.set_ylabel('LogMean(nUMI)')
    bp.set_xlabel('')
    bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold', fontsize=15)
    sns.move_legend(bp, loc='upper right', title='Condition', title_fontproperties={'weight': 'bold'}, frameon=False,
                    ncols=2,
                    bbox_to_anchor=(1.1, 1))
    bp.set_title(f'{gene}')
    plt.savefig(figure_path / f'FigureReviewerSex_Barplot_{gene}_MaleFemale_PseudoBulk.svg', bbox_inches='tight')


male = adata[(adata.obs.sex == 'male') & (adata.obs.age.isin(['3m', '18m']))].copy()
female = adata[(adata.obs.sex == 'female') & (adata.obs.age.isin(['3m', '18m']))].copy()

sc.tl.rank_genes_groups(male, groupby='age', method='wilcoxon', tie_correct=True)
sc.tl.rank_genes_groups(female, groupby='age', method='wilcoxon', tie_correct=True)

tmp = sc.get.rank_genes_groups_df(male, group='18m').set_index('names')
print(tmp.loc[['Per1', 'Usp2'],'pvals_adj'])
tmp = sc.get.rank_genes_groups_df(female, group='18m').set_index('names')
print(tmp.loc[['Per1', 'Usp2'],'pvals_adj'])
# </editor-fold>

# <editor-fold desc="FigD. Barplots - Proinflammatory and Sirpa Genes">
colors= {'3m':'sandybrown', '18m':'royalblue'}
genes = ['Sirpa', 'Cd209g', 'Ccl8', 'Fgr']

mp = adata[adata.obs.annotation.isin(['MP_resident', 'MP_recruit'])]
df_mp = davidrUtility.ExtractExpression(mp, groups=['sex', 'age'], features=genes)

for gene in genes:
    tdf = df_mp[df_mp.genes == gene]
    fig, axs = plt.subplots(1, 1, figsize=(3.6, 4.8))
    bp = sns.barplot(tdf, x='sex', hue='age', y='expr', palette=colors, capsize=.1)
    bp.set_ylabel('LogMean(nUMI)')
    bp.set_xlabel('')
    bp.set_xticklabels(bp.get_xticklabels(), fontweight='bold', fontsize=15)
    sns.move_legend(bp, loc='upper right', title='Condition', title_fontproperties={'weight': 'bold'}, frameon=False,
                    ncols=2,
                    bbox_to_anchor=(1.1, 1))
    bp.set_title(f'{gene} in MP')
    plt.savefig(figure_path / f'FigureReviewerSex_Barplot_{gene}_MaleFemale_MP.svg', bbox_inches='tight')


male = mp[mp.obs.sex == 'male'].copy()
female = mp[mp.obs.sex == 'female'].copy()

sc.tl.rank_genes_groups(male, groupby='age', method='wilcoxon', tie_correct=True)
sc.tl.rank_genes_groups(female, groupby='age', method='wilcoxon', tie_correct=True)

tmp = sc.get.rank_genes_groups_df(male, group='18m').set_index('names')
print(tmp.loc[genes,'pvals_adj'])
tmp = sc.get.rank_genes_groups_df(female, group='18m').set_index('names')
print(tmp.loc[genes,'pvals_adj'])
# </editor-fold>


########################################################################################################################
# - Figure for Reviewer - TAC
########################################################################################################################


# Fig A. TAC Adventitial - Total CellEvent normalised on DAPI count only and
# Cell Event splitting by CellType, normalised on DAPI Count

# Fig B. TAC Interstitital - Total CellEvent normalised on DAPI count only and
# Cell Event splitting by CellType, normalised on DAPI Count


########################################################################################################################
# - Figure for Reviewer - Arteries/Lymphatics/Veins
########################################################################################################################

# <editor-fold desc="Dotplot Senescence Vessels+Condition">
# Generate the Dotplot
aging.obs['vessel+cond'] = aging.obs['vessels'].astype(str) + '_' + aging.obs['condition'].astype(str)
adata = aging[aging.obs.vessels.isin(['Arteries', 'Veins', 'Lymphatics', 'nonVasc'])]
axs = sc.pl.dotplot(adata, groupby='vessel+cond', swap_axes=True,
                    var_names=['senescence'],
                    show=False, cmap='Reds',
                    colorbar_title='Scaled Mean score\n in group', figsize=(6, 2))
# Layout
ax = axs['mainplot_ax']  # Get Main axs
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)

# Get Current Figure
fig = plt.gcf()
# Add a subplot of top to add the brackets
pos = ax.get_position()
top_ax = fig.add_axes([pos.x0, pos.y1, pos.width, 0.1])  # Adjust height as needed

# Set the categories
labels = ['Arteries', 'Veins', 'Lymphatics', 'nonVasc']
bracket_positions = [(0, 2), (2, 4), (4, 6), (6, 8)]  # Tuples of (start, end) indices of clusters to group

# Add brackets on top for the groups
for (i, (x_start, x_end)) in enumerate(bracket_positions):
    path = davidrUtility.create_bracket(x_start, x_end)
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
ax.set_xticklabels([txt.get_text().split('_')[-1] for txt in ax.get_xticklabels()], fontweight='bold', rotation=75,
                   ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
plt.savefig(os.path.join(figure_path, 'Dotplot_SenescenceScore_VesselsAging.svg'), bbox_inches='tight')
# </editor-fold>

########################################################################################################################
# - Figure for Reviewer - DGE RV/LV Young and Old
########################################################################################################################

# <editor-fold desc="DGE Overall">
aging.obs['ventricles'] = aging.obs['AnatomicRegion'].map({'BG':'BG', 'LVi':'LV',
                                                           'LVm':'LV', 'LVo':'LV',
                                                           'RV':'RV', 'SEP':'SEP'})
davidrScanpy.rank_genes_groups(aging, groupby='ventricles', method='wilcoxon',
                               tie_correct=True, reference='LV', groups=['RV'], layer='logcounts')

table = sc.get.rank_genes_groups_df(aging, group=None)
table_up = table[(table.pvals_adj < 0.05) & (table.logfoldchanges > 0.25)]
table_down = table[(table.pvals_adj < 0.05) & (table.logfoldchanges < -0.25)]

table_combined = pd.DataFrame([list(table_up.names), list(table_down.names)], index=['UpRV', 'DownRV']).T

table_combined.to_excel(figure_path / 'DGE_LV_vs_RV.xlsx', index=False)

res_up = gp.enrichr(list(table_up.names), gene_sets='GO_Biological_Process_2023').results
res_down = gp.enrichr(list(table_down.names), gene_sets='GO_Biological_Process_2023').results

res_up = res_up[res_up['Adjusted P-value'] < 0.05]
res_down = res_down[res_down['Adjusted P-value'] < 0.05]
res_up['state'] = 'Up in RV'
res_down['state'] = 'Down in RV'
res = pd.concat([res_up, res_down])
res.Term = res.Term.str.split(" \(GO").str[0]
davidrPlotting.split_bar_gsea(res, 'Term', 'Combined Score',
                              'state', 'Up in RV',
                              path=figure_path, filename='SplitBar_GSEA_RV_vs_LV.svg')
# </editor-fold>

# <editor-fold desc="DGE Young/Old">
aging.obs['ventricles'] = aging.obs['AnatomicRegion'].map({'BG':'BG', 'LVi':'LV',
                                                           'LVm':'LV', 'LVo':'LV',
                                                           'RV':'RV', 'SEP':'SEP'})
aging_3m = aging[aging.obs.condition == '3m']
aging_18m = aging[aging.obs.condition == '18m']

# Test 3m slides
davidrScanpy.rank_genes_groups(aging_3m, groupby='ventricles', method='wilcoxon',
                               tie_correct=True, reference='LV', groups=['RV'], layer='logcounts')
table_3m = sc.get.rank_genes_groups_df(aging_3m, group=None)

# Test 18m slides
davidrScanpy.rank_genes_groups(aging_18m, groupby='ventricles', method='wilcoxon',
                               tie_correct=True, reference='LV', groups=['RV'], layer='logcounts')
table_18m = sc.get.rank_genes_groups_df(aging_18m, group=None)


table_main = pd.DataFrame()
for names, table in [('3m', table_3m), ('18m', table_18m)]:
    table_up = table[(table.pvals_adj < 0.05) & (table.logfoldchanges > 0.25)]
    table_down = table[(table.pvals_adj < 0.05) & (table.logfoldchanges < -0.25)]

    df_up = pd.DataFrame({f'UpRV_{names}': table_up.names.values})
    df_down = pd.DataFrame({f'DownRV_{names}': table_down.names.values})

    table_main = pd.concat([table_main, df_up, df_down], axis=1)

table_main.to_excel(figure_path / 'DGE_LV_vs_RV_SPlittingCondition.xlsx', index=False)
# </editor-fold>


########################################################################################################################
# - Check correlation
########################################################################################################################

import polars
path = Path('/media/Storage/DavidR')
import matplotlib.colors


old_mu = polars.read_excel(path / 'Visium_Old_mu.xlsx')
old_mu = old_mu.to_pandas()
old_mu.set_index('__UNNAMED__0', inplace=True)

old_data = polars.read_excel(path / 'Visium_Old_data.xlsx')
old_data = old_data.to_pandas()
old_data.set_index('__UNNAMED__0', inplace=True)

young_mu = polars.read_excel(path / 'Visium_Young_mu.xlsx')
young_mu = young_mu.to_pandas()
young_mu.set_index('__UNNAMED__0', inplace=True)

young_data = polars.read_excel(path / 'Visium_Young_data.xlsx')
young_data = young_data.to_pandas()
young_data.set_index('__UNNAMED__0', inplace=True)


fig, axs = plt.subplots(1, 1, figsize=(7, 6))
x = np.log10(old_data.values.flatten() + 1)
y = np.log10(old_mu.values.flatten() + 1)
# Compute Pearson correlation
corr = np.corrcoef(x, y)[0, 1]
# Plot 2D histogram
plt.hist2d(x, y, bins=50, norm=matplotlib.colors.LogNorm(), cmap='viridis')
plt.colorbar(label='log10(Counts)')
# Add correlation as text
plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
axs.set_xlabel('Observed Data')
axs.set_ylabel('Posterior expected Data')
axs.set_title('Reconstruction Accuracy\n18m Condition')
plt.savefig(figure_path / 'ReconstructionAccuracy_VisiumOld.svg', bbox_inches='tight')



fig, axs = plt.subplots(1, 1, figsize=(7, 6))
x = np.log10(young_data.values.flatten() + 1)
y = np.log10(young_mu.values.flatten() + 1)
# Compute Pearson correlation
corr = np.corrcoef(x, y)[0, 1]
# Plot 2D histogram
plt.hist2d(x, y, bins=50, norm=matplotlib.colors.LogNorm(), cmap='viridis')
plt.colorbar(label='log10(Counts)')
# Add correlation as text
plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
axs.set_xlabel('Observed Data')
axs.set_ylabel('Posterior expected Data')
axs.set_title('Reconstruction Accuracy\n3m Condition')
plt.savefig(figure_path / 'ReconstructionAccuracy_VisiumYoung.svg', bbox_inches='tight')

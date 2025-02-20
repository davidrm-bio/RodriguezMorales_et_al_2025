#!/usr/bin/env python

"""
Description: Correction of Scanpy Functions

Author: David Rodriguez Morales
Date Created: 11 - 02 - 25
Python Version: 3.11.8
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse
from pandas.api.types import is_numeric_dtype
from scanpy.plotting._anndata import _prepare_dataframe

from typing import TYPE_CHECKING
from scanpy.plotting._baseplot_class import BasePlot
from matplotlib import pyplot as plt
from scanpy._compat import  old_positionals
from scanpy.plotting._docs import doc_common_plot_args, doc_show_save_ax, doc_vboundnorm
from scanpy import logging as logg
from scanpy._settings import settings
from scanpy._utils import _doc_params, _empty
from scanpy.plotting._utils import (
    check_colornorm,
    fix_kwds,
    make_grid_spec,
    savefig_or_show,
)
from scanpy.plotting._baseplot_class import doc_common_groupby_plot_args

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal, Self

    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

    from scanpy._utils import Empty
    from scanpy.plotting._utils import ColorLike, _AxesSubplot


########################################################################################################################
#
#     CORRECTION FOR THE DOTPLOT CLASS TO CONSIDER LOGCOUNTS
#
########################################################################################################################

@_doc_params(common_plot_args=doc_common_plot_args)
class DotPlot(BasePlot):
    """\
    Allows the visualization of two values that are encoded as
    dot size and color. The size usually represents the fraction
    of cells (obs) that have a non-zero value for genes (var).

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all data belongs to a single category.

    .. note::
       A gene is considered expressed if the expression value in the `adata` (or
       `adata.raw`) is above the specified threshold which is zero by default.

    An example of dotplot usage is to visualize, for multiple marker genes,
    the mean value and the percentage of cells expressing the gene
    across multiple clusters.

    Parameters
    ----------
    {common_plot_args}
    title
        Title for the figure
    expression_cutoff
        Expression cutoff that is used for binarizing the gene expression and
        determining the fraction of cells expressing given genes. A gene is
        expressed only if the expression value is greater than this threshold.
    mean_only_expressed
        If True, gene expression is averaged only over the cells
        expressing the given genes.
    standard_scale
        Whether or not to standardize that dimension between 0 and 1,
        meaning for each variable or group,
        subtract the minimum and divide each by its maximum.
    kwds
        Are passed to :func:`matplotlib.pyplot.scatter`.

    See also
    --------
    :func:`~scanpy.pl.dotplot`: Simpler way to call DotPlot but with less options.
    :func:`~scanpy.pl.rank_genes_groups_dotplot`: to plot marker
        genes identified using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------

    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
    >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels').show()

    Using var_names as dict:

    >>> markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
    >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels').show()

    """
    DEFAULT_SAVE_PREFIX = "dotplot_"

    # default style parameters
    DEFAULT_COLORMAP = "Reds"
    DEFAULT_COLOR_ON = "dot"
    DEFAULT_DOT_MAX = None
    DEFAULT_DOT_MIN = None
    DEFAULT_SMALLEST_DOT = 0.0
    DEFAULT_LARGEST_DOT = 200.0
    DEFAULT_DOT_EDGECOLOR = "black"
    DEFAULT_DOT_EDGELW = 0.2
    DEFAULT_SIZE_EXPONENT = 1.5

    # default legend parameters
    DEFAULT_SIZE_LEGEND_TITLE = "Fraction of cells\nin group (%)"
    DEFAULT_COLOR_LEGEND_TITLE = "Mean expression\nin group"
    DEFAULT_LEGENDS_WIDTH = 1.5  # inches
    DEFAULT_PLOT_X_PADDING = 0.8  # a unit is the distance between two x-axis ticks
    DEFAULT_PLOT_Y_PADDING = 1.0  # a unit is the distance between two y-axis ticks

    @old_positionals(
        "use_raw",
        "log",
        "num_categories",
        "categories_order",
        "title",
        "figsize",
        "gene_symbols",
        "var_group_positions",
        "var_group_labels",
        "var_group_rotation",
        "layer",
        "expression_cutoff",
        "mean_only_expressed",
        "standard_scale",
        "dot_color_df",
        "dot_size_df",
        "ax",
        "vmin",
        "vmax",
        "vcenter",
        "norm",
    )
    def __init__(self,
                 adata: AnnData,
                 var_names: _VarNames | Mapping[str, _VarNames],
                 groupby: str | Sequence[str],
                 *,
                 use_raw: bool | None = None,
                 log: bool = False,
                 num_categories: int = 7,
                 categories_order: Sequence[str] | None = None,
                 title: str | None = None,
                 figsize: tuple[float, float] | None = None,
                 gene_symbols: str | None = None,
                 var_group_positions: Sequence[tuple[int, int]] | None = None,
                 var_group_labels: Sequence[str] | None = None,
                 var_group_rotation: float | None = None,
                 layer: str | None = None,
                 expression_cutoff: float = 0.0,
                 mean_only_expressed: bool = False,
                 standard_scale: Literal["var", "group"] | None = None,
                 dot_color_df: pd.DataFrame | None = None,
                 dot_size_df: pd.DataFrame | None = None,
                 ax: _AxesSubplot | None = None,
                 vmin: float | None = None,
                 vmax: float | None = None,
                 vcenter: float | None = None,
                 norm: Normalize | None = None,
                 logcounts: bool = True,
                 **kwds) -> None:
        BasePlot.__init__(self,
                          adata,
                          var_names,
                          groupby,
                          use_raw=use_raw,
                          log=log,
                          num_categories=num_categories,
                          categories_order=categories_order,
                          title=title,
                          figsize=figsize,
                          gene_symbols=gene_symbols,
                          var_group_positions=var_group_positions,
                          var_group_labels=var_group_labels,
                          var_group_rotation=var_group_rotation,
                          layer=layer,
                          ax=ax,
                          vmin=vmin,
                          vmax=vmax,
                          vcenter=vcenter,
                          norm=norm,
                          **kwds,
                          )

        # Prepare the plotting dataframe
        var_names = [var_names] if isinstance(var_names, str) else var_names
        adata_view = adata[:, adata.var_names.isin(var_names)]
        adata_view._sanitize()
        categories, obs_matrix = _prepare_dataframe(adata_view, var_names=var_names, groupby=groupby, layer=layer)

        # 1. compute fraction of cells having value > expression_cutoff
        obs_bool = obs_matrix > expression_cutoff

        if dot_size_df is None:
            dot_size_df = (
                    obs_bool.groupby(level=0, observed=True).sum() /
                    obs_bool.groupby(level=0, observed=True).count()
            )

        # 2. compute mean expression value
        if dot_color_df is None:
            if logcounts:  # Correction in case logcounts are provided
                obs_matrix = np.expm1(obs_matrix)
                if mean_only_expressed:
                    dot_color_df = np.log1p((
                        obs_matrix.mask(~obs_bool).groupby(level=0, observed=True).mean().fillna(0)
                    ))
                else:
                    dot_color_df = np.log1p(obs_matrix.groupby(level=0, observed=True).mean())

            # Scale the data
            if standard_scale == "group":
                dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
                dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
            elif standard_scale == "var":
                dot_color_df -= dot_color_df.min(0)
                dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
            elif standard_scale is None:
                pass
            else:
                logger.warning('Unknown type for standard_scale, ignored')
        else:
            assert dot_color_df.shape != dot_size_df.shape, 'The dot_color_df and dot_size_df have different shape'

            # Correction in case of duplicate genes
            unique_var_names, unique_idx = np.unique(
                dot_color_df.columns, return_index=True
            )
            if len(unique_var_names) != len(self.var_names):
                dot_color_df = dot_color_df.iloc[:, unique_idx]

            dot_color_df = dot_color_df.loc[dot_size_df.index][dot_size_df.columns]

        # Save in self the dot_color_df and seld.dot_size_df
        self.dot_color_df, self.dot_size_df = (
            df.loc[
                categories_order if categories_order is not None else self.categories  # Remove self if it does not work
            ]
            for df in (dot_color_df, dot_size_df)
        )

        # Save standard_scale argument
        self.standard_scale = standard_scale

        # Set default style parameters
        self.cmap = self.DEFAULT_COLORMAP
        self.dot_max = self.DEFAULT_DOT_MAX
        self.dot_min = self.DEFAULT_DOT_MIN
        self.smallest_dot = self.DEFAULT_SMALLEST_DOT
        self.largest_dot = self.DEFAULT_LARGEST_DOT
        self.color_on = self.DEFAULT_COLOR_ON
        self.size_exponent = self.DEFAULT_SIZE_EXPONENT
        self.grid = False
        self.plot_x_padding = self.DEFAULT_PLOT_X_PADDING
        self.plot_y_padding = self.DEFAULT_PLOT_Y_PADDING

        self.dot_edge_color = self.DEFAULT_DOT_EDGECOLOR
        self.dot_edge_lw = self.DEFAULT_DOT_EDGELW

        # set legend defaults
        self.color_legend_title = self.DEFAULT_COLOR_LEGEND_TITLE
        self.size_title = self.DEFAULT_SIZE_LEGEND_TITLE
        self.legends_width = self.DEFAULT_LEGENDS_WIDTH
        self.show_size_legend = True
        self.show_colorbar = True
        return

    @old_positionals(
        "cmap",
        "color_on",
        "dot_max",
        "dot_min",
        "smallest_dot",
        "largest_dot",
        "dot_edge_color",
        "dot_edge_lw",
        "size_exponent",
        "grid",
        "x_padding",
        "y_padding",
    )
    def style(
            self,
            *,
            cmap: Colormap | str | None | Empty = _empty,
            color_on: Literal["dot", "square"] | Empty = _empty,
            dot_max: float | None | Empty = _empty,
            dot_min: float | None | Empty = _empty,
            smallest_dot: float | Empty = _empty,
            largest_dot: float | Empty = _empty,
            dot_edge_color: ColorLike | None | Empty = _empty,
            dot_edge_lw: float | None | Empty = _empty,
            size_exponent: float | Empty = _empty,
            grid: bool | Empty = _empty,
            x_padding: float | Empty = _empty,
            y_padding: float | Empty = _empty,
    ) -> Self:
        r"""\
        Modifies plot visual parameters

        Parameters
        ----------
        cmap
            String denoting matplotlib color map.
        color_on
            By default the color map is applied to the color of the ``"dot"``.
            Optionally, the colormap can be applied to a ``"square"`` behind the dot,
            in which case the dot is transparent and only the edge is shown.
        dot_max
            If ``None``, the maximum dot size is set to the maximum fraction value found (e.g. 0.6).
            If given, the value should be a number between 0 and 1.
            All fractions larger than dot_max are clipped to this value.
        dot_min
            If ``None``, the minimum dot size is set to 0.
            If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        smallest_dot
            All expression fractions with `dot_min` are plotted with this size.
        largest_dot
            All expression fractions with `dot_max` are plotted with this size.
        dot_edge_color
            Dot edge color.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means that
            the edge color is white for darker colors and black for lighter background square colors.
        dot_edge_lw
            Dot edge line width.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means a line width of 1.5.
        size_exponent
            Dot size is computed as:
            fraction  ** size exponent and afterwards scaled to match the
            `smallest_dot` and `largest_dot` size parameters.
            Using a different size exponent changes the relative sizes of the dots
            to each other.
        grid
            Set to true to show grid lines. By default grid lines are not shown.
            Further configuration of the grid lines can be achieved directly on the
            returned ax.
        x_padding
            Space between the plot left/right borders and the dots center. A unit
            is the distance between the x ticks. Only applied when color_on = dot
        y_padding
            Space between the plot top/bottom borders and the dots center. A unit is
            the distance between the y ticks. Only applied when color_on = dot

        Returns
        -------
        :class:`~scanpy.pl.DotPlot`

        Examples
        -------

        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']

        Change color map and apply it to the square behind the dot

        >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels') \
        ...     .style(cmap='RdBu_r', color_on='square').show()

        Add edge to dots and plot a grid

        >>> sc.pl.DotPlot(adata, markers, groupby='bulk_labels') \
        ...     .style(dot_edge_color='black', dot_edge_lw=1, grid=True) \
        ...     .show()
        """
        super().style(cmap=cmap)

        if dot_max is not _empty:
            self.dot_max = dot_max
        if dot_min is not _empty:
            self.dot_min = dot_min
        if smallest_dot is not _empty:
            self.smallest_dot = smallest_dot
        if largest_dot is not _empty:
            self.largest_dot = largest_dot
        if color_on is not _empty:
            self.color_on = color_on
        if size_exponent is not _empty:
            self.size_exponent = size_exponent
        if dot_edge_color is not _empty:
            self.dot_edge_color = dot_edge_color
        if dot_edge_lw is not _empty:
            self.dot_edge_lw = dot_edge_lw
        if grid is not _empty:
            self.grid = grid
        if x_padding is not _empty:
            self.plot_x_padding = x_padding
        if y_padding is not _empty:
            self.plot_y_padding = y_padding

        return self

    def legend(
            self,
            *,
            show: bool | None = True,
            show_size_legend: bool | None = True,
            show_colorbar: bool | None = True,
            size_title: str | None = DEFAULT_SIZE_LEGEND_TITLE,
            colorbar_title: str | None = DEFAULT_COLOR_LEGEND_TITLE,
            width: float | None = DEFAULT_LEGENDS_WIDTH,
    ) -> Self:
        """\
        Configures dot size and the colorbar legends

        Parameters
        ----------
        show
            Set to `False` to hide the default plot of the legends. This sets the
            legend width to zero, which will result in a wider main plot.
        show_size_legend
            Set to `False` to hide the dot size legend
        show_colorbar
            Set to `False` to hide the colorbar legend
        size_title
            Title for the dot size legend. Use '\\n' to add line breaks. Appears on top
            of dot sizes
        colorbar_title
            Title for the color bar. Use '\\n' to add line breaks. Appears on top of the
            color bar
        width
            Width of the legends area. The unit is the same as in matplotlib (inches).

        Returns
        -------
        :class:`~scanpy.pl.DotPlot`

        Examples
        --------

        Set color bar title:

        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> markers = {'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}
        >>> dp = sc.pl.DotPlot(adata, markers, groupby='bulk_labels')
        >>> dp.legend(colorbar_title='log(UMI counts + 1)').show()
        """

        if not show:
            # turn of legends by setting width to 0
            self.legends_width = 0
        else:
            self.color_legend_title = colorbar_title
            self.size_title = size_title
            self.legends_width = width
            self.show_size_legend = show_size_legend
            self.show_colorbar = show_colorbar
        return self

    def _plot_size_legend(self, size_legend_ax: Axes):
        # for the dot size legend, use step between dot_max and dot_min
        # based on how different they are.
        diff = self.dot_max - self.dot_min
        if 0.3 < diff <= 0.6:
            step = 0.1
        elif diff <= 0.3:
            step = 0.05
        else:
            step = 0.2
        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        size_range = np.arange(self.dot_max, self.dot_min, step * -1)[::-1]
        if self.dot_min != 0 or self.dot_max != 1:
            dot_range = self.dot_max - self.dot_min
            size_values = (size_range - self.dot_min) / dot_range
        else:
            size_values = size_range

        size = size_values ** self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x * 100), decimals=0).astype(int)}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        size_legend_ax.tick_params(
            axis="y", left=False, labelleft=False, labelright=False
        )

        # remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(visible=False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    def _plot_legend(self, legend_ax, return_ax_dict, normalize):
        # to maintain the fixed height size of the legends, a
        # spacer of variable height is added at the bottom.
        # The structure for the legends is:
        # first row: variable space to keep the other rows of
        #            the same size (avoid stretching)
        # second row: legend for dot size
        # third row: spacer to avoid color and size legend titles to overlap
        # fourth row: colorbar

        cbar_legend_height = self.min_figure_height * 0.08
        size_legend_height = self.min_figure_height * 0.27
        spacer_height = self.min_figure_height * 0.3

        height_ratios = [
            self.height - size_legend_height - cbar_legend_height - spacer_height,
            size_legend_height,
            spacer_height,
            cbar_legend_height,
        ]
        fig, legend_gs = make_grid_spec(
            legend_ax, nrows=4, ncols=1, height_ratios=height_ratios
        )

        if self.show_size_legend:
            size_legend_ax = fig.add_subplot(legend_gs[1])
            self._plot_size_legend(size_legend_ax)
            return_ax_dict["size_legend_ax"] = size_legend_ax

        if self.show_colorbar:
            color_legend_ax = fig.add_subplot(legend_gs[3])

            self._plot_colorbar(color_legend_ax, normalize)
            return_ax_dict["color_legend_ax"] = color_legend_ax

    def _mainplot(self, ax: Axes):
        # work on a copy of the dataframes. This is to avoid changes
        # on the original data frames after repetitive calls to the
        # DotPlot object, for example once with swap_axes and other without

        _color_df = self.dot_color_df.copy()
        _size_df = self.dot_size_df.copy()
        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
            _size_df = _size_df.iloc[:, self.var_names_idx_order]

        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
            _size_df = _size_df.loc[self.categories_order, :]

        if self.are_axes_swapped:
            _size_df = _size_df.T
            _color_df = _color_df.T
        self.cmap = self.kwds.pop("cmap", self.cmap)

        normalize, dot_min, dot_max = self._dotplot(
            _size_df,
            _color_df,
            ax,
            cmap=self.cmap,
            color_on=self.color_on,
            dot_max=self.dot_max,
            dot_min=self.dot_min,
            standard_scale=self.standard_scale,
            edge_color=self.dot_edge_color,
            edge_lw=self.dot_edge_lw,
            smallest_dot=self.smallest_dot,
            largest_dot=self.largest_dot,
            size_exponent=self.size_exponent,
            grid=self.grid,
            x_padding=self.plot_x_padding,
            y_padding=self.plot_y_padding,
            vmin=self.vboundnorm.vmin,
            vmax=self.vboundnorm.vmax,
            vcenter=self.vboundnorm.vcenter,
            norm=self.vboundnorm.norm,
            **self.kwds,
        )

        self.dot_min, self.dot_max = dot_min, dot_max
        return normalize

    @staticmethod
    def _dotplot(
            dot_size: pd.DataFrame,
            dot_color: pd.DataFrame,
            dot_ax: Axes,
            *,
            cmap: Colormap | str | None,
            color_on: Literal["dot", "square"],
            dot_max: float | None,
            dot_min: float | None,
            standard_scale: Literal["var", "group"] | None,
            smallest_dot: float,
            largest_dot: float,
            size_exponent: float,
            edge_color: ColorLike | None,
            edge_lw: float | None,
            grid: bool,
            x_padding: float,
            y_padding: float,
            vmin: float | None,
            vmax: float | None,
            vcenter: float | None,
            norm: Normalize | None,
            **kwds,
    ):
        """\
        Makes a *dot plot* given two data frames, one containing
        the doc size and other containing the dot color. The indices and
        columns of the data frame are used to label the output image

        The dots are plotted using :func:`matplotlib.pyplot.scatter`. Thus, additional
        arguments can be passed.

        Parameters
        ----------
        dot_size
            Data frame containing the dot_size.
        dot_color
            Data frame containing the dot_color, should have the same,
            shape, columns and indices as dot_size.
        dot_ax
            matplotlib axis
        cmap
        color_on
        dot_max
        dot_min
        standard_scale
        smallest_dot
        edge_color
        edge_lw
        grid
        x_padding
        y_padding
            See `style`
        kwds
            Are passed to :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        matplotlib.colors.Normalize, dot_min, dot_max

        """
        assert dot_size.shape == dot_color.shape, (
            "please check that dot_size " "and dot_color dataframes have the same shape"
        )

        assert list(dot_size.index) == list(dot_color.index), (
            "please check that dot_size " "and dot_color dataframes have the same index"
        )

        assert list(dot_size.columns) == list(dot_color.columns), (
            "please check that the dot_size "
            "and dot_color dataframes have the same columns"
        )

        if standard_scale == "group":
            dot_color = dot_color.sub(dot_color.min(1), axis=0)
            dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
        elif standard_scale == "var":
            dot_color -= dot_color.min(0)
            dot_color = (dot_color / dot_color.max(0)).fillna(0)
        elif standard_scale is None:
            pass

        # make scatter plot in which
        # x = var_names
        # y = groupby category
        # size = fraction
        # color = mean expression

        # +0.5 in y and x to set the dot center at 0.5 multiples
        # this facilitates dendrogram and totals alignment for
        # matrixplot, dotplot and stackec_violin using the same coordinates.
        y, x = np.indices(dot_color.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        frac = dot_size.values.flatten()
        mean_flat = dot_color.values.flatten()
        cmap = plt.get_cmap(cmap)
        if dot_max is None:
            dot_max = np.ceil(max(frac) * 10) / 10
        else:
            if dot_max < 0 or dot_max > 1:
                raise ValueError("`dot_max` value has to be between 0 and 1")
        if dot_min is None:
            dot_min = 0
        else:
            if dot_min < 0 or dot_min > 1:
                raise ValueError("`dot_min` value has to be between 0 and 1")

        if dot_min != 0 or dot_max != 1:
            # clip frac between dot_min and  dot_max
            frac = np.clip(frac, dot_min, dot_max)
            old_range = dot_max - dot_min
            # re-scale frac between 0 and 1
            frac = (frac - dot_min) / old_range

        size = frac ** size_exponent
        # rescale size to match smallest_dot and largest_dot
        size = size * (largest_dot - smallest_dot) + smallest_dot
        normalize = check_colornorm(vmin, vmax, vcenter, norm)

        if color_on == "square":
            if edge_color is None:
                from seaborn.utils import relative_luminance

                # use either black or white for the edge color
                # depending on the luminance of the background
                # square color
                edge_color = []
                for color_value in cmap(normalize(mean_flat)):
                    lum = relative_luminance(color_value)
                    edge_color.append(".15" if lum > 0.408 else "w")

            edge_lw = 1.5 if edge_lw is None else edge_lw

            # first make a heatmap similar to `sc.pl.matrixplot`
            # (squares with the asigned colormap). Circles will be plotted
            # on top
            dot_ax.pcolor(dot_color.values, cmap=cmap, norm=normalize)
            for axis in ["top", "bottom", "left", "right"]:
                dot_ax.spines[axis].set_linewidth(1.5)
            kwds = fix_kwds(
                kwds,
                s=size,
                linewidth=edge_lw,
                facecolor="none",
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)
        else:
            edge_color = "none" if edge_color is None else edge_color
            edge_lw = 0.0 if edge_lw is None else edge_lw

            color = cmap(normalize(mean_flat))
            kwds = fix_kwds(
                kwds,
                s=size,
                color=color,
                linewidth=edge_lw,
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)

        y_ticks = np.arange(dot_color.shape[0]) + 0.5
        dot_ax.set_yticks(y_ticks)
        dot_ax.set_yticklabels(
            [dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False
        )

        x_ticks = np.arange(dot_color.shape[1]) + 0.5
        dot_ax.set_xticks(x_ticks)
        dot_ax.set_xticklabels(
            [dot_color.columns[idx] for idx, _ in enumerate(x_ticks)],
            rotation=90,
            ha="center",
            minor=False,
        )
        dot_ax.tick_params(axis="both", labelsize="small")
        dot_ax.grid(visible=False)

        # to be consistent with the heatmap plot, is better to
        # invert the order of the y-axis, such that the first group is on
        # top
        dot_ax.set_ylim(dot_color.shape[0], 0)
        dot_ax.set_xlim(0, dot_color.shape[1])

        if color_on == "dot":
            # add padding to the x and y lims when the color is not in the square
            # default y range goes from 0.5 to num cols + 0.5
            # and default x range goes from 0.5 to num rows + 0.5, thus
            # the padding needs to be corrected.
            x_padding = x_padding - 0.5
            y_padding = y_padding - 0.5
            dot_ax.set_ylim(dot_color.shape[0] + y_padding, -y_padding)

            dot_ax.set_xlim(-x_padding, dot_color.shape[1] + x_padding)

        if grid:
            dot_ax.grid(visible=True, color="gray", linewidth=0.1)
            dot_ax.set_axisbelow(True)

        return normalize, dot_min, dot_max


@old_positionals(
    "use_raw",
    "log",
    "num_categories",
    "expression_cutoff",
    "mean_only_expressed",
    "cmap",
    "dot_max",
    "dot_min",
    "standard_scale",
    "smallest_dot",
    "title",
    "colorbar_title",
    "size_title",
    # No need to have backwards compat for > 16 positional parameters
)
@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def dotplot(
        adata: AnnData,
        var_names: _VarNames | Mapping[str, _VarNames],
        groupby: str | Sequence[str],
        *,
        use_raw: bool | None = None,
        log: bool = False,
        num_categories: int = 7,
        categories_order: Sequence[str] | None = None,
        expression_cutoff: float = 0.0,
        mean_only_expressed: bool = False,
        standard_scale: Literal["var", "group"] | None = None,
        title: str | None = None,
        colorbar_title: str | None = "LogMean expression\nin group",
        size_title: str | None = "Fraction of cells\nin group (%)",
        figsize: tuple[float, float] | None = None,
        dendrogram: bool | str = False,
        gene_symbols: str | None = None,
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        layer: str | None = None,
        swap_axes: bool | None = False,
        dot_color_df: pd.DataFrame | None = None,
        show: bool | None = None,
        save: str | bool | None = None,
        ax: _AxesSubplot | None = None,
        return_fig: bool | None = False,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        # Style parameters
        cmap: Colormap | str | None = 'Reds',
        dot_max: float | None = None,
        dot_min: float | None = None,
        smallest_dot: float = 0.0,
        logcounts=True,
        **kwds,
) -> DotPlot | dict | None:
    """\
    Makes a *dot plot* of the expression values of `var_names`.

    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all data belongs to a single category.

    .. note::
       A gene is considered expressed if the expression value in the `adata` (or
       `adata.raw`) is above the specified threshold which is zero by default.

    An example of dotplot usage is to visualize, for multiple marker genes,
    the mean value and the percentage of cells expressing the gene
    across  multiple clusters.

    This function provides a convenient interface to the :class:`~scanpy.pl.DotPlot`
    class. If you need more flexibility, you should use :class:`~scanpy.pl.DotPlot`
    directly.

    Parameters
    ----------
    {common_plot_args}
    {groupby_plots_args}
    size_title
        Title for the size legend. New line character (\\n) can be used.
    expression_cutoff
        Expression cutoff that is used for binarizing the gene expression and
        determining the fraction of cells expressing given genes. A gene is
        expressed only if the expression value is greater than this threshold.
    mean_only_expressed
        If True, gene expression is averaged only over the cells
        expressing the given genes.
    dot_max
        If ``None``, the maximum dot size is set to the maximum fraction value found
        (e.g. 0.6). If given, the value should be a number between 0 and 1.
        All fractions larger than dot_max are clipped to this value.
    dot_min
        If ``None``, the minimum dot size is set to 0. If given,
        the value should be a number between 0 and 1.
        All fractions smaller than dot_min are clipped to this value.
    smallest_dot
        All expression levels with `dot_min` are plotted with this size.
    {show_save_ax}
    {vminmax}
    kwds
        Are passed to :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object,
    else if `show` is false, return axes dict

    See also
    --------
    :class:`~scanpy.pl.DotPlot`: The DotPlot class can be used to to control
        several visual parameters not available in this function.
    :func:`~scanpy.pl.rank_genes_groups_dotplot`: to plot marker genes
        identified using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------

    Create a dot plot using the given markers and the PBMC example dataset grouped by
    the category 'bulk_labels'.

    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.dotplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Using var_names as dict:

    .. plot::
        :context: close-figs

        markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
        sc.pl.dotplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Get DotPlot object for fine tuning

    .. plot::
        :context: close-figs

        dp = sc.pl.dotplot(adata, markers, 'bulk_labels', return_fig=True)
        dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()

    The axes used can be obtained using the get_axes() method

    .. code-block:: python

        axes_dict = dp.get_axes()
        print(axes_dict)

    """

    # backwards compatibility: previous version of dotplot used `color_map`
    # instead of `cmap`
    cmap = kwds.pop("color_map", cmap)

    dp = DotPlot(
        adata,
        var_names,
        groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        categories_order=categories_order,
        expression_cutoff=expression_cutoff,
        mean_only_expressed=mean_only_expressed,
        standard_scale=standard_scale,
        title=title,
        figsize=figsize,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        dot_color_df=dot_color_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        logcounts=logcounts,
        **kwds,
    )

    if dendrogram:
        dp.add_dendrogram(dendrogram_key=_dk(dendrogram))
    if swap_axes:
        dp.swap_axes()

    dp = dp.style(
        cmap=cmap,
        dot_max=dot_max,
        dot_min=dot_min,
        smallest_dot=smallest_dot,
        dot_edge_lw=kwds.pop("linewidth", _empty),
    ).legend(colorbar_title=colorbar_title, size_title=size_title)

    if return_fig:
        return dp
    else:
        dp.make_figure()
        savefig_or_show(DotPlot.DEFAULT_SAVE_PREFIX, show=show, save=save)
        show = settings.autoshow if show is None else show
        if not show:
            return dp.get_axes()


########################################################################################################################
#
#     CORRECTION FOR RANK GENES GROUPS
#
########################################################################################################################

from math import floor
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
from scipy.sparse import issparse, vstack

from scanpy import _utils
from scanpy import logging as logg
from scanpy._compat import old_positionals
from scanpy._utils import (
    check_nonnegative_integers,
    raise_not_implemented_error_if_backed_type,
)
from scanpy.get import _check_mask
from scanpy.preprocessing._utils import _get_mean_var

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from anndata import AnnData
    from numpy.typing import NDArray
    from scipy import sparse

    _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]

# Used with get_args
_Method = Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"]


def _undo_logspace(input_data):
    if isinstance(input_data, scipy.sparse.csr_matrix):
        input_data = scipy.sparse.csr_matrix.expm1(input_data)
    else:
        input_data = np.expm1(input_data)
    return input_data


def _redo_logspace(input_data):
    if isinstance(input_data, scipy.sparse.csr_matrix):
        input_data = scipy.sparse.csr_matrix.log1p(input_data)
    else:
        input_data = np.log1p(input_data)
    return input_data

def _select_top_n(scores: NDArray, n_top: int):
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]

    return global_indices


def _ranks(
        X: np.ndarray | sparse.csr_matrix | sparse.csc_matrix,
        mask_obs: NDArray[np.bool_] | None = None,
        mask_obs_rest: NDArray[np.bool_] | None = None,
):
    CONST_MAX_SIZE = 10000000

    n_genes = X.shape[1]

    if issparse(X):
        merge = lambda tpl: vstack(tpl).toarray()
        adapt = lambda X: X.toarray()
    else:
        merge = np.vstack
        adapt = lambda X: X

    masked = mask_obs is not None and mask_obs_rest is not None

    if masked:
        n_cells = np.count_nonzero(mask_obs) + np.count_nonzero(mask_obs_rest)
        get_chunk = lambda X, left, right: merge(
            (X[mask_obs, left:right], X[mask_obs_rest, left:right])
        )
    else:
        n_cells = X.shape[0]
        get_chunk = lambda X, left, right: adapt(X[:, left:right])

    # Calculate chunk frames
    max_chunk = floor(CONST_MAX_SIZE / n_cells)

    for left in range(0, n_genes, max_chunk):
        right = min(left + max_chunk, n_genes)

        df = pd.DataFrame(data=get_chunk(X, left, right))
        ranks = df.rank()
        yield ranks, left, right


def _tiecorrect(ranks):
    size = np.float64(ranks.shape[0])
    if size < 2:
        return np.repeat(ranks.shape[1], 1.0)

    arr = np.sort(ranks, axis=0)
    tf = np.insert(arr[1:] != arr[:-1], (0, arr.shape[0] - 1), True, axis=0)
    idx = np.where(tf, np.arange(tf.shape[0])[:, None], 0)
    idx = np.sort(idx, axis=0)
    cnt = np.diff(idx, axis=0).astype(np.float64)

    return 1.0 - (cnt**3 - cnt).sum(axis=0) / (size**3 - size)



class _RankGenes:
    def __init__(
        self,
        adata: AnnData,
        groups: Iterable[str] | Literal["all"],
        groupby: str,
        *,
        mask_var: NDArray[np.bool_] | None = None,
        reference: Literal["rest"] | str = "rest",
        use_raw: bool = True,
        layer: str | None = None,
        comp_pts: bool = False,
    ) -> None:
        self.mask_var = mask_var
        if (base := adata.uns.get("log1p", {}).get("base")) is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            self.expm1_func = np.expm1

        self.groups_order, self.groups_masks_obs = _utils.select_groups(
            adata, groups, groupby
        )

        # Singlet groups cause division by zero errors
        invalid_groups_selected = set(self.groups_order) & set(
            adata.obs[groupby].value_counts().loc[lambda x: x < 2].index
        )

        if len(invalid_groups_selected) > 0:
            raise ValueError(
                "Could not calculate statistics for groups {} since they only "
                "contain one sample.".format(", ".join(invalid_groups_selected))
            )

        adata_comp = adata
        if layer is not None:
            if use_raw:
                raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
            X = adata_comp.layers[layer]
        else:
            if use_raw and adata.raw is not None:
                adata_comp = adata.raw
            X = adata_comp.X
        raise_not_implemented_error_if_backed_type(X, "rank_genes_groups")

        # for correct getnnz calculation
        if issparse(X):
            X.eliminate_zeros()

        if self.mask_var is not None:
            self.X = X[:, self.mask_var]
            self.var_names = adata_comp.var_names[self.mask_var]

        else:
            self.X = X
            self.var_names = adata_comp.var_names

        self.ireference = None
        if reference != "rest":
            self.ireference = np.where(self.groups_order == reference)[0][0]

        self.means = None
        self.vars = None

        self.means_rest = None
        self.vars_rest = None

        self.comp_pts = comp_pts
        self.pts = None
        self.pts_rest = None

        self.stats = None

        # for logreg only
        self.grouping_mask = adata.obs[groupby].isin(self.groups_order)
        self.grouping = adata.obs.loc[self.grouping_mask, groupby]


    def _basic_stats(self) -> None:
        """Set self.{means,vars,pts}{,_rest} depending on X."""
        n_genes = self.X.shape[1]
        n_groups = self.groups_masks_obs.shape[0]

        self.means = np.zeros((n_groups, n_genes))
        self.vars = np.zeros((n_groups, n_genes))
        self.pts = np.zeros((n_groups, n_genes)) if self.comp_pts else None

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes))
            self.vars_rest = np.zeros((n_groups, n_genes))
            self.pts_rest = np.zeros((n_groups, n_genes)) if self.comp_pts else None
        else:
            mask_rest = self.groups_masks_obs[self.ireference]
            X_rest = self.X[mask_rest]

            # X_rest has logcounts, undo the log space
            X_rest = _undo_logspace(X_rest)

            self.means[self.ireference], self.vars[self.ireference] = _get_mean_var(
                X_rest)

            # Redo the log for the mean
            self.means[self.ireference] = _redo_logspace(self.means[self.ireference])

            # deleting the next line causes a memory leak for some reason
            del X_rest

        if issparse(self.X):
            get_nonzeros = lambda X: X.getnnz(axis=0)
        else:
            get_nonzeros = lambda X: np.count_nonzero(X, axis=0)

        for group_index, mask_obs in enumerate(self.groups_masks_obs):
            X_mask = self.X[mask_obs]

            X_mask = _undo_logspace(X_mask)

            if self.comp_pts:
                self.pts[group_index] = get_nonzeros(X_mask) / X_mask.shape[0]

            if self.ireference is not None and group_index == self.ireference:
                continue

            self.means[group_index], self.vars[group_index] = _get_mean_var(X_mask)

            # Redo the log for the mean
            self.means[group_index] = _redo_logspace(self.means[group_index])


            if self.ireference is None:
                mask_rest = ~mask_obs
                X_rest = self.X[mask_rest]
                X_rest = _undo_logspace(X_rest)

                (
                    self.means_rest[group_index],
                    self.vars_rest[group_index],
                ) = _get_mean_var(X_rest)

                self.means_rest[group_index] = _redo_logspace(self.means_rest[group_index])

                # this can be costly for sparse data
                if self.comp_pts:
                    self.pts_rest[group_index] = get_nonzeros(X_rest) / X_rest.shape[0]
                # deleting the next line causes a memory leak for some reason
                del X_rest

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> Generator[tuple[int, NDArray[np.floating], NDArray[np.floating]], None, None]:
        from scipy import stats

        self._basic_stats()

        for group_index, (mask_obs, mean_group, var_group) in enumerate(
            zip(self.groups_masks_obs, self.means, self.vars)
        ):
            if self.ireference is not None and group_index == self.ireference:
                continue

            ns_group = np.count_nonzero(mask_obs)

            if self.ireference is not None:
                mean_rest = self.means[self.ireference]
                var_rest = self.vars[self.ireference]
                ns_other = np.count_nonzero(self.groups_masks_obs[self.ireference])
            else:
                mean_rest = self.means_rest[group_index]
                var_rest = self.vars_rest[group_index]
                ns_other = self.X.shape[0] - ns_group

            if method == "t-test":
                ns_rest = ns_other
            elif method == "t-test_overestim_var":
                # hack for overestimating the variance for small groups
                ns_rest = ns_group
            else:
                raise ValueError("Method does not exist.")

            # TODO: Come up with better solution. Mask unexpressed genes?
            # See https://github.com/scipy/scipy/issues/10269
            with np.errstate(invalid="ignore"):
                scores, pvals = stats.ttest_ind_from_stats(
                    mean1=mean_group,
                    std1=np.sqrt(var_group),
                    nobs1=ns_group,
                    mean2=mean_rest,
                    std2=np.sqrt(var_rest),
                    nobs2=ns_rest,
                    equal_var=False,  # Welch's
                )

            # I think it's only nan when means are the same and vars are 0
            scores[np.isnan(scores)] = 0
            # This also has to happen for Benjamini Hochberg
            pvals[np.isnan(pvals)] = 1

            yield group_index, scores, pvals

    def wilcoxon(
        self, *, tie_correct: bool
    ) -> Generator[tuple[int, NDArray[np.floating], NDArray[np.floating]], None, None]:
        from scipy import stats

        self._basic_stats()

        n_genes = self.X.shape[1]
        # First loop: Loop over all genes
        if self.ireference is not None:
            # initialize space for z-scores
            scores = np.zeros(n_genes)
            # initialize space for tie correction coefficients
            T = np.zeros(n_genes) if tie_correct else 1

            for group_index, mask_obs in enumerate(self.groups_masks_obs):
                if group_index == self.ireference:
                    continue

                mask_obs_rest = self.groups_masks_obs[self.ireference]

                n_active = np.count_nonzero(mask_obs)
                m_active = np.count_nonzero(mask_obs_rest)

                if n_active <= 25 or m_active <= 25:
                    logg.hint(
                        "Few observations in a group for "
                        "normal approximation (<=25). Lower test accuracy."
                    )

                # Calculate rank sums for each chunk for the current mask
                for ranks, left, right in _ranks(self.X, mask_obs, mask_obs_rest):
                    scores[left:right] = ranks.iloc[0:n_active, :].sum(axis=0)
                    if tie_correct:
                        T[left:right] = _tiecorrect(ranks)

                std_dev = np.sqrt(
                    T * n_active * m_active * (n_active + m_active + 1) / 12.0
                )

                scores = (
                    scores - (n_active * ((n_active + m_active + 1) / 2.0))
                ) / std_dev
                scores[np.isnan(scores)] = 0
                pvals = 2 * stats.distributions.norm.sf(np.abs(scores))

                yield group_index, scores, pvals
        # If no reference group exists,
        # ranking needs only to be done once (full mask)
        else:
            n_groups = self.groups_masks_obs.shape[0]
            scores = np.zeros((n_groups, n_genes))
            n_cells = self.X.shape[0]

            if tie_correct:
                T = np.zeros((n_groups, n_genes))

            for ranks, left, right in _ranks(self.X):
                # sum up adjusted_ranks to calculate W_m,n
                for group_index, mask_obs in enumerate(self.groups_masks_obs):
                    scores[group_index, left:right] = ranks.iloc[mask_obs, :].sum(
                        axis=0
                    )
                    if tie_correct:
                        T[group_index, left:right] = _tiecorrect(ranks)

            for group_index, mask_obs in enumerate(self.groups_masks_obs):
                n_active = np.count_nonzero(mask_obs)

                T_i = T[group_index] if tie_correct else 1

                std_dev = np.sqrt(
                    T_i * n_active * (n_cells - n_active) * (n_cells + 1) / 12.0
                )

                scores[group_index, :] = (
                    scores[group_index, :] - (n_active * (n_cells + 1) / 2.0)
                ) / std_dev
                scores[np.isnan(scores)] = 0
                pvals = 2 * stats.distributions.norm.sf(np.abs(scores[group_index, :]))

                yield group_index, scores[group_index], pvals

    def logreg(
        self, **kwds
    ) -> Generator[tuple[int, NDArray[np.floating], None], None, None]:
        # if reference is not set, then the groups listed will be compared to the rest
        # if reference is set, then the groups listed will be compared only to the other groups listed
        from sklearn.linear_model import LogisticRegression

        # Indexing with a series causes issues, possibly segfault
        X = self.X[self.grouping_mask.values, :]

        if len(self.groups_order) == 1:
            raise ValueError("Cannot perform logistic regression on a single cluster.")

        clf = LogisticRegression(**kwds)
        clf.fit(X, self.grouping.cat.codes)
        scores_all = clf.coef_
        # not all codes necessarily appear in data
        existing_codes = np.unique(self.grouping.cat.codes)
        for igroup, cat in enumerate(self.groups_order):
            if len(self.groups_order) <= 2:  # binary logistic regression
                scores = scores_all[0]
            else:
                # cat code is index of cat value in .categories
                cat_code: int = np.argmax(self.grouping.cat.categories == cat)
                # index of scores row is index of cat code in array of existing codes
                scores_idx: int = np.argmax(existing_codes == cat_code)
                scores = scores_all[scores_idx]
            yield igroup, scores, None

            if len(self.groups_order) <= 2:
                break

    def compute_statistics(
        self,
        method: _Method,
        *,
        corr_method: _CorrMethod = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        **kwds,
    ) -> None:
        if method in {"t-test", "t-test_overestim_var"}:
            generate_test_results = self.t_test(method)
        elif method == "wilcoxon":
            generate_test_results = self.wilcoxon(tie_correct=tie_correct)
        elif method == "logreg":
            generate_test_results = self.logreg(**kwds)

        self.stats = None

        n_genes = self.X.shape[1]

        for group_index, scores, pvals in generate_test_results:
            group_name = str(self.groups_order[group_index])

            if n_genes_user is not None:
                scores_sort = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(scores_sort, n_genes_user)
                first_col = "names"
            else:
                global_indices = slice(None)
                first_col = "scores"

            if self.stats is None:
                idx = pd.MultiIndex.from_tuples([(group_name, first_col)])
                self.stats = pd.DataFrame(columns=idx)

            if n_genes_user is not None:
                self.stats[group_name, "names"] = self.var_names[global_indices]

            self.stats[group_name, "scores"] = scores[global_indices]

            if pvals is not None:
                self.stats[group_name, "pvals"] = pvals[global_indices]
                if corr_method == "benjamini-hochberg":
                    from statsmodels.stats.multitest import multipletests

                    pvals[np.isnan(pvals)] = 1
                    _, pvals_adj, _, _ = multipletests(
                        pvals, alpha=0.05, method="fdr_bh"
                    )
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals * n_genes, 1.0)
                self.stats[group_name, "pvals_adj"] = pvals_adj[global_indices]

            if self.means is not None:
                mean_group = self.means[group_index]
                if self.ireference is None:
                    mean_rest = self.means_rest[group_index]
                else:
                    mean_rest = self.means[self.ireference]
                foldchanges = (self.expm1_func(mean_group) + 1e-9) / (
                    self.expm1_func(mean_rest) + 1e-9
                )  # add small value to remove 0's
                self.stats[group_name, "logfoldchanges"] = np.log2(
                    foldchanges[global_indices]
                )

        if n_genes_user is None:
            self.stats.index = self.var_names


@old_positionals(
    "mask",
    "use_raw",
    "groups",
    "reference",
    "n_genes",
    "rankby_abs",
    "pts",
    "key_added",
    "copy",
    "method",
    "corr_method",
    "tie_correct",
    "layer",
)
def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    mask_var: NDArray[np.bool_] | str | None = None,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    copy: bool = False,
    method: _Method | None = None,
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    **kwds,
) -> AnnData | None:
    """\
    Rank genes for characterizing groups.

    Expects logarithmized data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    mask_var
        Select subset of genes to use in statistical tests.
    use_raw
        Use `raw` attribute of `adata` if present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups. Note that if
        `reference='rest'` all groups will still be used as the reference, not
        just those specified in `groups`.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables.
        Defaults to all genes.
    method
        The default method is `'t-test'`,
        `'t-test_overestim_var'` overestimates variance of each group,
        `'wilcoxon'` uses Wilcoxon rank-sum,
        `'logreg'` uses logistic regression. See :cite:t:`Ntranos2019`,
        `here <https://github.com/scverse/scanpy/issues/95>`__ and `here
        <https://www.nxn.se/valent/2018/3/5/actionable-scrna-seq-clusters>`__,
        for why this is meaningful.
    corr_method
        p-value correction method.
        Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
    tie_correct
        Use tie correction for `'wilcoxon'` scores.
        Used only for `'wilcoxon'`.
    rankby_abs
        Rank genes by the absolute value of the score, not by the
        score. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` information is saved to.
    copy
        Whether to copy `adata` or modify it inplace.
    kwds
        Are passed to test methods. Currently this affects only parameters that
        are passed to :class:`sklearn.linear_model.LogisticRegression`.
        For instance, you can pass `penalty='l1'` to try to come up with a
        minimal set of genes that are good predictors (sparse solution meaning
        few non-zero fitted coefficients).

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.uns['rank_genes_groups' | key_added]['names']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['logfoldchanges']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group. Ordered according to
        scores. Only provided if method is 't-test' like.
        Note: this is an approximation calculated from mean-log values.
    `adata.uns['rank_genes_groups' | key_added]['pvals']` : structured :class:`numpy.ndarray` (dtype `float`)
        p-values.
    `adata.uns['rank_genes_groups' | key_added]['pvals_adj']` : structured :class:`numpy.ndarray` (dtype `float`)
        Corrected p-values.
    `adata.uns['rank_genes_groups' | key_added]['pts']` : :class:`pandas.DataFrame` (dtype `float`)
        Fraction of cells expressing the genes for each group.
    `adata.uns['rank_genes_groups' | key_added]['pts_rest']` : :class:`pandas.DataFrame` (dtype `float`)
        Only if `reference` is set to `'rest'`.
        Fraction of cells from the union of the rest of each group
        expressing the genes.

    Notes
    -----
    There are slight inconsistencies depending on whether sparse
    or dense data are passed. See `here <https://github.com/scverse/scanpy/blob/main/tests/test_rank_genes_groups.py>`__.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(adata, 'bulk_labels', method='wilcoxon')
    >>> # to visualize the results
    >>> sc.pl.rank_genes_groups(adata)
    """
    if mask_var is not None:
        mask_var = _check_mask(adata, mask_var, "var")

    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw is True and adata.raw is None:
        raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")

    if method is None:
        method = "t-test"

    if "only_positive" in kwds:
        rankby_abs = not kwds.pop("only_positive")  # backwards compat

    start = logg.info("ranking genes")
    avail_methods = set(get_args(_Method))
    if method not in avail_methods:
        raise ValueError(f"Method must be one of {avail_methods}.")

    avail_corr = {"benjamini-hochberg", "bonferroni"}
    if corr_method not in avail_corr:
        raise ValueError(f"Correction method must be one of {avail_corr}.")

    adata = adata.copy() if copy else adata
    _utils.sanitize_anndata(adata)
    # for clarity, rename variable
    if groups == "all":
        groups_order = "all"
    elif isinstance(groups, str | int):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != "rest" and reference not in set(groups_order):
            groups_order += [reference]
    if reference != "rest" and reference not in adata.obs[groupby].cat.categories:
        cats = adata.obs[groupby].cat.categories.tolist()
        raise ValueError(
            f"reference = {reference} needs to be one of groupby = {cats}."
        )

    if key_added is None:
        key_added = "rank_genes_groups"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        groupby=groupby,
        reference=reference,
        method=method,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
    )

    test_obj = _RankGenes(
        adata,
        groups_order,
        groupby,
        mask_var=mask_var,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        comp_pts=pts,
    )

    if check_nonnegative_integers(test_obj.X) and method != "logreg":
        logg.warning(
            "It seems you use rank_genes_groups on the raw count data. "
            "Please logarithmize your data before calling rank_genes_groups."
        )

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    # defaults to all genes
    if n_genes_user is None or n_genes_user > test_obj.X.shape[1]:
        n_genes_user = test_obj.X.shape[1]

    logg.debug(f"consider {groupby!r} groups:")
    logg.debug(f"with sizes: {np.count_nonzero(test_obj.groups_masks_obs, axis=1)}")

    test_obj.compute_statistics(
        method,
        corr_method=corr_method,
        n_genes_user=n_genes_user,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        **kwds,
    )

    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    test_obj.stats.columns = test_obj.stats.columns.swaplevel()

    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    for col in test_obj.stats.columns.levels[0]:
        adata.uns[key_added][col] = test_obj.stats[col].to_records(
            index=False, column_dtypes=dtypes[col]
        )

    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{key_added!r}]`\n"
            "    'names', sorted np.recarray to be indexed by group ids\n"
            "    'scores', sorted np.recarray to be indexed by group ids\n"
            + (
                "    'logfoldchanges', sorted np.recarray to be indexed by group ids\n"
                "    'pvals', sorted np.recarray to be indexed by group ids\n"
                "    'pvals_adj', sorted np.recarray to be indexed by group ids"
                if method in {"t-test", "t-test_overestim_var", "wilcoxon"}
                else ""
            )
        ),
    )
    return adata if copy else None


def _calc_frac(X):
    n_nonzero = X.getnnz(axis=0) if issparse(X) else np.count_nonzero(X, axis=0)
    return n_nonzero / X.shape[0]


@old_positionals(
    "key",
    "groupby",
    "use_raw",
    "key_added",
    "min_in_group_fraction",
    "min_fold_change",
    "max_out_group_fraction",
    "compare_abs",
)
def filter_rank_genes_groups(
    adata: AnnData,
    *,
    key: str | None = None,
    groupby: str | None = None,
    use_raw: bool | None = None,
    key_added: str = "rank_genes_groups_filtered",
    min_in_group_fraction: float = 0.25,
    min_fold_change: int | float = 1,
    max_out_group_fraction: float = 0.5,
    compare_abs: bool = False,
) -> None:
    """\
    Filters out genes based on log fold change and fraction of genes expressing the
    gene within and outside the `groupby` categories.

    See :func:`~scanpy.tl.rank_genes_groups`.

    Results are stored in `adata.uns[key_added]`
    (default: 'rank_genes_groups_filtered').

    To preserve the original structure of adata.uns['rank_genes_groups'],
    filtered genes are set to `NaN`.

    Parameters
    ----------
    adata
    key
    groupby
    use_raw
    key_added
    min_in_group_fraction
    min_fold_change
    max_out_group_fraction
    compare_abs
        If `True`, compare absolute values of log fold change with `min_fold_change`.

    Returns
    -------
    Same output as :func:`scanpy.tl.rank_genes_groups` but with filtered genes names set to
    `nan`

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(adata, 'bulk_labels', method='wilcoxon')
    >>> sc.tl.filter_rank_genes_groups(adata, min_fold_change=3)
    >>> # visualize results
    >>> sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered')
    >>> # visualize results using dotplot
    >>> sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered')
    """
    if key is None:
        key = "rank_genes_groups"

    if groupby is None:
        groupby = adata.uns[key]["params"]["groupby"]

    if use_raw is None:
        use_raw = adata.uns[key]["params"]["use_raw"]

    same_params = (
        adata.uns[key]["params"]["groupby"] == groupby
        and adata.uns[key]["params"]["reference"] == "rest"
        and adata.uns[key]["params"]["use_raw"] == use_raw
    )

    use_logfolds = same_params and "logfoldchanges" in adata.uns[key]
    use_fraction = same_params and "pts_rest" in adata.uns[key]

    # convert structured numpy array into DataFrame
    gene_names = pd.DataFrame(adata.uns[key]["names"])

    fraction_in_cluster_matrix = pd.DataFrame(
        np.zeros(gene_names.shape),
        columns=gene_names.columns,
        index=gene_names.index,
    )
    fraction_out_cluster_matrix = pd.DataFrame(
        np.zeros(gene_names.shape),
        columns=gene_names.columns,
        index=gene_names.index,
    )

    if use_logfolds:
        fold_change_matrix = pd.DataFrame(adata.uns[key]["logfoldchanges"])
    else:
        fold_change_matrix = pd.DataFrame(
            np.zeros(gene_names.shape),
            columns=gene_names.columns,
            index=gene_names.index,
        )

        if (base := adata.uns.get("log1p", {}).get("base")) is not None:
            expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            expm1_func = np.expm1

    logg.info(
        f"Filtering genes using: "
        f"min_in_group_fraction: {min_in_group_fraction} "
        f"min_fold_change: {min_fold_change}, "
        f"max_out_group_fraction: {max_out_group_fraction}"
    )

    for cluster in gene_names.columns:
        # iterate per column
        var_names = gene_names[cluster].values

        if not use_logfolds or not use_fraction:
            sub_X = adata.raw[:, var_names].X if use_raw else adata[:, var_names].X
            in_group = adata.obs[groupby] == cluster
            X_in = sub_X[in_group]
            X_out = sub_X[~in_group]

        if use_fraction:
            fraction_in_cluster_matrix.loc[:, cluster] = (
                adata.uns[key]["pts"][cluster].loc[var_names].values
            )
            fraction_out_cluster_matrix.loc[:, cluster] = (
                adata.uns[key]["pts_rest"][cluster].loc[var_names].values
            )
        else:
            fraction_in_cluster_matrix.loc[:, cluster] = _calc_frac(X_in)
            fraction_out_cluster_matrix.loc[:, cluster] = _calc_frac(X_out)

        if not use_logfolds:
            # compute mean value
            mean_in_cluster = np.ravel(X_in.mean(0))
            mean_out_cluster = np.ravel(X_out.mean(0))
            # compute fold change
            fold_change_matrix.loc[:, cluster] = np.log2(
                (expm1_func(mean_in_cluster) + 1e-9)
                / (expm1_func(mean_out_cluster) + 1e-9)
            )

    if compare_abs:
        fold_change_matrix = fold_change_matrix.abs()
    # filter original_matrix
    gene_names = gene_names[
        (fraction_in_cluster_matrix > min_in_group_fraction)
        & (fraction_out_cluster_matrix < max_out_group_fraction)
        & (fold_change_matrix > min_fold_change)
    ]
    # create new structured array using 'key_added'.
    adata.uns[key_added] = adata.uns[key].copy()
    adata.uns[key_added]["names"] = gene_names.to_records(index=False)





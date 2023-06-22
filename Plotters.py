#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import pandas as pd
import os
import os.path as op
import warnings

warnings.filterwarnings('ignore')


def AccioFigure(nSubplots, fontdict=None, polar=False, resolution=100):
    """ Returns a figure with appropriate properties for plotting analyses results

        INPUTS:
            - nSubplots: a tuple where the first element is the number of rows for the subplots and the second element
                        is the number of columns for the subplots

            - fontdict: a dictionary with font properties (default is {'size': 15, 'weight': 'normal'})
            - resolution: the resolution in dpi (defualt is 200)

        OUTPUTS:
            - fig: a figure handle
            - axs: a handle to the axes (this is an array with the same dimensions as the elements of nSubplots
    """

    if fontdict is None:
        fontdict = {'size': 8, 'weight': 'normal'}

    plt.rc('font', **fontdict)
    if not polar:
        fig, axs = plt.subplots(int(nSubplots[0]), int(nSubplots[1]), dpi=resolution)
    else:
        fig, axs = plt.subplots(int(nSubplots[0]), int(nSubplots[1]), dpi=resolution,
                                subplot_kw=dict(projection='polar'))
    mng = plt.get_current_fig_manager()
    pltBackend = plt.get_backend()
    # set figure size to maximum (using three most common backends to try to be as backend-agnostic as possible)
    if pltBackend == 'TkAgg':
        mng.window.wm_geometry("+%d+%d" % (0, 0))
        mng.resize(*mng.window.maxsize())
    elif pltBackend == 'wxAgg':
        mng.window.SetPosition((0, 0))
        mng.frame.Maximize(True)
    elif pltBackend == 'Qt4Agg':
        mng.window.move(0, 0)
        mng.window.showMaximized()

    return fig, axs


def SaveThyFigure(fig, filename, filepath, saveEPS, eps_dpi=250, png_dpi=200):
    """ save the current figure
    :param saveEPS:
    :param png_dpi:
    :param eps_dpi:
    :param fig: the figure to save
    :parameter filename: the name of the file (without the extension)
    :parameter filepath: the path to which to save the file
    """
    fig.savefig(filepath + op.sep + filename + ".png", dpi=png_dpi, format='png', backend='agg')
    if saveEPS:
        fig.savefig(filepath + op.sep + filename + ".eps", dpi=eps_dpi, format='eps')


def show():
    """ show current figures
    """
    plt.show()


def tighten(pads=None):
    """
    tighten the figure to avoid overlapping subplots or cut-out plots
    :parameter pads: this is how much to pad in all directions (see the documentation of this function on the matplotlib
                website)
    """
    if pads is not None:
        plt.gcf().tight_layout(pad=pads[0], w_pad=pads[1], h_pad=pads[2])
    else:
        plt.gcf().tight_layout()


def ErrorLinePlot(x, y, se, title, xlabel, ylabel, annotate=True, annotx=None, annot_text=None, conditions=None,
                  ax=None):
    """
    This function plot a line plot with a shaded error area defined by sem.
    :param ax: axis on which to plot. if not provided, the function will create its own and return a handle to it
    :param x: the x-axis data
    :param y: the y. This should be a NxD array where N is the number of conditions (e.g.
                durations) and D is the number of data points
    :param se: the standard errors. This should be the same dimensions as dist
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param annotx: a list of the x-points at which to add annotation lines
    :param annot_text: a list of strings. these will be the texts added to the annotation lines so it should have the
                       same number of elements as annotx
    :param conditions: a list of strings containing each condition (e.g. duration), this should have the same number of
                       elements as the first dimension in y.
    :param annotate: a flag to indicate whether or not to annotate
    :return: axes: the axes handle to the plot
    """

    sns.set_style('dark')
    sns.reset_orig()

    # get a figure if an axis handle isn't provided
    if ax is None:
        fig, ax = AccioFigure((1, 1))

    # get a color palette
    linepal = sns.color_palette(n_colors=y.shape[0])
    errpal = sns.color_palette('pastel', n_colors=y.shape[0])

    # loop through the durations and plot
    for cond in range(0, y.shape[0]):
        # plot the lines
        if conditions is not None:
            sns.lineplot(x=x, y=y[cond, :], linewidth=0.6, color=linepal[cond], label=conditions[cond], ax=ax)
        else:
            sns.lineplot(x=x, y=y[cond, :], linewidth=0.6, color=linepal[cond], ax=ax)
        # plot the errors
        ax.fill_between(x, y1=y[cond, :] - se[cond, :], y2=y[cond, :] + se[cond, :], facecolor=errpal[cond], alpha=0.35)

    # annotate
    if annotate:
        for ln, txt in zip(annotx, annot_text):
            ax.axvline(x=ln, color='r', linewidth=0.4, alpha=0.7)
            ymax = ax.get_ylim()[1]
            ymin = ax.get_ylim()[0]
            ax.text(ln + 0.05, (ymax + ymin) + 0.35, txt, rotation=90, color='k',
                    fontsize='small')

    # title, limits, etc
    ax.set_title(title, fontdict={'weight': 'bold'})
    ax.set_xlabel(xlabel, fontdict={'size': 8, 'weight': 'bold'})
    ax.set_ylabel(ylabel, fontdict={'size': 8, 'weight': 'bold'})
    ax.set_xlim(min(x), max(x))
    #ax.set_ylim(0.0, 0.5)
    if conditions is not None:
        ax.legend(prop={'size': 6}, loc="upper right")

    return ax


def HeatMap(data, title, roiSize, dims, ax=None):
    """
    This function plots a heatmap of data and shows an ractangular roi defined by roiSize.
    :param ax:
    :param data:
    :param title:
    :param roiSize:
    :param dims:
    :return:
    """

    # get a figure and an axis if none is provided
    if ax is None:
        fig, ax = AccioFigure((1, 1), fontdict={'size': 10, 'weight': 'bold'})

    # plot
    arr_img = plt.imread("fix_bcg.png", format='png')
    ax.imshow(arr_img, aspect='equal', extent=[20, dims[0], 20, dims[1]],alpha =1)
    data[data < 0.002] = np.nan
    ax.imshow(data, aspect='equal', extent=[20, dims[0], 20, dims[1]],alpha =1,interpolation = 'none', vmin = 0)


    #plt.gcf().colorbar(pos, ax=ax, boundaries=[0,.05,.1, .15])

    ax.grid(False)

    # draw center lines
    ax.axhline(y=dims[1] / 2, color='w', linewidth=0.3, alpha=0.5)
    ax.axvline(x=dims[0] / 2, color='w', linewidth=0.3, alpha=0.5)

    
    # draw roi circle for fixation cutoff (2 degrees of visual angle)
    xycir = (dims[0] / 2, dims[1] / 2)
    circ = patches.Circle(xycir, roiSize/2, fill=False, linewidth=1, ec='g', alpha=0.8)
    ax.add_patch(circ)

    # remove ticks
    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                   labelright=False, labelleft=False, which='both')

    # title
    ax.set_title(title, fontdict={'size': 10, 'weight': 'bold'})


    return ax


def PolarPlot(theta, r, title, conditions, ax=None):
    """

    :param theta:
    :param r:
    :param title:
    :param conditions:
    :param ax:
    :return:
    """

    # get a figure and an axis if none is provided
    if ax is None:
        fig, ax = AccioFigure((1, 1), polar=True)

    # get a color palette
    linepal = sns.color_palette(n_colors=r.shape[0])

    # loop through the durations and plot
    for cond in range(0, r.shape[0]):
        # plot the lines
        if conditions is not None:
            ax.plot(theta, r[cond, :], linewidth=0.8, color=linepal[cond], label=conditions[cond])
        else:
            ax.plot(theta, r[cond, :], linewidth=0.8, color=linepal[cond])

    # title etc
    ax.grid(True)
    ax.tick_params(which='both', pad=-4)
    ax.set_rticks(np.around(np.linspace(min(r.flatten()), max(r.flatten()), 4), 2))
    ax.set_title(title, fontdict={'weight': 'bold'}, pad=10)
    if conditions is not None:
        ax.legend(prop={'size': 4, 'weight': 'bold'})

    return ax


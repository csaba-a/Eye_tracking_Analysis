#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math


def deg2pix(viewDistance, degrees, cmPerPixel):
    """
    converts degrees to pixels
    :param viewDistance: viewer distance from the display screen
    :param degrees: degrees visual angle to be converted to no. of pixels
    :param cmPerPixel: the size of one pixel in centimeters
    :return: pixels: the number of pixels corresponding to the degrees visual angle specified
    """

    # get the size of the visual field
    centimeters = math.tan(math.radians(degrees) / 2) * (2 * viewDistance)

    # now convert the centimeters to pixels
    pixels = round(centimeters / cmPerPixel[0])

    return pixels


def CalcFixationDensity(gaze, scale, screenDims):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin
    :param gaze: a tuple where the first element is gazeX and the second is gazeY. gazeX and gazeY are both NxD matrices
                where N is ntrials and D is number of timepoints
    :param scale:
    :param screenDims:
    :return:
    """
    # make sure inputs are arrays
    gazeX = np.array(gaze[0]).flatten()
    gazeY = np.array(gaze[1]).flatten()

    # initialize the fixation density matrix
    fixDensity = np.zeros((int(np.ceil(screenDims[1] / scale)), int(np.ceil(screenDims[0] / scale))))

    # loop through the bins
    L = len(gazeX)
    for i in range(0, fixDensity.shape[1]):
        for j in range(0, fixDensity.shape[0]):
            fixDensity[j, i] = np.sum(((gazeX >= scale * i) & (gazeX <= scale * (i + 1))) &
                                      ((gazeY >= scale * j) & (gazeY <= scale * (j + 1)))) / L

    
    return fixDensity


def plot_group(group_plot_variables,modality, trialInfo,trialInfo_group, time, dir, params, saveEPS=False):

    import os
    import numpy as np
    import pandas as pd
    import Plotters
    import matplotlib.pyplot as plt
    import seaborn as sns
    import Analyzer_support as AS
    import AnalysisHelpers
    import scipy.stats as stats
    ###Gaze plotting
    stimDur = 0.5
    scDims = params['ScreenResolution']
    picSize = params['PictureSizePixels']

    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    savePath ='/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET/Figures'
    
    #ASSC 25 histogram
    
    bins = 160
     # Font sizes parameters
    SMALL_SIZE = 6
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 17
    
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=9)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure

    gazeX = pd.DataFrame(group_plot_variables['gazeX']) * params['DegreesPerPixel']
    gazeY = pd.DataFrame(group_plot_variables['gazeY']) * params['DegreesPerPixel']

    gazeDataPerCond = {'All': {}, 'Relevance': {'Target': {}, 'NonTarget': {}, 'Irrelevant': {}},
                       'Category': {'Face': {}, 'Object': {}, 'Letter': {}, 'False': {}},
                       'Orientation': {'Center': {}, 'Left': {}, 'Right': {}}}
    center_function = lambda x: x - np.nanmean(x)


    colors = sns.color_palette("colorblind", 6)


    if modality == 'MEG':

        for dur in range(0, len(allConditions['Duration'])):
            # get the masks to index the relevant trials

            msk = trialInfo['Duration'] == allConditions['Duration'][dur]
            tmask = (time > 0) & (time < stimDur * (dur + 1))
            # get the gaze data and append it to the gaze data per condition
            horizontal_gaze = gazeX.loc[msk, tmask]
            vertical_gaze = gazeY.loc[msk, tmask]
            horiz_gaze = center_function(horizontal_gaze.to_numpy().flatten())
            vert_gaze = center_function(vertical_gaze.to_numpy().flatten())

            # plotting histograms
            plt.subplot(1, 3, dur + 1)
            plt.hist(horiz_gaze, bins=bins, range=(-5, 5), color=colors[3],
                     label='Horizontal Fixation')
            plt.hist(vert_gaze, bins=bins, range=(-5, 5), color=colors[4],
                     label='Vertical Fixation')

            plt.title(allConditions['Duration'][dur],fontsize=8)
            # plt.axvline(np.nanmean(fix_distance_SE), color='k', linestyle='dashed', linewidth=2)

            plt.legend(loc='upper right')
            ax = plt.gca()

            ax.axes.yaxis.set_ticklabels([])
            ax.tick_params(axis='x', labelsize=8)
            plt.ylabel("Trials (1440)")
            plt.xlim(-5, 5)
            plt.xlabel("Fixation in visual angle")
        plt.suptitle('MEG raw fixation', fontsize=8)
        Plotters.SaveThyFigure(plt.gcf(), 'histogram_fix_dist', savePath, saveEPS=False)
    elif modality == 'fMRI':
        

        for dur in range(0, len(allConditions['Duration'])):
            # get the masks to index the relevant trials

            msk = trialInfo['Duration'] == allConditions['Duration'][dur]
            tmask = (time > 0) & (time < stimDur * (dur + 1))
            # get the gaze data and append it to the gaze data per condition
            horizontal_gaze = gazeX.loc[msk, tmask]
            vertical_gaze = gazeY.loc[msk, tmask]
            horiz_gaze = center_function(horizontal_gaze.to_numpy().flatten())
            vert_gaze = center_function(vertical_gaze.to_numpy().flatten())

            # plotting histograms
            plt.subplot(1, 3, dur + 1)
            plt.hist(horiz_gaze, bins=bins, range=(-5, 5), color=colors[3],
                     label='Horizontal Fixation')
            plt.hist(vert_gaze, bins=bins, range=(-5, 5), color=colors[4],
                     label='Vertical Fixation')

            plt.title(allConditions['Duration'][dur],fontsize=8)
            # plt.axvline(np.nanmean(fix_distance_SE), color='k', linestyle='dashed', linewidth=2)

            plt.legend(loc='upper right')
            ax = plt.gca()

            ax.axes.yaxis.set_ticklabels([])
            ax.tick_params(axis='x', labelsize=8)
            plt.ylabel("Trials (576)")
            plt.xlim(-5, 5)
            plt.xlabel("Fixation in visual angle")
        plt.suptitle('fMRI raw fixation', fontsize=10)
        Plotters.SaveThyFigure(plt.gcf(), 'histogram_fix_dist', savePath, saveEPS=False)
    else:

        
        for dur in range(0, len(allConditions['Duration'])):
            # get the masks to index the relevant trials

            msk = trialInfo['Duration'] == allConditions['Duration'][dur]
            tmask = (time > 0) & (time < stimDur * (dur + 1))
            # get the gaze data and append it to the gaze data per condition
            horizontal_gaze = gazeX.loc[msk, tmask]
            vertical_gaze = gazeY.loc[msk, tmask]
            horiz_gaze = center_function(horizontal_gaze.to_numpy().flatten())
            vert_gaze = center_function(vertical_gaze.to_numpy().flatten())

            # plotting histograms
            plt.subplot(1, 3, dur + 1)
            plt.hist(horiz_gaze, bins=bins, range=(-5, 5), color=colors[3],
                     label='Horizontal Fixation')
            plt.hist(vert_gaze, bins=bins, range=(-5, 5), color=colors[4],
                     label='Vertical Fixation')

            plt.title(allConditions['Duration'][dur],fontsize=8)
            # plt.axvline(np.nanmean(fix_distance_SE), color='k', linestyle='dashed', linewidth=2)

            plt.legend(loc='upper right')
            ax = plt.gca()

            ax.axes.yaxis.set_ticklabels([])
            ax.tick_params(axis='x', labelsize=8)
            plt.ylabel("Trials (720)")
            plt.xlim(-5, 5)
            plt.xlabel("Fixation in visual angle")

        plt.suptitle('ECoG raw fixation', fontsize=8)
        Plotters.SaveThyFigure(plt.gcf(), 'histogram_fix_dist', savePath, saveEPS=False)
    
    
    ############Group plotting starts
    
    figDensity, axsDensity = Plotters.AccioFigure((1, 3))
    axsDensityF = [fx for fx in axsDensity]
    fixDensityScale = 20
    # this will contain the gaze data segemented by condition

    gazeX = pd.DataFrame(group_plot_variables['gazeX'])
    gazeY = pd.DataFrame(group_plot_variables['gazeY'])

    for dur in range(0, len(allConditions['Duration'])):
        # get the masks to index the relevant trials
        
        msk = trialInfo['Duration'] == allConditions['Duration'][dur]
        tmask = (time > 0) & (time < stimDur * (dur + 1))
        # get the gaze data and append it to the gaze data per condition
        gazeDataPerCond['All'][allConditions['Duration'][dur]] = (gazeX.loc[msk, tmask],gazeY.loc[msk, tmask])
        # get the fixation density
        fixDensity = AnalysisHelpers.CalcFixationDensity(gazeDataPerCond['All'][allConditions['Duration'][dur]],
                                                         fixDensityScale, scDims)
           # now plot fixation density
        Plotters.HeatMap(fixDensity, allConditions['Duration'][dur], picSize, scDims, ax=axsDensityF[dur])

    # adjust the fix density plot and add a title to it
    # figDensity.tight_layout()
    # plt.subplots_adjust(top=0.75)
    figDensity.suptitle('Fixation Density During Stimulus Presentation Across Duration', fontsize=12)

    Plotters.SaveThyFigure(figDensity, 'FixationDensityDuringStimulusPresentationAcrossDuration', savePath, saveEPS)
    
    # plot the distance from fixation
    ax=Plotters.ErrorLinePlot(time, group_plot_variables['meanall_avg'], group_plot_variables['semall_avg'],
                           'Mean Euclidean Distance from Fixation Across Conditions', 'Time', 'Distance (Deg)',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    ax.set_ylim(-0.5, 3)
    Plotters.SaveThyFigure(plt.gcf(), 'MeanEuclideanDistancefromFixationAcrossConditions', savePath, saveEPS)

    ##Fixation distance use again
    fix_dist_all_g = pd.DataFrame(group_plot_variables['fix_dist_all_g'])
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        # acquire a figure with subplots for plotting the distance from fixation
        if len(allConditions[condition]) == 3:
            fig, axs = Plotters.AccioFigure((3, 1))
        elif len(allConditions[condition]) == 4:
            fig, axs = Plotters.AccioFigure((2, 2))
        else:
            fig, axs = Plotters.AccioFigure((1, 1))
        # flatten
        axsf = [fx for fx in axs.flat]

        # acquire a figure with subplots for plotting the fixation distances
        figd, axsd = Plotters.AccioFigure((len(allConditions[condition]), 3), fontdict={'size': 10, 'weight': 'normal'})
        for s in range(0, len(allConditions[condition])):
            # get the trial type wrt current condition
            subCond = allConditions[condition][s]
            meanDist = np.zeros((len(allConditions['Duration']), fix_dist_all_g.shape[1]))
            semDist = np.zeros((len(allConditions['Duration']), fix_dist_all_g.shape[1]))
              
            # loop for each duration
            for dur in range(0, len(allConditions['Duration'])):
                # now plot fixation density
                # initialize the mean and sem arrays
                # get the mask to index the relevant trials
                msk = (trialInfo['Duration'] == allConditions['Duration'][dur]) & (trialInfo[condition] == subCond)
                tmask = (time > 0) & (time < stimDur * (dur + 1))
                # get the fixation density
                # get the fixation distance

                fixDist = fix_dist_all_g.loc[msk, :]
                # calculate the mean and sem
                meanDist[dur, :] = np.array(fixDist.mean(axis=0, skipna=True))
                semDist[dur, :] = stats.sem(fixDist, axis=0, ddof=1, nan_policy='omit')
                fixDensity = AnalysisHelpers.CalcFixationDensity((gazeX.loc[msk, tmask], gazeY.loc[msk, tmask]),fixDensityScale, scDims)
                 
                Plotters.HeatMap(fixDensity, ('%s, %s' % (subCond, allConditions['Duration'][dur][0])),
                                 picSize, scDims, ax=axsd[s, dur])

            # TODO: equate colorbar scale across subplots in the same figure
            # adjustments to fixation density figure
            # figd.tight_layout()
            figd.subplots_adjust(top=0.9)
            figd.suptitle('Fixation Density During Stimulus Presentation Across %s' % condition, fontsize=12)
            Plotters.SaveThyFigure(figd, 'FixationDensityDuringStimulusPresentationAcross%s' % condition, savePath,
                                   saveEPS)
            
            # plot distance from fixation
            ax=Plotters.ErrorLinePlot(time, meanDist, semDist,
                                   '', 'Time', 'Distance (Deg)',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=allConditions['Duration'], ax=axsf[s])
            ax.set_ylim(-0.5, 3)

        fig.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        #fig.suptitle(('Distance from Fixation for each %s' % condition))

        Plotters.SaveThyFigure(fig, ('DistancefromFixationforeach%s' % condition), savePath, saveEPS)
    # SACCADES

    # initialize some paramteres
    dsacc = 100  # downsampling rate for saccade data
    dsacc_dir = 12  # bin width for directional distribution
    binTimes = np.arange(0, params['TrialTimePts'], dsacc)
    binTimes_dir = np.arange(0, 360, dsacc_dir)
    time = np.linspace(-params['PreStim'], params['PostStim'], len(binTimes) - 1)
    theta = np.linspace(0, 360, len(binTimes_dir) - 1)
    thetaRad = np.deg2rad(theta)
    stimDur = 0.5
    trialInfo_c=pd.concat(trialInfo_group)
    saccAmpPerCond, saccRatePerCond, saccDirPerCond, meanSacc = AS.GetSaccData(group_plot_variables['allSaccades_c'], binTimes, binTimes_dir,
                                                                            trialInfo_c, params, dsacc)
    
    # plot across conditions first # TODO: equate scales across subplots in each figure
    ax=Plotters.ErrorLinePlot(time, saccAmpPerCond['All'][0], saccAmpPerCond['All'][1],
                           'Mean Saccade Amplitude Change by Duration Across All Conditions', 'Time', 'Amplitude (Deg)',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    ax.set_ylim(0, 3)

    Plotters.SaveThyFigure(plt.gcf(), 'MeanSaccadeAmplitudeChangebyDurationAcrossAllConditions', savePath, saveEPS)

    # plot directional distribution
    ax=Plotters.PolarPlot(thetaRad, saccDirPerCond['All'],
                       'Directional Distribution of Saccades Post-stim Across All Conditions',
                       ['Short', 'Medium', 'Long'])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    Plotters.SaveThyFigure(plt.gcf(), 'DirectionalDistributionofSaccadesPost-stimAcrossAllConditions', savePath,
                           saveEPS)

    for condition in saccAmpPerCond.keys():
        if condition == 'All':
            continue

        subConds = list(saccAmpPerCond[condition].keys())

        # acquire a figure with subplots for plotting the saccade amplitudes and another for the saccade rates
        if len(saccAmpPerCond[condition]) == 3:
            figA, axsA = Plotters.AccioFigure((3, 1))
            figR, axsR = Plotters.AccioFigure((3, 1))
            figD, axsD = Plotters.AccioFigure((1, 3), polar=True)
        elif len(saccAmpPerCond[condition]) == 4:
            figA, axsA = Plotters.AccioFigure((2, 2))
            figR, axsR = Plotters.AccioFigure((2, 2))
            figD, axsD = Plotters.AccioFigure((2, 2), polar=True)
        else:
            figA, axsA = Plotters.AccioFigure((1, 1))
            figR, axsR = Plotters.AccioFigure((1, 2))
            figD, axsD = Plotters.AccioFigure((2, 2), polar=True)
        # flatten
        axsAF = [fx for fx in axsA.flat]
        axsRF = [fx for fx in axsR.flat]
        axsDF = [fx for fx in axsD.flat]

        for s in range(0, len(subConds)):
            # plot amplitude
            ax=Plotters.ErrorLinePlot(time, saccAmpPerCond[condition][subConds[s]][0],
                                   saccAmpPerCond[condition][subConds[s]][1],
                                   '', 'Time', 'Amplitude (Deg)',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsAF[s])
            ax.set_ylim(0, 3)

            # plot rate
            Plotters.ErrorLinePlot(time, saccRatePerCond[condition][subConds[s]],
                                   np.zeros(saccRatePerCond[condition][subConds[s]].shape),
                                   '', 'Time', 'Rate (Deg)',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsRF[s])

            # plot directional distributions
            ax=Plotters.PolarPlot(thetaRad, saccDirPerCond[condition][subConds[s]],
                               ('%s' % subConds[s]), ['Short', 'Medium', 'Long'], ax=axsDF[s])
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
        # figure adjustments, titles, etc
        figA.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        figR.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        if len(saccAmpPerCond[condition]) == 4:
            figD.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        else:
            figD.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        figA.suptitle(('Mean Saccade Amplitude Change for each %s' % condition))
        figR.suptitle(('Mean Saccade Rate for each %s' % condition))
        figD.suptitle(('Directional Distribution of Saccades Post-Stim for each %s' % condition))
        Plotters.SaveThyFigure(figA, ('MeanSaccadeAmplitudeChangeforeach%s' % condition), savePath, saveEPS)
        Plotters.SaveThyFigure(figR, ('MeanSaccadeRateforeach%s' % condition), savePath, saveEPS)
        Plotters.SaveThyFigure(figD, ('DirectionalDistributionPost-Stimforeach%s' % condition), savePath, saveEPS)

    #Blinks
    stimDur = 0.5
    time = np.linspace(-params['PreStim'], params['PostStim'], group_plot_variables['blink_plot_c'].shape[1])

    # all conditions for plotting
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # plot across all conditions first
    # initialize arrays to hold the mean and sem
    meanBlinkAll = np.zeros((len(allConditions['Duration']), group_plot_variables['blink_plot_c'].shape[1]))
    semBlinkAll = np.zeros((len(allConditions['Duration']), group_plot_variables['blink_plot_c'].shape[1]))

    # loop through durations
    for dur in range(0, len(allConditions['Duration'])):
        # get the mask to index the relevant trials
        msk = trialInfo_c['Duration'] == allConditions['Duration'][dur]
        # get the fixation distance for the relevant trials
        currBlinks = pd.DataFrame(group_plot_variables['blink_plot_c'][msk, :])
        round_currBlinks=currBlinks.round()
        # calculate the mean and sem
        meanBlinkAll[dur, :] = np.nanmean(round_currBlinks, axis=0)
        semBlinkAll[dur, :] = np.nanstd(round_currBlinks, axis=0, ddof=1) / np.sqrt(currBlinks.shape[0])

    # now plot
    ax=Plotters.ErrorLinePlot(time, meanBlinkAll, semBlinkAll,
                           'Percentage of blinks Over Time Across all Conditions',
                           'Time (s)', 'Percentage of blinks',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    ax.set_ylim(0.0, 0.4)
    #fig.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
    Plotters.SaveThyFigure(plt.gcf(), 'MeanNumberofBlinksOverTimeAcrossallConditions', savePath, saveEPS)


    # now do the same for each condition
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        # acquire a figure with subplots
        if len(allConditions[condition]) == 3:
            fig, axs = Plotters.AccioFigure((3, 1))
        elif len(allConditions[condition]) == 4:
            fig, axs = Plotters.AccioFigure((2, 2))
        else:
            fig, axs = Plotters.AccioFigure((1, 1))
        # flatten
        axsf = [fx for fx in axs.flat]

        for s in range(0, len(allConditions[condition])):
            # get the trial type wrt current condition
            subCond = allConditions[condition][s]
            # initialize arrays to hold the mean and sem
            meanBlink = np.zeros((len(allConditions['Duration']), group_plot_variables['blink_plot_c'].shape[1]))
            semBlink = np.zeros((len(allConditions['Duration']), group_plot_variables['blink_plot_c'].shape[1]))

            for dur in range(0, len(allConditions['Duration'])):
                # get the mask to index the relevant trials
                msk = (trialInfo_c['Duration'] == allConditions['Duration'][dur]) & (trialInfo_c[condition] == subCond)
                # get the fixation distance for the relevant trials
                currBlinks = pd.DataFrame(group_plot_variables['blink_plot_c'][msk, :])
                round_currBlinks= currBlinks.round()
                # calculate the mean and sem
                meanBlink[dur, :] = np.nanmean(round_currBlinks, axis=0)
                semBlink[dur, :] = np.nanstd(round_currBlinks, axis=0, ddof=1) / np.sqrt(currBlinks.shape[0])

            # now plot
            ax=Plotters.ErrorLinePlot(time, meanBlink, semBlink,
                                   '',
                                   'Time (s)', 'Percentage of blinks',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsf[s])
            ax.set_ylim(0.0, 0.4)

        # figure adjustments
        fig.subplots_adjust(top=0.9, left=0.07, right=0.97, hspace=0.45, wspace=0.3)
        #fig.suptitle(('Percentage of blinks for each %s' % condition), fontsize=10)

        Plotters.SaveThyFigure(fig, ('MeanNumberofBlinksforeach%s' % condition), savePath, saveEPS)

    ####Pupils#######


    time = np.linspace(-params['PreStim'], params['PostStim'], group_plot_variables['all_pupil_c'].shape[1])

    # all conditions for plotting
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # plot across all conditions first
    # initialize arrays to hold the mean and sem
    meanPupilAll = np.zeros((len(allConditions['Duration']), group_plot_variables['all_pupil_c'].shape[1]))
    semPupilAll = np.zeros((len(allConditions['Duration']), group_plot_variables['all_pupil_c'].shape[1]))

    # loop through durations
    for dur in range(0, len(allConditions['Duration'])):
        # get the mask to index the relevant trials
        msk = trialInfo_c['Duration'] == allConditions['Duration'][dur]
        # get the fixation distance for the relevant trials
        currPupil = group_plot_variables['all_pupil_c'][msk, :]
        # calculate the mean and sem
        meanPupilAll[dur, :] = np.nanmean(currPupil, axis=0)
        semPupilAll[dur, :] = np.nanstd(currPupil, axis=0, ddof=1) / np.sqrt(currPupil.shape[0])

    # now plot
    ax=Plotters.ErrorLinePlot(time, meanPupilAll, semPupilAll,
                           'Mean Pupil Size Over Time Across all Conditions',
                           'Time (s)', 'Mean Pupil Size',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    ax.set_ylim(0.0, 0.03)
    #fig.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
    Plotters.SaveThyFigure(plt.gcf(), 'MeanPupilSizeOverTimeAcrossallConditions', savePath, saveEPS)

    # now do the same for each condition
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        # acquire a figure with subplots
        if len(allConditions[condition]) == 3:
            fig, axs = Plotters.AccioFigure((3, 1))
        elif len(allConditions[condition]) == 4:
            fig, axs = Plotters.AccioFigure((2, 2))
        else:
            fig, axs = Plotters.AccioFigure((1, 1))
        # flatten
        axsf = [fx for fx in axs.flat]

        for s in range(0, len(allConditions[condition])):
            # get the trial type wrt current condition
            subCond = allConditions[condition][s]
            # initialize arrays to hold the mean and sem
            meanPupil = np.zeros((len(allConditions['Duration']), group_plot_variables['all_pupil_c'].shape[1]))
            semPupil = np.zeros((len(allConditions['Duration']), group_plot_variables['all_pupil_c'].shape[1]))

            for dur in range(0, len(allConditions['Duration'])):
                # get the mask to index the relevant trials
                msk = (trialInfo_c['Duration'] == allConditions['Duration'][dur]) & (trialInfo_c[condition] == subCond)
                # get the fixation distance for the relevant trials
                currPupil = group_plot_variables['all_pupil_c'][msk, :]
                # calculate the mean and sem
                meanPupil[dur, :] = np.nanmean(currPupil, axis=0)
                semPupil[dur, :] = np.nanstd(currPupil, axis=0, ddof=1) / np.sqrt(currPupil.shape[0])

            # now plot
            ax=Plotters.ErrorLinePlot(time, meanPupil, semPupil,
                                   '',
                                   'Time (s)', 'Mean Pupil Size',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsf[s])
            ax.set_ylim(0.0, 0.03)

        # figure adjustments
        #fig.subplots_adjust(top=0.9, left=0.07, right=0.97, hspace=0.45, wspace=0.3)
        #fig.suptitle(('Mean Pupil Size for each %s' % condition))

        Plotters.SaveThyFigure(plt.gcf(), ('MeanPupilSizeforeach%s' % condition), savePath, saveEPS)


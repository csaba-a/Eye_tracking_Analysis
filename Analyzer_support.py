#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.stats as stats
import pandas as pd
import Plotters
import matplotlib.pyplot as plt
import AnalysisHelpers
import copy
import mne as mne

import warnings


# sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings('ignore')


def AnalyzeGaze(modality,trialData, trialInfo, params, savePath, saveEPS=False):
    """

    :param saveEPS:
    :param savePath:
    :param trialData:
    :param trialInfo:
    :param modality:
    :param params:
    :return:
    """
    
    # initialize some parameters
    stimDur = 0.5
    fixDensityScale = 20
    center = params['ScreenCenter']  # center screen
    scDims = params['ScreenResolution']
    picSize = params['PictureSizePixels']

    # calculate the distance from fixation for all trials
    if np.isnan(trialData[0].LX.T.T).values.all():
        # organize gaze into a NxD dataframe where N is ntrials and D is timepoints
        gazeX = pd.DataFrame([np.array(df['RX']) for df in trialData])
        gazeY = pd.DataFrame([np.array(df['RY']) for df in trialData])
        
        
    else:
        # organize gaze into a NxD dataframe where N is ntrials and D is timepoints
        gazeX = pd.DataFrame([np.array(df['LX']) for df in trialData])
        gazeY = pd.DataFrame([np.array(df['LY']) for df in trialData])
        
    if modality == 'MEG':
            gazeX = gazeX.iloc[:,0:3500]
            gazeY = gazeY.iloc[:,0:3500]
    elif modality == 'ECoG':
            gazeX = gazeX.iloc[:,0:1750]
            gazeY = gazeY.iloc[:,0:1750]
    elif modality == 'fMRI':
            gazeX = gazeX.iloc[:,0:3500]
            gazeY = gazeY.iloc[:,0:3500]




            

      # get the fixation density
    fixDistanceAll = np.sqrt((gazeX - center[0]) ** 2 + (gazeY - center[1]) ** 2)

    # convert to degrees
    fixDistanceAll = fixDistanceAll * params['DegreesPerPixel']
    # Baseline Normalization
    fixDistanceAll_bsl = pd.DataFrame()
    if modality == 'ECoG':
        for i in range(0, fixDistanceAll.shape[0]):
            baseline_mean = np.nanmean(fixDistanceAll.loc[i, 0:250])

            for k in range(0, fixDistanceAll.shape[1]):
                fixDistanceAll_bsl.loc[i, k] = fixDistanceAll.loc[i, k] - baseline_mean


    else:
        for i in range(0, fixDistanceAll.shape[0]):
            baseline_mean = np.nanmean(fixDistanceAll.loc[i, 0:500])
            for k in range(0, fixDistanceAll.shape[1]):
                fixDistanceAll_bsl.loc[i, k] = fixDistanceAll.loc[i, k] - baseline_mean
    fixDistanceAll = fixDistanceAll_bsl
    # all conditions for plotting
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # this will contain the gaze data segemented by condition
    gazeDataPerCond = {'All': {}, 'Relevance': {'Target': {}, 'NonTarget': {}, 'Irrelevant': {}},
                       'Category': {'Face': {}, 'Object': {}, 'Letter': {}, 'False': {}},
                       'Orientation': {'Center': {}, 'Left': {}, 'Right': {}}}

    # initialize a time array
    time = np.linspace(-params['PreStim'], params['PostStim'], fixDistanceAll.shape[1])

    # initialize arrays to hold the mean and sem distance across all conditions
    meanDistAll = np.zeros((len(allConditions['Duration']), fixDistanceAll.shape[1]))
    semDistAll = np.zeros((len(allConditions['Duration']), fixDistanceAll.shape[1]))
    

    
    
    # get a figure and axes for fixation density plots
    figDensity, axsDensity = Plotters.AccioFigure((1, 3))
    axsDensityF = [fx for fx in axsDensity]
    
    
    
    # separate for each duration across all conditions
    for dur in range(0, len(allConditions['Duration'])):
        # get the masks to index the relevant trials
        msk = trialInfo['Duration'] == allConditions['Duration'][dur]
        tmask = (time > 0) & (time < stimDur * (dur + 1))
        # get the gaze data and append it to the gaze data per condition


        gazeDataPerCond['All'][allConditions['Duration'][dur]] = (gazeX.loc[msk, tmask], gazeY.loc[msk, tmask])

        # get the fixation distance for the relevant trials
        fixDist = fixDistanceAll.loc[msk, :]
        # calculate the mean and sem
        meanDistAll[dur, :] = np.array(fixDist.mean(axis=0, skipna=True))
        semDistAll[dur, :] = stats.sem(fixDist, axis=0, ddof=1, nan_policy='omit')

        # get the fixation density
        fixDensity = AnalysisHelpers.CalcFixationDensity(gazeDataPerCond['All'][allConditions['Duration'][dur]],
                                                         fixDensityScale, scDims)
        # now plot fixation density
        Plotters.HeatMap(fixDensity, allConditions['Duration'][dur], picSize, scDims, ax=axsDensityF[dur])

    # adjust the fix density plot and add a title to it
    #figDensity.tight_layout()
    #plt.subplots_adjust(top=0.75)
    figDensity.suptitle('Fixation Density During Stimulus Presentation Across Duration', fontsize=12)

    Plotters.SaveThyFigure(figDensity, 'FixationDensityDuringStimulusPresentationAcrossDuration', savePath, saveEPS)

    # plot the distance from fixation
    Plotters.ErrorLinePlot(time, meanDistAll, semDistAll,
                           'Mean Euclidean Distance from Fixation Across Conditions', 'Time', 'Distance (Deg)',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    Plotters.SaveThyFigure(plt.gcf(), 'MeanEuclideanDistancefromFixationAcrossConditions', savePath, saveEPS)



    # separate for each duration and each condition
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
            # initialize the mean and sem arrays
            meanDist = np.zeros((len(allConditions['Duration']), fixDistanceAll.shape[1]))
            semDist = np.zeros((len(allConditions['Duration']), fixDistanceAll.shape[1]))
            # loop for each duration
            for dur in range(0, len(allConditions['Duration'])):
                # get the mask to index the relevant trials
                msk = (trialInfo['Duration'] == allConditions['Duration'][dur]) & (trialInfo[condition] == subCond)
                tmask = (time > 0) & (time < stimDur * (dur + 1))
                # get and append the relevant gaze data for this condition
                gazeDataPerCond[condition][subCond][allConditions['Duration'][dur]] = (gazeX.loc[msk, tmask],
                                                                                       gazeY.loc[msk, tmask])
                # get the fixation distance
                fixDist = fixDistanceAll.loc[msk, :]
                # calculate the mean and sem
                meanDist[dur, :] = np.array(fixDist.mean(axis=0, skipna=True))
                semDist[dur, :] = stats.sem(fixDist, axis=0, ddof=1, nan_policy='omit')


                # get the fixation density
                fixDensity = AnalysisHelpers.CalcFixationDensity((gazeX.loc[msk, tmask], gazeY.loc[msk, tmask]),
                                                                 fixDensityScale, scDims)
                # now plot fixation density
                Plotters.HeatMap(fixDensity, ('%s, %s' % (subCond, allConditions['Duration'][dur][0])),
                                 picSize, scDims, ax=axsd[s, dur])

            # TODO: equate colorbar scale across subplots in the same figure
            # adjustments to fixation density figure
            #figd.tight_layout()
            figd.subplots_adjust(top=0.9)
            figd.suptitle('Fixation Density During Stimulus Presentation Across %s' % condition, fontsize=12)
            Plotters.SaveThyFigure(figd, 'FixationDensityDuringStimulusPresentationAcross%s' % condition, savePath,
                                   saveEPS)

            # plot distance from fixation
            Plotters.ErrorLinePlot(time, meanDist, semDist,
                                   ('%s' % subCond), 'Time', 'Distance (Deg)',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=allConditions['Duration'], ax=axsf[s])

        fig.subplots_adjust(top=0.9, left=0.07, right=0.99, hspace=0.45, wspace=0.3)
        fig.suptitle(('Distance from Fixation for each %s' % condition))

        Plotters.SaveThyFigure(fig, ('DistancefromFixationforeach%s' % condition), savePath, saveEPS)
    
    #Trimming gaze data to standardized size to be able to compile data across all subjects
    if modality == 'MEG':
        if len(gazeX.index) !=1440:
                gazeX = gazeX.reindex(range(1440))
                gazeY = gazeY.reindex(range(1440))
            
    elif modality == 'ECoG':
        if len(gazeX.index) !=720:
            gazeX = gazeX.reindex(range(720))
            gazeY = gazeY.reindex(range(720))
    else:
        if len(gazeX.index) !=576:
            gazeX = gazeX.reindex(range(576))
            gazeY = gazeY.reindex(range(576))
            
     # get the fixation density
    fixDistanceAll = np.sqrt((gazeX - center[0]) ** 2 + (gazeY - center[1]) ** 2)

    # convert to degrees
    fixDistanceAll = fixDistanceAll * params['DegreesPerPixel']

    # Baseline Normalization
    fixDistanceAll_bsl = pd.DataFrame()
    if modality == 'ECoG':
        for i in range(0, fixDistanceAll.shape[0]):
            baseline_mean = np.nanmean(fixDistanceAll.loc[i, 0:250])

            for k in range(0, fixDistanceAll.shape[1]):
                fixDistanceAll_bsl.loc[i, k] = fixDistanceAll.loc[i, k] - baseline_mean


    else:
        for i in range(0, fixDistanceAll.shape[0]):
            baseline_mean = np.nanmean(fixDistanceAll.loc[i, 0:500])
            for k in range(0, fixDistanceAll.shape[1]):
                fixDistanceAll_bsl.loc[i, k] = fixDistanceAll.loc[i, k] - baseline_mean
    fixDistanceAll = fixDistanceAll_bsl
    
        
    return fixDistanceAll, gazeX, gazeY, time, meanDistAll, semDistAll


def AnalyzeSaccades(trialInfo, saccadeInfo, params, savePath, saveEPS=False):
    """

    :param saveEPS:
    :param savePath:
    :param saccadeInfo:
    :param trialInfo:
    :param params:
    :return:
    """

    # initialize some paramteres
    dsacc = 100  # downsampling rate for saccade data
    dsacc_dir = 12  # bin width for directional distribution
    binTimes = np.arange(0, params['TrialTimePts'], dsacc)
    binTimes_dir = np.arange(0, 360, dsacc_dir)
    time = np.linspace(-params['PreStim'], params['PostStim'], len(binTimes) - 1)
    theta = np.linspace(0, 360, len(binTimes_dir) - 1)
    thetaRad = np.deg2rad(theta)
    stimDur = 0.5

    # get all saccades in one table
    if 'SA' in params['ParticipantName']:
        binocular = True
    elif 'SB' in params['ParticipantName']:
        binocular = True
    else:
        binocular = False

    if binocular:
        eyestr = 'both'
    else:
        if saccadeInfo[0]['Saccades']['right'] is None:
            eyestr = 'left'
        else:
            eyestr = 'right'
    
    sacc_num=next((i for i in range(len(saccadeInfo)) if saccadeInfo[i]['Saccades'][eyestr] is not None))
    allSaccades = pd.DataFrame(columns=saccadeInfo[sacc_num]['Saccades'][eyestr].columns)

    saccTrials = np.array([])
    for tr in range(0, len(saccadeInfo)):
        currSaccades = copy.deepcopy(saccadeInfo[tr]['Saccades'][eyestr])
        if currSaccades is not None:
            saccTrials = np.hstack((saccTrials, np.ones(currSaccades.shape[0]) * tr))
            allSaccades = allSaccades.append(currSaccades, ignore_index=True)

    allSaccades = allSaccades.drop_duplicates().reset_index(drop=True)
    allSaccades['trial'] = pd.Series(saccTrials)

    # get mean saccade amplitudes and rate for each condition
    saccAmpPerCond, saccRatePerCond, saccDirPerCond, meanSacc = GetSaccData(allSaccades, binTimes, binTimes_dir,
                                                                  trialInfo, params, dsacc)

    # plot across conditions first # TODO: equate scales across subplots in each figure
    Plotters.ErrorLinePlot(time, saccAmpPerCond['All'][0], saccAmpPerCond['All'][1],
                           'Mean Saccade Amplitude Change by Duration Across All Conditions', 'Time', 'Amplitude (Deg)',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
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
            figA, axsA = Plotters.AccioFigure((1, 3))
            figR, axsR = Plotters.AccioFigure((1, 3))
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
            Plotters.ErrorLinePlot(time, saccAmpPerCond[condition][subConds[s]][0],
                                   saccAmpPerCond[condition][subConds[s]][1],
                                   ('%s' % subConds[s]), 'Time', 'Amplitude (Deg)',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsAF[s])

            # plot rate
            Plotters.ErrorLinePlot(time, saccRatePerCond[condition][subConds[s]],
                                   np.zeros(saccRatePerCond[condition][subConds[s]].shape),
                                   ('%s' % subConds[s]), 'Time', 'Rate (Deg)',
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
            figD.subplots_adjust(top=0.9, left=0.07, right=0.95, hspace=0.45, wspace=0.3)
        else:
            figD.subplots_adjust(top=0.85, left=0.07, right=0.95, hspace=0.45, wspace=0.3)
        figA.suptitle(('Mean Saccade Amplitude Change for each %s' % condition))
        figR.suptitle(('Mean Saccade Rate for each %s' % condition))
        figD.suptitle(('Directional Distribution of Saccades Post-Stim for each %s' % condition))
        Plotters.SaveThyFigure(figA, ('MeanSaccadeAmplitudeChangeforeach%s' % condition), savePath, saveEPS)
        Plotters.SaveThyFigure(figR, ('MeanSaccadeRateforeach%s' % condition), savePath, saveEPS)
        Plotters.SaveThyFigure(figD, ('DirectionalDistributionPost-Stimforeach%s' % condition), savePath, saveEPS)

    return allSaccades

def GetSaccData(allSaccades, binTimes, binTimes_direction, trialInfo, params, dsacc):
    """
    This function computes mean saccade amplitude over trials separated by conditions and durations
    :param binTimes_direction:
    :param dsacc:
    :param params:
    :param trialInfo:
    :param allSaccades:
    :param binTimes:
    :return:
    """

    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    saccAmpPerCond = {'All': {}, 'Relevance': {'Target': {}, 'NonTarget': {}, 'Irrelevant': {}},
                      'Category': {'Face': {}, 'Object': {}, 'Letter': {}, 'False': {}},
                      'Orientation': {'Center': {}, 'Left': {}, 'Right': {}}}

    saccRatePerCond = {'All': {}, 'Relevance': {'Target': {}, 'NonTarget': {}, 'Irrelevant': {}},
                       'Category': {'Face': {}, 'Object': {}, 'Letter': {}, 'False': {}},
                       'Orientation': {'Center': {}, 'Left': {}, 'Right': {}}}

    saccDirPerCond = {'All': {}, 'Relevance': {'Target': {}, 'NonTarget': {}, 'Irrelevant': {}},
                      'Category': {'Face': {}, 'Object': {}, 'Letter': {}, 'False': {}},
                      'Orientation': {'Center': {}, 'Left': {}, 'Right': {}}}

    # find the indices in time corresponding to the different events of interest
    stimDur = 0.5
    time = np.linspace(-params['PreStim'], params['PostStim'], int(params['TrialTimePts']))
    startTS = int(np.argwhere(time >= 0)[0])
    endTS = [int(np.argwhere(time >= stimDur)[0]), int(np.argwhere(time >= stimDur * 2)[0]),
             int(np.argwhere(time >= stimDur * 3)[0])]

    # get the mean saccade amplitudes, rates and directions across conditions first
    meanSaccAll = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
    semSaccAll = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
    saccRateAll = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
    saccDirAll = np.zeros((len(allConditions['Duration']), binTimes_direction.shape[0] - 1))
    for dur in range(0, len(allConditions['Duration'])):
        relevantTrials = np.array(trialInfo.loc[trialInfo['Duration'] == allConditions['Duration'][dur], :].index)
        trialMask = allSaccades['trial'].isin(relevantTrials)
        for bn in range(0, binTimes.shape[0] - 1):
            # the masks that will index the relevant trials
            timeMask = (allSaccades['start'] >= binTimes[bn]) & (allSaccades['end'] <= binTimes[bn + 1])
            # get the relevant series of saccades
            relevantSacc = np.array(allSaccades.loc[trialMask & timeMask, 'distance_to_fixation'])
            relevantSacc = relevantSacc.astype(int)
            if not len(relevantSacc) == 0:
                # compute the mean
                meanSaccAll[dur, bn] = np.nanmean(abs(relevantSacc))
                semSaccAll[dur, bn] = np.std(abs(relevantSacc), ddof=1) / np.sqrt(len(relevantSacc))
                # compute the rate
                saccRateAll[dur, bn] = (len(relevantSacc) / len(relevantTrials)) * (1000 / dsacc)
            else:
                meanSaccAll[dur, bn] = 0
                semSaccAll[dur, bn] = 0
                saccDirAll[dur, bn] = 0

        # now do the same for the direction
        timeMask_dir = (allSaccades['start'] > startTS) & (allSaccades['start'] < endTS[dur])
        relevantSacc_dir = np.array(allSaccades.loc[trialMask & timeMask_dir, 'direction'].dropna())
        # loop through the bins and compute the directional distribution
        for bndir in range(0, binTimes_direction.shape[0] - 1):
            saccDirAll[dur, bndir] = np.nansum((relevantSacc_dir >= binTimes_direction[bndir]) &
                                               (relevantSacc_dir <= binTimes_direction[bndir + 1])) / len(
                relevantSacc_dir)

    saccAmpPerCond['All'] = (meanSaccAll, semSaccAll)
    saccRatePerCond['All'] = saccRateAll
    saccDirPerCond['All'] = saccDirAll


    # now do the same for the other conditions
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        for subcond in allConditions[condition]:
            meanSacc = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
            semSacc = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
            rateSacc = np.zeros((len(allConditions['Duration']), binTimes.shape[0] - 1))
            saccDir = np.zeros((len(allConditions['Duration']), binTimes_direction.shape[0] - 1))
            for dur in range(0, len(allConditions['Duration'])):
                condMask = (trialInfo['Duration'] == allConditions['Duration'][dur]) & (trialInfo[condition] == subcond)
                relevantTrials = np.array(trialInfo.loc[condMask, :].index)
                trialMask = allSaccades['trial'].isin(relevantTrials)
                for bn in range(0, binTimes.shape[0] - 1):
                    # the masks that will index the relevant trials
                    timeMask = (allSaccades['start'] >= binTimes[bn]) & (allSaccades['end'] <= binTimes[bn + 1])
                    # get the relevant series of saccades
                    relevantSacc = allSaccades.loc[trialMask & timeMask, 'distance_to_fixation']
                    relevantSacc = abs(relevantSacc)
                    if not len(relevantSacc) == 0:
                        # compute the mean
                        meanSacc[dur, bn] = relevantSacc.mean(skipna=True)
                        relevantSacc = np.array(relevantSacc)
                        if not len(relevantSacc) == 1:
                            semSacc[dur, bn] = np.std(relevantSacc, ddof=1) / np.sqrt(len(relevantSacc))
                        else:
                            semSacc[dur, bn] = np.std(relevantSacc, ddof=0) / np.sqrt(len(relevantSacc))
                        # compute the rate
                        rateSacc[dur, bn] = (len(relevantSacc) / len(relevantTrials)) * (1000 / dsacc)
                    else:
                        meanSacc[dur, bn] = 0
                        semSacc[dur, bn] = 0
                        rateSacc[dur, bn] = 0

                # now do the same for the direction
                timeMask_dir = (allSaccades['start'] > startTS) & (allSaccades['start'] < endTS[dur])
                relevantSacc_dir = np.array(allSaccades.loc[trialMask & timeMask_dir, 'direction'].dropna())
                # loop through the bins and compute the directional distribution
                for bndir in range(0, binTimes_direction.shape[0] - 1):
                    if len(relevantSacc_dir) == 0:
                        saccDir[dur, bndir] = 0
                    else:
                        saccDir[dur, bndir] = np.nansum((relevantSacc_dir >= binTimes_direction[bndir]) &
                                                        (relevantSacc_dir <= binTimes_direction[bndir + 1])) / \
                                              len(relevantSacc_dir)

            saccAmpPerCond[condition][subcond] = (meanSacc, semSacc)
            saccRatePerCond[condition][subcond] = rateSacc
            saccDirPerCond[condition][subcond] = saccDir

    return saccAmpPerCond, saccRatePerCond, saccDirPerCond, meanSacc


def AnalyzeBlinks(BlinkArray, trialInfo, params, savePath, saveEPS=False):
    """

    :param saveEPS:
    :param savePath:
    :param BlinkArray:
    :param trialInfo:
    :param params:
    :return:
    """

    # initialize some parameters
    stimDur = 0.5
    time = np.linspace(-params['PreStim'], params['PostStim'], BlinkArray.shape[1])

    # all conditions for plotting
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # plot across all conditions first
    # initialize arrays to hold the mean and sem
    meanBlinkAll = np.zeros((len(allConditions['Duration']), BlinkArray.shape[1]))
    semBlinkAll = np.zeros((len(allConditions['Duration']), BlinkArray.shape[1]))

    # loop through durations
    for dur in range(0, len(allConditions['Duration'])):
        # get the mask to index the relevant trials
        msk = trialInfo['Duration'] == allConditions['Duration'][dur]
        # get the fixation distance for the relevant trials
        currBlinks = BlinkArray[msk, :]
        # calculate the mean and sem
        meanBlinkAll[dur, :] = np.nanmean(currBlinks, axis=0)
        semBlinkAll[dur, :] = np.nanstd(currBlinks, axis=0, ddof=1) / np.sqrt(currBlinks.shape[0])

    # now plot
    Plotters.ErrorLinePlot(time, meanBlinkAll, semBlinkAll,
                           'Percentage of Blinks Over Time Across all Conditions',
                           'Time (s)', 'Percentage of Blinks',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    Plotters.SaveThyFigure(plt.gcf(), 'MeanNumberofBlinksOverTimeAcrossallConditions', savePath, saveEPS)

    
    # now do the same for each condition
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        # acquire a figure with subplots
        if len(allConditions[condition]) == 3:
            fig, axs = Plotters.AccioFigure((1, 3))
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
            meanBlink = np.zeros((len(allConditions['Duration']), BlinkArray.shape[1]))
            semBlink = np.zeros((len(allConditions['Duration']), BlinkArray.shape[1]))

            for dur in range(0, len(allConditions['Duration'])):
                # get the mask to index the relevant trials
                msk = (trialInfo['Duration'] == allConditions['Duration'][dur]) & (trialInfo[condition] == subCond)
                # get the fixation distance for the relevant trials
                currBlinks = BlinkArray[msk, :]
                # calculate the mean and sem
                meanBlink[dur, :] = np.nanmean(currBlinks, axis=0)
                semBlink[dur, :] = np.nanstd(currBlinks, axis=0, ddof=1) / np.sqrt(currBlinks.shape[0])


            # now plot
            Plotters.ErrorLinePlot(time, meanBlink, semBlink,
                                   ('%s' % subCond),
                                   'Time (s)', 'Percentage of Blinks',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsf[s])

        # figure adjustments
        fig.subplots_adjust(top=0.9, left=0.07, right=0.97, hspace=0.45, wspace=0.3)
        fig.suptitle(('Percentage of Blinks for each %s' % condition), fontsize=10)

        Plotters.SaveThyFigure(fig, ('MeanNumberofBlinksforeach%s' % condition), savePath, saveEPS)
    for condition in allConditions.keys():
        for subcond in allConditions[condition]:

            # get the mask to index the relevant trials
            msk = trialInfo[condition] == subcond
            # get the blinks for the relevant trials
            currBlinks = BlinkArray[msk, :]
            
    
    return BlinkArray, trialInfo 


def AnalyzePupil(modality,trialData, trialInfo, params, savePath, saveEPS=False):
    """

    :param saveEPS:
    :param savePath:
    :param trialData:
    :param trialInfo:
    :param params:
    :return:
    """

    # construct a pupil size matrix
    if params['Eye'] == 'L':
        pupil = pd.DataFrame([np.array(df['LPupil']) for df in trialData])
    else:
        pupil = pd.DataFrame([np.array(df['RPupil']) for df in trialData])
    if modality == 'MEG':
            pupil = pupil.iloc[:,0:3500]
            
    elif modality == 'ECoG':
            pupil = pupil.iloc[:,0:1750]
            
    pupil = np.array(pupil)

    # initialize some parameters
    stimDur = 0.5

    time = np.linspace(-params['PreStim'], params['PostStim'], pupil.shape[1])

    # all conditions for plotting
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # plot across all conditions first
    # initialize arrays to hold the mean and sem
    meanPupilAll = np.zeros((len(allConditions['Duration']), pupil.shape[1]))
    semPupilAll = np.zeros((len(allConditions['Duration']), pupil.shape[1]))

    # loop through durations
    for dur in range(0, len(allConditions['Duration'])):
        # get the mask to index the relevant trials
        msk = trialInfo['Duration'] == allConditions['Duration'][dur]
        # get the fixation distance for the relevant trials
        currPupil = pupil[msk, :]
        # calculate the mean and sem
        meanPupilAll[dur, :] = np.nanmean(currPupil, axis=0)
        semPupilAll[dur, :] = np.nanstd(currPupil, axis=0, ddof=1) / np.sqrt(currPupil.shape[0])

    # now plot
    Plotters.ErrorLinePlot(time, meanPupilAll, semPupilAll,
                           'Mean Pupil Size Over Time Across all Conditions',
                           'Time (s)', 'Mean Pupil Size',
                           annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                           annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                           conditions=['Short', 'Medium', 'Long'])
    Plotters.SaveThyFigure(plt.gcf(), 'MeanPupilSizeOverTimeAcrossallConditions', savePath, saveEPS)

    # now do the same for each condition
    for condition in allConditions.keys():
        if condition == 'Duration':
            continue

        # acquire a figure with subplots
        if len(allConditions[condition]) == 3:
            fig, axs = Plotters.AccioFigure((1, 3))
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
            meanPupil = np.zeros((len(allConditions['Duration']), pupil.shape[1]))
            semPupil = np.zeros((len(allConditions['Duration']), pupil.shape[1]))

            for dur in range(0, len(allConditions['Duration'])):
                # get the mask to index the relevant trials
                msk = (trialInfo['Duration'] == allConditions['Duration'][dur]) & (trialInfo[condition] == subCond)
                # get the fixation distance for the relevant trials
                currPupil = pupil[msk, :]
                # calculate the mean and sem
                meanPupil[dur, :] = np.nanmean(currPupil, axis=0)
                semPupil[dur, :] = np.nanstd(currPupil, axis=0, ddof=1) / np.sqrt(currPupil.shape[0])

            # now plot
            Plotters.ErrorLinePlot(time, meanPupil, semPupil,
                                   ('%s' % subCond),
                                   'Time (s)', 'Mean Pupil Size',
                                   annotx=[0, stimDur, stimDur * 2, stimDur * 3],
                                   annot_text=['Stimulus On', 'Short Off', 'Medium Off', 'Long Off'],
                                   conditions=['Short', 'Medium', 'Long'], ax=axsf[s])

        # figure adjustments
        fig.subplots_adjust(top=0.9, left=0.07, right=0.97, hspace=0.45, wspace=0.3)
        fig.suptitle(('Mean Pupil Size for each %s' % condition))

        Plotters.SaveThyFigure(fig, ('MeanPupilSizeforeach%s' % condition), savePath, saveEPS)
    return pupil



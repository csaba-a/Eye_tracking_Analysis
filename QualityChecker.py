#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fnmatch
import os
import time as tm
import pickle
import numpy as np
import pandas as pd
import DataParser
from pandas.errors import EmptyDataError

def check_percent_fixation(modality,trialData, trialInfo, params, reference_angle):

    import itertools
    """
    this function calculates and saves the amount of time spent within a specified range (reference angle) based on
    the distance from fixation
    :param trialData:
    :param trialInfo:
    :param params:
    :param reference_angle:
    :return:
    """

    # initialize some parameters
    center = params['ScreenCenter']  # center screen

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
        gazeX = gazeX.iloc[:, 0:3500]
        gazeY = gazeY.iloc[:, 0:3500]
    elif modality == 'ECoG':
        gazeX = gazeX.iloc[:, 0:1750]
        gazeY = gazeY.iloc[:, 0:1750]

    # get the fixation density
    fixDistanceAll = np.sqrt((gazeX - center[0]) ** 2 + (gazeY - center[1]) ** 2)
    # convert to degrees
    fixDistanceAll = fixDistanceAll * params['DegreesPerPixel']
    # Separate non baseline normalized  fixation distance for stats
    fixDistanceAll_no_normalized = fixDistanceAll
    # Baseline Normalization
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
    # all conditions
    allConditions = {'Relevance': params['EventTypes'][0], 'Duration': params['EventTypes'][1],
                     'Orientation': params['EventTypes'][2], 'Category': ['Face', 'Object', 'Letter', 'False']}

    # this will contain the gaze data segemented by condition
    percentFixation = {'Overall': {}, 'Relevance': dict.fromkeys(params['EventTypes'][0]),
                       'Category': dict.fromkeys(allConditions['Category']),
                       'Orientation': dict.fromkeys(allConditions['Orientation']),
                       'Duration': dict.fromkeys(allConditions['Duration'])}
    # this will contain the gaze data segemented by condition
    meanDist_r = {'Overall': {}, 'Relevance': dict.fromkeys(params['EventTypes'][0]),
                       'Category': dict.fromkeys(allConditions['Category']),
                       'Orientation': dict.fromkeys(allConditions['Orientation']),
                       'Duration': dict.fromkeys(allConditions['Duration'])}

    # remove nans
    fixDistanceAllFlat = fixDistanceAll.to_numpy().flatten()
    fixDistanceAllFlat = fixDistanceAllFlat[~np.isnan(fixDistanceAllFlat)]

    # get the percent fixation overall
    numel = fixDistanceAllFlat.shape[0]
    percentFixation['Overall'] = np.nansum(fixDistanceAllFlat <= reference_angle) / numel
    meanDist_r['Overall'] = np.nanmean(fixDistanceAllFlat)



    for condition in allConditions.keys():
        for subcond in allConditions[condition]:
            # get the masks to index the relevant trials
            msk = trialInfo[condition] == subcond

            # get the fixation distance for the relevant trials
            fixDist = fixDistanceAll.loc[msk, :]

            # remove nans
            fixDistFlat = fixDist.to_numpy().flatten()
            fixDistFlat = fixDistFlat[~np.isnan(fixDistFlat)]

            # get the percent fixation for this condition
            numel = fixDistFlat.shape[0]
            percentFixation[condition][subcond] = np.nansum(fixDistFlat <= reference_angle) / numel
            mean_fixd = np.nanmean(fixDistanceAll, axis = 1)
            meanDist_r[condition][subcond] = np.nanmean(fixDistFlat)


    return percentFixation, meanDist_r, mean_fixd, fixDistanceAll, fixDistanceAll_no_normalized


def main(modality, group_QC):
    import EDFConverter
    import Analyzers
    import pingouin as pg
    import pandas as pd

    basepath = '/mnt/beegfs/XNAT/COGITATE/'
    modality_spec_dir = basepath + '/' + modality + '/' + 'Raw' + '/' + 'projects' + '/' + 'CoG_' + modality + '_PhaseII'
    


    # list all subject folders
    subFolders = [fldr for fldr in os.listdir(modality_spec_dir) if os.path.isdir(modality_spec_dir + os.path.sep + fldr) and
                  fnmatch.fnmatch(fldr, '*')]

    # assumes the edf conversion exe is in the same directory as this
    edfConvExePath = os.path.dirname(os.path.abspath(__file__))

    # initialize parameters
    SCREEN_SIZE = {'SA': np.array([78.7, 44.6]), 'SB': np.array([64, 36]), 'SC': np.array([69.8, 39.3]),
                   'SD': np.array([58.5, 32.9]), 'SE': np.array([34.5, 19.5]), 'SF': np.array([34.5, 19.5]),
                   'SX': np.array([34.5, 19.5]), 'SZ': np.array([41.0, 26.0])}
    # VIEWING DISTANCE IN CM:
    # taken from : https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit#gid=0
    VIEWING_DIST = {'SA': 119, 'SB': 100, 'SC': 144, 'SD': 123, 'SE': 80, 'SF': 80, 'SX': 69.5, 'SZ': 66}

    # PARAMETERS
    fs = 500
    fixRefAngle = 3
    qc_s =[]
    qc_fix =[]
    qc_p =[]
    skip_et = ["QC_group_summary", "Analysis", "SD103","SD118",'SD161','SE109']
    subFolders = [e for e in subFolders if e not in skip_et]# and e.startswith('SE')]
    # loop through subjects
    for count, folder in enumerate(subFolders):
        
        print('Quality-checking subject: %s' % folder)
        t = tm.time()
        if modality == 'MEG':
                sess = '_MEEG_V1'
        elif modality == 'fMRI':
                sess = '_MR_V1'
        else:
                sess = '_ECOG_V1'

        behPath = modality_spec_dir + '/' + folder + '/' + folder + sess + '/' + 'RESOURCES' + '/' + 'BEH' + '/' #+ os.path.sep + 'files' + os.path.sep #+ 'BEH' + os.path.sep + pt
            
        etPath = modality_spec_dir + '/' + folder + '/' + folder + sess + '/' + 'RESOURCES' + '/' + 'ET' + '/' #+ os.path.sep + 'files' + os.path.sep #+ 'BEH' + os.path.sep + pt
            
        # PARAMETERS_2
        lab_name = folder[:2]
        scw = SCREEN_SIZE[lab_name][0] * 10
        sch = SCREEN_SIZE[lab_name][1] * 10
        vd = VIEWING_DIST[lab_name] * 10

        try:
        # get a list of asc files and if none exists then convert the edf files to ascii
            ascFiles = [fl for fl in os.listdir(etPath) if os.path.isfile(etPath + os.sep + fl) and fl.endswith('asc')
                    and 'DurR' in fl]
            
                
        except OSError:
            
            continue
            
        if len(ascFiles) == 0:
            print('No ascii files found. Will convert EDF files to ascii.')
            edfFiles = [fl for fl in os.listdir(etPath) if os.path.isfile(etPath + os.sep + fl) and fl.endswith('edf')]
            print('Converting %d EDF files...' % len(edfFiles))
            for edf in edfFiles:
                EDFConverter.convert_file(etPath, edf, edfConvExePath)

            ascFiles = [fl for fl in os.listdir(etPath) if os.path.isfile(etPath + os.sep + fl)
                        and fl.endswith('asc') and 'DurR' in fl]

        # put together the behavioral log files
        behFiles = [bhfl for bhfl in os.listdir(behPath) if os.path.isfile(behPath + os.sep + bhfl)
                    and bhfl.endswith('csv') and 'Raw' in bhfl]
        if len(behFiles) == 0:
            raise DataParser.InputError('No behavioral logs found in for subject %s' % folder)
        else:
            print("Concatenating behavioral log files...")
            behLogs = pd.DataFrame()
            for behFile in behFiles:
                behLogs = behLogs.append(pd.read_csv(behPath + os.sep + behFile), ignore_index=True)

        # continue parameter initiliazation
        pname = folder
        
        pickle_folder = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET_pickles/' + '/' + folder
        if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
        # if there is a pickle file with the subject's data, grab it
        pickFile = [f for f in os.listdir(pickle_folder) if fnmatch.fnmatch(f, '%sEyeTrackingData.pickle' % folder)]


        if not len(pickFile) == 0:
            print('Found subject saved data. Loading now...')
            pickPath = pickle_folder + '/' + pickFile[0]
            fl = open(pickPath, 'rb')
            subData = pickle.load(fl)
            trialDataNbNs = subData['trialDataNbNs']
            trialInfo = subData['trialInfo']
            saccadeInfo = subData['saccadeInfo']
            blinkArray = subData['blinkArray']
            if np.isnan(subData['trialDataNbNs'][0].LX).all():
                eye = 'R'
            else:
                eye = 'L'
            params = DataParser.InitParams(scw, sch, vd, pname, fs, eye)
            fl.close()
        else:
            # loop through the asc files for each run and read the data
            print('Parsing subject eye tracking data...')
            eyeData = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
            for fl in ascFiles:
                try:
                   flEyeData = DataParser.ParseEyeLinkAsc(etPath + os.sep + fl)
                   for df in range(0, len(flEyeData)):
                        eyeData[df] = eyeData[df].append(flEyeData[df], ignore_index=True)
                except EmptyDataError:
                    
                    continue

            print('Extracting timestamps...')

            eye = flEyeData[2].eye[0]
            params = DataParser.InitParams(scw, sch, vd, pname, fs, eye)
            try:
                timeStamps = DataParser.ExtractTimeStamps(eyeData, params['Triggers'], logDF=behLogs)
            except IndexError or AttributeError:
                continue
            print('Sequencing trials...')
           
            trialData, trialInfo = DataParser.SequenceEyeData(params, eyeData, timeStamps)
            
            print('Removing blinks...')
            trialDataNoBlinks, Blinks, blinkArray = DataParser.RemoveBlinks(trialData, params)

            print('Removing saccades...')
            gazeData = [df[['LX', 'LY', 'RX', 'RY']] for df in trialDataNoBlinks]
            saccadeInfo = DataParser.ExtractSaccades(gazeData, params, getBinocular=True)
            trialDataNbNs = DataParser.RemoveSaccades(trialDataNoBlinks, params, binocular=True,
                                                      saccade_info=saccadeInfo)

            # pickle the data
            print('Saving subject eye tracking data...')
            fl = open(pickle_folder + os.path.sep + ('%sEyeTrackingData.pickle' % folder), 'ab')
            subData = {'trialDataNbNs': trialDataNbNs, 'trialInfo': trialInfo, 'saccadeInfo': saccadeInfo,
                       'blinkArray': blinkArray}
            pickle.dump(subData, fl)
            fl.close()

        # initialize the data dict which will be saved in excel sheet
        resultsData = {'Check': [], 'Overall': []}
        for tp in params['EventTypes']:
            for tt in tp:
                resultsData[tt] = []
        resultsData['Face'] = []
        resultsData['Object'] = []
        resultsData['Letter'] = []
        resultsData['False'] = []

        # initialize the data dict which will be saved in excel sheet
        resultsData2 = {'Check': [], 'Overall': []}
        for tp in params['EventTypes']:
            for tt in tp:
                resultsData[tt] = []
        resultsData2['Face'] = []
        resultsData2['Object'] = []
        resultsData2['Letter'] = []
        resultsData2['False'] = []

        resultsData3 = {'Check': []}
        for zz in params['EventTypes']:
            for za in zz:
                resultsData3[za] = []
        resultsData3['Face'] = []
        resultsData3['Object'] = []
        resultsData3['Letter'] = []
        resultsData3['False'] = []

        # percent fixation check
        print('Checking Percent Fixation and mean distance from fixation...')
        percentFixation, meanDist_r, mean_fixd, fixDistanceAll = check_percent_fixation(trialDataNbNs, trialInfo, params, fixRefAngle)

        print('Checking mean Blink...')
        saveDir_g = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET/QC_group_summary'
        if not os.path.exists(saveDir_g):
                os.makedirs(saveDir_g)
        meanblink_r, BlinkArray, trialInfo, mean_blinks = Analyzers.AnalyzeBlinks(blinkArray, trialInfo, params, saveDir_g)

        # print overall percent fixation
        print('Percent Time within %0.1f degrees from fixation = %0.2f' % (fixRefAngle, percentFixation['Overall']))

        # add the data to the results dict
        resultsData['Check'].append('Percent Fixation')
        resultsData2['Check'].append('Mean distance in Degree')
        resultsData3['Check'].append('Percent Mean Blink')
        resultsData['Overall'].append(percentFixation['Overall'])
        resultsData2['Overall'].append(meanDist_r['Overall'])


        for cond in percentFixation.keys():
            if cond == 'Overall':
                continue
            for subcond in percentFixation[cond].keys():
                resultsData[subcond] = percentFixation[cond][subcond]

        for cond in meanDist_r.keys():
            if cond == 'Overall':
                continue
            for subcond_m in meanDist_r[cond].keys():
                resultsData2[subcond_m] = meanDist_r[cond][subcond_m]

        for cond in meanblink_r.keys():
            for subcond in meanblink_r[cond].keys():
                resultsData3[subcond] = meanblink_r[cond][subcond]
        # save
        print('Saving QC results...')
        DataF = pd.DataFrame(resultsData)
        DataF2 = pd.DataFrame(resultsData2)
        DataF3 = pd.DataFrame(resultsData3)
        orig_cols = DataF2.columns.to_frame().index
        result = pd.concat([DataF, DataF2, DataF3], sort=False).reindex(orig_cols, axis=1)

        # save as an excel sheet in either one QC folder or in the particular subject`s folder
        if group_QC:
            # create a directory to hold the QC measures in the subject's eye tracker folder
            saveDir_g = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET_QC_group_summary'
            if not os.path.exists(saveDir_g):
                os.makedirs(saveDir_g)
            #with pd.ExcelWriter(saveDir_g + os.sep + "QC_results_{0}.xlsx".format(folder)) as xwriter:
            result.to_csv(saveDir_g + '/' + folder + 'ET_QC.csv')
        else:
            saveDir = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/sub-' + folder + '/' + 'ses-v1' + '/' + 'et' + '/' 
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            #with pd.ExcelWriter(saveDir_g + os.sep + "QC_results_{0}.xlsx".format(folder)) as xwriter:
            result.to_csv(saveDir + '/' + folder + 'ET_QC.csv')

        if percentFixation['Overall']< 0.90:
                s=0
                b=resultsData['Overall']
        else: 
                s=1
                b=resultsData['Overall']
                
        qc_s.append(s)
        qc_p.append(pname)
        qc_fix.append(b)
    qc_output = pd.DataFrame(qc_s)
    qc_output['Fixation'] = qc_fix
    qc_output['participants']=qc_p
    qc_output.to_csv('/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/et_output_V1.csv')
   



    print('Finished subject: %s, took %0.1fs' % (folder, (tm.time() - t)))


if __name__ == '__main__':
    main(modality='fMRI', group_QC=0)


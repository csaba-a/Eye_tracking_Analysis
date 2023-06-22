#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fnmatch
import os
import time as tm
import pickle
import numpy as np

import DataParser



def main(modality,stats,plotting):
    import EDFConverter
    import pandas as pd
    import QualityChecker as Qc
    import AnalysisHelpers
    from pandas.errors import EmptyDataError
    from sklearn import preprocessing
    import Analyzer_support as AS
    import numpy.ma as ma
    from itertools import zip_longest

    basepath = '/mnt/beegfs/XNAT/COGITATE/'
    directory = basepath + '/' + modality + '/' + 'Raw' + '/' + 'projects' + '/' + 'CoG_' + modality + '_PhaseII'


    # list all subject folders
    subFolders = os.listdir(directory) #[fldr for fldr in os.listdir(directory) if os.path.isdir(directory + os.path.sep + fldr) and
                  #fnmatch.fnmatch(fldr, labcode + '*')]

    # assumes the edf conversion exe is in the same directory as this
    edfConvExePath = os.path.dirname(os.path.abspath(__file__))

    # initialize parameters
    SCREEN_SIZE = {'SA': np.array([78.7, 44.6]), 'SB': np.array([64, 36]), 'SC': np.array([69.8, 39.3]),
                   'SD': np.array([58.5, 32.9]), 'SE': np.array([34.5, 19.5]), 'SF': np.array([34.5, 19.5]),
                   'SX': np.array([53, 30]), 'SZ': np.array([41.0, 26.0])}
    # VIEWING DISTANCE IN CM:
    # taken from : https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit#gid=0
    VIEWING_DIST = {'SA': 119, 'SB': 100, 'SC': 144, 'SD': 123, 'SE': 80, 'SF': 80, 'SX': 69.5, 'SZ': 71}

    # PARAMETERS



    fs = 500
    fixRefAngle = 3

    Statistics = []
    Statistics_sacc = []
    gazeX_g = []
    gazeY_g = []
    fixdist_all_g = []
    meanDistAll_g = []
    semDistAll_g = []
    allSaccades_g = []
    trialInfo_group = []
    blink_plot =[]
    all_pupils = []
    fix_sd = []
    fixation_distance_only_Trial = []
    blinks_trialsXTimepoints_only_Trial = []

    
    # loop through subjects
    for folder in subFolders:
        if folder == "QC_group_summary":
            continue
        elif folder == "Stats_results":
            continue
        elif folder == "Figures":
            continue
        print('Analyzing subject: %s' % folder)
        t = tm.time()

        lab_name = folder[:2]
        scw = SCREEN_SIZE[lab_name][0] * 10
        sch = SCREEN_SIZE[lab_name][1] * 10
        vd = VIEWING_DIST[lab_name] * 10
        
    #Skip these participants/patients due to the fractioned data or no data collected for ET
    
    skip_et = ['SA114','SA167','SA170','SB084','SC151','SD163','SD111','SD109','SD189','SD176','SD185_MR_V1','SD119','SD173',"SD103","SD118",'SD161', 'SD156','SE109','SE118']
    
    
    subFolders = [e for e in subFolders if e not in skip_et]# and e.startswith('SE')]
    
    # loop through subjects
    for count, folder in enumerate(subFolders):
        
        print('Analyzing subject: %s' % folder)
        t = tm.time()
        if modality == 'MEG':
                sess = '_MEEG_V1'
        elif modality == 'fMRI':
                sess = '_MR_V1'
        else:
                sess = '_ECOG_V1'

        behPath = directory + '/' + folder + '/' + folder + sess + '/' + 'RESOURCES' + '/' + 'BEH' + '/' #+ os.path.sep + 'files' + os.path.sep #+ 'BEH' + os.path.sep + pt
        etPath = directory + '/' + folder + '/' + folder + sess + '/' + 'RESOURCES' + '/' + 'ET' + '/' #+ os.path.sep + 'files' + os.path.sep #+ 'BEH' + os.path.sep + pt
        #etPath = directory + os.sep + folder + os.path.sep + 'resources' + os.path.sep + 'ET' + os.path.sep + 'files'
        #behPath = directory + os.sep + folder + os.path.sep + 'resources' + os.path.sep + 'BEH' + os.path.sep + 'files'

        # get a list of asc files and if none exists then convert the edf files to ascii
        try:
            ascFiles = [fl for fl in os.listdir(etPath) if os.path.isfile(etPath + os.sep + fl) and fl.endswith('asc')
                    and 'DurR' in fl]
        except FileNotFoundError:
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

        # create a directory to hold the figures in the subject's eye tracker folder
        saveDir = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET/'+ folder + '/' + 'Figures'
        #saveDir = dir + os.path.sep + folder + os.path.sep + 'Figures'
        if not os.path.exists(saveDir):
                os.makedirs(saveDir)

        # continue parameter initiliazation
        pname = folder
        pickle_folder = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/ET_pickles/' + '/' + folder
        if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
        # if there is a pickle file with the subject's data, grab it
        pickFile = [f for f in os.listdir(pickle_folder) if fnmatch.fnmatch(f, '%sEyeTrackingData.pickle' % folder)]
        # continue parameter initiliazation
        pname = folder

        # if there is a pickle file with the subject's data, grab it
        #pickFile = [f for f in os.listdir(etPath) if fnmatch.fnmatch(f, '%sEyeTrackingData.pickle' % folder)]
        if not len(pickFile) == 0:
            print('Found subject saved data. Loading now...')
            pickPath = pickle_folder + '/' + pickFile[0]
            #pickPath =  etPath + os.path.sep + pickFile[0]
            fl = open(pickPath, 'rb')
            subData = pickle.load(fl)

            trialDataNbNs = subData['trialDataNbNs']
            trialInfo = subData['trialInfo']
            saccadeInfo = subData['saccadeInfo']
            blinkArray = subData['blinkArray']
            if np.isnan(subData['trialDataNbNs'][0].LX).any():
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
            trialDataNoBlinks, _, blinkArray = DataParser.RemoveBlinks(trialData, params, modality)

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

        if plotting:
            # gaze analysis
            print('Analyzing gaze...')
            fixDistanceAll,gazeX, gazeY, time, meanDistAll, semDistAll = AS.AnalyzeGaze(modality,trialDataNbNs, trialInfo, params, savePath=saveDir)
            # Accumulate data across subjects
            gazeX_g.append(gazeX)
            gazeY_g.append(gazeY)
            fixdist_all_g.append(fixDistanceAll)
            meanDistAll_g.append(meanDistAll)
            semDistAll_g.append(semDistAll)
            

                
            # saccade analysis
            print('Analyzing saccades...')
            allSaccades = AS.AnalyzeSaccades(trialInfo, saccadeInfo, params, saveDir)
    
            # Accumulate data across subjects
            allSaccades_g.append(allSaccades.T.T)
    
    
    
            # blink analysis
            print('Analyzing blinks...')
    
            AS.AnalyzeBlinks(blinkArray, trialInfo, params, saveDir)
            # Pupil size analysis
            print('Analyzing pupil size...')
    
            pupil = AS.AnalyzePupil(modality,trialDataNbNs, trialInfo, params, saveDir)
            # Normailize pupils
            pupss = pd.DataFrame(pupil)
            pupss.replace(float("nan"), 0, inplace=True)
            pupil = preprocessing.normalize(pupss)
            all_pupils.append(pupil)
    
    
            # Calculate fixation distance and blinks and saccades #
            """
            This section is dedicated for accumulating Fixation , blink, pupil and saccade data across all the variables 
            and save the statistics table for analysis in R
            """

        if stats:
            # fixation
            percentFixation, meanDist_r, mean_fixd, fixDistanceAll, fixDistanceAll_no_normalized = Qc.check_percent_fixation(modality,trialDataNbNs, trialInfo, params, fixRefAngle)
            pupil = AS.AnalyzePupil(modality,trialDataNbNs, trialInfo, params, saveDir)
            blink_plot.append(blinkArray)
            
            ###Save Fixation distance matrix Trials X timepoints

            if modality=='ECoG':
                fixation_distance_matrix = fixDistanceAll_no_normalized.iloc[:, 250:1750]
            else:
                fixation_distance_matrix = fixDistanceAll_no_normalized.iloc[:, 500:2000]

            fixation_distance_trialsXTimepoints = pd.concat([trialInfo, fixation_distance_matrix], axis=1)
            fixation_distance_only_Trial.append(fixation_distance_trialsXTimepoints)

            ###Save Blinks(0/1) matrix Trials X timepoints
            blinks_all = pd.DataFrame(blinkArray)
            if modality == 'ECoG':
                blinks_matrix = blinks_all.iloc[:, 250:1750]
            else:
                blinks_matrix = blinks_all.iloc[:, 500:2000]

            blinks_matrix_trialsXTimepoints = pd.concat([trialInfo, blinks_matrix], axis=1)
            blinks_trialsXTimepoints_only_Trial.append(blinks_matrix_trialsXTimepoints)
            
            
            #Saccades amplitude
            
            if np.isnan(subData['trialDataNbNs'][0].LX).any():
                sacc_eye = 'right'
            else:
                sacc_eye = 'left'
            
            sacc_amp = []
            msks = []
            for i in range(0, len(saccadeInfo)):
                if saccadeInfo[i]['Saccades'][sacc_eye] is None:
                    msks.append(i)
    
                if saccadeInfo[i]['Saccades'][sacc_eye] is not None:
                    sacc_amp.append(np.mean(saccadeInfo[i]['Saccades'][sacc_eye].total_amplitude))
    
            trialInfo_sacc = trialInfo.drop(msks)
            
            #FIX
            msks_short = []
            msks_long = []
            msks_medium = []
            time = np.linspace(-params['PreStim'], params['PostStim'], fixDistanceAll_no_normalized.shape[1])
            for i in range(0, len(time)):
                if 0 < time[i] < 0.5:
                    msks_short.append(i)
                elif 0.5 < time[i] < 1.0:
                    msks_medium.append(i)
                elif 1 < time[i] < 1.5:
                    msks_long.append(i)

            short = fixDistanceAll_no_normalized.iloc[:,msks_short]
            medium = fixDistanceAll_no_normalized.iloc[:, msks_medium]
            long = fixDistanceAll_no_normalized.iloc[:, msks_long]
            
            
            
            Stats_table_ET_fix_blk_pup = trialInfo
            Stats_table_ET_fix_blk_pup['Short_fix_mean']= np.nanmean(short, axis=1)
            Stats_table_ET_fix_blk_pup['Short_fix_max'] = np.amax(short, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_fix_mean'] = np.nanmean(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_fix_max'] = np.amax(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Long_fix_mean'] = np.nanmean(long, axis=1)
            Stats_table_ET_fix_blk_pup['Long_fix_max'] = np.amax(long, axis=1)
            
            ###Calcualte fixation if it is 2SDs above baseline SD
            for i in range(0, fixDistanceAll_no_normalized.shape[0]):
                if modality == 'ECoG' and (np.std(fixDistanceAll_no_normalized.iloc[i, 250:1750]) > (np.std(fixDistanceAll_no_normalized.iloc[i, 0:250]) * 2)):
                    fix_sd_value = 1
                elif (np.std(fixDistanceAll_no_normalized.iloc[i, 500:2000]) > (np.std(fixDistanceAll_no_normalized.iloc[i, 0:500]) * 2)):
                    fix_sd_value = 1
                else:
                    fix_sd_value = 0
                fix_sd.append(fix_sd_value)
            Stats_table_ET_fix_blk_pup['Fix_deviation_2SD'] = fix_sd
            fix_sd = []
            
            #BLINK
            blinks = pd.DataFrame(blinkArray)
            short = blinks.iloc[:, msks_short]
            medium = blinks.iloc[:, msks_medium]
            long = blinks.iloc[:, msks_long]
            
            Stats_table_ET_fix_blk_pup['Short_Blk_mean'] = np.nanmean(short, axis=1)
            Stats_table_ET_fix_blk_pup['Short_Blk_max'] = np.amax(short, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_Blk_mean'] = np.nanmean(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_Blk_max'] = np.amax(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Long_Blk_mean'] = np.nanmean(long, axis=1)
            Stats_table_ET_fix_blk_pup['Long_Blk_max'] = np.amax(long, axis=1)

            #Pupil

            pups = pd.DataFrame(pupil)
            pups.replace(float("nan"), 0, inplace=True)
            normalized = preprocessing.normalize(pups)
            pups = pd.DataFrame(normalized)

            short = pups.iloc[:, msks_short]
            medium = pups.iloc[:, msks_medium]
            long = pups.iloc[:, msks_long]

            Stats_table_ET_fix_blk_pup['Short_Pup_mean'] = np.nanmean(short, axis=1)
            Stats_table_ET_fix_blk_pup['Short_Pup_max'] = np.amax(short, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_Pup_mean'] = np.nanmean(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Medium_Pup_max'] = np.amax(medium, axis=1)
            Stats_table_ET_fix_blk_pup['Long_Pup_mean'] = np.nanmean(long, axis=1)
            Stats_table_ET_fix_blk_pup['Long_Pup_max'] = np.amax(long, axis=1)
            #Final step add participant IDs
            Stats_table_ET_fix_blk_pup['Id'] = pname # add participant name

            #SACCADES
            Stats_table_ET_sacc = trialInfo_sacc
            Stats_table_ET_sacc['saccades'] = sacc_amp # extract saccade amplitude
            
            Stats_table_ET_sacc['Id'] = pname
            trialInfo_group.append(subData['trialInfo'])
           # for xz, values in trialInfo.iterrows():
            #   if trialInfo['Relevance'][xz] == 'NonTarget' or trialInfo['Relevance'][xz] == 'Target':
             #     trialInfo['Relevance'][xz] = 'Relevant'
            Statistics.append(Stats_table_ET_fix_blk_pup)


        #Saccades


            #for xz, values in trials.iterrows():
             #   if trials['Relevance'][xz] == 'NonTarget' or trials['Relevance'][xz] == 'Target':
              #      trials['Relevance'][xz] = 'Relevant'

            Statistics_sacc.append(Stats_table_ET_sacc)

        print('Finished subject: %s, took %0.1fs' % (folder, (tm.time() - t)))
    if plotting:

        group_plot_variables={"gazeX":np.nanmean(gazeX_g,axis=0),
                              "gazeY":np.nanmean(gazeY_g,axis=0),
                              'fix_dist_all_g': np.nanmean(fixdist_all_g,axis=0),
                              "meanall_avg":np.nanmean(meanDistAll_g,axis=0),
                              "semall_avg":np.nanmean(semDistAll_g,axis=0),
                              "allSaccades_c":pd.concat(allSaccades_g),
                              "blink_plot_c":np.concatenate(blink_plot),
                              "all_pupil_c":np.concatenate(all_pupils,axis=0)}
       

        
        AnalysisHelpers.plot_group(group_plot_variables,modality, trialInfo,trialInfo_group, time, dir, params)
        
        
    if stats:
        saveDir = '/mnt/beegfs/XNAT/COGITATE/' + modality + '/phase_2/processed/bids/derivatives/qcs/population_analysis/Stats_results'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        Statistics = pd.concat(Statistics)
        Statistics_sacc = pd.concat(Statistics_sacc)
        FD_matrix = pd.concat(fixation_distance_only_Trial)
        BLK_matrix = pd.concat(blinks_trialsXTimepoints_only_Trial)
        FD_matrix.to_csv(saveDir + os.path.sep + 'FixationDistance_Matrix_TrialByTimepoints.csv', index=False)
        BLK_matrix.to_csv(saveDir + os.path.sep + 'Blinks_trialsXTimepoints.csv', index=False)
        Statistics_sacc.to_csv(saveDir + os.path.sep + 'ET_stats_Sacc.csv', index=False)
        Statistics.to_csv(saveDir + os.path.sep + 'ET_stats_fix_blk_pup.csv', index=False)
        
       
if __name__ == '__main__':
    main(modality=r"MEG",stats=True,plotting=True)


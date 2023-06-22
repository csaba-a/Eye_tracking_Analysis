#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string
import numpy as np
import pandas as pd
import time
import AnalysisHelpers
import copy
from scipy import stats
#from progress.bar import IncrementalBar as ibar
from based_noise_blinks_detection import based_noise_blinks_detection


class Error(Exception):
    pass


class InputError(Error):
    def __init__(self, msg):
        self.message = msg


class RepresentationError(Error):
    def __init__(self, msg):
        self.message = msg


def InitParams(screenWidth, screenHeight, viewDistance, participantName, fs, eye):
    """ defines and returns the parameters that will be used for analysis base on the experiment being analyzed
    :param screenWidth: width of the display screen
    :param screenHeight: height of the display screen
    :param viewDistance: viewer distance from the display screen
    :param participantName: name of the participant whose data is being analyzed
    :param fs: sampling rate of the eye tracker
    :param eye: either 'L' or 'R' for left or right eye
    :return: params: a dictionary holding the parameters that will be used for analysis
    """

    # initialize params as an empty dictionary
    params = dict([])

    params['ScreenWidth'] = screenWidth
    params['ScreenHeight'] = screenHeight
    params['ViewDistance'] = viewDistance
    params['ParticipantName'] = participantName
    params['SamplingFrequency'] = fs
    params['Eye'] = eye
    params['ScreenResolution'] = np.array([1920, 1080])
    params['cmPerPixel'] = np.array([screenWidth, screenHeight]) / params['ScreenResolution']
    params['ScreenCenter'] = params['ScreenResolution'] / 2
    params['EventTypes'] = [['Target', 'NonTarget', 'Irrelevant'],
                            ['Short', 'Medium', 'Long'],
                            ['Center', 'Left', 'Right']]
    params['AcceptedShift'] = 2  # max shift from fixation circle that we accept (2 deg on each side)
    params['PictureSize'] = 2  # size of the picture in degrees (4 x 4 degrees)

    # convert visual angles to pixels
    params['FixationWindow'] = AnalysisHelpers.deg2pix(viewDistance, params['AcceptedShift'], params['cmPerPixel'])

    params['PictureSizePixels'] = AnalysisHelpers.deg2pix(viewDistance, params['PictureSize'], params['cmPerPixel'])

    params['DegreesPerPixel'] = params['PictureSize'] / params['PictureSizePixels']

    # define the time before and after the stimulus presentation over which the analysis is performed
    # the post stim duration depends on the duration category (short, medium, long) but we first segment the
    # entire trial from the whole data and then segment the trial itself again into 500ms segments
    params['PreStim'] = 0.5
    params['PostStim'] = 3
    # total number of time points for each trial
    params['TrialTimePts'] = (params['PreStim'] + params['PostStim']) * fs

    # Here, we are going to define a dictionary object that can be indexed with a stimulus ID to return the
    # category of that stimulus. This will be useful for when try to decode the eye tracker messages
    class Triggers:
        # a class to act as a struct object that will hold the ID information for stimuli, orientation, duration,
        # and task
        pass

    # stimuli
    Triggers.Stimuli = [None] * 81
    # faces
    Triggers.Stimuli[1:21] = ['Face'] * 20
    # objects
    Triggers.Stimuli[21:41] = ['Object'] * 20
    # letters
    Triggers.Stimuli[41:61] = ['Letter'] * 20
    # false fonts
    Triggers.Stimuli[61:81] = ['False'] * 20

    # trial onsets (which according to our current scheme is the same as the stimulus category trigger)
    Triggers.TrialOnset = np.arange(1, 81)

    # trial numbers
    Triggers.TrialIDs = np.arange(111, 149)

    # orientation
    Triggers.Orientation = {'101': 'Center', '102': 'Left', '103': 'Right'}

    # duration
    Triggers.Duration = {'151': 'Short', '152': 'Medium', '153': 'Long'}

    # task
    Triggers.Task = {'201': 'Target', '202': 'NonTarget', '203': 'Irrelevant'}

    params['Triggers'] = Triggers

    # parameters relevant for saccade and microsaccade detection
    params['SaccadeDetection'] = {'threshold': 1,  # upper cutoff for microsaccades (in degrees)
                                  'msOverlap': 2,  # number of overlapping points to count as a binocular saccade
                                  'vfac': 5,  # will be multiplied by E&K criterion to get velocity threshold
                                  'mindur': 5,  # minimum duration of a microsaccade (in indices or samples)
                                  }
    return params


def ExtractTimeStamps(eyeDFs, triggers, log_file_path=None, logDF=None):
    """
    this function extracts the onset, stimulus ID, and trial type of each trial
    :param eyeDFs: the list of data frames that has the eye tracker data. this should be the output of the asc file
    parser function
    :param triggers: this holds the trigger IDs and their corresponding lables. it should be the Triggers key in the
    params dictionary
    :param log_file_path: the path to the behavioral log file for the subject
    :param logDF: alternatively, you can provide a dataframe holding the log data if not providing the log path
    :return:TimeStamps: a dataframe with a number of rows equal to the number of trials and 4 columns corresponding to
    trial number, stimulus ID, trial onset, and event type
    """

    # get the msgs data frame
    msgs = eyeDFs[1]

    # get the indices of the TRIALID messages
    trialOnsetTriggers = [str(a) for a in triggers.TrialOnset]
    trialIDs = [str(b) for b in triggers.TrialIDs]
    trialIndcs = msgs[np.isin(msgs.text, trialOnsetTriggers)].index.values

    # get the behavioral logs
    if log_file_path is not None:
        if log_file_path.endswith('csv'):
            behLogs = pd.read_csv(log_file_path)
        else:
            behLogs = pd.read_excel(log_file_path)
    else:
        if logDF is None:
            raise InputError('A behavioral log file is required for timestamp extraction but was not provided.')
        else:
            behLogs = logDF
    # get only the stimuli rows
    behLogs = behLogs.loc[behLogs['eventType'] == 'Stimulus', :]
    behLogs.reset_index(drop=True, inplace=True)

    # initialize the arrays that will hold trial information
    trialNo = np.zeros(trialIndcs.shape, dtype='int')
    stimID = np.empty(trialIndcs.shape, dtype='int')
    stimType = [None] * trialIndcs.shape[0]
    trialOnset = np.empty(trialIndcs.shape)
    relevance = [None] * trialIndcs.shape[0]  # target, irrelevant, nontarget
    duration = [None] * trialIndcs.shape[0]  # short, medium, long
    orientation = [None] * trialIndcs.shape[0]  # center, left, right
    targetStatus = [None] * trialIndcs.shape[0]  # previous target, current target, non-target

    # loop through the trial messages
    for trial in range(0, len(trialIndcs)):

        trialIdx = trialIndcs[trial]
        if trial < len(trialIndcs) - 1:
            nextTrialIdx = trialIndcs[trial + 1]
            # get the trial triggers
            trialTriggs = msgs.loc[trialIdx: nextTrialIdx - 1]
        else:
            # get the trial triggers
            trialTriggs = msgs.loc[trialIdx:]
        trialTriggs.reset_index(drop=True, inplace=True)   # reset index

        # get the event characterstics
        relx = trialTriggs[np.isin(trialTriggs.text, list(triggers.Task.keys()))].index.values[0]
        durx = trialTriggs[np.isin(trialTriggs.text, list(triggers.Duration.keys()))].index.values[0]
        orix = trialTriggs[np.isin(trialTriggs.text, list(triggers.Orientation.keys()))].index.values[0]
        relevance[trial] = triggers.Task[str(trialTriggs.text[relx])]
        duration[trial] = triggers.Duration[str(trialTriggs.text[durx])]
        orientation[trial] = triggers.Orientation[str(trialTriggs.text[orix])]
        # get the trial number
        trialNo[trial] = int(trialTriggs.text[np.isin(trialTriggs.text, trialIDs)])

        # get the current stimulus and its block
        currStim = str(int((behLogs['event'][trial])))
        currBlock = behLogs['miniBlock'][trial]

        # get the stimuli from the previous block and compare
        if currBlock > 1:
            preTargets = behLogs.loc[behLogs['miniBlock'] == currBlock - 1, ['targ1', 'targ2']]

            # compare to curr stimulus
            preTarg1 = str(preTargets['targ1'].iloc[-1])
            preTarg2 = str(preTargets['targ1'].iloc[-1])
            if (preTarg1[0] == currStim[0] and preTarg1[2:3] == currStim[2:3]) or (preTarg2[0] == currStim[0] and
                                                                                   preTarg2[2:3] == currStim[2:3]):
                targetStatus[trial] = 'Previous'
            if relevance[trial] == 'Target':
                targetStatus[trial] = 'Target'

            if not targetStatus[trial] == 'Previous' and not targetStatus[trial] == 'Target':
                targetStatus[trial] = 'NeverTarget'
        else:
            if not relevance[trial] == 'Target':
                targetStatus[trial] = 'NeverTarget'
            else:
                targetStatus[trial] = 'Target'

        # get the stimulus ID and type
        stimID[trial] = int(trialTriggs.text[0])
        stimType[trial] = triggers.Stimuli[stimID[trial]]

        # get the trial onset
        trialOnset[trial] = trialTriggs.time[0]

    # construct the data frame and return it
    TimeStamps = pd.DataFrame({'TrialNumber': trialNo, 'StimulusID': stimID, 'Category': stimType,
                               'TrialOnset': trialOnset, 'Relevance': relevance, 'Duration': duration,
                               'Orientation': orientation, 'TargetStatus': targetStatus})

    return TimeStamps


def SequenceEyeData(params, eyeDFs, timestamps):
    """
    segments the full eye data into trials. so at each trial timestamp, we grab the data from some pre-onset until
    sometime post-onset to get a window of data that associate with that trial
    :param params: the parameters dictionary. should be the output of initParams
    :param eyeDFs: the data frames with the eye tracker data. should be the ouptut of the asc parser function
    :param timestamps: the data frame holding information about trial timestamps and event types. should be the output
    of extractTimeStamps
    :return: TrialData: a list of dataframes holding the samples for each trial
    :return: TrialInfo: an updated version of the timestamps dataframe; updated to hold the TrialWindowStart and
    TrialWindowEnd columns
    """

    # get the dataframe with the data/samples
    data = eyeDFs[5]

    # get the trial window start and end for each trial
    trialOnset = np.array(timestamps['TrialOnset'])
    trialBegin = trialOnset - params['PreStim'] * 1000
    trialEnd = trialOnset + params['PostStim'] * 1000

    # initialize outputs
    TrialData = [None] * len(trialOnset)
    TrialInfo = timestamps.copy(deep=True)

    TrialInfo['TrialWindowStart'] = trialBegin
    TrialInfo['TrialWindowEnd'] = trialEnd

    print('num trials = ' + str(len(trialOnset)))

    # loop through trials
    # start tracking progress
    #bar = ibar('Sequencing', max=len(trialOnset), suffix='%(percent)d%%')
    for trial in range(0, len(trialOnset)):

        # get the indices for the start and end with which to index the data array
        stIdx = data.iloc[:, 0] == trialBegin[trial]
        endIdx = data.iloc[:, 0] == trialEnd[trial]

        if not np.sum(np.array(stIdx)) or not np.sum(np.array(endIdx)):
            stIdx = data.iloc[:, 0] == trialBegin[trial] - 1
            endIdx = data.iloc[:, 0] == trialEnd[trial] - 1

        if not np.sum(np.array(stIdx)) or not np.sum(np.array(endIdx)):  # at this point, something is wrong, raise
            # an error
            raise InputError('The indice of trial start and end could not be found for trial %d,'
                             ' check the raw data' % trial)

        # convert logical indices to numeric indices
        stIdx = stIdx[stIdx].index.values[0]
        endIdx = endIdx[endIdx].index.values[0]

        # get the trial's data
        TrialData[trial] = data.iloc[stIdx:endIdx, :]
        TrialData[trial].reset_index(drop=True, inplace=True)

        # raise an error if the trial data is empty
        #if TrialData[trial].empty:
            #raise InputError('Trial no. %d has no samples, check the raw data' % trial)

       # bar.next()
  #  bar.finish()

    return TrialData, TrialInfo


def RemoveBlinks(TrialData, params,modality):
    """
    this function takes in the sequenced TrialData and for each trial, it detects and removes blinks
    the blink detection script and algorithm used here are based on Hershman et al. 2018 (https://osf.io/jyz43/)
    :param TrialData: a list of dataframes wihere each element corresponds to the data frame of a single trial. this
    should be the output of the sequenceEyeData
    :param params: the parameters dictionary. should be the output of initParams
    :return: TrialDataNoBlinks: an updated version of trial data where the blinks have been removed
    :return: Blinks: a list of dataframes where each entry corresponds to a trial and holds the information about
    the starting time/index and ending time/index of each blink. the number of rows in a dataframe is the number of
    blinks in that trial.
    :return BlinkArray: an array with rows of zeros and ones with ones at the indices of blinks. Each row in
    the array is a trial and the number of columns is the total time points or number of samples associated with a
    trial.
    """
    # TODO: the way this is currently implemented, if a blink starts in a trial/segment and ends in another, it's
    #  counted twice, this needs to be solved

    # initialize the no blinks trial data
    TrialDataNoBlinks = copy.deepcopy(TrialData)

    # initialize the list of blinks
    Blinks = [None] * len(TrialData)

    # initialize the blink array
    BlinkArray = np.zeros([TrialData[0].shape[0], len(TrialData)])

    # loop through the trials
    for trial in range(0, len(TrialData)):

        # get the pupil size array for the eye of interest
        if modality == 'MEG':
            pupilArray = np.array((TrialDataNoBlinks[trial].LPupil+TrialDataNoBlinks[trial].RPupil)/2)
        elif params['Eye'] == 'L':
            pupilArray = np.array(TrialDataNoBlinks[trial].LPupil)
        else:
            pupilArray = np.array(TrialDataNoBlinks[trial].RPupil)

        # replace nan values with zeros
        pupilArray[np.isnan(pupilArray)] = 0

        # detect blinks
        trialBlinks = based_noise_blinks_detection(pupilArray, params['SamplingFrequency'])

        # put into the Blinks array
        Blinks[trial] = pd.DataFrame({'Onset': trialBlinks['blink_onset'], 'Offset': trialBlinks['blink_offset']})

        # loop through the blinks and replace them
        for bOn, bOff in zip(trialBlinks['blink_onset'], trialBlinks['blink_offset']):
            # replace with nans
            TrialDataNoBlinks[trial].iloc[int(bOn):int(bOff), :] = np.nan

            # add 1s in the blink array where there is a blink
            BlinkArray[int(bOn):int(bOff), trial] = 1

    return TrialDataNoBlinks, Blinks, BlinkArray.T


def RemoveSaccades(trialData, params, binocular=False, saccade_info=None):
    """
    This function takes in the sequenced trial data (after or before blinks are removed) and removes saccade data. If
    no saccade info (output of ExtractSaccades) is provided. The function makes the call to ExtractSaccades.
    :param trialData: a list of dataframes wihere each element corresponds to the data frame of a single trial. this
    should be the output of either the sequenceEyeData or RemoveBlinks
    :param params: the parameters dictionary. should be the output of initParams
    :param binocular: a flag to indicate whether or not to get binocular saccades. if true, the function will ignore
    the 'Eye' field in the parameters dict and will get microsaccades for both eyes, then detect binocluar ones based
    on overlap
    :param saccade_info: a list of dictionaries, one for each trial. A trial's dictionary holds the following fields:
            - 'Velocity': left, right, both
            - 'Microsaccades': left, right, both
            - 'Saccades': left, right, both
            - 'Threshold': in x and y, the treshold used for classifiying saccades
    :return: TrialDataNoSaccades: a list of dataframes (same structure as trialData) with the saccades removed.
    :return: saccade_info: if no saccade_info is provided as input, the function will create it and return it.
    """

    # initialize output
    TrialDataNoSaccades = copy.deepcopy(trialData)

    # get saccade info if not already given
    returnInfo = False
    if saccade_info is None:
        returnInfo = True
        # get the gaze data
        gazedata = [df[['LX', 'LY', 'RX', 'RY']] for df in trialData]
        saccade_info = ExtractSaccades(gazedata, params, binocular)

    # remove saccades (replacing them with nans)
    for trial in range(0, len(saccade_info)):
        if saccade_info[trial]['Saccades']['both'] is not None:
            allSaccSt = np.array(saccade_info[trial]['Saccades']['both']['start'])
            allSaccEnd = np.array(saccade_info[trial]['Saccades']['both']['end'])
        else:
            allSaccSt = np.array([])
            allSaccEnd = np.array([])
        if not len(allSaccSt) == 0:
            for sac in range(0, len(allSaccSt)):
                TrialDataNoSaccades[trial].iloc[int(allSaccSt[sac]):int(allSaccEnd[sac]), :] = np.nan

    if not returnInfo:
        return TrialDataNoSaccades
    else:
        return TrialDataNoSaccades, saccade_info


def ExtractSaccades(gazeData, params, getBinocular=False):
    """
    This is the main function responsible for extracting saccadees and microsaccades. it calls several other helper
    functions that do some heavy lifting.
    :param gazeData: the gaze data segmented per trial. this should be a a list of ntrials dataframes with each
    dataframe holding 4 columns: LX, LY, RX, RY
    :param params: the parameters dictionary. should be the output of initParams
    :param getBinocular: a flag to indicate whether or not to get binocular saccades. if true, the function will ignore
    the 'Eye' field in the parameters dict and will get microsaccades for both eyes, then detect binocluar ones based
    on overlap
    :return: SaccadeInfo: a list of dictionaries, one for each trial. A trial's dictionary holds the following fields:
            - 'Velocity': left, right, both
            - 'Microsaccades': left, right, both
            - 'Saccades': left, right, both
            - 'Threshold': in x and y, the treshold used for classifiying saccades
    """
    # get the parameters relevant for saccade detection
    saccParams = params['SaccadeDetection']

    SaccadeInfo = [None] * len(gazeData)

    # loop through the trials' gaze data
    for trial in range(0, len(gazeData)):
        # initialize the Saccades list
        saccDict = {'Velocity': {'left': None, 'right': None},
                    'Microsaccades': {'left': None, 'right': None, 'both': None},
                    'Saccades': {'left': None, 'right': None, 'both': None},
                    'Threshold': None}
        # get the trials gaze data and convert it to degrees relative to screen center
        trialGaze = np.array(gazeData[trial])
        # convert zeros to nans
        trialGaze[trialGaze == 0] = np.nan
        # make it relative to screen center
        trialGaze[:, [0, 2]] = trialGaze[:, [0, 2]] - (params['ScreenResolution'][0] / 2)
        trialGaze[:, [1, 3]] = -(trialGaze[:, [1, 3]]) - (params['ScreenResolution'][1] / 2)  # y data is inverted
        trialGaze = trialGaze * params['cmPerPixel'][0]  # convert to cm
        trialGaze = np.degrees(np.arctan(trialGaze / params['ViewDistance']))  # convert to degrees

        # if we are not getting binocular microsaccades, get the gaze for the specified eye in parameters
        if not getBinocular:
            if params['Eye'] == 'L':
                eyeGazes = [trialGaze[:, [0, 1]]]
                eyes = ['left']
            else:
                eyeGazes = [trialGaze[:, [2, 3]]]
                eyes = ['right']
        else:
            eyeGazes = [trialGaze[:, [0, 1]], trialGaze[:, [2, 3]]]
            eyes = ['left', 'right']

        # get monocular microsaccades
        for eye in range(0, len(eyes)):  # for each eye
            # gaze data for this eye
            eyeGaze = eyeGazes[eye]
            # get velocity
            velocity, speed = GetVelocity(eyeGaze, params['SamplingFrequency'])
            # get the saccades
            trialsacc, radius = ExtractMonocularMS(eyeGaze, velocity, params)[0:2]

            # fill out the saccadeinfo list
            saccDict['Threshold'] = radius
            saccDict['Velocity'][eyes[eye]] = velocity
            saccDict['Saccades'][eyes[eye]] = trialsacc

            # get the indices of the microsaccades (saccades less than threshold in amplitude)
            if trialsacc is not None:
                indMS = trialsacc['total_amplitude'] < saccParams['threshold']
                saccDict['Microsaccades'][eyes[eye]] = trialsacc.loc[indMS, :].reset_index()

        # get binocular saccades
        saccLeft = saccDict['Saccades']['left']
        saccRight = saccDict['Saccades']['right']
        microLeft = saccDict['Microsaccades']['left']
        microRight = saccDict['Microsaccades']['right']
        if saccLeft is not None and saccRight is not None:
            # for microsaccades only (saccades less than threshold)
            if not microLeft.empty and not microRight.empty:
                ind_both = []
                for k in range(0, microLeft.shape[0]):
                    # get the maximum overlap between this left ms and all right ms
                    max_intersect = 0
                    for j in range(0, microRight.shape[0]):
                        L = len(np.intersect1d(np.arange(microLeft['start'][k], microLeft['end'][k]),
                                               np.arange(microRight['start'][j], microRight['end'][j])))
                        if L > max_intersect:
                            max_intersect = L

                    # check overlap criteria
                    if max_intersect >= saccParams['msOverlap']:
                        ind_both.append(k)

                # add the binocular microsaccades
                saccDict['Microsaccades']['both'] = microLeft.iloc[ind_both, :]

            # for all saccades
            ind_both = []
            for k in range(0, saccLeft.shape[0]):
                # get the maximum overlap between this left saccade and all right saccades
                max_intersect = 0
                for j in range(0, saccRight.shape[0]):
                    L = len(np.intersect1d(np.arange(saccLeft['start'][k], saccLeft['end'][k]),
                                           np.arange(saccRight['start'][j], saccRight['end'][j])))
                    if L > max_intersect:
                        max_intersect = L

                # check overlap criteria
                if max_intersect >= saccParams['msOverlap']:
                    ind_both.append(k)

            # add the binocular saccades
            saccDict['Saccades']['both'] = saccLeft.iloc[ind_both, :]

        SaccadeInfo[trial] = saccDict

    return SaccadeInfo


def GetVelocity(eyeGaze, fs):
    """

    :param eyeGaze:
    :param fs:
    :return:
    """
    # initialize outputs
    velocity = np.zeros(eyeGaze.shape)
    speed = np.zeros((velocity.shape[0], 1))

    # loop through the data points and calculate a moving average of velocities over 5
    # data samples
    for n in range(2, eyeGaze.shape[0] - 2):
        velocity[n, :] = (eyeGaze[n + 1, :] + eyeGaze[n + 2, :] - eyeGaze[n - 1, :] - eyeGaze[n - 2, :]) * (fs / 6)

    # calculate speed
    speed[:, 0] = np.sqrt(np.power(velocity[:, 0], 2) + np.power(velocity[:, 1], 2))

    return velocity, speed


def ExtractMonocularMS(eyeGaze, velocity, params, msdx=None, msdy=None):
    """
    This function extracts microsaccades (and generally all saccades) in one eye.
    This is based on Engbert, R., & Mergenthaler, K. (2006) Microsaccades are triggered by low retinal image slip.
    Proceedings of the National Academy of Sciences of the United States of America, 103: 7192-7197.

    :param eyeGaze:
    :param params:
    :param velocity:
    :param msdx:
    :param msdy:
    :return:
    """

    # saccade extraction parameters
    saccParams = params['SaccadeDetection']

    # get the velocity thresholds if they are not given as inputs
    if msdx is None or msdy is None:
        # if only one data point exists, replace entire trial with nans
        if sum(~np.isnan(velocity[:, 0])) == 1 or sum(~np.isnan(velocity[:, 1])) == 1:
            velocity[:, 0] = np.nan
            velocity[:, 1] = np.nan
        elif all(velocity[~np.isnan(velocity[:, 0]), 0] == 0) or all(velocity[~np.isnan(velocity[:, 1]), 1] == 0):
            velocity[:, 0] = np.nan
            velocity[:, 1] = np.nan

        MSDX, MSDY, stddev, maddev = GetVelocityThreshold(velocity)
        if msdx is None:
            msdx = MSDX
        if msdy is None:
            msdy = MSDY

    else:
        _, _, stddev, maddev = GetVelocityThreshold(velocity)

    # begin saccade detection
    radiusx = saccParams['vfac'] * msdx
    radiusy = saccParams['vfac'] * msdy
    radius = np.array([radiusx, radiusy])

    # compute test criterion: ellipse equation
    test = np.power((velocity[:, 0] / radiusx), 2) + np.power((velocity[:, 1] / radiusy), 2)
    indx = np.argwhere(test > 1)

    # determine saccades
    N = len(indx)
    sac = np.zeros((1, 10))
    nsac = 0
    dur = 1
    a = 0
    k = 0
    while k < N - 1:
        if indx[k + 1] - indx[k] == 1:
            dur = dur + 1
        else:
            if dur >= saccParams['mindur']:
                nsac = nsac + 1
                b = k
                if nsac == 1:
                    sac[0][0] = indx[a]
                    sac[0][1] = indx[b]
                else:
                    sac = np.vstack((sac, np.array([indx[a], indx[b], 0, 0, 0, 0, 0, 0, 0, 0])))
            a = k + 1
            dur = 1
        k = k + 1

    # check duration criterion for the last microsaccade
    if dur >= saccParams['mindur']:
        nsac = nsac + 1
        b = k
        if nsac == 1:
            sac[0][0] = indx[a]
            sac[0][1] = indx[b]
        else:
            sac = np.vstack((sac, np.array([indx[a], indx[b], 0, 0, 0, 0, 0, 0, 0, 0])))

    # compute peak velocity, horizontal and vertical components, amplitude, and gaze direction
    if nsac > 0:
        for s in range(0, nsac):
            # Onset and offset
            a = int(sac[s][0])
            b = int(sac[s][1])
            idx = range(a, b)

            # peak velocity
            peakvel = max(np.sqrt(velocity[idx, 0] ** 2 + velocity[idx, 1] ** 2))
            sac[s][2] = peakvel

            # horz and vert components
            dx = eyeGaze[b, 0] - eyeGaze[a, 0]
            dy = eyeGaze[b, 1] - eyeGaze[a, 1]
            sac[s][3] = dx
            sac[s][4] = dy

            # amplitude (dX,dY)
            minx = min(eyeGaze[idx, 0])
            maxx = max(eyeGaze[idx, 0])
            miny = min(eyeGaze[idx, 1])
            maxy = max(eyeGaze[idx, 1])
            ix1 = np.argmin(eyeGaze[idx, 0])
            ix2 = np.argmax(eyeGaze[idx, 0])
            iy1 = np.argmin(eyeGaze[idx, 1])
            iy2 = np.argmax(eyeGaze[idx, 1])
            dX = np.sign(ix2 - ix1) * (maxx - minx)
            dY = np.sign(iy2 - iy1) * (maxy - miny)
            sac[s][5] = dX
            sac[s][6] = dY

            # total amplitude
            sac[s][7] = np.sqrt(dX ** 2 + dY ** 2)

            # saccade distance to fixation (screen center)
            gazeOnset = eyeGaze[a, :]
            gazeOffset = eyeGaze[b, :]
            distToFixOnset = np.sqrt((gazeOnset[0] - params['ScreenCenter'][0] * params['DegreesPerPixel']) ** 2 +
                                     (gazeOnset[1] - params['ScreenCenter'][1] * params['DegreesPerPixel']) ** 2)
            distToFixOffset = np.sqrt((gazeOffset[0] - params['ScreenCenter'][0] * params['DegreesPerPixel']) ** 2 +
                                      (gazeOffset[1] - params['ScreenCenter'][1] * params['DegreesPerPixel']) ** 2)
            distToFix = (distToFixOffset - distToFixOnset)
            sac[s][8] = distToFix

            # saccade direction
            rad = np.arccos((gazeOffset[0] - gazeOnset[0]) / np.sqrt((gazeOffset[0] - gazeOnset[0]) ** 2 +
                                                                     (gazeOffset[1] - gazeOnset[1]) ** 2))
            angle = np.degrees(rad)
            if (gazeOffset[1] - gazeOnset[1]) >= 0:
                sac[s][9] = angle
            else:
                sac[s][9] = 360 - angle

        # convert to a dataframe
        sacdf = pd.DataFrame(data=sac,
                             columns=['start', 'end', 'peak_velocity', 'dx', 'dy', 'x_amplitude', 'y_amplitude',
                                      'total_amplitude', 'distance_to_fixation', 'direction'])
        sacdf['start'] = sacdf['start'].apply(int)
        sacdf['end'] = sacdf['end'].apply(int)

    else:
        sacdf = None

    # return all values that were relevant for detection, if a user function doesn't want all values it can just select
    return sacdf, radius, msdx, msdy, stddev, maddev


def GetVelocityThreshold(velocity):
    """

    :param velocity:
    :return: msdx:
    :return msdy:
    :return stddev:
    :return maddev:
    """
    # compute threshold
    msdx = np.sqrt(np.nanmedian(np.power(velocity[:, 0], 2)) - np.power(np.nanmedian(velocity[:, 0]), 2))
    msdy = np.sqrt(np.nanmedian(np.power(velocity[:, 1], 2)) - np.power(np.nanmedian(velocity[:, 1]), 2))

    if msdx < np.finfo('float').tiny:  # if less than the smallest usable float
        # switch to a mean estimator instead and see
        msdx = np.sqrt(np.nanmean(np.power(velocity[:, 0], 2)) - np.power(np.nanmean(velocity[:, 0]), 2))
        # raise an error if still smaller
        if msdx < np.finfo('float').tiny:
            raise RepresentationError('Calculated velocity threshold (msdx) was smaller than the smallest '
                                      'positive representable floating-point number. Did you exclude blinks/'
                                      'missing data before saccade detection?')

    # do the same for the y-component
    if msdy < np.finfo('float').tiny:  # if less than the smallest usable float
        # switch to a mean estimator instead and see
        msdy = np.sqrt(np.nanmean(np.power(velocity[:, 1], 2)) - np.power(np.nanmean(velocity[:, 1]), 2))
        # raise an error if still smaller
        if msdy < np.finfo('float').tiny:
            raise RepresentationError('Calculated velocity threshold (msdy) was smaller than the smallest '
                                      'positive representable floating-point number. Did you exclude blinks/'
                                      'missing data before saccade detection?')

    # compute the standard deviation and the median abs deviation for the velocity values in both components
    stddev = np.nanstd(velocity, axis=0, ddof=1)
    maddev = stats.median_abs_deviation(velocity, axis=0, nan_policy='omit')

    return msdx, msdy, stddev, maddev


def ParseEyeLinkAsc(elFilename):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file 
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples

    # Read in EyeLink file
    print('Reading in EyeLink file %s...' % elFilename)
    t = time.time()
    f = open(elFilename, 'r')
    fileTxt0 = f.read().splitlines(True)  # split into lines
    fileTxt0 = list(filter(None, fileTxt0))  # remove emptys
    fileTxt0 = np.array(fileTxt0)  # concert to np array for simpler indexing
    f.close()
    print('Done! Took %f seconds.' % (time.time() - t))

    # Separate lines into samples and messages
    print('Sorting lines...')
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER'] * nLines, dtype='object')
    iStartRec = list([])
    t = time.time()
    for iLine in range(nLines):
        if len(fileTxt0[iLine]) < 3:
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif bool(len(fileTxt0[iLine][0])) and fileTxt0[iLine][0].isdigit():
            fileTxt0[iLine] = fileTxt0[iLine].replace(' . ', ' NaN ')
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if '!MODE RECORD' in fileTxt0[iLine]:  # TODO is this the best way to detect the start of recording?
            iStartRec.append(iLine + 1)

    iStartRec = iStartRec[0]
    print('Done! Took %f seconds.' % (time.time() - t))

    # ===== PARSE EYELINK FILE ===== #
    t = time.time()
    # Trials
    print('Parsing recording markers...')
    iNotStart = np.nonzero(lineType != 'START')[0]
    dfRecStart = pd.read_csv(elFilename, skiprows=iNotStart, header=None, delim_whitespace=True, usecols=[1])
    dfRecStart.columns = ['tStart']
    iNotEnd = np.nonzero(lineType != 'END')[0]
    dfRecEnd = pd.read_csv(elFilename, skiprows=iNotEnd, header=None, delim_whitespace=True, usecols=[1, 5, 6])
    dfRecEnd.columns = ['tEnd', 'xRes', 'yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart, dfRecEnd], axis=1)
    nRec = dfRec.shape[0]
    print('%d recording periods found.' % nRec)

    # Import Messages
    print('Parsing stimulus messages...')
    t = time.time()
    iMsg = np.nonzero(lineType == 'MSG')[0]
    # set up
    tMsg = []
    txtMsg = []
    t = time.time()
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    # Convert dict to dataframe
    dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})
    print('Done! Took %f seconds.' % (time.time() - t))

    # Import Fixations
    print('Parsing fixations...')
    t = time.time()
    iNotEfix = np.nonzero(lineType != 'EFIX')[0]
    dfFix = pd.read_csv(elFilename, skiprows=iNotEfix, header=None, delim_whitespace=True, usecols=range(1, 8))
    dfFix.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']
    nFix = dfFix.shape[0]
    print('Done! Took %f seconds.' % (time.time() - t))

    # Saccades
    print('Parsing saccades...')
    t = time.time()
    iNotEsacc = np.nonzero(lineType != 'ESACC')[0]
    dfSacc = pd.read_csv(elFilename, skiprows=iNotEsacc, header=None, delim_whitespace=True, usecols=range(1, 11))
    dfSacc.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']
    print('Done! Took %f seconds.' % (time.time() - t))

    # Blinks
    print('Parsing blinks...')
    iNotEblink = np.nonzero(lineType != 'EBLINK')[0]
    dfBlink = pd.read_csv(elFilename, skiprows=iNotEblink, header=None, delim_whitespace=True, usecols=range(1, 5))
    dfBlink.columns = ['eye', 'tStart', 'tEnd', 'duration']
    print('Done! Took %f seconds.' % (time.time() - t))

    # determine sample columns based on eyes recorded in file
    eyesInFile = np.unique(dfFix.eye)
    if eyesInFile.size == 2:
        print('binocular data detected.')
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile[0]
        print('monocular data detected (%c eye).' % eye)
        cols = ['tSample', '%cX' % eye, '%cY' % eye, '%cPupil' % eye]
    # Import samples
    print('Parsing samples...')
    t = time.time()
    iNotSample = np.nonzero(np.logical_or(lineType != 'SAMPLE', np.arange(nLines) < iStartRec))[0]
    dfSamples = pd.read_csv(elFilename, skiprows=iNotSample, header=None, delim_whitespace=True,
                            usecols=range(0, len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyesInFile:
            dfSamples['%cX' % eye] = pd.to_numeric(dfSamples['%cX' % eye], errors='coerce')
            dfSamples['%cY' % eye] = pd.to_numeric(dfSamples['%cY' % eye], errors='coerce')
            dfSamples['%cPupil' % eye] = pd.to_numeric(dfSamples['%cPupil' % eye], errors='coerce')
        else:
            dfSamples['%cX' % eye] = np.nan
            dfSamples['%cY' % eye] = np.nan
            dfSamples['%cPupil' % eye] = np.nan

    print('Done! Took %.1f seconds.' % (time.time() - t))

    # Return new compilation dataframe
    return dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples


def main():
    """ for debugging purposes """

    elFilename = r'Data/Exp 1/MP101/eyetracker/MP101.asc'

    # important parameters
    scw = 53
    sch = 30
    vd = 68
    eye = 'L'
    fs = 500
    pname = 'MP101'
    exp = 1
    logPath = r'Data/Exp 1/MP101/behavioral/Seattle_ResultsS101_b1_101-02-Oct-2019.xlsx'

    # parse
    dataEye = ParseEyeLinkAsc(elFilename)

    # initialize all parameters
    params = InitParams(scw, sch, vd, pname, fs, eye)

    # event timestamps
    print('Extracting event timestamps...')
    timestamps = ExtractTimeStamps(dataEye, params['Triggers'], logPath)
    print('Done!')

    # trial data
    print('Sequencing data into trials...')
    trialdata = SequenceEyeData(params, dataEye, timestamps)
    print('Done!')

    # blink removal
    print('Removing blinks...')
    trialdataNoBlinks = RemoveBlinks(trialdata[0], params)[0]
    print('Done!')

    # saccade detection
    gazedata = [df[['LX', 'LY', 'RX', 'RY']] for df in trialdataNoBlinks]
    print('Extracting saccades...')
    saccadeinfo = ExtractSaccades(gazedata, params, getBinocular=True)
    trialDataNbNs = RemoveSaccades(trialdataNoBlinks, params, binocular=True, saccade_info=saccadeinfo)
    print('Done!')


if __name__ == '__main__':
    main()


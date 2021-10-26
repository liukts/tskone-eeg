# coding=utf-8
'''
Reach-to-grasp IO module

This module provides an IO to load data recorded in the context of the reach-
to-grasp experiments conducted by Thomas Brochier and Alexa Riehle at the
Institute de Neurosciences de la Timone. The IO is based on the BlackrockIO of
the Neo library, which is used in the background to load the primary data, and
utilized the odML library to load metadata information. Specifically, this IO
annotates the Neo object returned by BlackrockIO with semantic information,
e.g., interpretation of digital event codes, and key-value pairs found in the
corresponding odML file are attached to relevant Neo objects as annotations.

Authors: Julia Sprenger, Lyuba Zehl, Michael Denker


Copyright (c) 2017, Institute of Neuroscience and Medicine (INM-6),
Forschungszentrum Juelich, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import glob
import os
import re

import numpy as np
import odml.tools
import quantities as pq

import neo

# Using old version of IO as long as available - should be deleted at some point
try:
    from neo.io import OldBlackrockIO as BlackrockIO
except ImportError:
    from neo.io.blackrockio import BlackrockIO


class ReachGraspIO(BlackrockIO):
    """
    Derived class from Neo's BlackrockIO to load recordings obtained from the
    reach-to-grasp experiments.

    Args:
        filename (string):
            File name (without extension) of the set of Blackrock files to
            associate with. Any .nsX or .nev, .sif, or .ccf extensions are
            ignored when parsing this parameter. Note: unless the parameter
            nev_override is given, this IO will load the nev file containing
            the most recent spike sorted data of all nev files found in the
            same directory as filename. The spike sorting version is attached
            to filename by a postfix '-XX', where XX is the version, e.g.,
            l101010-001-02 for spike sorting version 2 of file l101010-001. If
            an odML file is specified, the version must be listed in the odML
            entries at
            "/PreProcessing/OfflineSpikeSorting/Sortings"
            and relates to the section
            "/PreProcessing/OfflineSpikeSorting/Sorting-XX".
            If no odML is present, no information on the spike sorting (e.g.,
            if a unit is SUA or MUA) is provided by this IO.
        odml_directory (string):
            Alternative directory where the odML file is stored. If None, the
            directory is assumed to be the same as the .nev and .nsX data
            files. Default: None.
        nsx_override (string):
            File name of the .nsX files (without extension). If None,
            filename is used.
            Default: None.
        nev_override (string):
            File name of the .nev file (without extension). If None, the
            current spike-sorted version filename is used (see parameter
            filename above). Default: None.
        sif_override (string):
            File name of the .sif file (without extension). If None,
            filename is used.
            Default: None.
        ccf_override (string):
            File name of the .ccf file (without extension). If None,
            filename is used.
            Default: None.
        odml_override (string):
            File name of the .odml file (without extension). If None,
            filename is used.
            Default: None.
        verbose (boolean):
            If True, the class will output additional diagnostic
            information on stdout.
            Default: False

    Returns:
        -

    Attributes:
        condition_str (dict):
            Dictionary containing a list of string codes reflecting the trial
            types that occur in recordings in a certain condition code
            (dictionary keys). For example, for condition 1 (all grip first
            conditions), condition_str[1] contains the list
            ['SGHF', 'SGLF', 'PGHF', 'PGLF'].
            Possible conditions:
               0:[]
                 No trials, or condition not conclusive from file
               4 types (two_cues_task):
                 1: all grip-first trial types with two different cues
                 2: all force-first trial types with two different cues
               2 types (two_cues_task):
                 11: grip-first, but only LF types
                 12: grip-first, but only HF types
                 13: grip-first, but only SG types
                 14: grip-first, but only PG types
               2 types (two_cues_task):
                 21: force-first, but only LF types
                 22: force-first, but only HF types
                 23: force-first, but only SG types
                 24: force-first, but only PG types
               1 type (two_cues_task):
                 131: grip-first, but only SGLF type
                 132: grip-first, but only SGHF type
                 141: grip-first, but only PGLF type
                 142: grip-first, but only PGHF type
                 213: force-first, but only LFSG type
                 214: force-first, but only LFPG type
                 223: force-first, but only HFSG type
                 224: force-first, but only HFPG type
               1 type (one_cue_task):
                 133: SGSG, only grip info, force unknown
                 144: PGPG, only grip info, force unknown
                 211: LFLF, only force info, grip unknown
                 222: HFHF, only force info, grip unknown
        event_labels_str (dict):
            Provides a text label for each digital event code returned as
            events by the parent BlackrockIO. For example,
            event_labels_str['65296'] contains the string 'TS-ON'.
        event_labels_codes (dict):
            Reverse of `event_labels_str`: Provides a list of event codes
            related to a specific text label for a trial event. For example,
            event_labels_codes['TS-ON'] contains the list ['65296']. In
            addition to the detailed codes, for convenience the meta codes
            'CUE/GO', 'RW-ON', and 'SR' summarizing a set of digital events are
            defined for easier access.
        trial_const_sequence_str (dict):
            Dictionary contains the ordering of selected constant trial events
            for correct trials, e.g., as TS is the first trial event in a
            correct trial, trial_const_sequence_codes['TS'] is 0.
        trial_const_sequence_codes (dict):
            Reverse of trial_const_sequence_str: Dictionary contains the
            ordering of selected constant trial events for correct trials,
            e.g., trial_const_sequence_codes[0] is 'TS'.
        performance_str (dict):
            Text strings to help interpret the performance code of a trial. For
            example, correct trials have a performance code of 255, and thus
            performance_str[255] == 'correct_trial'
        performance_codes (dict):
            Reverse of performance_const_sequence_str. Returns the performance
            code of a given text string indicating trial performance. For
            example, performance_str['correct_trial'] == 255
    """

    # Create a dictionary of conditions (i.e., the trial types presented in a
    # given recording session)
    condition_str = {
        0: [],
        1: ['SGHF', 'SGLF', 'PGHF', 'PGLF'],
        2: ['HFSG', 'HFPG', 'LFSG', 'LFPG'],
        11: ['SGLF', 'PGLF'],
        12: ['SGHF', 'PGHF'],
        13: ['SGHF', 'SGLF'],
        14: ['PGHF', 'PGLF'],
        21: ['LFSG', 'LFPG'],
        22: ['HFSG', 'HFPG'],
        23: ['HFSG', 'LFSG'],
        24: ['HFPG', 'LFPG'],
        131: ['SGLF'],
        132: ['SGHF'],
        133: ['SGSG'],
        141: ['PGLF'],
        142: ['PGHF'],
        144: ['PGPG'],
        211: ['LFLF'],
        213: ['LFSG'],
        214: ['LFPG'],
        222: ['HFHF'],
        223: ['HFSG'],
        224: ['HFPG']}

    ###########################################################################
    # event labels, the corresponding first 8 digits of their binary
    # representation and their meaning
    #
    #         R L T T L L L L
    #         w E a r E E E E
    #         P D S S D D D D                                               in
    #         u c w t b t t b                                               mo-
    #                 l r l r                                               nk-
    # label:| ^ ^ ^ ^ ^ ^ ^ ^ | status of devices:    | trial event label:| ey
    # 65280 < 0 0 0 0 0 0 0 0 > TS-OFF                > TS-OFF/STOP       > L,T
    # 65296 < 0 0 0 1 0 0 0 0 > TS-ON                 > TS-ON             > all
    # 65312 < 0 0 1 0 0 0 0 0 > TaSw                  > STOP              > all
    # 65344 < 0 1 0 0 0 0 0 0 > LEDc       (+TS-OFF)  > WS-ON/CUE-OFF     > L,T
    # 65349 < 0 1 0 0 0 1 0 1 > LEDc|rt|rb (+TS-OFF)  > PG-ON (CUE/GO-ON) > L,T
    # 65350 < 0 1 0 0 0 1 1 0 > LEDc|tl|tr (+TS-OFF)  > HF-ON (CUE/GO-ON) > L,T
    # 65353 < 0 1 0 0 1 0 0 1 > LEDc|bl|br (+TS-OFF)  > LF-ON (CUE/GO-ON) > L,T
    # 65354 < 0 1 0 0 1 0 1 0 > LEDc|lb|lt (+TS-OFF)  > SG-ON (CUE/GO-ON) > L,T
    # 65359 < 0 1 0 0 1 1 1 1 > LEDall                > ERROR-FLASH-ON    > L,T
    # 65360 < 0 1 0 1 0 0 0 0 > LEDc       (+TS-ON)   > WS-ON/CUE-OFF     > N
    # 65365 < 0 1 0 1 0 1 0 1 > LEDc|rt|rb (+TS-ON)   > PG-ON (CUE/GO-ON) > N
    # 65366 < 0 1 0 1 0 1 1 0 > LEDc|tl|tr (+TS-ON)   > HF-ON (CUE/GO-ON) > N
    # 65369 < 0 1 0 1 1 0 0 1 > LEDc|bl|br (+TS-ON)   > LF-ON (CUE/GO-ON) > N
    # 65370 < 0 1 0 1 1 0 1 0 > LEDc|lb|lt (+TS-ON)   > SG-ON (CUE/GO-ON) > N
    # 65376 < 0 1 1 0 0 0 0 0 > LEDc+TaSw             > GO-OFF/RW-OFF     > all
    # 65381 < 0 1 1 0 0 1 0 1 > TaSw (+LEDc|rt|rb)    > SR (+PG)          > all
    # 65382 < 0 1 1 0 0 1 1 0 > TaSw (+LEDc|tl|tr)    > SR (+HF)          > all
    # 65383 < 0 1 1 0 0 1 1 1 > TaSw (+LEDc|rt|rb|tl) > SR (+PGHF/HFPG)   >
    # 65385 < 0 1 1 0 1 0 0 1 > TaSw (+LEDc|bl|br)    > SR (+LF)          > all
    # 65386 < 0 1 1 0 1 0 1 0 > TaSw (+LEDc|lb|lt)    > SR (+SG)          > all
    # 65387 < 0 1 1 0 1 0 1 1 > TaSw (+LEDc|lb|lt|br) > SR (+SGLF/LGSG)   >
    # 65389 < 0 1 1 0 1 1 0 1 > TaSw (+LEDc|rt|rb|bl) > SR (+PGLF/LFPG)   >
    # 65390 < 0 1 1 0 1 1 1 0 > TaSw (+LEDc|lb|lt|tr) > SR (+SGHF/HFSG)   >
    # 65391 < 0 1 1 0 1 1 1 1 > LEDall (+TaSw)        > ERROR-FLASH-ON    > L,T
    # 65440 < 1 0 1 0 0 0 0 0 > RwPu (+TaSw)          > RW-ON (noLEDs)    > N
    # 65504 < 1 1 1 0 0 0 0 0 > RwPu (+LEDc)          > RW-ON (-CONF)     > L,T
    # 65509 < 1 1 1 0 0 1 0 1 > RwPu (+LEDcr)         > RW-ON (+CONF-PG)  > all
    # 65510 < 1 1 1 0 0 1 1 0 > RwPu (+LEDct)         > RW-ON (+CONF-HF)  > N?
    # 65513 < 1 1 1 0 1 0 0 1 > RwPu (+LEDcb)         > RW-ON (+CONF-LF)  > N?
    # 65514 < 1 1 1 0 1 0 1 0 > RwPu (+LEDcl)         > RW-ON (+CONF-SG)  > all
    #         ^ ^ ^ ^ ^ ^ ^ ^
    #        label binary code
    #
    # ABBREVIATIONS:
    # c (central), l (left), t (top), b (bottom),  r (right),
    # HF (high force, LEDt), LF (low force, LEDb), SG (side grip, LEDl),
    # PG (precision grip, LEDr), RwPu (reward pump), TaSw (table switch),
    # TS (trial start), SR (switch release), WS (warning signal), RW (reward),
    # L (Lilou), T (Tanya t+a), N (Nikos n+i)
    ###########################################################################

    # Create dictionaries for event labels
    event_labels_str = {
        '65280': 'TS-OFF/STOP',
        '65296': 'TS-ON',
        '65312': 'STOP',
        '65344': 'WS-ON/CUE-OFF',
        '65349': 'PG-ON',
        '65350': 'HF-ON',
        '65353': 'LF-ON',
        '65354': 'SG-ON',
        '65359': 'ERROR-FLASH-ON',
        '65360': 'WS-ON/CUE-OFF',
        '65365': 'PG-ON',
        '65366': 'HF-ON',
        '65369': 'LF-ON',
        '65370': 'SG-ON',
        '65376': 'GO/RW-OFF',
        '65381': 'SR (+PG)',
        '65382': 'SR (+HF)',
        '65383': 'SR (+PGHF/HFPG)',
        '65385': 'SR (+LF)',
        '65386': 'SR (+SG)',
        '65387': 'SR (+SGLF/LFSG)',
        '65389': 'SR (+PGLF/LFPG)',
        '65390': 'SR (+SGHF/HFSG)',
        '65391': 'ERROR-FLASH-ON',
        '65440': 'RW-ON (noLEDs)',
        '65504': 'RW-ON (-CONF)',
        '65509': 'RW-ON (+CONF-PG)',
        '65510': 'RW-ON (+CONF-HF)',
        '65513': 'RW-ON (+CONF-LF)',
        '65514': 'RW-ON (+CONF-SG)'}
    event_labels_codes = dict(
        [(k, []) for k in np.unique(list(event_labels_str.values()))]) #inefficient in Python2
    for k in list(event_labels_codes):
        for l, v in event_labels_str.items(): #inefficient in Python2
            if v == k:
                event_labels_codes[k].append(l)

    # additional summaries
    event_labels_codes['CUE/GO'] = \
        event_labels_codes['SG-ON'] + \
        event_labels_codes['PG-ON'] + \
        event_labels_codes['LF-ON'] + \
        event_labels_codes['HF-ON']
    event_labels_codes['RW-ON'] = \
        event_labels_codes['RW-ON (+CONF-PG)'] + \
        event_labels_codes['RW-ON (+CONF-HF)'] + \
        event_labels_codes['RW-ON (+CONF-LF)'] + \
        event_labels_codes['RW-ON (+CONF-SG)'] + \
        event_labels_codes['RW-ON (-CONF)'] + \
        event_labels_codes['RW-ON (noLEDs)']
    event_labels_codes['SR'] = \
        event_labels_codes['SR (+PG)'] + \
        event_labels_codes['SR (+HF)'] + \
        event_labels_codes['SR (+LF)'] + \
        event_labels_codes['SR (+SG)'] + \
        event_labels_codes['SR (+PGHF/HFPG)'] + \
        event_labels_codes['SR (+SGHF/HFSG)'] + \
        event_labels_codes['SR (+PGLF/LFPG)'] + \
        event_labels_codes['SR (+SGLF/LFSG)']
    del k, l, v

    # Create dictionaries for constant trial sequences (in all monkeys)
    # (bit position (value) set if trial event (key) occurred)
    trial_const_sequence_codes = {
        'TS-ON': 0,
        'WS-ON': 1,
        'CUE-ON': 2,
        'CUE-OFF': 3,
        'GO-ON': 4,
        'SR': 5,
        'RW-ON': 6,
        'STOP': 7}
    trial_const_sequence_str = dict(
        (v, k) for k, v in trial_const_sequence_codes.items()) #inefficient in Python2

    # Create dictionaries for trial performances
    # (resulting decimal number from binary number created from trial_sequence)
    performance_codes = {
        'incomplete_trial': 0,
        'error<SR-ON': 159,
        'error<WS': 161,
        'error<CUE-ON': 163,
        'error<CUE-OFF': 167,
        'error<GO-ON': 175,
        'grip_error': 191,
        'correct_trial': 255}
    performance_str = dict((v, k) for k, v in performance_codes.items())  #inefficient in Python2

    def __init__(
            self, filename, odml_directory=None,
            nsx_override=None, nev_override=None,
            sif_override=None, ccf_override=None, odml_filename=None,
            verbose=False):
        """
        Constructor
        """

        # Remember choice whether to print diagnostic messages or not
        self._verbose = verbose

        # Remove known extensions from input filename
        for ext in self.extensions:
            filename = re.sub(os.path.extsep + ext + '$', '', filename)

        if nev_override:
            # check if sorting postfix is appended to nev_override name
            if nev_override[-3] == '-':
                sorting_postfix = nev_override[-2:]
            else:
                sorting_postfix = None
            sorting_version = nev_override
        else:
            # find most recent spike sorting version
            nev_versions = [re.sub(
                os.path.extsep + 'nev$', '', p) for p in glob.glob(
                    filename + '*.nev')]
            nev_versions = [p.replace(filename, '') for p in nev_versions]
            if len(nev_versions):
                sorting_postfix = sorted(nev_versions)[-1]
            else:
                sorting_postfix = ''
            sorting_version = filename + sorting_postfix

        # Initialize file
        BlackrockIO.__init__(
            self, filename, nsx_override=nsx_override,
            nev_override=sorting_version, sif_override=sif_override,
            ccf_override=ccf_override, verbose=verbose)

        # if no odML directory is specified, use same directory as main files
        if not odml_directory:
            odml_directory = os.path.dirname(self.filename)[:-1]

        # remove extensions from odml override
        filen = os.path.split(self.filename)[-1]
        if odml_filename:
            self._filenames['odml'] = ''.join(
                [odml_directory, os.path.sep, odml_filename])
        else:
            self._filenames['odml'] = ''.join(
                [odml_directory, os.path.sep, filen])

        file2check = ''.join([self._filenames['odml'], os.path.extsep, 'odml'])
        if os.path.exists(file2check):
            self._avail_files['odml'] = True
            self.odmldoc = odml.load(file2check)
        else:
            self._avail_files['odml'] = False
            self.odmldoc = None

        # If we did not specify an explicit sorting version, and there is an
        # odML, then make sure the detected sorting version matches the odML
        if self.odmldoc:
            if sorting_postfix not in self.odmldoc.sections['PreProcessing'].sections[
                    'OfflineSpikeSorting'].properties['Sortings'].values:
                self._print_verbose(
                    "Attempting to utilize the most recent "
                    "sorting version in file %s, but the sorting version "
                    "specified in odML is %s" % (
                        sorting_version,
                        self.odmldoc.sections['PreProcessing'].sections[
                            'OfflineSpikeSorting'].properties['Sortings'].values))
                self._load_spikesorting_info = False
            else:
                self._load_spikesorting_info = True
        else:
            self._load_spikesorting_info = False

    def __is_set(self, flag, pos):
        """
        Checks if bit is set at the given position for flag. If flag is an
        array, an array will be returned.
        """
        return flag & (1 << pos) > 0

    def __set_bit(self, flag, pos):
        """
        Returns the given flag with an additional bit set at the given
        position. for flag. If flag is an array, an array will be returned.
        """
        return flag | (1 << pos)

    def __add_rejection_to_event(self, event):
        """
        Given an event with annotation trial_id, adds information on whether to
        reject the trial or not.
        """
        if self.odmldoc:
            # Get rejection bands
            sec = self.odmldoc['PreProcessing']
            bands = sec.properties['LFPBands'].values

            for band in bands:
                sec = self.odmldoc['PreProcessing'][band]

                if type(sec.properties['RejTrials'].values) != [-1]:
                    rej_trials = [int(_) for _ in sec.properties['RejTrials'].values]
                    rej_index = np.in1d(event.annotations['trial_id'], rej_trials)
                elif sec.properties['RejTrials'].values == [-1]:
                    rej_index = np.zeros((len(event.annotations['trial_id'])), dtype=bool)
                else:
                    raise ValueError(
                        "Invalid entry %s in odML for rejected trials in LFP  band %s." %
                        (sec.properties['RejTrials'].values, band))
                event.annotate(**{str('trial_reject_' + band): list(rej_index)})

    def __extract_task_condition(self, trialtypes):
        """
        Extracts task condition from trialtypes.
        """
        occurring_trtys = np.unique(trialtypes).tolist()

        # reduce occurring_trtys to actual trialtypes
        # (remove all not identifiable trialtypes (incomplete/error trial))
        if 'NONE' in occurring_trtys:
            occurring_trtys.remove('NONE')
        # (remove all trialtypes where only the CUE was detected (error trial))
        if 'SG' in occurring_trtys:
            occurring_trtys.remove('SG')
        if 'PG' in occurring_trtys:
            occurring_trtys.remove('PG')
        if 'LF' in occurring_trtys:
            occurring_trtys.remove('LF')
        if 'HF' in occurring_trtys:
            occurring_trtys.remove('HF')

        # first set to unidentified task condition
        task_condition = 0

        if len(occurring_trtys) > 0:
            for cnd, trtys in self.condition_str.items():  #inefficient in Python2
                if set(trtys) == set(occurring_trtys):
                    # replace with detected task condition
                    task_condition = cnd

        return task_condition

    def __extract_analog_events_from_odml(self, t_start, t_stop):

        event_name = []
        event_time = []
        trial_id = []
        trial_timestamp_id = []
        performance_code = []
        trial_type = []

        # Look for all Trial Sections
        sec = self.odmldoc['Recording']['TaskSettings']
        ff = lambda x: x.name.startswith('Trial_')
        tr_secs = sec.itersections(filter_func=ff)
        for trial_sec in tr_secs:
            for signalname in ['GripForceSignals', 'DisplacementSignal']:
                for analog_events in trial_sec[
                        'AnalogEvents'][signalname].properties:

                    time = analog_events.values * \
                        pq.CompoundUnit(analog_events.unit)
                    if time >= t_start and time < t_stop:
                        event_name.append(analog_events.name)
                        event_time.append(time)
                        trial_id.extend(
                            trial_sec.properties['TrialID'].values)
                        trial_timestamp_id.extend(
                            trial_sec.properties[
                                'TrialTimestampID'].values)
                        performance_code.extend(
                            trial_sec.properties['PerformanceCode'].values)
                        trial_type.extend(
                            trial_sec.properties['TrialType'].values)

        # Create event object with analog events
        analog_events = neo.Event(
            times=pq.Quantity(
                [_.magnitude for _ in event_time],
                units=event_time[0].units).rescale('ms').flatten(),
            labels=np.array(event_name),
            name='AnalogTrialEvents',
            description='Events extracted from analog signals')

        analog_events.annotate(
            trial_id=trial_id,
            trial_timestamp_id=trial_timestamp_id,
            performance_in_trial=performance_code,
            belongs_to_trialtype=trial_type,
            trial_event_labels=event_name)

        return analog_events

    def __annotate_dig_trial_events(self, events):
        """
        Modifies events of digital input port to trial events of the
        reach-to-grasp project.
        """
        # Modifiy name and description
        events.name = "DigitalTrialEvents"
        events.description = "Trial " + events.description.lower()

        # Uncomment for event and trial sequence debugging
#        for ev in events.labels:
#            if ev in list(self.event_labels_str):
#                print ev, self.event_labels_str[ev]
#            else:
#                print ev

        # Extract beginning of first complete trial
        tson_label = self.event_labels_codes['TS-ON'][0]
        if tson_label in events.labels:
            first_TSon_idx = list(events.labels).index(tson_label)
        else:
            first_TSon_idx = len(events.labels)
        # Extract end of last complete trial
        stop_label = self.event_labels_codes['STOP'][0]
        if stop_label in events.labels:
            last_WSoff_idx = len(events.labels) - \
                             list(events.labels[::-1]).index(stop_label) - 1
        else:
            last_WSoff_idx = -1

        # Annotate events with modified labels, trial ids, and trial types
        trial_event_labels = []
        trial_ID = []
        trial_timestamp_ID = []
        trialtypes = {-1: 'NONE'}
        trialsequence = {-1: 0}
        for i, l in enumerate(events.labels):
            if i < first_TSon_idx or i > last_WSoff_idx:
                trial_event_labels.append('NONE')
                trial_ID.append(-1)
                trial_timestamp_ID.append(-1)
            else:
                # interpretation of TS-ON
                if self.event_labels_str[l] == 'TS-ON':
                    if i > 0:
                        prev_ev = events.labels[i - 1]
                        if self.event_labels_str[prev_ev] in \
                                ['STOP', 'TS-OFF/STOP']:
                            timestamp_id = int(events.times[i].rescale(
                                self._BlackrockIO__nev_params(
                                    'event_unit')).item())
                            trial_timestamp_ID.append(timestamp_id)
                            trial_event_labels.append('TS-ON')
                            trialsequence[timestamp_id] = self.__set_bit(
                                0, self.trial_const_sequence_codes['TS-ON'])
                        else:
                            timestamp_id = trial_timestamp_ID[-1]
                            trial_timestamp_ID.append(timestamp_id)
                            trial_event_labels.append('TS-ON-ERROR')
                    else:
                        timestamp_id = int(events.times[i].rescale(
                            self._BlackrockIO__nev_params(
                                'event_unit')).item())
                        trial_timestamp_ID.append(timestamp_id)
                        trial_event_labels.append('TS-ON')
                        trialsequence[timestamp_id] = self.__set_bit(
                            0, self.trial_const_sequence_codes['TS-ON'])

                    # Identify trial ID if odML exists
                    ID = -1
                    if self.odmldoc:
                        sec = self.odmldoc['Recording']['TaskSettings']
                        ff = lambda x: x.name.startswith('Trial_')
                        tr_secs = sec.itersections(filter_func=ff)
                        for trial_sec in tr_secs:
                            if trial_sec.properties[
                                    'TrialTimestampID'].values[0] == \
                                    timestamp_id:
                                ID = trial_sec.properties['TrialID'].values[0]
                    trial_ID.append(ID)
                # interpretation of GO/RW-OFF
                elif self.event_labels_str[l] == 'GO/RW-OFF':
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    trial_event_labels.append('GO/RW-OFF')
                # interpretation of ERROR-FLASH-ON
                elif l in self.event_labels_codes['ERROR-FLASH-ON']:
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    trial_event_labels.append('ERROR-FLASH-ON')
                    # Error-Flash hides too early activation of SR
                    # SR is set to 1 here to match perf codes between monkeys
                    trialsequence[timestamp_id] = self.__set_bit(
                        trialsequence[timestamp_id],
                        self.trial_const_sequence_codes['SR'])
                # TS-OFF/STOP
                elif self.event_labels_str[l] == 'TS-OFF/STOP':
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prev_ev = events.labels[i - 1]
                    if self.event_labels_str[prev_ev] == 'TS-ON':
                        trial_event_labels.append('TS-OFF')
                    elif prev_ev in self.event_labels_codes['ERROR-FLASH-ON']:
                        trial_event_labels.append('STOP')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['STOP'])
                    else:
                        trial_event_labels.append('STOP')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['STOP'])
                # interpretation of WS-ON/CUE-OFF
                elif self.event_labels_str[l] == 'WS-ON/CUE-OFF':
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prev_ev = events.labels[i - 1]
                    if self.event_labels_str[prev_ev] in \
                            ['TS-ON', 'TS-OFF/STOP']:
                        trial_event_labels.append('WS-ON')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['WS-ON'])
                    elif (prev_ev in self.event_labels_codes['CUE/GO'] or
                          prev_ev in self.event_labels_codes['GO/RW-OFF']):
                        trial_event_labels.append('CUE-OFF')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['CUE-OFF'])
                    else:
                        raise ValueError("Unknown trial event sequence.")
                # interpretation of CUE and GO events and trialtype detection
                elif l in self.event_labels_codes['CUE/GO']:
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prprev_ev = events.labels[i - 2]
                    if self.event_labels_str[prprev_ev] in \
                            ['TS-ON', 'TS-OFF/STOP']:
                        trial_event_labels.append('CUE-ON')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['CUE-ON'])
                        trialtypes[timestamp_id] = self.event_labels_str[l][:2]
                    elif prprev_ev in self.event_labels_codes['CUE/GO']:
                        trial_event_labels.append('GO-ON')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['GO-ON'])
                        trialtypes[timestamp_id] += \
                            self.event_labels_str[l][:2]
                    else:
                        raise ValueError("Unknown trial event sequence.")
                # interpretation of WS-OFF
                elif self.event_labels_str[l] == 'STOP':
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prev_ev = self.event_labels_str[events.labels[i - 1]]
                    if prev_ev == 'ERROR-FLASH-ON':
                        trial_event_labels.append('ERROR-FLASH-OFF')
                    else:
                        trial_event_labels.append('STOP')
                    trialsequence[timestamp_id] = self.__set_bit(
                        trialsequence[timestamp_id],
                        self.trial_const_sequence_codes['STOP'])
                # interpretation of SR events
                elif l in self.event_labels_codes['SR']:
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prev_ev = events.labels[i - 1]
                    if prev_ev in self.event_labels_codes['SR']:
                        trial_event_labels.append('SR-REP')
                    elif prev_ev in self.event_labels_codes['RW-ON']:
                        trial_event_labels.append('RW-OFF')
                    else:
                        trial_event_labels.append('SR')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['SR'])
                # interpretation of RW events
                elif l in self.event_labels_codes['RW-ON']:
                    trial_timestamp_ID.append(timestamp_id)
                    trial_ID.append(ID)
                    prev_ev = events.labels[i - 1]
                    if prev_ev in self.event_labels_codes['RW-ON']:
                        trial_event_labels.append('RW-ON-REP')
                    else:
                        trial_event_labels.append('RW-ON')
                        trialsequence[timestamp_id] = self.__set_bit(
                            trialsequence[timestamp_id],
                            self.trial_const_sequence_codes['RW-ON'])
                else:
                    raise ValueError("Unknown event label.")

        # add modified trial_event_labels to annotations
        events.annotate(trial_event_labels=trial_event_labels)

        # add trial timestamp IDs
        events.annotate(trial_timestamp_id=trial_timestamp_ID)

        # add trial IDs
        events.annotate(trial_id=trial_ID)

        # add modified belongs_to_trialtype to annotations
        for tid in trial_timestamp_ID:
            if tid not in list(trialtypes):
                trialtypes[tid] = 'NONE'
        belongs_to_trialtype = [
            trialtypes[tid] for tid in trial_timestamp_ID]
        events.annotate(belongs_to_trialtype=belongs_to_trialtype)

        # add modified trial_performance_codes to annotations
        performance_in_trial = [
            trialsequence[tid] for tid in trial_timestamp_ID]
        events.annotate(performance_in_trial=performance_in_trial)

    def __annotate_units_with_odml(self, units):
        """
        Annotates units with metadata from odml file.
        """
        # Can the spike sorting info from the odML be matched with the odML?
        if not self._load_spikesorting_info:
            return

        for un in units:
            an_dict = dict(
                sua=False,
                mua=False,
                noise=False)

            try:
                sec = self.odmldoc['UtahArray']['Array'][
                    'Electrode_%03d' % un.annotations['channel_id']][
                    'OfflineSpikeSorting']
            except KeyError:
                return

            suaids = sec.properties['SUAIDs'].values
            muaid = sec.properties['MUAID'].values[0]
            noiseids = sec.properties['NoiseIDs'].values

            if un.annotations['unit_id'] in suaids:
                an_dict['sua'] = True
            elif un.annotations['unit_id'] in noiseids:
                an_dict['noise'] = True
            elif un.annotations['unit_id'] == muaid:
                an_dict['mua'] = True
            else:
                raise ValueError(
                    "Unit %i is not registered for channel %i in odML file."
                    % (un.annotations['unit_id'],
                       un.annotations['channel_id']))

            if ('Unit_%02i' % un.annotations['unit_id']) in sec.sections:
                unit_sec = sec['Unit_%02i' % un.annotations['unit_id']]
                if an_dict['sua']:
                    an_dict['SNR'] = unit_sec.properties['SNR'].values[0]

                    # TODO: Add units here
                    an_dict['spike_duration'] = unit_sec.properties['SpikeDuration'].values[0]
                    an_dict['spike_amplitude'] = unit_sec.properties['SpikeAmplitude'].values[0]
                    an_dict['spike_count'] = unit_sec.properties['SpikeCount'].values[0]

            # Annotate Unit and all children for convenience
            un.annotate(**an_dict)
            for st in un.spiketrains:
                st.annotate(**an_dict)

    def __annotate_spiketrains_with_odml(self, sts):
        """
        Annotates spiketrains with metadata from odml file.
        """

    def __annotate_analogsignals_with_odml(self, asig):
        """
        Annotates analogsignals with metadata from odml file.
        """
        if self.odmldoc and asig.annotations['channel_id'] in range(1, 129):
            # Annotate filter settings from odML
            sec = self.odmldoc[
                'Cerebus']['NeuralSignalProcessor']['NeuralSignals'][
                'Filter_ns%i' % asig.annotations['nsx']]
            asig.annotate(
                filter_hi_pass_freq=pq.Quantity(
                    sec.properties['HighPassFreq'].values[0],
                    sec.properties['HighPassFreq'].unit),
                filter_lo_pass_freq=pq.Quantity(
                    sec.properties['LowPassFreq'].values[0],
                    sec.properties['LowPassFreq'].unit),
                filter_hi_pass_order=sec.properties[
                    'HighPassOrder'].values[0],
                filter_lo_pass_order=sec.properties[
                    'LowPassOrder'].values[0],
                filter_type=sec.properties[
                    'Type'].values[0])

    def __annotate_channelindex_with_odml(self, chidx):
        """
        Annotates channelindex with metadata from odml file.
        """

        if self.odmldoc:
            # Get rejection bands
            sec = self.odmldoc['PreProcessing']
            bands = sec.properties['LFPBands'].values

            if hasattr(bands, '__iter__'):
                for band in bands:
                    sec = self.odmldoc['PreProcessing'][band]

                    if type(sec.properties['RejElectrodes'].values) is list:
                        rej_electrodes = [int(_) for _ in sec.properties[
                            'RejElectrodes'].values]
                        rej = chidx.channel_ids[0] in rej_electrodes
                    elif sec.properties['RejElectrodes'].values == -1:
                        rej = False
                    elif sec.properties['RejElectrodes'].values >= 0:
                        rej_electrodes = sec.properties[
                            'RejElectrodes'].values
                        rej = (chidx.channel_ids[0] == rej_electrodes)
                    else:
                        raise ValueError(
                            "Invalid entry %s in odML for rejected electrodes "
                            "in LFP band %s." % (
                                sec.properties['RejElectrodes'].values,
                                band))

                    rej_dict = {str('electrode_reject_' + band): rej}

                    # Annotate ChannelIndex and all children for convenience
                    chidx.annotate(**rej_dict)
                    for asig in chidx.analogsignals:
                        asig.annotate(**rej_dict)
                    for unit in chidx.units:
                        unit.annotate(**rej_dict)
                        for st in unit.spiketrains:
                            st.annotate(**rej_dict)

            # Annotate connector aligned ID to channel
            if chidx.channel_ids[0] in \
                    chidx.block.annotations['avail_electrode_ids']:
                ca_dict = {
                    'connector_aligned_id': chidx.block.annotations[
                        'avail_electrode_ids'].index(chidx.channel_ids[0])+1}
                chidx.coordinates = pq.Quantity(np.array([
                    np.mod(ca_dict['connector_aligned_id']-1, 10)*.4,
                    (ca_dict['connector_aligned_id']-1)/10*.4]),
                    units=pq.mm)

                chidx.annotate(**ca_dict)
                for asig in chidx.analogsignals:
                    asig.annotate(**ca_dict)
                for unit in chidx.units:
                    unit.annotate(**ca_dict)
                    for st in unit.spiketrains:
                        st.annotate(**ca_dict)

    def __annotate_block_with_odml(self, bl):
        """
        Annotates block with metadata from odml file.
        """
        sec = self.odmldoc['Project']
        bl.annotate(
            project_name=sec.properties['Name'].values,
            project_type=sec.properties['Type'].values,
            project_subtype=sec.properties['Subtype'].values)

        sec = self.odmldoc['Project']['TaskDesigns']
        bl.annotate(
            taskdesigns=[v for v in sec.properties['UsedDesign'].values])

        sec = self.odmldoc['Subject']
        bl.annotate(
            subject_name=sec.properties['GivenName'].values,
            subject_gender=sec.properties['Gender'].values,
            subject_activehand=sec.properties['ActiveHand'].values,
            subject_birthday=sec.properties['Birthday'].values)

        sec = self.odmldoc['Setup']
        bl.annotate(setup_location=sec.properties['Location'].values)

        sec = self.odmldoc['UtahArray']
        bl.annotate(array_serialnum=sec.properties['SerialNo'].values)

        sec = self.odmldoc['UtahArray']['Connector']
        bl.annotate(connector_type=sec.properties['Style'].values)

        sec = self.odmldoc['UtahArray']['Array']
        bl.annotate(arraygrids_tot_num=sec.properties['GridCount'].values)

        sec = self.odmldoc['UtahArray']['Array']['Grid_01']
        bl.annotate(
            electrodes_tot_num=sec.properties['ElectrodeCount'].values,
            electrodes_pitch=pq.Quantity(
                sec.properties['ElectrodePitch'].values,
                units=sec.properties['ElectrodePitch'].unit),
            arraygrid_row_num=sec.properties['GridRows'].values,
            arraygrid_col_num=sec.properties['GridColumns'].values)

        secs = self.odmldoc['UtahArray']['Array'].sections
        bl.annotate(
            avail_electrode_ids=[])
        for i in range(1, 101):
            elidx = [s.properties['ID'].values for s in secs if
                     s.name.startswith('Electrode') and
                     s.properties['ConnectorAlignedID'].values[0] == i]
            if len(elidx) == 0:
                bl.annotations['avail_electrode_ids'].append(-1)
            elif len(elidx) == 1:
                bl.annotations['avail_electrode_ids'].append(elidx[0])
            else:
                raise ValueError("Electrode IDs in odML file are corrupt. "
                                 "ID %i occurs %i times" % (i, len(elidx)))

        # TODO: add list of behavioral channels
        # bl.annotate(avail_behavsig_indexes=[])

    def __correct_filter_shifts(self, asig):
        if self.odmldoc and asig.annotations['channel_id'] in range(1, 129):
            # Get and correct for shifts
            sec = self.odmldoc[
                'Cerebus']['NeuralSignalProcessor']['NeuralSignals'][
                'Filter_ns%i' % asig.annotations['nsx']]
            shift = pq.Quantity(
                sec.properties['EstimatedShift'].values[0],
                sec.properties['EstimatedShift'].unit)
            asig.t_start = asig.t_start - shift
            # Annotate shift
            asig.annotate(filter_shift_correction=shift)

    def __merge_digital_analog_events(self, events):
        """
        Merge the two event arrays AnalogTrialEvents and DigitalTrialEvents
        into one common event array TrialEvents.
        """
        event_name = []
        event_time = None
        trial_id = []
        trial_timestamp_id = []
        performance_code = []
        trial_type = []

        for event in events:
            if event.name in ['AnalogTrialEvents', 'DigitalTrialEvents']:
                # Extract event times
                if event_time is None:
                    event_time = event.times.magnitude
                    event_units = event.times.units
                else:
                    event_time = np.concatenate((
                        event_time,
                        event.times.rescale(event_units).magnitude.flatten()))

                # Transfer annotations
                trial_id.extend(
                    event.annotations['trial_id'])
                trial_timestamp_id.extend(
                    event.annotations['trial_timestamp_id'])
                performance_code.extend(
                    event.annotations['performance_in_trial'])
                trial_type.extend(
                    event.annotations['belongs_to_trialtype'])
                event_name.extend(
                    event.annotations['trial_event_labels'])

        # Sort time stamps and save sort order
        sort_idx = np.argsort(event_time)
        event_time = event_time[sort_idx]

        # Create event object with analog events
        merged_event = neo.Event(
            times=pq.Quantity(event_time, units=event_units),
            labels=np.array([event_name[_] for _ in sort_idx]),
            name='TrialEvents',
            description='All trial events (digital and analog)')

        merged_event.annotate(
            trial_id=[trial_id[_] for _ in sort_idx],
            trial_timestamp_id=[trial_timestamp_id[_] for _ in sort_idx],
            performance_in_trial=[performance_code[_] for _ in sort_idx],
            belongs_to_trialtype=[trial_type[_] for _ in sort_idx],
            trial_event_labels=[event_name[_] for _ in sort_idx])

        return merged_event

    def read_block(
            self, index=None, name=None, description=None, nsx_to_load='none',
            n_starts=None, n_stops=None, channels=range(1, 97), units='none',
            load_waveforms=False, load_events=False, scaling='raw',
            correct_filter_shifts=True, lazy=False, cascade=True):
        """
        Reads file contents as a Neo Block.

        The Block contains one Segment for each entry in zip(n_starts,
        n_stops). If these parameters are not specified, the default is
        to store all data in one Segment.

        The Block contains one ChannelIndex per channel.

        Args:
            index (None, int):
                If not None, index of block is set to user input.
            name (None, str):
                If None, name is set to default, otherwise it is set to user
                input.
            description (None, str):
                If None, description is set to default, otherwise it is set to
                user input.
            nsx_to_load (int, list, str):
                ID(s) of nsx file(s) from which to load data, e.g., if set to
                5 only data from the ns5 file are loaded. If 'none' or empty
                list, no nsx files and therefore no analog signals are loaded.
                If 'all', data from all available nsx are loaded.
            n_starts (None, Quantity, list):
                Start times for data in each segment. Number of entries must be
                equal to length of n_stops. If None, intrinsic recording start
                times of files set are used.
            n_stops (None, Quantity, list):
                Stop times for data in each segment. Number of entries must be
                equal to length of n_starts. If None, intrinsic recording stop
                times of files set are used.
            channels (int, list, str):
                Channel id(s) from which to load data. If 'none' or empty list,
                no channels and therefore no analog signal or spiketrains are
                loaded. If 'all', all available channels are loaded. By
                default, all neural channels (1-96) are loaded.
            units (int, list, str, dict):
                ID(s) of unit(s) to load. If 'none' or empty list, no units and
                therefore no spiketrains are loaded. If 'all', all available
                units are loaded. If dict, the above can be specified
                individually for each channel (keys), e.g. {1: 5, 2: 'all'}
                loads unit 5 from channel 1 and all units from channel 2.
            load_waveforms (boolean):
                If True, waveforms are attached to all loaded spiketrains.
            load_events (boolean):
                If True, all recorded events are loaded.
            scaling (str):
                Determines whether time series of individual
                electrodes/channels are returned as AnalogSignals containing
                raw integer samples ('raw'), or scaled to arrays of floats
                representing voltage ('voltage'). Note that for file
                specification 2.1 and lower, the option 'voltage' requires a
                nev file to be present.
            correct_filter_shifts (bool):
                If True, shifts of the online-filtered neural signals (e.g.,
                ns2, channels 1-128) are corrected by time-shifting the signal
                by a heuristically determined estimate stored in the metadata,
                in the property EstimatedShift, under the path
                /Cerebus/NeuralSignalProcessor/NeuralSignals/Filter_nsX/
            lazy (bool):
                If True, only the shape of the data is loaded.
            cascade (bool or "lazy"):
                If True, only the block without children is returned.

        Returns:
            Block (neo.segment.Block):
                Block linking to all loaded Neo objects.

                Block annotations:
                    avail_file_set (list of str):
                        List of file extensions of the files found to be
                        associated to the project, and which are used in
                        loading the data, e.g., ccf, odml, nev, ns2,...
                    avail_nsx (list of int):
                        List of integers specifying the .nsX files available,
                        e.g., [2, 5] indicates that an ns2 and and ns5 file are
                        available.
                    avail_nev (bool):
                        True if a .nev file is available.
                    avail_ccf (bool):
                        True if a .ccf file is available.
                    avail_sif (bool):
                        True if a .sif file is available.
                    nb_segments (int):
                        Number of segments created after merging recording
                        times specified by user with the intrinsic ones of the
                        file set.
                    project_name (str):
                        Identifier for the project/experiment.
                    project_type (str):
                        Identifier for the type of project/experiment.
                    project_subtype (str):
                        Identifier of the subtype of the project/experiment.
                    taskdesigns (list of str):
                        List of strings identifying the task designed presented
                        during the recording. The standard task reach-to-grasp
                        is denoted by the string "TwoCues".
                    conditions (list of int):
                        List of condition codes (each code describing the set
                        of trial types presented to the subject during a
                        segment of the recording) present during the recording.
                        For a mapping of condition codes to trial types, see
                        the condition_str attribute of the ReachGraspIO class.
                    subject_name (str):
                        Name of the recorded subject.
                    subject_gender (bool):
                        'male' or 'female'.
                    subject_birthday (datetime):
                        Birthday of the recorded subject.
                    subject_activehand (str):
                        Handedness of the subject.
                    setup_location (str):
                        Physical location of the recording setup.
                    avail_electrode_ids (list of int):
                        List of length 100 of electrode channel IDs (Blackrock
                        IDs) ordered corresponding to the connector-aligned
                        linear electrode IDs. The connector-aligned IDs start
                        at 1 in the bottom left corner, and increase from left
                        to right, and from bottom to top assuming the array is
                        placed in front of the observer pins facing down,
                        connector extruding to the right:

                        91 92 ... 99 100  \
                        81 82 ... 89  90   \
                        ...          ...    --- Connector Wires
                        11 12 ... 19  20   /
                         1  2 ...  9  10  /

                        Thus,
                          avail_electrode_ids[k-1]
                        is the Blackrock channel ID corresponding to connector-
                        aligned ID k. Unconnected/unavailable channels are
                        marked by -1.
                    arraygrids_tot_num (int):
                        Number of Utah arrays (not necessarily all connected).
                    electrodes_tot_num (int):
                        Number of electrodes of the Utah array (not necessarily
                        all connected).
                    electrodes_pitch (float):
                        Distance in micrometers between neighboring electrodes
                        in one row/column.
                    array_serial_num (str):
                        Serial number of the recording array.
                    array_grid_col_num, array_grid_row_num (int):
                        Number of columns / rows of the array.
                    connector_type (str):
                        Type of connector used for recording.
                    rec_pauses (bool):
                        True if the session contains a recording pause (i.e.,
                        multiple segments).

                Segment annotations:
                    condition (int):
                        Condition code (describing the set of trial types
                        presented to the subject) of this segment. For a
                        mapping of condition codes to trial types, see the
                        condition_str attribute of the ReachGraspIO class.

                ChannelIndex annotations:
                    connector_aligned_id (int):
                        Connector-aligned channel ID from which the spikes were
                        loaded. This is a channel ID between 1 and 100 that is
                        related to the location of an electrode on the Utah
                        array and thus common across different arrays
                        (independent of the Blackrock channel ID). The ID
                        considers a top-view of the array with the connector
                        wires extruding to the right. Electrodes are then
                        numbered from bottom left to top right:

                        91 92 ... 99 100  \
                        81 82 ... 89  90   \
                        ...          ...    --- Connector Wires
                        11 12 ... 19  20   /
                         1  2 ...  9  10  /

                        Note: The Blackrock IDs are given in the 'channel_ids'
                        property of the ChannelIndex object.
                    waveform_size (Quantitiy):
                        Length of time used to save spike waveforms (in units
                        of 1/30000 s).
                    nev_hi_freq_corner (Quantitiy),
                    nev_lo_freq_corner (Quantitiy),
                    nev_hi_freq_order (int), nev_lo_freq_order (int),
                    nev_hi_freq_type (str), nev_lo_freq_type (str),
                    nev_hi_threshold, nev_lo_threshold,
                    nev_energy_threshold (Quantity):
                        Indicates parameters of spike detection.
                    nev_dig_factor (int):
                        Digitization factor in microvolts of the nev file, used
                        to convert raw samples to volt.
                    connector_ID, connector_pinID (int):
                        ID of connector and pin on the connector where the
                        channel was recorded from.
                    nb_sorted_units (int):
                        Number of sorted units on this channel (noise, mua and
                        sua).
                    electrode_reject_XXX (bool):
                        For different filter ranges XXX (as defined in the odML
                        file), if this variable is True it indicates whether
                        the spikes were recorded on an electrode that should be
                        rejected based on preprocessing analysis for removing
                        electrodes due to noise/artefacts in the respective
                        frequency range.

                Unit annotations:
                    coordinates (Quantity):
                        Contains the x and y coordinate of the electrode in mm
                        (spacing: 0.4mm). The coordinates use the same
                        representation as the connector_aligned_id with the
                        origin located at the bottom left electrode. Thus,
                        e.g., connector aligned ID 14 is at coordinates:
                            (1.2 mm, 0.4 mm)
                    unit_id (int):
                        ID of the unit.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the unit was
                        loaded (equiv. to the single list entry in the
                        attribute channel_ids of ChannelIndex parent).
                    connector_aligned_id (int):
                        Connector-aligned channel ID from which the unit was
                        loaded. This is a channel ID between 1 and 100 that is
                        related to the location of an electrode on the Utah
                        array and thus common across different arrays
                        (independent of the Blackrock channel ID). The ID
                        considers a top-view of the array with the connector
                        wires extruding to the right. Electrodes are then
                        numbered from bottom left to top right:

                        91 92 ... 99 100  \
                        81 82 ... 89  90   \
                        ...          ...    --- Connector Wires
                        11 12 ... 19  20   /
                         1  2 ...  9  10  /

                    electrode_reject_XXX (bool):
                        For different filter ranges XXX (as defined in the odML
                        file), if this variable is True it indicates whether
                        the spikes were recorded on an electrode that should be
                        rejected based on preprocessing analysis for removing
                        electrodes due to noise/artefacts in the respective
                        frequency range.
                    noise, mua, sua (bool):
                        True, if the unit is classified as a noise unit, i.e.,
                        not considered neural activity (noise), a multi-unit
                        (mua), or a single unit (sua).
                    SNR (float):
                        Signal to noise ratio of SUA/MUA waveforms. A higher
                        value indicates that the unit could be better
                        distinguished in the spike detection and spike sorting
                        procedure.
                    spike_duration (float):
                        Approximate duration of the spikes of SUAs/MUAs in
                        microseconds.
                    spike_amplitude (float):
                        Maximum amplitude of the spike waveform.
                    spike_count (int):
                        Number of spikes sorted into this unit.

                AnalogSignal annotations:
                    nsx (int):
                        nsX file the signal was loaded from, e.g., 5 indicates
                        the .ns5 file.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the signal was
                        loaded.
                    connector_aligned_id (int):
                        Connector-aligned channel ID from which the signal was
                        loaded. This is a channel ID between 1 and 100 that is
                        related to the location of an electrode on the Utah
                        array and thus common across different arrays
                        (independent of the Blackrock channel ID). The ID
                        considers a top-view of the array with the connector
                        wires extruding to the right. Electrodes are then
                        numbered from bottom left to top right:

                        91 92 ... 99 100  \
                        81 82 ... 89  90   \
                        ...          ...    --- Connector Wires
                        11 12 ... 19  20   /
                         1  2 ...  9  10  /

                    electrode_reject_XXX (bool):
                        For different filter ranges XXX (as defined in the odML
                        file), if this variable is True it indicates whether
                        the spikes were recorded on an electrode that should be
                        rejected based on preprocessing analysis for removing
                        electrodes due to noise/artefacts in the respective
                        frequency range.
                    filter_shift_correction (Quantity):
                        If the parameter correct_filter_shift is True, and a
                        shift estimate was found in the odML, this annotation
                        indicates the amount of time by which the signal was
                        shifted. I.e., adding this number to t_start will
                        result in the uncorrected, originally recorded time
                        axis.

                Spiketrain annotations:
                    unit_id (int):
                        ID of the unit from which the spikes were recorded.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the spikes were
                        loaded.
                    connector_aligned_id (int):
                        Connector-aligned channel ID from which the spikes were
                        loaded. This is a channel ID between 1 and 100 that is
                        related to the location of an electrode on the Utah
                        array and thus common across different arrays
                        (independent of the Blackrock channel ID). The ID
                        considers a top-view of the array with the connector
                        wires extruding to the right. Electrodes are then
                        numbered from bottom left to top right:

                        91 92 ... 99 100  \
                        81 82 ... 89  90   \
                        ...          ...    --- Connector Wires
                        11 12 ... 19  20   /
                         1  2 ...  9  10  /

                    electrode_reject_XXX (bool):
                        For different filter ranges XXX (as defined in the odML
                        file), if this variable is True it indicates whether
                        the spikes were recorded on an electrode that should be
                        rejected based on preprocessing analysis for removing
                        electrodes due to noise/artefacts in the respective
                        frequency range.
                    noise, mua, sua (bool):
                        True, if the unit is classified as a noise unit, i.e.,
                        not considered neural activity (noise), a multi-unit
                        (mua), or a single unit (sua).
                    SNR (float):
                        Signal to noise ratio of SUA/MUA waveforms. A higher
                        value indicates that the unit could be better
                        distinguished in the spike detection and spike sorting
                        procedure.
                    spike_duration (float):
                        Approximate duration of the spikes of SUAs/MUAs in
                        microseconds.
                    spike_amplitude (float):
                        Maximum amplitude of the spike waveform.
                    spike_count (int):
                        Number of spikes sorted into this unit.

                Event annotations:
                    The resulting Block contains three Event objects with the
                    following names:
                    "DigitalTrialEvents' contains all digitally recorded events
                        returned by BlackrockIO, annotated with semantic labels
                        in accordance with the reach-to-grasp experiment (e.g.,
                        'TS-ON').
                    'AnalogTrialEvents' contains events extracted from the
                        analog behavioral signals during preprocessing and
                        stored in the odML (e.g., 'OT').
                    'TrialEvents' contains all events of DigitalTrialEvents and
                        AnalogTrialEvents merged into a single Neo object.

                    Each annotation is a list containing one entry per time
                    point stored in the event.

                    trial_event_labels (list of str):
                        Name identifying the name of the event, e.g., 'TS-ON'.
                    trial_id (list of int):
                        Trial ID the event belongs to.
                    trial_timestamp_id (list of int):
                        Timestamp-based trial ID (equivalent to the time of TS-
                        ON of a trial) the event belongs to.
                    belongs_to_trialtype (str):
                        String identifying the trial type (e.g., SGHF) the
                        trial belongs to.
                    performance_in_trial (list of int):
                        Performance code of the trial that the event belongs
                        to. Compare to the performance_codes and
                        performance_str attributes of ReachGraspIO class.
                    trial_reject_XXX:
                        For different filter ranges XXX (defined in the odML
                        file), if True this variable indicates whether the
                        trial was rejected based on preprocessing analysis.
        """
        if not name:
            name = 'Reachgrasp Recording Data Block'
        if not description:
            description = \
                "Block of reach-to-grasp project data from Blackrock file set."

        # Load neo block
        bl = BlackrockIO.read_block(
            self, index=index, name=name, description=description,
            nsx_to_load=nsx_to_load, n_starts=n_starts, n_stops=n_stops,
            channels=channels, units=units, load_waveforms=load_waveforms,
            load_events=load_events, scaling=scaling, lazy=lazy,
            cascade=cascade)

        bl.annotate(conditions=[])
        for seg in bl.segments:
            if load_events and not lazy:
                if 'condition' in list(seg.annotations):
                    bl.annotations['conditions'].append(
                        seg.annotations['condition'])

        if self.odmldoc:
            self.__annotate_block_with_odml(bl)
            for chidx in bl.channel_indexes:
                self.__annotate_channelindex_with_odml(chidx)
                self.__annotate_units_with_odml(chidx.units)

        return bl

    def read_segment(
            self, n_start, n_stop, name=None, description=None, index=None,
            nsx_to_load='none', channels=range(1, 97), units='none',
            load_waveforms=False, load_events=False, scaling='raw',
            correct_filter_shifts=True, lazy=False, cascade=True):
        """
        Reads file contents as a Neo Block.

        The Block contains one Segment for each entry in zip(n_starts,
        n_stops). If these parameters are not specified, the default is
        to store all data in one Segment.

        The Block contains one ChannelIndex per channel.

        Args:
            n_start (Quantity):
                Start time of maximum time range of signals contained in this
                segment.
            n_stop (Quantity):
                Stop time of maximum time range of signals contained in this
                segment.
            name (None, string):
                If None, name is set to default, otherwise it is set to user
                input.
            description (None, string):
                If None, description is set to default, otherwise it is set to
                user input.
            index (None, int):
                If not None, index of segment is set to user index.
            nsx_to_load (int, list, str):
                ID(s) of nsx file(s) from which to load data, e.g., if set to
                5 only data from the ns5 file are loaded. If 'none' or empty
                list, no nsx files and therefore no analog signals are loaded.
                If 'all', data from all available nsx are loaded.
            channels (int, list, str):
                Channel id(s) from which to load data. If 'none' or empty list,
                no channels and therefore no analog signal or spiketrains are
                loaded. If 'all', all available channels are loaded.  By
                default, all neural channels (1-96) are loaded.
            units (int, list, str, dict):
                ID(s) of unit(s) to load. If 'none' or empty list, no units and
                therefore no spiketrains are loaded. If 'all', all available
                units are loaded. If dict, the above can be specified
                individually for each channel (keys), e.g. {1: 5, 2: 'all'}
                loads unit 5 from channel 1 and all units from channel 2.
            load_waveforms (boolean):
                If True, waveforms are attached to all loaded spiketrains.
            load_events (boolean):
                If True, all recorded events are loaded.
            scaling (str):
                Determines whether time series of individual
                electrodes/channels are returned as AnalogSignals containing
                raw integer samples ('raw'), or scaled to arrays of floats
                representing voltage ('voltage'). Note that for file
                specification 2.1 and lower, the option 'voltage' requires a
                nev file to be present.
            correct_filter_shifts (bool):
                If True, shifts of the online-filtered neural signals (e.g.,
                ns2, channels 1-128) are corrected by time-shifting the signal
                by a heuristically determined estimate stored in the metadata,
                in the property EstimatedShift, under the path
                /Cerebus/NeuralSignalProcessor/NeuralSignals/Filter_nsX/
            lazy (boolean):
                If True, only the shape of the data is loaded.
            cascade (boolean):
                If True, only the segment without children is returned.

        Returns:
            Segment (neo.segment.Segment):
                Segment linking to all loaded Neo objects. See documentation of
                read_block() for a full list of annotations per Neo object.
        """
        # Load neo block
        seg = BlackrockIO.read_segment(
            self, n_start, n_stop, name=name, description=description,
            index=index, nsx_to_load=nsx_to_load, channels=channels,
            units=units, load_waveforms=load_waveforms,
            load_events=load_events, scaling=scaling, lazy=lazy,
            cascade=cascade)

        for asig in seg.analogsignals:
            self.__annotate_analogsignals_with_odml(asig)
            if correct_filter_shifts:
                self.__correct_filter_shifts(asig)

        if load_events and not lazy:
            for ev in seg.events:
                # Modify digital trial events to include semantic event
                # informations
                if ev.name == 'digital_input_port':
                    self.__annotate_dig_trial_events(ev)
                    self.__add_rejection_to_event(ev)

                    cnd = self.__extract_task_condition(
                        ev.annotations['belongs_to_trialtype'])
                    seg.annotate(condition=cnd)

            # If digital trial events exist, extract analog events from odML
            # and create one common event array
            if len(seg.events) > 0 and self.odmldoc:
                analog_event = self.__extract_analog_events_from_odml(
                    seg.t_start, seg.t_stop)
                self.__add_rejection_to_event(analog_event)
                seg.events.append(analog_event)

                merged_event = self.__merge_digital_analog_events(
                    seg.events)
                self.__add_rejection_to_event(merged_event)
                seg.events.append(merged_event)

        return seg


if __name__ == '__main__':
    pass

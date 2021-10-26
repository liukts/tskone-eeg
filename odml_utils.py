# -*- coding: utf-8 -*-
'''
Convenience functions to work with the odML metadata collection of the reach-
to-grasp experiment.

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

import itertools
import numpy as np


def get_TrialCount(doc, trialtype=None, performance_code=None):
    """
    Returns a list of trials ids

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        trialtype (str or int):
            If stated, returns only count of trials with given trial type
        performance_code (int):
            If stated, returns only count of trials with given performance code

    Returns:
        (int):
            Number of specified trials
    """
    sec = doc['Recording']['TaskSettings']

    if performance_code == 255:
        output = sec.properties['CorrectTrialCount'].values[0]

    elif performance_code == 191:
        output = sec.properties['GripErrorTrialCount'].values[0]

    elif performance_code in [0, 159, 161, 163, 167, 175]:
        subsec = sec['TrialTypeSettings']

    else:
        output = sec.properties['TotalTrialCount'].values[0]

    # TODO: extend to trial types and other performance codes

    return output


def get_TrialIDs(doc, idtype='TrialID'):
    """
    Returns a list of trials ids

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project

    Returns:
        (list of int):
            Trial id list
    """
    output = []

    sec = doc['Recording']['TaskSettings']

    def ff(x): return x.name.startswith('Trial_')
    for trsec in sec.itersections(filter_func=ff):
        def FF(x): return x.name == idtype
        output.append(
            [p for p in trsec.iterproperties(filter_func=FF)][0].values[0])

    return sorted(output)


def get_TrialType(doc, trialid, code=True):
    """
    Returns trial type (code or abbreviation) for wanted trial

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        trialid (int):
            ID of wanted trial
        code (boolean):
            If True (default), integer code of trial type is returned
            If False, string abbreviation of trial type is returned

    Returns:
        (int or str):
            trial type for wanted trial
    """
    def ff(x): return x.name == 'Trial_%03i' % trialid
    sec = [s for s in doc.itersections(filter_func=ff)][0]

    output = sec.properties['TrialType'].values[0]

    return output


def get_PerformanceCode(doc, trialid, code=True):
    """
    Returns the performance of the monkey in the given trial either as code or
    abbreviation.

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        trialid (int):
            ID of wanted trial
        code (boolean):
            If True (default), integer code of trial performance is returned
            If False, abbreviation of trial performance is returned

    Returns:
        (int or string):
            performance code or abbreviation for wanted trial
    """
    def ff1(x): return x.name == 'Trial_%03i' % trialid
    sec = [s for s in doc.itersections(filter_func=ff1)][0]

    def ff2(x): return x.name == 'PerformanceCode'
    output = [p for p in sec.iterproperties(filter_func=ff2)][0].values[0]

    if code:
        return output

    else:
        def ff3(x): return x.name == 'PerformanceCodes'
        sec = [s for s in doc.itersections(filter_func=ff3)][0]

        def ff4(x): return x.name == 'pc_%i' % output
        output = [p for p in sec.iterproperties(filter_func=ff4)][0].values[0]

        return output


def get_OccurringTrialTypes(doc, code=True):
    """
    Returns all occurring trial types (code or abbreviations)

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        code (boolean):
            If True, integer code of trial type is returned
            If False, string abbreviation of trial type is returned

    Returns:
        (list of int or str):
            list of occurring trial types
    """
    trial_id_list = get_TrialIDs(doc)

    output = np.unique([get_TrialType(doc, trid, code=code) for trid in
                        trial_id_list]).tolist()

    return output


def get_trialids_pc(doc, performance_code):
    """
    Returns a list of trials ids which have the given performance code

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        trialtype (int or str):
            trial type of wanted trials

    Returns:
        (list of int):
            Trial id list with the given trial type
    """
    trialids = get_TrialIDs(doc)

    if isinstance(performance_code, int):
        code = True
    else:
        code = False

    output = []
    for trid in trialids:
        if get_PerformanceCode(doc, trid, code) == performance_code:
            output.append(trid)

    return output


def get_trialids_trty(doc, trialtype):
    """
    Returns a list of trials ids which have the given trial type

    Args:
        doc (odml.doc.BaseDocument):
            odML Document of reach-to-grasp project
        trialtype (int or str):
            trial type of wanted trials

    Returns:
        (list of int):
            Trial id list with the given trial type
    """
    trialids = get_TrialIDs(doc)

    if isinstance(trialtype, int):
        code = True
    else:
        code = False

    output = []
    for trid in trialids:
        if get_TrialType(doc, trid, code) == trialtype:
            output.append(trid)

    return output

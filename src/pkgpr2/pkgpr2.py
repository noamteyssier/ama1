#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math

from numba import jit
from tqdm import tqdm
from scipy.optimize import minimize

pd.plotting.register_matplotlib_converters()
sns.set(rc={'figure.figsize': (30, 30), 'lines.linewidth': 2})

# Cohort Level


class Individual(object):
    """
    Class to handle methods related to an individual in the cohort
    """

    def __init__(self, cid_frame,
                 skip_threshold=3, impute_missing=True, drop_missing=True,
                 haplodrop=False
                 ):
        """ Inititalization of Individual object

        Parameters
        ----------
        cid_frame : pd.DataFrame
            dataframe subsetted for a specific cohortid
        skip_threshold : int
            Number of skips allowed before considering a new infection
        impute_missing : bool
            Impute missing genotyped data from qpcr
        drop_missing : bool
            Drop dates with positive qpcr but missing genotyping data
            from skip calculation

        Returns
        -------
        Individual
            An object of class individual to organize haplotype timelines
            burnin dates, haplotypes, dates, and infection labels

        """
        self.frame = cid_frame
        self.skip_threshold = skip_threshold
        self.impute_missing = impute_missing
        self.drop_missing = drop_missing
        self.haplodrop = haplodrop

        self.cid = cid_frame.cohortid.unique()[0]
        self.burnin = cid_frame.burnin.unique()[0]
        self.dates = np.sort(cid_frame.date.unique())
        self.hids = np.sort(cid_frame.h_popUID.unique())

        self.to_drop = np.array([])

        self.timeline = pd.DataFrame()
        self.skip_frame = pd.DataFrame()
        self.labels = pd.DataFrame()

        self.QpcrTimeline()
        self.PostBurnin()
        if self.haplodrop:
            self.plot_haplodrop(save=True)

    def BuildTimeline(self, cid_frame,
                      index='h_popUID',
                      column='date',
                      value='pass_qpcr'
                      ):
        """
        Awesome numpy pivot function found @
        https://stackoverflow.com/questions/48527091/efficiently-and-simply-convert-from-long-format-to-wide-format-in-pandas-and-or
        """
        arr = cid_frame.values
        idx_index, idx_column, idx_value = [
            np.where(cid_frame.columns == v)[0] for v in [index, column, value]
            ]

        rows, row_pos = np.unique(arr[:, idx_index], return_inverse=True)
        cols, col_pos = np.unique(arr[:, idx_column], return_inverse=True)
        pivot_arr = np.zeros((rows.size, cols.size))
        pivot_arr[row_pos, col_pos] = arr[:, idx_value[0]]

        frame = pd.DataFrame(
            pivot_arr,
            columns=cols,
            index=rows,
            dtype=bool
            )
        frame.index.name = index
        return frame

    def QpcrTimeline(self):
        """
        Build timeline based on whether values pass qpcr_threshold
        """

        self.timeline = self.BuildTimeline(
            self.frame, value='pass_qpcr'
            )

        self.timeline.loc['nan'] = self.timeline.max(axis=0)

    def PostBurnin(self):
        """
        Find first date past the burnin,
        if burnin is between two visits mark postburnin as midway
        """
        try:
            self.post_burnin = np.where(self.dates >= self.burnin)[0].min()

        except ValueError:
            return False

        if self.dates[self.post_burnin] > self.burnin:
            self.post_burnin -= 0.5

        return self.post_burnin

    def PositionalDifference(self, x,
                             position=0, add_one=False,
                             skip_drop=True
                             ):
        """
        For a bool array x : calculate number of skips between each truth
        calculate first truth with respect to the number of visits post burnin
        """
        truth = np.where(x)[0]

        # no qpcr positive events
        if truth.size == 0:
            return np.array([])

        # remove positive qpcr but no genotyping data from skip calculation
        if self.drop_missing and not skip_drop:

            # find dates not already inserted to truth
            hid_to_drop = self.to_drop[
                ~np.isin(self.to_drop, truth)
                ]

            # reduce position of skips where necessary
            for i, x in enumerate(truth):
                truth[i] = x - np.where(
                    (hid_to_drop < x)
                    )[0].size

        skip_arr = []
        for i, pos in enumerate(truth):

            # first occurence of a haplotype
            if i == 0:

                # make all skips zero pre burnin
                if pos < position:
                    skips = 0

                # first visit after burnin is equal to visit number
                else:
                    skips = pos

            # every other occurence
            else:
                # disregard pre-burnin skips
                if (truth[i - 1] < position) & (pos >= position):
                    skips = pos - truth[i-1] - 1

                # calculate skips in pre-burnin period
                else:
                    skips = pos - truth[i-1] - 1

            skip_arr.append(skips)

        skip_arr = np.array(skip_arr)

        if add_one:
            skip_arr = skip_arr + 1

        return skip_arr

    def FillMissing(self, truth, impute=False):
        """
        Generator to take positive qpcr events that are missing genotyping data
        and yield positions that must be inserted into the truth array of a
        clone

        Parameters
        ----------
        truth : np.array
            All positions of a clone where qpcr positive events occur

        Yields
        -------
        int
            Positions that must be inserted or appended into the truth
            array to reflect positive qpcr events that are missing
            genotyping

        """

        def evaluate(x, impute):
            """
            evaluation function to insert a qpcr positive date.

            x : query date
            q+ : qpcr positive dates

            condition 1 : the q+ - x to be greater than zero
            condition 2 : the q+ - x less than skips threshold

            return : True if both, False otherwise
            """
            eval = self.to_drop - x
            possible = np.where(eval > 0)[0]
            within_range = np.where(eval[possible] <= self.skip_threshold)[0]

            if within_range.size == 0:
                return within_range

            possible_values = eval[possible]

            possible_range = np.zeros(possible_values.max() + 1, dtype=bool)

            possible_range[possible_values] = True

            possible_skips = self.PositionalDifference(
                possible_range, 0, skip_drop=True
                )

            skip_limit = np.where(possible_skips > self.skip_threshold)[0]
            if skip_limit.size > 0:

                result = self.to_drop[
                    possible[:skip_limit.min()]
                    ]
            else:

                result = self.to_drop[
                    possible
                    ]

            if not impute:
                result = result[result <= x.max()]

            return result

        to_fill = map(
            lambda x: evaluate(x, impute),
            truth
            )

        known = set()
        for i in to_fill:
            for j in i:
                if j not in known:
                    known.add(j)
                    yield j

    def DropMissing(self, recurse=False, impute=False):
        """
        Drop qpcr positive dates with no genotyping data
        """

        if not recurse:
            qpcr_positive = np.where(
                self.timeline.loc['nan']
                )[0]

            single_positive = np.where(
                self.timeline.sum(axis=0) == 1
                )[0]

            self.to_drop = qpcr_positive[
                np.isin(qpcr_positive, single_positive)
                ]

            self.DropMissing(recurse=True, impute=impute)

        else:

            for hid in self.hids[self.hids != 'nan']:

                # all q+ dates for haplotype
                truth = np.where(
                    self.timeline.loc[hid]
                    )[0]

                # q+ dates inserted if conditions met
                filled_truth = np.sort(
                    np.concatenate([
                            truth,
                            [i for i in self.FillMissing(truth, impute=impute)]
                        ]).astype(int)
                    )

                # new array created
                replacement = np.zeros(self.timeline.loc[hid].size, dtype=bool)
                replacement[filled_truth.astype(int)] = True

                # replace original array with inserted values
                self.timeline.loc[hid] = replacement

    def HID_Skips(self):
        """
        Generator object that calculates number of skips in a timeline
        by clone.

        Yields
        -------
        dict
            dictionary organized as columns for skip frame construction

        """

        for hid in self.hids:

            hid_arr = self.timeline.loc[hid]

            # calculate skips
            skips = self.PositionalDifference(
                hid_arr, self.post_burnin
                )

            # find dates where h_popUID is present
            hid_dates = self.dates[hid_arr]

            # report visit number of hid_dates
            visit_numbers = np.where(hid_arr)[0]

            # append to growing skip dataframe
            for idx in np.arange(skips.size):
                yield {
                    'cohortid': self.cid,
                    'h_popUID': hid,
                    'date': hid_dates[idx],
                    'visit_number': visit_numbers[idx],
                    'skips': skips[idx]
                    }

    def SkipsByClone(self, impute=False):
        """
        - Drop qpcr positive missing genotyping visits if required
        - Impute missing qpcr positive dates if required
        - Build skip frame over all haplotypes
        - Drop qpcr timeline if required
        """

        if self.drop_missing:
            self.DropMissing(impute=impute)

        self.skip_frame = pd.DataFrame([
            h for h in self.HID_Skips() if h
            ])

        if (not self.skip_frame.empty) & (not impute) & (self.impute_missing):
            self.skip_frame = self.skip_frame[
                self.skip_frame.h_popUID != 'nan'
                ]

    def LabelInfectionEvents(self):
        """
        Finds all infection events and collapse by date.
        Label infection events numerically
        """

        def label_by_date(ifx_dates, ifx_events):
            """
            return numeric infection events labels by date of infections
            """

            ifx_name_list = []

            for date, sub in ifx_events.groupby('date'):
                ifx_name = 'ifx_event.{}'.format(
                     np.where(ifx_dates == date.to_datetime64())[0][0] + 1
                     )

                padded_ifx_name = np.full(sub.shape[0], ifx_name)
                ifx_name_list.append(padded_ifx_name)

            return np.concatenate(ifx_name_list)

        # get infection events
        ifx_events = self.labels[['h_popUID', 'date']][
            self.labels.infection_event
            ].\
            sort_values('date')

        # get unique ifx dates
        ifx_dates = ifx_events.date.unique()

        # if there are no infection events quit
        if ifx_dates.size == 0:
            self.labels['ie'] = 'ifx_events.0'
            return

        # label infection events by date
        ifx_events['ie'] = label_by_date(ifx_dates, ifx_events)

        # merge infection events with labels
        self.labels = self.labels.merge(
            ifx_events,
            how='left',
            ).fillna('ifx_event.0')

        # label subsequent infection events as the same infection event
        for hdi, sub in ifx_events.groupby(['h_popUID', 'date', 'ie']):
            hid, date, ie = hdi
            self.labels.ie[
                (self.labels.h_popUID == hid) & (self.labels.date >= date)
                ] = ie

        # find last baseline visit number
        baseline_final_date = self.labels[
            (self.labels.ie == 'ifx_event.0') & (
                self.labels.h_popUID != 'nan')
            ].visit_number.max()

        # drop nan labels for infection event 0 past the end of
        # the imputed genotyped ie
        self.labels = self.labels[
            (self.labels.ie != 'ifx_event.0') | ~(
                    (self.labels.h_popUID == 'nan') & (
                        self.labels.visit_number > baseline_final_date)
                )
            ]

    def NestInfectionEvents(self, infection_mins):
        """
        Calculate skips at infection event level and nest infection events
        that do not exceed the skip threshold (+1)
        """

        # build bool array for skips on infection events
        date_pos_arr = np.zeros(
            infection_mins.date_pos.max() + 1, dtype=bool
            )
        date_pos_arr[infection_mins.date_pos] = True

        # calculate skips
        infection_mins['date_skips'] = self.PositionalDifference(
                date_pos_arr, 0, add_one=True
                )

        # initialize rename column
        infection_mins['rename'] = infection_mins.h_popUID.copy()

        # find where skips do not exceed skip threshold
        to_merge = np.where(
            infection_mins.date_skips <= self.skip_threshold
            )[0]

        # nest infection events within threshold
        for i in to_merge:
            if i == 0:
                continue
            else:

                # post baseline infections cannot be grouped into baseline
                if infection_mins['rename'].iloc[i-1] == 'ifx_event.0':
                    continue

                infection_mins['rename'].iloc[i] = \
                    infection_mins['rename'].iloc[i-1]

        # merge infection minds with labels dataframe
        self.labels = self.labels.merge(
            infection_mins, how='left'
            )

        # convert infection events to renamed infection events
        self.labels['h_popUID'] = self.labels['rename']

        # drop extraneous columns
        self.labels = self.labels.drop(
            columns=['date_pos', 'date_skips', 'rename']
            )

    def ValidateNestedInfections(self):
        """
        final validation and filtering for nested infection events
        """

        def remove_extraneous_infection_events():
            """
            zero out infection events that appear twice in the same
            group (artifact of merging)
            """
            ifx_size = self.labels.\
                groupby('h_popUID').\
                apply(
                    lambda x: x.infection_event.cumsum()
                    ).values

            ifx_size = ifx_size.reshape(-1)

            self.labels.infection_event[
                np.where(ifx_size > 1)[0]
                ] = False

        def drop_multiple_date_values():
            """
            collapses nested infection events by date so no
            duplicated visit numbers are present
            """
            group = ['cohortid', 'h_popUID', 'date']
            self.labels = self.labels.groupby(group).agg({
                'skips': 'min',
                'visit_number': 'min',
                'enrolldate': 'min',
                'burnin': 'min',
                'gender': 'min',
                'agecat': 'min',
                'infection_event': 'max',
                'active_new_infection': 'max',
                'active_baseline_infection': 'max'
                }).reset_index()

        remove_extraneous_infection_events()
        drop_multiple_date_values()

    def CollapseInfectionEvents(self):
        """
        Collapses infection events that fall within the skip threshold of one
        another.
        """

        self.LabelInfectionEvents()

        # collapse duplicate ifx_event dates together
        self.labels = self.labels.\
            groupby(['cohortid', 'ie', 'date']).\
            agg({
                'skips': 'min',
                'visit_number': 'min',
                'enrolldate': 'min',
                'burnin': 'min',
                'gender': 'min',
                'agecat': 'min',
                'infection_event': 'min',
                'active_new_infection': 'min',
                'active_baseline_infection': 'min',
                }).\
            reset_index().\
            rename(columns={'ie': 'h_popUID'})

        # group infection events to their minimum visit_number
        infection_mins = self.labels.\
            groupby('h_popUID').\
            agg({'visit_number': 'min'}).\
            reset_index().\
            rename(columns={'visit_number': 'date_pos'})

        # continue only if there is more than one infection
        if infection_mins.shape[0] <= 1:
            return

        self.NestInfectionEvents(infection_mins)
        self.ValidateNestedInfections()

    def FillMissingDates(self):
        """
        Find visit numbers that were overlooked in infectio event
        nesting and fill in missing dates from dataframe
        """
        to_add = []

        for hid, sub in self.labels.groupby('h_popUID'):

            ie_min = sub.visit_number.min()
            ie_max = sub.visit_number.max()

            values = sub.iloc[-1][sub.columns[5:]]

            for i in np.arange(ie_min, ie_max):

                if i not in sub.visit_number.values:
                    date = self.dates[i]
                    row = {
                        'cohortid': self.cid,
                        'h_popUID': hid,
                        'date': date,
                        'skips': 0,
                        'visit_number': i,
                        'enrolldate': values[0],
                        'burnin': values[1],
                        'gender': values[2],
                        'agecat': values[3],
                        'infection_event': values[4],
                        'active_new_infection': values[5],
                        'active_baseline_infection': values[6]
                        }
                    to_add.append(row)

        for row in to_add:
            self.labels = self.labels.append(row, ignore_index=True)

        self.labels.sort_values(['h_popUID', 'date'], inplace=True)

    def ActiveInfection(self, group):
        """
        Label all timepoints where an infection is still active
        """

        infections = np.where(group.infection_event)[0]

        active_array = np.zeros(group.infection_event.size)

        # for each infection event, label all following infections as active
        for i in np.arange(infections.size):
            if i < infections.size - 1:
                active_array[infections[i]: infections[i+1]] = 1
            else:
                active_array[infections[i]:] = 1

        return active_array

    def LabelActiveInfections(self):
        """
        Label active baseline and new infections
        """
        self.labels.sort_values(['h_popUID', 'date'], inplace=True)
        self.labels['active_new_infection'] = np.concatenate(
            self.labels.groupby(['cohortid', 'h_popUID']).apply(
                lambda x: self.ActiveInfection(x).astype(bool)
                ).values
            )
        self.labels['active_baseline_infection'] = (
            ~self.labels.active_new_infection
            )

    def getLabel(self, row):
        """label infections as true or false"""

        # visits before burnin are false
        if row.date <= row.burnin:
            return False

        # first infection occurs at a timepoint past the allowed skips
        elif row.skips > self.skip_threshold:
            return True

        # if infection is never seen before and after burnin then true
        elif row.hid_visit_number == 1 and row.date > row.burnin:
            return True

        else:
            return False

    def LabelInfections(self, by_clone=True, impute=False):
        merge_cols = ['cohortid', 'enrolldate', 'burnin', 'gender', 'agecat']

        if by_clone:
            self.SkipsByClone(impute=impute)

            if not self.skip_frame.empty:

                self.labels = self.skip_frame.merge(
                    self.frame[merge_cols].iloc[:1],
                    how='left'
                    )

                self.labels['hid_visit_number'] = self.labels.\
                    groupby(['h_popUID']).\
                    apply(lambda x: x.visit_number.rank()).\
                    values.ravel().astype(int)

                self.labels['infection_event'] = self.labels.apply(
                    lambda x: self.getLabel(x),
                    axis=1
                    )

                self.LabelActiveInfections()

        else:
            self.LabelInfections(by_clone=True, impute=True)

            if not self.labels.empty:
                self.CollapseInfectionEvents()
                self.FillMissingDates()

        return self.labels

    def plot_haplodrop(self, infection_event=True, save=False, prefix=None):
        """
        Plot an individuals timeline
        """
        if infection_event:
            timeline = self.timeline.copy()
        else:
            timeline = self.infection_event_timeline.copy()

        if timeline.shape[0] == 0:
            return

        sns.heatmap(
            timeline, square=True, linewidths=1,
            cbar=False, xticklabels=np.where(timeline.columns)[0],
            yticklabels=False,
            annot=True
            )
        if save:
            name = '../plots/cid_haplodrop/{}.png'.format(self.cid)
            if prefix:
                name = '../plots/cid_haplodrop/{}.{}.png'.format(
                    prefix, self.cid
                    )

            print('saving haplodrop : {}'.format(name))
            plt.savefig(name)
        else:
            plt.show()

        plt.close()


class InfectionLabeler(object):
    """
    Label Infection events given a qpcr threshold,
    burnin period, and allowed skips
    """
    pd.options.mode.chained_assignment = None

    def __init__(self, sdo, meta,
                 qpcr_threshold=0, burnin=2, skip_threshold=3,
                 by_infection_event=False, impute_missing=True,
                 agg_infection_event=True, haplodrop=False,
                 drop_missing=True):

        self.sdo = sdo
        self.meta = meta
        self.qpcr_threshold = qpcr_threshold
        self.burnin = burnin
        self.skip_threshold = skip_threshold
        self.by_infection_event = by_infection_event
        self.impute_missing = impute_missing
        self.agg_infection_event = agg_infection_event
        self.drop_missing = drop_missing
        self.haplodrop = haplodrop

        # post processed sdo + meta
        self.frame = pd.DataFrame()

        self.cohort = list()

        # annotated skips dataframe
        self.skips = pd.DataFrame()

        # annotated infection events dataframe
        self.labels = pd.DataFrame()

        # dictionary of timelines by id
        self.id_dates = dict()

        self.InitFrames()

    def PrepareSDO(self):
        """
        - Split cohortid and date from sample name
        - Convert dates to pd.datetime
        - Convert cohortid to int
        """
        split_date_cid = np.vstack(self.sdo.s_Sample.str.split('-'))

        # split date and cid from s_Sample
        self.sdo['date'] = np.array([
            '-'.join(date) for date in split_date_cid[:, :3]
            ]).\
            astype('datetime64')
        self.sdo['cohortid'] = split_date_cid[:, -1].\
            astype(str)

    def PrepareMeta(self):
        """
        - Convert date to pd.datetime
        - Convert enrolldate to pd.datetime
        - Convert cohortid to int
        - Sort values by date
        - Add burnin date
        """

        self.meta['date'] = self.meta['date'].astype('datetime64')
        self.meta['enrolldate'] = self.meta['enrolldate'].astype('datetime64')
        self.meta['cohortid'] = self.meta['cohortid'].astype(str)

        self.meta.sort_values(['cohortid', 'date'], inplace=True)

        return self.meta

    def MarkQPCR(self):
        """
        Keep samples above qpcr threshold (removes NaN as well)
        """
        self.meta['pass_qpcr'] = self.meta.qpcr > self.qpcr_threshold

    def MergeFrames(self):
        """
        - Merge seekdeep output and meta data
        - Update h_popUID type
        """
        self.frame = self.meta.merge(
            self.sdo,
            left_on=['cohortid', 'date'],
            right_on=['cohortid', 'date'],
            how='left'
            )
        self.frame['h_popUID'] = self.frame['h_popUID'].astype(str)
        self.frame.date = self.frame.date

        self.frame = self.frame[
            ~np.isnan(self.frame.qpcr)
            ]

    def AddBurnin(self):
        """
        Generate burnin and add to Meta

        - Take minimum enrolldate per cohortid
        - Add {burnin} months to enrolldate
        - Merge with meta frame
        """

        cid_enroll = self.meta.\
            groupby('cohortid').\
            agg({'enrolldate': 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + \
            pd.DateOffset(months=self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on=['cohortid', 'enrolldate'],
            right_on=['cohortid', 'enrolldate']
            )

    def InitFrames(self):
        """
        - Initialize Seekdeep Output & Meta
        - Add burnin dates to Meta
        - Mark QPCR passing threshold
        - Merge
        """
        self.PrepareSDO()
        self.PrepareMeta()
        self.AddBurnin()
        self.MarkQPCR()
        self.MergeFrames()
        self.InitializeCohort()

    def InitializeCohort(self):
        """
        Create Individual objects for each individual in the cohort
        """

        # self.frame = self.frame[self.frame.cohortid == '3147']

        iter_frame = tqdm(
            self.frame.groupby('cohortid'),
            desc='initializing cohort'
            )

        for cid, cid_frame in iter_frame:
            t = Individual(
                cid_frame,
                skip_threshold=self.skip_threshold,
                drop_missing=self.drop_missing,
                haplodrop=self.haplodrop,
                impute_missing=self.impute_missing
                )
            self.cohort.append(t)

    def AddMeta(self):
        """
        Add cohortid meta data to labels dataframe
        """

        # adds malaria category
        self.labels = self.labels.merge(
            self.meta[['cohortid', 'date', 'malariacat']],
            how='left',
            left_on=['cohortid', 'date'],
            right_on=['cohortid', 'date']
            )

    def LabelInfections(self, by_clone=True):
        """
        Label infections for all individuals in the cohort

        Returns
        -------
        pd.Dataframe
            Returns concatenated labels for all infections in the cohort

        """

        iter_cohort = tqdm(
            self.cohort,
            desc='labeling infections'
            )

        self.labels = pd.concat([
            c.LabelInfections(by_clone=by_clone) for c in iter_cohort
            ])

        self.AddMeta()
        return self.labels


# Metrics


class FOI(object):

    def __init__(self, labels, meta, burnin=3):
        self.labels = labels
        self.meta = meta
        self.burnin = burnin

        self.frame = pd.DataFrame()

        self.prepareData()

    def prepareData(self):
        """
        validate column types
        add burnin to meta
        merge labels with meta
        """
        self.labels.date = self.labels.date.astype('datetime64')
        self.labels.enrolldate = self.labels.enrolldate.astype('datetime64')
        self.labels.burnin = self.labels.burnin.astype('datetime64')

        self.meta.date = self.meta.date.astype('datetime64')
        self.meta.enrolldate = self.meta.enrolldate.astype('datetime64')
        self.meta['year_month'] = pd.DatetimeIndex(self.meta.date).\
            to_period('M')

        self.AddBurnin()

        merge_columns = [
            'cohortid', 'date', 'enrolldate',
            'burnin', 'gender', 'agecat'
            ]

        self.frame = self.meta.merge(
                self.labels,
                left_on=merge_columns,
                right_on=merge_columns,
                how='left'
                )

    def AddBurnin(self):
        """
        Generate burnin and add to Meta

        - Take minimum enrolldate per cohortid
        - Add {burnin} months to enrolldate
        - Merge with meta frame
        """

        cid_enroll = self.meta.\
            groupby('cohortid').\
            agg({'enrolldate': 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + \
            pd.DateOffset(months=self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on=['cohortid', 'enrolldate'],
            right_on=['cohortid', 'enrolldate']
            )

    def getDurations(self, group=None, working_frame=None):
        """
        return durations across a group or a singular value
        for the full dataset
        """
        # if type(working_frame) == type(None):
        if not isinstance(working_frame, pd.core.frame.DataFrame):
            working_frame = self.frame.copy()

        working_frame = working_frame[
            working_frame.date >= working_frame.burnin
            ]

        if group:
            durations = working_frame.\
                groupby(group).\
                apply(lambda x: self.getDurations(working_frame=x))

        else:
            durations = working_frame.date.max() - working_frame.date.min()
            durations = durations.days / 365.25

        return durations

    def getInfections(self, group=None, working_frame=None):
        """
        return number of infections across a group
        or a singular value for the full dataset
        """
        if not isinstance(working_frame, pd.core.frame.DataFrame):
            working_frame = self.frame.copy()

        working_frame = working_frame[
            working_frame.date >= working_frame.burnin
            ]

        if group:
            events = working_frame.\
                groupby(group).\
                apply(lambda x: self.getInfections(working_frame=x))
        else:
            events = working_frame.infection_event.sum()

        return events

    def getExposure(self, group=None, working_frame=None):
        """
        return number of exposed individuals across a group
        or a singular value for the full dataset
        """
        if not isinstance(working_frame, pd.core.frame.DataFrame):
            working_frame = self.frame.copy()

        working_frame = working_frame[
            working_frame.date >= working_frame.burnin
            ]

        if group:
            exposure = working_frame.\
                groupby(group).\
                apply(lambda x: self.getExposure(working_frame=x))
        else:
            exposure = working_frame['cohortid'].unique().size

        return exposure

    def fit(self, group=None):
        """
        calculate force of infection across dataset given groups
        """

        durations = self.getDurations(group=group)
        events = self.getInfections(group=group)
        exposure = self.getExposure(group=group)

        foi = events / (exposure * durations)

        if group:
            grouped_results = np.vstack([
                i.values for i in [events, durations, exposure, foi]
            ]).T
            foi = pd.DataFrame(
                grouped_results,
                columns=['events', 'durations', 'exposure', 'FOI'],
                index=foi.index
                )

        else:
            foi = pd.DataFrame(
                np.array([events, durations, exposure, foi]).reshape(1, -1),
                columns=['events', 'durations', 'exposure', 'FOI']
            )

        return foi.reset_index()


class ExponentialDecay(object):

    def __init__(self, infections,
                 left_censor='2018-01-01', right_censor='2019-04-01',
                 minimum_duration=30, seed=None, skips=3
                 ):

        if seed:
            np.random.seed(seed)

        self.infections = infections
        self.study_start = pd.to_datetime(left_censor) if left_censor \
            else infections.date.min()
        self.study_end = pd.to_datetime(right_censor) if right_censor \
            else infections.date.max()
        self.study_censor = self.study_end - pd.Timedelta(
                '{} Days'.format(skips * 28)
                )
        self.minimum_duration = pd.Timedelta(
            '{} Days'.format(minimum_duration)
            )

        self.durations = []
        self.num_classes = np.zeros(5)
        self.optimizers = []

        self.DropMalaria()

    def DropMalaria(self):
        """
        Drop any infections with a malaria event
        """

        to_drop = self.infections.\
            groupby(['cohortid', 'h_popUID']).\
            apply(
                lambda x: np.any(x.malariacat == 'Malaria')
                ).reset_index()

        self.infections = self.infections.\
            merge(to_drop, how='left')

        self.infections = self.infections[~self.infections[0]].\
            drop(columns=0)

    def BootstrapInfections(self, frame):
        """Bootstrap on Cohortid"""

        mat = frame.values

        cids = np.unique(mat[:, 0])

        cid_choice = np.random.choice(cids, cids.size)

        bootstrap = np.concatenate([
            mat[mat[:, 0] == i] for i in cid_choice
            ])

        return pd.DataFrame(bootstrap, columns=frame.columns)

    def AddClassifications(self, class_vec):
        """
        Add number of observations of each classification
        """

        classification, counts = np.unique(
            class_vec, return_counts=True
            )

        for i, j in enumerate(classification):
            self.num_classes[j] += counts[i]

    def SplitLikelihoods(self, durations):
        """
        Split durations into different arrays for vectorized liklihood
        calculations
        """

        durations = durations[durations[:, 0] != 0]

        l1_durations = durations[durations[:, 0] <= 2][:, 1]

        l2_durations = durations[durations[:, 0] > 2][:, 1]

        self.durations.append([l1_durations, l2_durations])

        return l1_durations, l2_durations

    def GetInfectionDurations(self, infection_frame):
        """for each clonal infection calculate duration"""

        def ClassifyInfection(ifx_min, ifx_max, study_start, study_end,
                              study_censor, minimum_duration):
            """
            Classify an infection type by whether or not the start date of the
            infection is observed in a given period and return the duration by
            the class
            """

            active = (ifx_max >= study_start) & (ifx_min <= study_end)
            start_observed = (ifx_min >= study_start)
            end_observed = (ifx_max <= study_censor)

            classification = 0
            duration = minimum_duration

            # infection not active in period
            if not active:
                duration = -1
                return classification, duration

            # Start and End Observed in Period
            elif start_observed and end_observed:
                classification = 1
                duration = ifx_max - ifx_min

            # Unobserved Start + Observed End in Period
            elif not start_observed and end_observed:
                classification = 2
                duration = ifx_max - study_start

            # Observed Start + Unobserved End in Period
            elif start_observed and not end_observed:
                classification = 3
                if ifx_max < study_end:
                    duration = ifx_max - ifx_min
                else:
                    duration = study_end - ifx_min

            # Unobserved Start + Unobserved End in Period
            elif not start_observed and not end_observed:
                classification = 4
                if ifx_max < study_end:
                    duration = ifx_max - study_start
                else:
                    duration = study_end - study_start

            duration = np.timedelta64(duration, 'D').astype(int)

            return classification, duration

        study_end = self.study_end.asm8
        study_censor = self.study_censor.asm8
        minimum_duration = self.minimum_duration.asm8

        infection_minimums = infection_frame.\
            groupby(['cohortid', 'h_popUID']).\
            apply(lambda x: (
                x.date.values.min(),
                x.date.values.max(),
                x.burnin.values.min()
                )
            )

        durations = infection_minimums.apply(
            lambda x: ClassifyInfection(
                x[0], x[1], x[2], study_end,
                study_censor, minimum_duration
                )
            )

        durations = np.vstack(durations.values)

        # print(durations.shape)
        # sys.exit(durations)

        self.AddClassifications(durations[:, 0])

        l1_durations, l2_durations = self.SplitLikelihoods(durations)

        return l1_durations, l2_durations, durations

    def GetConfidenceIntervals(self, min=5, max=95):
        """
        Return confidence intervals for a bootstrapped array
        """
        ci_min, ci_max = np.percentile(self.bootstrapped_lams, [min, max])
        return ci_min, ci_max

    def fit(self, frame=None, bootstrap=False, n_iter=200):
        """
        Fit Exponential Model
        """

        def decay_function(lam, l1_durations, l2_durations):
            """
            Exponential Decay Function as Log Likelihood
            """

            l1_llk = np.log(lam) - (lam * l1_durations)

            l2_llk = (-1 * lam) * l2_durations

            llk = l1_llk.sum() + l2_llk.sum()

            return -1 * llk

        if not isinstance(frame, pd.core.frame.DataFrame):
            frame = self.infections.copy()
        if bootstrap:
            bootstrapped_lams = [
                self.fit(frame=self.BootstrapInfections(frame))
                for _ in tqdm(range(n_iter))
                ]

        # generate durations and initial guess
        l1_d, l2_d, durations = self.GetInfectionDurations(frame)
        lam = np.random.random()

        # run minimization of negative log likelihood
        opt = minimize(
            decay_function,
            lam,
            args=(l1_d, l2_d),
            method='L-BFGS-B',
            bounds=((1e-6, None), )
            )
        self.optimizers.append(opt)
        self.estimated_lam = opt.x[0]

        if bootstrap:
            self.bootstrapped_lams = np.array(bootstrapped_lams)
            return (self.estimated_lam, self.bootstrapped_lams)
        else:
            return self.estimated_lam

    def plot(self):
        """
        Generate a plot of the distribution of bootstrapped lambdas
        """
        ci_min, ci_max = self.GetConfidenceIntervals()
        estim_l_str = "Estimated Lambda : {:.4f}e-3 ({:.4f}e-3 -- {:.4f}e-3)"
        estim_d_str = "Estimated Days : {:.1f} ({:.1f} -- {:.1f})"

        sns.distplot(1 / self.bootstrapped_lams, color='teal', bins=30)
        plt.axvline(1 / self.estimated_lam, color='teal', linestyle=':', lw=8)
        plt.xlabel("Calculated Days (1 / lambda)")
        plt.title(
            '\n'.join([estim_l_str, estim_d_str]).format(
                self.estimated_lam * 1e3, ci_min * 1e3, ci_max * 1e3,
                1/self.estimated_lam, 1/ci_max, 1/ci_min
                )
            )
        plt.show()
        plt.close()


# Survival


class Survival(object):

    def __init__(self, infections, meta,
                 burnin=3, skip_threshold=3,
                 bootstrap=False, n_iter=200
                 ):
        self.infections = infections
        self.meta = meta
        self.burnin = burnin
        self.skip_threshold = skip_threshold
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.minimum_duration = pd.Timedelta('15 days')

        self.date_bins = pd.DataFrame()
        self.ym_counts = pd.Series()
        self.treatments = pd.DataFrame()

        # dataframe to store original infection results
        self.original_results = pd.DataFrame()

        # dataframe to store bootstrapped infection results
        self.bootstrap_results = pd.DataFrame()

        self.ValidateInfections()
        self.ValidateMeta()
        self.ymCounts()

        self.original_infections = self.infections.copy()

    def ValidateInfections(self):
        """
        validate labeled infections for information required
        """

        # convert infection date to date
        self.infections.date = self.infections.date.astype('datetime64')

        # convert burnin to date if not already
        self.infections.burnin = self.infections.burnin.astype('datetime64')

        self.infections['year_month'] = pd.DatetimeIndex(
            self.infections.date
            ).\
            to_period('M')
        self.infections['year_month'] = [
            i.to_timestamp() for i in self.infections.year_month.values
            ]
        self.infections['year_month'] = self.infections['year_month'].\
            astype('datetime64')

    def ValidateMeta(self):
        """
        validate meta dataframe for information required
        """

        self.meta.date = self.meta.date.astype('datetime64')
        self.meta.enrolldate = self.meta.enrolldate.astype('datetime64')
        self.meta['burnin'] = self.meta.enrolldate + \
            pd.DateOffset(months=self.burnin)

        self.meta['year_month'] = pd.DatetimeIndex(self.meta.date).\
            to_period('M')
        self.meta['year_month'] = [
            i.to_timestamp() for i in self.meta.year_month.values
            ]
        self.meta['year_month'] = self.meta['year_month'].\
            astype('datetime64')

    def ymCounts(self):
        """
        count number of people a year_month
        """
        self.ym_counts = self.meta.\
            groupby('year_month').\
            apply(
                lambda x: x['cohortid'].unique().size
                )

    def BootstrapInfections(self, frame=None):
        """
        randomly sample with replacement on CID
        """
        if not frame:
            frame = self.original_infections.copy()

        c = frame.cohortid.unique().copy()
        rc = np.random.choice(c, c.size)
        self.infections = frame.copy()

        # calculate index size for each cohortid in random choice
        cid_size = np.array([
            np.where(self.infections.cohortid == i)[0].size for i in rc
            ])

        # set index to cohortid
        self.infections = self.infections.set_index('cohortid')

        # generate bootstrap
        self.infections = self.infections.loc[rc]

        # create array of new cid_id with expected length
        new_cid = np.concatenate([
            np.full(cid_size[i], i) for i in range(cid_size.size)
            ]).ravel()

        # have hashable id num to cid
        self.bootstrap_id_dates = pd.Series(rc)

        self.infections = self.infections.reset_index()
        self.infections['cohortid'] = new_cid


class FractionOldNew(Survival):
    """
    Survival object that calculates the fraction of old vs new clones
    by year_month
    """

    def __init__(self, *args, **kwargs):
        Survival.__init__(self, *args, **kwargs)

    def uniqueCidHid(self, frame):
        """
        Return number of unique cohortid~h_popUID pairs
        """
        return frame[['cohortid', 'h_popUID']].drop_duplicates().shape[0]

    def FractionByPeriod(self, frame):
        """
        Calculate percentage of Old / New clones by year_month period
        """
        ym_frame = frame.\
            groupby(['year_month', 'active_new_infection']).\
            apply(lambda x: self.uniqueCidHid(x)).\
            reset_index().\
            rename(columns={0: 'counts'})

        ym_frame['pc'] = ym_frame[['year_month', 'counts']].\
            groupby('year_month').\
            apply(lambda x: x/x.sum())

        return ym_frame[ym_frame.active_new_infection]

    def RunBootstraps(self):
        """
        Calculate fraction by period on bootstrapped infections
        """

        bootstraps = list()

        for i in tqdm(range(self.n_iter), desc='bootstrapping...'):
            self.BootstrapInfections()
            bootstraps.append(
                self.FractionByPeriod(self.infections)
                )

        # merge bootstraps
        self.bootstrap_results = pd.concat(bootstraps)

        # calculate confidence intervals
        self.bootstrap_results = self.bootstrap_results.\
            groupby(['active_new_infection', 'year_month']).\
            apply(
                lambda x: np.percentile(x.pc, [2.5, 97.5])
                )

    def fit(self):
        """
        Run fraction by period on original dataframe,
        and bootstraps if option supplied
        """
        self.original_results = self.FractionByPeriod(self.original_infections)
        if self.bootstrap:
            self.RunBootstraps()

    def plot(self):

        if self.bootstrap:

            for v in self.original_results.active_new_infection.unique():

                sns.lineplot(
                    data=self.original_results[
                        self.original_results.active_new_infection == v
                        ],
                    x='year_month',
                    y='pc',
                    label=v,
                    lw=4
                    )

                plt.fill_between(
                    self.bootstrap_results[v].index,
                    [i for i, j in self.bootstrap_results[v].values],
                    [j for i, j in self.bootstrap_results[v].values],
                    alpha=0.5)
        else:
            sns.lineplot(
                data=self.original_results,
                x='year_month',
                y='pc',
                hue='active_new_infection')

        plt.ylim(0, 1)

        plt.xlabel('Date')
        plt.ylabel('Percentage')
        plt.title('Fraction of Old Clones In Infected Population')
        plt.show()
        plt.close()


class OldNewSurival(Survival):
    """
    Survival objects that calculates the fraction of old, new, and mixed clones
    in total population
    """

    def __init__(self, *args, **kwargs):
        Survival.__init__(self, *args, **kwargs)

        self.infection_category = np.array(['old', 'new', 'mix'])

    def ClassifyInfection(self, x):
        """
        classify cohortid infection type as:
        1 : only old
        2 : only new
        3 : mixed infections
        """
        m = x.active_new_infection.mean()
        if (m == 0) | (m == 1):
            return self.infection_category[m.astype(int)]
        else:
            return self.infection_category[2]

    def UniqueCID(self, x):
        """
        return number of unique cids in dataframe
        """
        return x.cohortid.unique().size

    def CalculatePercentageOldNewMix(self, frame):
        """
        Calculate percentage of old, new, and mixed infections as fraction
        of total population at each year_month timepoint
        """

        # classify cohortid by infection type
        mix_counts = frame.\
            groupby(['cohortid', 'year_month']).\
            apply(lambda x: self.ClassifyInfection(x)).\
            reset_index().\
            rename(columns={0: 'cid_active_new_infection'})

        # count infection types by year month
        date_counts = mix_counts.groupby([
                'year_month', 'cid_active_new_infection'
                ]).\
            apply(lambda x: self.UniqueCID(x)).\
            reset_index().\
            rename(columns={0: 'counts'})

        # calculate as percentage
        date_counts['pc'] = date_counts.apply(
            lambda x: x.counts / self.ym_counts[x.year_month],
            axis=1
            )

        return date_counts

    def RunBootstraps(self):
        """
        Calculate fraction by period on bootstrapped infections
        """

        bootstraps = list()

        for i in tqdm(range(self.n_iter), desc='bootstrapping...'):
            self.BootstrapInfections()
            bootstraps.append(
                self.CalculatePercentageOldNewMix(self.infections)
                )

        # merge bootstraps
        self.bootstrap_results = pd.concat(bootstraps)

        # calculate confidence intervals
        self.bootstrap_results = self.bootstrap_results.\
            groupby(['cid_active_new_infection', 'year_month']).\
            apply(
                lambda x: np.percentile(x.pc, [2.5, 97.5])
                )

    def fit(self):
        self.original_results = self.CalculatePercentageOldNewMix(
            self.original_infections
            )
        if self.bootstrap:
            self.RunBootstraps()

    def plot(self):
        if self.bootstrap:

            for v in self.original_results.cid_active_new_infection.unique():
                sns.lineplot(
                    data=self.original_results[
                        self.original_results.cid_active_new_infection == v
                        ],
                    x='year_month', y='pc',
                    label=v, lw=4
                    )
                plt.fill_between(
                    self.bootstrap_results[v].index,
                    [i for i, j in self.bootstrap_results[v].values],
                    [j for i, j in self.bootstrap_results[v].values],
                    alpha=0.3)

        else:
            sns.lineplot(
                data=self.original_results,
                x='year_month', y='pc',
                hue='cid_active_new_infection'
                )

        plt.xlabel('Date')
        plt.ylabel('Percentage')
        plt.title('Fraction of New and Old Clones by Individual')
        plt.show()
        plt.close()


class OldWaning(Survival):
    """
    calculate fraction of old clones remaining across each month past
    the burnin period
    """

    def __init__(self, *args, **kwargs):
        Survival.__init__(self, *args, **kwargs)

    def MonthlyKept(self, x):
        """
        return number of old infections past burnin
        """
        monthly_old = x[(x.active_baseline_infection) & (x.date >= x.burnin)]
        return monthly_old[['cohortid', 'h_popUID']].drop_duplicates().shape[0]

    def CalculateWaning(self, frame):
        """
        Calculate percentage of old clones kept by year month
        """
        monthly_counts = frame.\
            groupby('year_month').\
            apply(lambda x: self.MonthlyKept(x))

        monthly_pc = (monthly_counts / monthly_counts.values.max()).\
            reset_index().\
            rename(columns={0: 'pc'})

        return monthly_pc

    def RunBootstraps(self):
        """
        Calculate fraction by period on bootstrapped infections
        """

        bootstraps = list()

        for i in tqdm(range(self.n_iter), desc='bootstrapping...'):
            self.BootstrapInfections()
            bootstraps.append(
                self.CalculateWaning(self.infections)
                )

        # merge bootstraps
        self.bootstrap_results = pd.concat(bootstraps)

        # calculate confidence intervals
        self.bootstrap_results = self.bootstrap_results.\
            groupby(['year_month']).\
            apply(
                lambda x: np.percentile(x.pc, [2.5, 97.5])
                )

    def fit(self):
        self.original_results = self.CalculateWaning(self.original_infections)
        if self.bootstrap:
            self.RunBootstraps()

    def plot(self):
        sns.lineplot(
            data=self.original_results,
            x='year_month', y='pc',
            legend=False, lw=5
            )

        if self.bootstrap:
            plt.fill_between(
                self.bootstrap_results.index,
                y1=[i for i, j in self.bootstrap_results.values],
                y2=[j for i, j in self.bootstrap_results.values],
                alpha=0.3
                )

        plt.xlabel('Date')
        plt.title('Percentage')
        plt.title('Fraction of Old Clones Remaining')
        plt.show()
        plt.close()

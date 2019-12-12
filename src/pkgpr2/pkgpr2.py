#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math

from multiprocess import Pool
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

        self.rename_dict = {}

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
        Recursive implementation of nesting infection events

        Calculate skips
        If collapse necessary
            collapse
            recurse
        Apply changes

        Parameters
        ----------
        infection_mins : pd.DataFrame
            Dataframe of infection events and date positions

        """
        def apply_rename(x):
            if x.h_popUID in self.rename_dict:
                return self.rename_dict[x.h_popUID]
            else:
                return x.h_popUID

        date_pos_arr = np.zeros(
            infection_mins.date_pos.max() + 1, dtype=bool
            )
        date_pos_arr[infection_mins.date_pos] = True

        # calculate skips
        skips = self.PositionalDifference(
                date_pos_arr, 0, add_one=True
                )

        skips = skips + 1

        # find where skips do not exceed skip threshold
        to_merge = np.where(
            skips <= self.skip_threshold
            )[0]

        to_merge = to_merge[to_merge != 0]

        if to_merge.size > 0:
            merge_from = infection_mins.iloc[to_merge[0]].h_popUID
            merge_to = infection_mins.iloc[to_merge[0] - 1].h_popUID

            self.rename_dict[merge_from] = merge_to

            # drop infection event from infection mins and reset index
            infection_mins = infection_mins.drop(to_merge[0]).\
                reset_index()
            infection_mins = infection_mins.iloc[:, 1:]

            self.NestInfectionEvents(infection_mins)

        self.labels['h_popUID'] = self.labels.apply(
            lambda x: apply_rename(x), axis=1
            )

    def oldNestInfectionEvents(self, infection_mins):
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

    def MakeLongForm(self):

        def convert_dataframe(hid, x, ifx_idx):
            frame_dict = []
            for i in ifx_idx[:-1]:
                frame_dict.append({
                    'cohortid': self.cid,
                    'h_popUID': hid,
                    'date': self.dates[i],
                    'end_date': self.dates[i+1],
                    'agecat': x.agecat.unique()[0],
                    'gender': x.gender.unique()[0],
                    'active_baseline_infection': x.active_baseline_infection.max(),
                    'terminal': False
                    })

            if ifx_idx[-1] == len(self.dates) - 1:
                frame_dict[-1]['terminal'] = True
            else:
                frame_dict.append({
                    'cohortid': self.cid,
                    'h_popUID': hid,
                    'date': self.dates[ifx_idx[-1]],
                    'end_date': self.dates[ifx_idx[-1] + 1],
                    'agecat': x.agecat.unique()[0],
                    'gender': x.gender.unique()[0],
                    'active_baseline_infection': x.active_baseline_infection.max(),
                    'terminal': True
                    })

            frame = pd.DataFrame(frame_dict)
            return frame

        def convert_longform(hid, x):
            interval_min = x.date.min().to_datetime64()
            interval_max = x.date.max().to_datetime64()

            ifx_idx = np.where(
                (self.dates >= interval_min) &
                (self.dates <= interval_max)
                )[0]

            frame = convert_dataframe(hid, x, ifx_idx)
            return frame

        if self.labels.empty:
            return

        hid_frames = []
        for hid, g in self.labels.groupby(['h_popUID']):
            hid_frames.append(convert_longform(hid, g))

        self.labels = pd.concat(hid_frames)

    def LabelInfections(self, by_clone=True, long_form=False, impute=False):
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

        if long_form:
            self.MakeLongForm()
        # sys.exit()
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

        # self.frame = self.frame[self.frame.cohortid == '3597']

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
            self.meta[
                ['cohortid', 'date', 'malariacat', 'qpcr', 'hhid', 'hemoglobin']
                ],
            how='left',
            left_on=['cohortid', 'date'],
            right_on=['cohortid', 'date']
            )

    def LabelInfections(self, by_clone=True, long_form=False):
        """
        Label infections for all individuals in the cohort

        Returns
        -------
        pd.Dataframe
            Returns concatenated labels for all infections in the cohort

        """
        def pooled_run(cid):
            return cid.LabelInfections(by_clone=by_clone, long_form=long_form)

        iter_cohort = tqdm(
            self.cohort,
            desc='labeling infections'
            )

        p = Pool()
        self.labels = pd.concat(
            p.map(pooled_run, iter_cohort)
            )

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
            'burnin', 'gender', 'agecat', 'malariacat'
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

        sys.exit()

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

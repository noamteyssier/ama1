#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import *
import sys, math, warnings
from scipy.optimize import minimize
from scipy.special import expit

from multiprocessing import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(20, 20), 'lines.linewidth': 2})

class InfectionLabeler:
    """
    Label Infection events given a qpcr threshold, burnin period, and allowed skips
    """
    pd.options.mode.chained_assignment = None
    def __init__(self, sdo, meta, qpcr_threshold = 0, burnin=3, allowedSkips = 3, by_infection_event=False, impute_missing=True, agg_infection_event=True, haplodrops=False):

        self.sdo = sdo
        self.meta = meta
        self.qpcr_threshold = qpcr_threshold
        self.burnin = burnin
        self.allowedSkips = allowedSkips
        self.by_infection_event = by_infection_event
        self.impute_missing = impute_missing
        self.agg_infection_event = agg_infection_event
        self.haplodrops = haplodrops
        self.is_bootstrap = False

        # post processed sdo + meta
        self.frame = pd.DataFrame()

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
        self.sdo['date'] = pd.to_datetime(['-'.join(date) for date in split_date_cid[:,:3]])
        self.sdo['cohortid'] = split_date_cid[:,-1].astype(str)
    def PrepareMeta(self):
        """
        - Convert date to pd.datetime
        - Convert enrolldate to pd.datetime
        - Convert cohortid to int
        - Sort values by date
        - Add burnin date
        """

        self.meta['date'] = pd.to_datetime(self.meta['date'])
        self.meta['enrolldate'] = pd.to_datetime(self.meta['enrolldate'])
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
            self.sdo, how = 'left',
            left_on = ['cohortid', 'date'],
            right_on = ['cohortid', 'date']
            )
        self.frame['h_popUID'] = self.frame['h_popUID'].astype(str)
        self.frame.date = self.frame.date


        if 'pseudo_cid' in self.frame.columns:
            self.is_bootstrap = True
            self.frame.pseudo_cid = self.frame.pseudo_cid.astype(str)
    def AddBurnin(self):
        """
        Generate burnin and add to Meta

        - Take minimum enrolldate per cohortid
        - Add {burnin} months to enrolldate
        - Merge with meta frame
        """

        cid_enroll = self.meta.\
            groupby('cohortid').\
            agg({'enrolldate' : 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + pd.DateOffset(months = self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on = ['cohortid', 'enrolldate'],
            right_on = ['cohortid', 'enrolldate']
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
    def PostBurnin(self, dates, burnin):
        """
        Find first date past the burnin,
        if burnin is between two visits mark postburnin as midway
        """
        try:
            post_burnin = np.where(dates >= burnin)[0].min()

        except ValueError:
            return False


        if dates[post_burnin] > burnin:
            post_burnin -= 0.5

        return post_burnin
    def PositionalDifference(self, x, post_burnin):
        """
        For a bool array x : calculate number of skips between each truth
        calculate first truth with respect to the number of visits post burnin
        """
        truth = np.where(x)[0]

        # no qpcr positive events
        if truth.size == 0:
            return np.array([])

        skip_arr = []
        for i, pos in enumerate(truth):

            # first occurence of a haplotype
            if i == 0:

                # make all skips zero pre burnin
                if pos < post_burnin:
                    skips = 0

                # first visit after burnin is equal to visit number
                else:
                    skips = pos


            # every other occurence
            else:
                # disregard pre-burnin skips
                if (truth[i - 1] < post_burnin) & (pos >= post_burnin):
                    skips = pos - math.ceil(post_burnin)

                # calculate skips in pre-burnin period
                else:
                    skips = pos - truth[i-1] - 1

            skip_arr.append(skips)

        return np.array(skip_arr)
    def BuildTimeline(self, cid_frame, index='h_popUID', column = 'date', value='pass_qpcr'):
        """
        Awesome numpy pivot function found @
        https://stackoverflow.com/questions/48527091/efficiently-and-simply-convert-from-long-format-to-wide-format-in-pandas-and-or
        """
        arr = cid_frame.values
        idx_index, idx_column, idx_value = [np.where(cid_frame.columns == v)[0] for v in [index, column, value]]


        rows, row_pos = np.unique(arr[:, idx_index], return_inverse=True)
        cols, col_pos = np.unique(arr[:, idx_column], return_inverse=True)
        pivot_arr = np.zeros((rows.size, cols.size))
        pivot_arr[row_pos, col_pos] = arr[:, idx_value[0]]

        frame = pd.DataFrame(
            pivot_arr,
            columns = cols,
            index = rows,
            dtype=bool
            )
        frame.index.name = index
        return frame
    def DropEmptyGenotyping(self, cid_timeline):
        """
        Have qpcr row inherit genotyping passing qpcr
        Remove empty genotyping results with positive qpcr
        """

        if self.by_infection_event == False or self.impute_missing == False:
            return cid_timeline[cid_timeline.index != 'nan']
        else:
            cid_timeline.loc['nan'] = cid_timeline.max(axis=0)
            return cid_timeline
    def SkipsByHaplotype(self, hid_timelines, cid, post_burnin):
        """
        Calculate Skips for each haplotype given an HID_timeline
        """

        hids = hid_timelines.index
        dates = hid_timelines.columns

        for idx, hid_arr in enumerate(hid_timelines.values):

            skips = self.PositionalDifference(hid_arr, post_burnin)
            hid_dates = dates[hid_arr]
            visit_numbers = np.where(hid_arr)[0]

            if skips.size > 0:
                self.skip_frame.append(
                    [[cid, hids[idx], hid_dates[idx_skips], skips[idx_skips], visit_numbers[idx_skips]]
                    for idx_skips,_ in enumerate(skips)]
                )
    def CalculateSkips(self):
        """
        - Build timelines for each h_popUID~cohortid combination
        - Calculate number of skips between each passing qpcr event
        - Calculate visit number of events
        - Compile dataframe of skips and visit number
        """

        self.skip_frame = []
        cid_group = 'cohortid' if not self.is_bootstrap else 'pseudo_cid'
        # self.frame = self.frame[self.frame.cohortid == '3786']

        for cid, cid_frame in tqdm(self.frame.groupby([cid_group]), desc='calculating skips'):
            burnin = cid_frame.burnin.values[0]
            self.id_dates[cid] = cid_frame.date.unique()

            # convert long dates to wide
            cid_timeline = self.BuildTimeline(cid_frame)

            # no genotyping inherit qpcr + drop qpcr if not by_infection_event or not impute_missing
            cid_timeline = self.DropEmptyGenotyping(cid_timeline)

            post_burnin = self.PostBurnin(cid_timeline.columns, burnin)

            self.SkipsByHaplotype(cid_timeline, cid, post_burnin)

            if self.haplodrops:
                self.plot_haplodrop(cid_timeline, save=cid)

        self.skips = pd.DataFrame(
            np.vstack(self.skip_frame),
            columns = ['cohortid', 'h_popUID', 'date', 'skips', 'visit_number']
            )

        self.skips.date = pd.to_datetime(self.skips.date)
    def AggregateInfectionEventDate(self):
        """
        Aggregate infection events by date, relabel h_popUID as aggregate over id~date
        """

        rows = []
        for cid_date, cid_date_frame in self.labels.groupby(['cohortid', 'date']):
            cid, date = cid_date
            agg_name = 'agg_{}_{}'.format(cid, date.date())

            rows.append([
                cid, agg_name, date,
                cid_date_frame.skips.max(), cid_date_frame.visit_number.min(),
                cid_date_frame.enrolldate.min(), cid_date_frame.burnin.min(),
                cid_date_frame.infection_event.max()
            ])

        self.labels = pd.DataFrame(rows, columns = self.labels.columns)
    def AggregateInfectionEvents(self):
        """
        Aggregate infection events using skip rule
        - find all infection events
        - calculate distance between sequential events
        - aggregate events not passing skips
        """

        for cid, cid_frame in tqdm(self.labels.groupby(['cohortid']), desc='aggregating infection events'):

            # build infection event dataframe
            hid_frame = self.BuildTimeline(cid_frame, value = 'infection_event')

            # fill missing dates
            missing_dates = self.id_dates[cid][
                np.isin(self.id_dates[cid], hid_frame.columns, invert=True)
                ]
            for d in missing_dates:
                hid_frame[d] = False

            # sort columns by date
            hid_frame = hid_frame.loc[:,np.sort(hid_frame.columns.values)]

            # all infection events
            infection_points = hid_frame.max(axis=0)

            # positive indices
            pos_infection = np.where(infection_points)[0]

            # skips between infection events
            point_skips = self.PositionalDifference(infection_points.values, 0)

            # skips past threshold
            passing_skips = np.where(point_skips > self.allowedSkips)[0]


            # dates passing skip threshold
            passing_dates = infection_points[pos_infection[passing_skips]].index.values

            # subset labels, negate all infection events, flip only those passing aggregation
            self.labels.infection_event[
                (self.labels.cohortid == cid) & \
                np.isin(self.labels.date, passing_dates, invert=True)
                ] = False

            if self.haplodrops:
                self.plot_haplodrop(
                    self.BuildTimeline(self.labels[self.labels.cohortid == cid], value='infection_event'),
                    save=cid,
                    prefix='aggIE'
                )
    def AggregateInfections(self):
        """
        Perform aggregation depending on flags given
        """
        if self.by_infection_event:
            self.AggregateInfectionEventDate()

            if self.agg_infection_event:
                self.AggregateInfectionEvents()
    def getLabel(self, row):
        """label infections as true or false"""

        # visits before burnin are false
        if row.date <= row.burnin:
            return False

        # first infection occurs at a timepoint past the allowed skips
        elif row.skips > self.allowedSkips :
            return True

        # if infection is never seen before and after burnin then true
        elif row.skips == row.visit_number:
            return True

        else:
            return False
    def ActiveInfection(self, group):
        """
        Label all timepoints where an infection is still active
        """

        infections = np.where(group.infection_event)[0]

        active_array = np.zeros(group.infection_event.size)

        # for each infection event, label all following infections as active
        for i in np.arange(infections.size):
            if i < infections.size - 1:
                active_array[infections[i] : infections[i+1]] = 1
            else:
                active_array[infections[i] : ] = 1


        return active_array
    def getColumns(self):
        """
        Returns columns to merge on and columns to keep from frame
        """
        merging = ['cohortid', 'date']
        to_keep = ['cohortid', 'date', 'enrolldate', 'burnin']
        if not self.by_infection_event:
            merging.append('h_popUID')
            to_keep.append('h_popUID')
        if self.is_bootstrap:
            to_keep.append('pseudo_cid')
        return merging, to_keep
    def LabelActiveInfections(self):
        """
        Label active baseline and new infections
        """
        self.labels['active_new_infection'] = np.concatenate(
            self.labels.groupby(['cohortid', 'h_popUID']).apply(
                lambda x : self.ActiveInfection(x).astype(bool)
                ).values
            )
        self.labels['active_baseline_infection'] = (~self.labels.active_new_infection)
    def LabelInfections(self):
        """
        Label timepoints as infection events
        &
        Label subsequence timepoints of infection events as active infections
        """
        self.CalculateSkips()

        merging, to_keep = self.getColumns()

        self.labels = self.skips.merge(
            self.frame[to_keep],
            left_on = merging,
            right_on = merging,
            how='inner'
            ).drop_duplicates()

        if self.labels.shape[0] == 0:
            merge_right_pseudo = merging.copy()
            merge_right_pseudo[0] = 'pseudo_cid'
            self.labels = self.skips.merge(
                self.frame[to_keep],
                left_on = merging,
                right_on = merge_right_pseudo,
                how='inner'
                ).drop_duplicates().\
                drop(
                    columns = 'cohortid_x'
                ).\
                rename(columns = {
                    'cohortid_y' : 'cohortid'
                })

        self.labels['infection_event'] = self.labels.apply(
            lambda x : self.getLabel(x),
            axis = 1
            )

        self.AggregateInfections()
        self.LabelActiveInfections()

        return self.labels
    def plot_haplodrop(self, cid_timeline, save=False, prefix=None):
        if cid_timeline.shape[0] == 0:
            return
        sns.heatmap(
            cid_timeline, square=True, linewidths=1,
            cbar=False, xticklabels=False, yticklabels=False,
            annot=True
            )
        if save:
            name = '../plots/cid_haplodrop/{}.png'.format(save)
            if prefix:
                name = '../plots/cid_haplodrop/{}.{}.png'.format(prefix, save)

            print('saving haplodrop : {}'.format(name))
            plt.savefig(name)
        else:
            plt.show()

        plt.close()

class FOI:
    def __init__(self, labels, meta, burnin=3):
        self.labels = labels
        self.meta = meta
        self.burnin = burnin

        self.frame = pd.DataFrame()
        self.is_bootstrap = False

        self.prepareData()
    def prepareBootstrap(self):
        """
        Merge frames on pseudo cid if column found in either
        """
        self.is_bootstrap = True

        # case where both have bootstrapped values
        if ('pseudo_cid' not in self.meta.columns):
            self.meta.pseudo_cid = self.meta.apply(
                lambda x : str(x.cohortid) + '_0' if str(x.pseudo_cid) == 'nan' else x.pseudo_cid,
                axis=1
            )
        if ('pseudo_cid' not in self.labels.columns):
            self.labels.pseudo_cid = self.meta.apply(
                lambda x : str(x.cohortid) + '_0' if str(x.pseudo_cid) == 'nan' else x.pseudo_cid,
                axis=1
            )

        self.frame = self.meta.merge(
            self.labels,
            left_on = ['pseudo_cid', 'date', 'enrolldate', 'burnin'],
            right_on = ['pseudo_cid', 'date', 'enrolldate', 'burnin'],
            how = 'left'
            )

        return
    def prepareData(self):
        """
        validate column types
        add burnin to meta
        merge labels with meta
        """
        self.labels.date = pd.to_datetime(self.labels.date)
        self.labels.enrolldate = pd.to_datetime(self.labels.enrolldate)
        self.labels.burnin = pd.to_datetime(self.labels.burnin)

        self.meta.date = pd.to_datetime(self.meta.date)
        self.meta.enrolldate = pd.to_datetime(self.meta.enrolldate)
        self.meta['year_month'] = pd.DatetimeIndex(self.meta.date).to_period('M')

        self.AddBurnin()
        if ('pseudo_cid' in self.meta.columns) | ('pseudo_cid' in self.labels.columns):
            self.prepareBootstrap()

        else:
            self.frame = self.meta.merge(
                self.labels,
                left_on = ['cohortid', 'date', 'enrolldate', 'burnin'],
                right_on = ['cohortid', 'date', 'enrolldate', 'burnin'],
                how = 'left'
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
            agg({'enrolldate' : 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + pd.DateOffset(months = self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on = ['cohortid', 'enrolldate'],
            right_on = ['cohortid', 'enrolldate']
            )
    def getDurations(self, group=None, working_frame=None):
        """
        return durations across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()

        working_frame = working_frame[working_frame.date >= working_frame.burnin]

        if group:
            durations = working_frame.\
                groupby(group).\
                apply(lambda x : self.getDurations(working_frame=x))

        else:
            durations = working_frame.date.max() - working_frame.date.min()
            durations = durations.days / 365.25

        return durations
    def getInfections(self, group=None, working_frame=None):
        """
        return number of infections across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()
        working_frame = working_frame[working_frame.date >= working_frame.burnin]

        if group:
            events = working_frame.\
                groupby(group).\
                apply(lambda x : self.getInfections(working_frame=x))
        else:
            events = working_frame.infection_event.sum()

        return events
    def getExposure(self, group=None, working_frame=None):
        """
        return number of exposed individuals across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()

        working_frame = working_frame[working_frame.date >= working_frame.burnin]
        exposure_group = 'cohortid' if not self.is_bootstrap else 'pseudo_cid'

        if group:
            exposure = working_frame.\
                groupby(group).\
                apply(lambda x : self.getExposure(working_frame=x))
        else:
            exposure = working_frame[exposure_group].unique().size

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
                columns = ['events', 'durations', 'exposure', 'FOI'],
                index = foi.index
                )

        else:
            foi = pd.DataFrame(
                np.array([events, durations, exposure, foi]).reshape(1,-1),
                columns = ['events', 'durations', 'exposure', 'FOI']
            )

        return foi.reset_index()

class BootstrapCID:
    """
    Perform bootstrapping on a dataframe by cohortid
    requires : `cohortid` as column
    """
    def __init__(self, dataframe, grouping='cohortid', seed=None):
        self.frame = dataframe.set_index(grouping)
        self.cid = self.frame.index.unique()
        self.grouping = grouping

        if seed:
            np.random.seed(seed)
    def sampleCID(self, size = 0, replace=True):
        """
        Random choice of COI in set
        """
        if size == 0:
            size = self.cid.size
        return np.random.choice(self.cid, size, replace=replace)
    def labelPseudo(self, bootstrap):
        """
        Label multiple occurences of a Cohortid with a pseudonym
        """
        group = ['cohortid', 'date']

        pseudo_label = bootstrap.groupby(group).apply(
            lambda x : np.arange(x.shape[0])
        )

        bootstrap.sort_values(group, inplace=True)

        bootstrap['pseudo_cid'] = bootstrap.cohortid.astype(str) + '_' + np.hstack(pseudo_label).astype(str)


        return bootstrap
    def getSample(self, size=0, pseudo_id=True):
        """
        Select cohortids found in sampling and return
        """

        bootstrap = self.frame.loc[self.sampleCID()].\
            reset_index()
        if pseudo_id:
            return self.labelPseudo(bootstrap)

        return bootstrap
    def getIter(self, num_iter=100):
        """
        Iterable generator for bootstraps
        """
        for _ in tqdm(np.arange(num_iter)):
            yield self.getSample()

class Survival:
    np.seterr(divide = 'ignore')
    def __init__(self, infections, meta, burnin=3, allowedSkips=3):
        self.infections = infections
        self.meta = meta
        self.burnin = burnin
        self.allowedSkips = allowedSkips
        self.minimum_duration = pd.Timedelta('15 days')


        self.date_bins = pd.DataFrame()
        self.ym_counts = pd.Series()
        self.cid_dates = pd.DataFrame()
        self.treatments = pd.DataFrame()


        self.ValidateInfections()
        self.ValidateMeta()
        self.ymCounts()
        self.MakeCohortidDates()
        self.MarkTreatments()
        self.original_infections = self.infections.copy()
    def ValidateInfections(self):
        """validate labeled infections for information required"""

        # convert infection date to date if not already
        self.infections.date = pd.to_datetime(self.infections.date)

        # convert burnin to date if not already
        self.infections.burnin = pd.to_datetime(self.infections.burnin)

        self.infections['year_month'] = pd.DatetimeIndex(self.infections.date).to_period('M')
    def ValidateMeta(self):
        """validate meta dataframe for information required"""

        self.meta.date = pd.to_datetime(self.meta.date)
        self.meta.enrolldate = pd.to_datetime(self.meta.enrolldate)
        self.meta['year_month'] = pd.DatetimeIndex(self.meta.date).to_period('M')
        self.meta['burnin'] = self.meta.enrolldate + pd.DateOffset(months = self.burnin)
    def ymCounts(self):
        """count number of people a year_month past burnin"""
        #self.meta[self.meta.date >= self.meta.burnin].\
        self.ym_counts = self.meta.\
            groupby('year_month').apply(lambda x : x['cohortid'].unique().size)
    def MakeCohortidDates(self):
        """create a series indexed by cid for all dates of that cid"""
        self.cid_dates = self.infections[['cohortid', 'date']].\
            drop_duplicates().\
            groupby('cohortid').\
            apply(lambda x : x.date.values)
    def MarkTreatments(self):
        """mark all malaria dates where treatment was given"""
        def treatments(x):
            return x[x.malariacat == 'Malaria'].date.unique()
        self.treatments = self.meta.groupby(['cohortid']).apply(lambda x : treatments(x))
    def MarkAges(self):
        """create a series indexed by cid for final age of that cid"""

        self.cid_ages = self.meta[['cohortid', 'date', 'ageyrs']].\
            drop_duplicates().\
            groupby(['cohortid']).\
            apply(lambda x : x.tail(1).ageyrs)
    def RemoveTreated(self):
        """
        Remove individuals who received treatment at any point in the study
        """
        self.treated_individuals = self.treatments[
            self.treatments.apply(lambda x : len(x) > 0)
            ].index
        self.original_infections = self.original_infections[
            ~self.original_infections.cohortid.isin(self.treated_individuals)
            ]
    def BootstrapInfections(self, frame=None):
        """randomly sample with replacement on CID"""
        if type(frame) == type(None):
            frame = self.original_infections.copy()

        c = frame.cohortid.unique().copy()
        rc = np.random.choice(c, c.size)
        self.infections = frame.copy()

        # calculate index size for each cohortid in random choice
        cid_size = np.array([np.where(self.infections.cohortid == i)[0].size for i in rc])

        # set index to cohortid
        self.infections = self.infections.set_index('cohortid')

        # generate bootstrap
        self.infections = self.infections.loc[rc]

        # create array of new cid_id with expected length
        new_cid = np.concatenate([np.full(cid_size[i], i) for i in range(cid_size.size)]).ravel()

        # have hashable id num to cid
        self.bootstrap_id_dates = pd.Series(rc)

        self.infections = self.infections.reset_index()
        self.infections['cohortid'] = new_cid
    def OldNewSurvival(self, bootstrap=False, n_iter=200):
        """plot proportion of haplotypes are old v new in population"""
        def cid_hid_cat_count(x):
            return x[['cohortid','h_popUID']].drop_duplicates().shape[0]
        def calculate_percentages(df):
            chc_counts = df.\
                groupby(['year_month', 'active_new_infection']).\
                apply(lambda x : cid_hid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})
            chc_counts['pc'] = chc_counts[['year_month','counts']].\
                groupby('year_month').\
                apply(lambda x : x / x.sum())

            chc_counts['active_new_infection'] = chc_counts.active_new_infection.apply(lambda x : 'new' if x else 'old')
            return chc_counts
        def plot_ons(odf, boots=pd.DataFrame()):
            odf['year_month'] = [i.to_timestamp() for i in odf.year_month.values]
            odf['year_month'] = pd.to_datetime(odf['year_month'])

            if not boots.empty:
                for v in odf.active_new_infection.unique():
                    sns.lineplot(data=odf[odf.active_new_infection == v],  x='year_month', y='pc', label=v, lw=4)
                    plt.fill_between(
                        boots[v].index.to_timestamp(),
                        [i for i,j in boots[v].values],
                        [j for i,j in boots[v].values],
                        alpha = 0.5)
            else:
                sns.lineplot(data=odf, x='year_month', y = 'pc', hue='active_new_infection')
            plt.xlabel('Date')
            plt.ylabel('Percentage')
            plt.title('Fraction of Old Clones In Infected Population')
            plt.savefig("../plots/survival/hid_survival.pdf")
            plt.show()
            plt.close()
        odf = calculate_percentages(self.original_infections)
        if bootstrap:
            boots = []
            for i in tqdm(range(n_iter), desc = 'bootstrapping'):
                self.BootstrapInfections()
                df = calculate_percentages(self.infections)
                boots.append(df)

            boots = pd.concat(boots)
            boots = boots.groupby(['active_new_infection', 'year_month']).apply(lambda x : np.percentile(x.pc, [2.5, 97.5]))
            plot_ons(odf, boots)
        else:
            plot_ons(odf)
    def CID_oldnewsurvival(self, bootstrap=False, n_iter=200):
        """plot proportion of people with old v new v mixed"""
        def cid_cat_count(x, mix=True):
            return (x['active_new_infection'] + 1).unique().sum()
        def date_cat_count(x):
            return x.cohortid.drop_duplicates().shape[0]
        def calculate_percentages(df):
            mix_counts = df.\
                groupby(['cohortid', 'year_month']).\
                apply(lambda x : cid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'cid_active_new_infection'})

            date_counts = mix_counts.groupby(['year_month', 'cid_active_new_infection']).\
                apply(lambda x : date_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})

            piv = date_counts.\
                pivot(index='year_month', columns='cid_active_new_infection', values='counts').\
                rename(columns = {1 : 'old', 2 : 'new', 3 : 'mix'}).\
                fillna(0)

            if piv.shape[1] == 3:
                piv['mix_old'] = piv.old + piv.mix
                piv['mix_new'] = piv.new + piv.mix
                piv = piv.drop(columns = 'mix')
            else:
                piv['mix_old'] = piv.old
                piv['mix_new'] = piv.new

            # convert old/new/mix values into fractions of datebin active population
            piv = piv.apply(
                lambda x : x.values / self.ym_counts[x.index],
                axis=0
                ).reset_index()

            df = pd.melt(piv, id_vars = 'year_month')
            df['mixed'] = df.cid_active_new_infection.apply(lambda x : 'mix' in x)

            return df
        def plot_cons(odf, boots = pd.DataFrame()):
            odf['year_month'] = [i.to_timestamp() for i in odf.year_month.values]
            odf['year_month'] = pd.to_datetime(odf['year_month'])
            if not boots.empty:
                lines = []
                colors = []
                for v in odf.cid_active_new_infection.unique():
                    ls = ':' if 'mix' in v else '-'
                    cl = 'teal' if 'old' in v else 'coral'
                    ax = sns.lineplot(
                        data=odf[odf.cid_active_new_infection == v],
                        x='year_month',
                        y='value',
                        label=v,
                        # color=cl,
                        lw=4)
                    plt.fill_between(
                        boots[v].index.to_timestamp(),
                        [i for i,j in boots[v].values],
                        [j for i,j in boots[v].values],
                        alpha = 0.3)
                    lines.append(ls)
                    colors.append(cl)
                [ax.lines[i].set_linestyle(lines[i]) for i in range(len(lines))]
                [ax.lines[i].set_color(colors[i]) for i in range(len(lines))]
            else:
                sns.lineplot(data=odf, x='year_month', y ='value', hue='cid_active_new_infection', style='mixed')


            plt.xlabel('Date')
            plt.ylabel('Percentage')
            plt.title('Fraction of New and Old Clones by Individual')
            plt.savefig("../plots/survival/CID_survival.pdf")
            plt.show()
            plt.close()

        odf = calculate_percentages(self.original_infections)
        if bootstrap :
            boots = []
            for i in tqdm(range(n_iter), desc = 'bootstrapping'):
                self.BootstrapInfections()
                df = calculate_percentages(self.infections)
                boots.append(df)
            boots = pd.concat(boots)
            boots = boots.groupby(['cid_active_new_infection', 'year_month']).apply(lambda x : np.percentile(x.value, [2.5, 97.5]))
            plot_cons(odf, boots)
        else:
            plot_cons(odf)
    def OldWaning(self, bootstrap=False, n_iter=200):
        """calculate fraction of old clones remaining across each month past the burnin (use october meta)"""
        def monthly_kept(x):
            """return number of old infections past burnin"""
            monthly_old = x[(~x.active_new_infection) & (x.date >= x.burnin)]
            return monthly_old[['cohortid', 'h_popUID']].drop_duplicates().shape[0]
        def waning(df):
            monthly_counts = df.groupby('year_month').apply(lambda x : monthly_kept(x))
            return monthly_counts / monthly_counts.values.max()
        def plot_wane(omc, boots=pd.DataFrame()):
            # omc['year_month'] = [i.to_timestamp() for i in omc.year_month.values]
            # omc['year_month'] = pd.to_datetime(omc['year_month'])
            g = sns.lineplot(x = omc.index.to_timestamp(), y = omc.values, legend=False, lw=5)
            if not boots.empty:
                plt.fill_between(
                    boots.index.to_timestamp(),
                    y1=[i for i,j in boots.values],
                    y2=[j for i,j in boots.values],
                    alpha=0.3
                    )
            plt.xlabel('Date')
            plt.title('Percentage')
            plt.title('Fraction of Old Clones Remaining')
            g.get_figure().savefig("../plots/survival/oldClones.pdf")
            plt.show()
            plt.close()

        omc = waning(self.original_infections)
        if bootstrap:
            boots = []
            for i in tqdm(range(n_iter), desc='bootstrapping'):
                self.BootstrapInfections()
                mc = waning(self.infections)
                boots.append(mc)
            boots = pd.concat(boots)
            boots = boots.groupby(level = 0).apply(lambda x : np.percentile(x, [2.5, 97.5]))
            plot_wane(omc, boots)
        else:
            plot_wane(omc)


def dev_infectionLabeler():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta,
        by_infection_event=False, qpcr_threshold=0.1,
        burnin=2, haplodrops=False)
    labels = il.LabelInfections()

def dev_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)
    # labels = InfectionLabeler(sdo, meta)


    # labels = pd.read_csv('temp/labels.tab', sep="\t")
    foi = FOI(labels, meta, burnin=2)

    full = foi.fit(group = ['year_month'])
    labels.infection_event
    labels[labels.date <= pd.to_datetime('2019-04-01')].infection_event.sum()

def dev_BootstrapCID():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)
    labels = pd.read_csv('temp/labels.tab', sep="\t")

    bl = BootstrapCID(meta, seed=42)

    total = []
    for b_meta in bl.getIter(10):
        b_labels = InfectionLabeler(sdo, b_meta).LabelInfections()
        foi = FOI(b_labels, b_meta)
        full = foi.fit(group=None)
        total.append(full)
    bs_foi = pd.concat(total)


    true_foi = FOI(labels, meta)
    true_fit = true_foi.fit()

    sns.distplot(bs_foi.FOI)
    plt.axvline(true_fit.FOI[0])

def dev_Survival():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta,
        by_infection_event=False, qpcr_threshold=0.1,
        burnin=2, haplodrops=False)
    labels = il.LabelInfections()

    s = Survival(labels, meta, burnin=2)
    # s.OldNewSurvival(bootstrap=True, n_iter=100)
    # s.CID_oldnewsurvival(bootstrap=True, n_iter=100)
    # s.OldWaning(bootstrap=True)

def worker_foi(sdo, meta, group):
    labels = InfectionLabeler(sdo, meta, by_infection_event=True, impute_missing=True).LabelInfections()
    foi = FOI(labels, meta)
    full = foi.fit(group=group)
    return full
def multiprocess_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    bl = BootstrapCID(meta, seed=42)

    p = Pool(processes=7)

    group = ['gender']
    bootstrapped_meta = [[sdo, bl.getSample(), group] for _ in tqdm(range(100))]
    results = p.starmap(worker_foi, bootstrapped_meta)

    bootstrapped_foi = pd.concat(results)

    labels = InfectionLabeler(sdo, meta, by_infection_event=True, impute_missing=True).LabelInfections()
    foi = FOI(labels, meta)
    true_foi = foi.fit(group=group)

    for g, sub in bootstrapped_foi.groupby(group):
        sns.distplot(sub.FOI.values, label=g)
    [plt.axvline(val) for val in true_foi.FOI.values]
    plt.legend()


if __name__ == '__main__':
    # dev_infectionLabeler()
    dev_Survival()
    # dev_FOI()
    # dev_BootstrapLabels()
    # multiprocess_FOI()
    pass

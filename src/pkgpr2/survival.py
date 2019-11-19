#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm
from multiprocess import Pool

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



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

    def plot(self, save=None):
        self._plot()

        plt.xlabel('Date')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if not save:
            plt.show()
            plt.close()
        else:
            plt.savefig(save)
            plt.close()


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

    def _plot(self):

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

        plt.title('Fraction of New Clones In Infected Population')


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

    def _plot(self):
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

        plt.title('Fraction of New and Old Clones by Individual')


class OldWaning(Survival):
    """
    calculate fraction of old clones remaining across each month past
    the burnin period
    """

    def __init__(self, *args, **kwargs):
        Survival.__init__(self, *args, **kwargs)

    def CalculateWaning(self, frame):
        """
        calculate percentage of old clones remaining at each year month
        """

        # select only old infections
        old_infections = frame[frame.active_baseline_infection]

        # find maximum year~month for infections
        infection_maximums = old_infections.\
            groupby(['cohortid', 'burnin', 'h_popUID']).\
            apply(lambda x : x.year_month.max()).\
            reset_index().\
            rename(columns={0: 'max_ym'})

        # select those past burnin
        infection_maximums = infection_maximums[
            infection_maximums.max_ym > infection_maximums.burnin
            ]

        # take unique year months
        year_months = np.sort(infection_maximums.max_ym.unique())

        # number of infections
        im_size = infection_maximums.shape[0]


        # compile data into dataframe
        frame = []
        for ym in year_months:
            num_infections = np.where(
                infection_maximums.max_ym >= ym
                )[0].size
            frame.append({
                'year_month': ym,
                'pc': num_infections / im_size
                })

        monthly_pc = pd.DataFrame(frame)

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

    def _plot(self):
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

        plt.title('Fraction of Old Clones Remaining')

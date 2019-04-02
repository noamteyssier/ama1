#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, warnings, time
from tqdm import tqdm

from scipy.optimize import minimize

# np.random.seed(42)
sns.set(rc={'figure.figsize':(15, 12), 'lines.linewidth': 2})
sns.set_style('ticks')

class Survival:
    pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def __init__(self, sdo, meta, burnin='2018-01-01', allowedSkips=3, fail_flag=True, qpcr_threshold = 0.1):
        self.sdo = sdo
        self.meta = meta
        self.burnin = pd.to_datetime(burnin)
        self.allowedSkips = allowedSkips
        self.fail_flag = fail_flag
        self.qpcr_threshold = qpcr_threshold
        self.minimum_duration = pd.Timedelta('15 days')

        self.pr2 = pd.DataFrame()
        self.timelines = pd.DataFrame()
        self.cid_dates = pd.Series()
        self.treatments = pd.Series()
        self.infections = pd.DataFrame()
        self.mass_reindex = []
        self.date_bins = np.array([])
        self.column_bins = np.array([])

        self.__prepare_df__()
        self.__timeline__()
        self.__cid_dates__()
        self.__mark_treatments__()
        self.__label_new_infections__()
        self.__bin_dates__()

        self.original_infections = self.infections.copy()
    def __prepare_df__(self):
        """prepare dataframe for timeline creation"""
        self.__prepare_sdo__()
        self.__prepare_meta__()

        # filter qpcr dates
        if self.qpcr_threshold > 0:
            self.meta = self.meta[(self.meta.qpcr == 0) | (self.meta.qpcr >= self.qpcr_threshold)]

        # merge meta and sdo
        self.pr2 = self.meta.merge(self.sdo, how='left')

        # filter all positive qpcr dates with failed sequencing
        if self.fail_flag:
            self.pr2 = self.pr2[~((self.pr2.qpcr > 0) & np.isnan(self.pr2.c_AveragedFrac))]

        self.pr2.date = pd.to_datetime(self.pr2.date)
    def __prepare_meta__(self):
        """prepare meta data for usage in timeline generation"""
        self.agecat_rename = {
            '< 5 years'  : 1,
            '5-15 years' : 2,
            '16 years or older' : 3}
        self.meta = self.meta[['date', 'cohortid', 'qpcr', 'agecat', 'malariacat']]
        self.meta['date'] = self.meta['date'].astype('str')
        self.meta['cohortid'] = self.meta['cohortid'].astype('int')
        self.meta['agecat'] = self.meta.agecat.apply(lambda x : self.agecat_rename[x])
        self.meta.sort_values(by='date', inplace=True)
        self.meta = self.meta[~self.meta.qpcr.isna()]
        self.cid_count = self.meta['cohortid'].unique().size
    def __prepare_sdo__(self, controls=False):
        """prepare seekdeep output dataframe for internal usage"""
        # keep only patient samples and normalize dataframe
        if controls == False:
            self.sdo = self.sdo[~self.sdo.s_Sample.str.contains('ctrl|neg')]
        else:
            self.sdo = self.sdo

        # split cid and date
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x),
            axis = 1, result_type = 'expand')

        self.sdo['cohortid'] = self.sdo.cohortid.astype('int')

        # select columns of interest
        self.sdo = self.sdo[['cohortid', 'date', 'h_popUID', 'c_AveragedFrac']]
    def __split_cid_date__(self, row):
        """convert s_Sample to date and cohortid"""
        a = row.s_Sample.split('-')
        date, cid = '-'.join(a[:3]), a[-1]
        return [date, cid]
    def __timeline__(self):
        """generate timelines for each cohortid"""
        self.timelines = self.pr2.pivot_table(
            values = 'c_AveragedFrac',
            index=['cohortid', 'h_popUID'],
            columns='date', dropna=False).\
            dropna(axis=0, how='all')
        self.timelines = (self.timelines > 0).astype('int')
    def __cid_dates__(self):
        """create a series indexed by cid for all dates of that cid"""
        self.cid_dates = self.pr2[['cohortid', 'date']].\
            drop_duplicates().\
            groupby('cohortid').\
            apply(lambda x : x.date.values)
    def __bin_dates__(self):
        """bin columns (dates) into 12 evenly spaced windows based on first and last visit"""
        dates = pd.to_datetime(self.infections.date.unique())
        self.date_bins = pd.date_range(dates.min(), dates.max(), periods=13)

        dfs = []
        for i in range(1, self.date_bins.size):
            a = np.where((dates < self.date_bins[i]) & (dates >= self.date_bins[i-1]))[0]
            p = pd.DataFrame({'date_bin' : self.date_bins[i], 'date' : dates[a]})
            dfs.append(p)

        self.date_bins = pd.concat(dfs)
        self.infections = self.infections.\
            merge(self.date_bins)
    def __mark_treatments__(self):
        """mark all malaria dates where treatment was given"""
        def treatments(x):
            return x[x.malariacat == 'Malaria'].date.unique()
        self.treatments = self.pr2.groupby(['cohortid']).apply(lambda x : treatments(x))
    def __get_skips__(self, x):
        """return skips for timeline row"""
        # find all infection events
        i_event = x.values.nonzero()[1]

        # find distances between infection events
        vals = np.diff(i_event)

        # subtract one from distances to reflect true skip size
        return vals - 1
    def __infection_duration__(self, x, skips):
        """split infections into single or multiple dependant on skips"""
        i_event = x.values.nonzero()[1]
        if np.all(skips <= self.allowedSkips):
            # one continuous infection
            return np.array([i_event, np.nan])
        else:
            # reinfection occurs under allowed skips
            return np.split(i_event, np.where(skips > self.allowedSkips)[0] + 1)
    def __timeline_marker__(self, timeline, ifx):
        """
        given an infection timeline,
        label old or new if first infection is
        before or after burnin
        1 if old
        2 if new
        """
        t = pd.to_datetime(timeline.columns[ifx[0]])
        ifx = range(ifx.min(), ifx.max() + 1)
        if t <= self.burnin:
            timeline.iloc[:,ifx] = 1
        else:
            timeline.iloc[:,ifx] = 2

        return timeline
    def __type_labeller__(self, x):
        """
        calculate skips
        split infection timelines if reinfection (based on skips)
        label infection timelines as old or new
        return timeline
        """
        cid, hid = x.name

        timeline = x.loc[:,self.cid_dates[cid]]

        skips = self.__get_skips__(timeline)
        ifx = self.__infection_duration__(timeline, skips)

        return ifx
    def __mark_timeline__(self, x):
        cid, hid, i = x.name
        vals = x.values[0][0]
        ifx = np.arange(vals.min(), vals.max() + 1)
        dates = self.cid_dates[cid][ifx]
        if pd.to_datetime(dates[0]) <= self.burnin:
            i_state = 1
        else:
            i_state = 2

        self.mass_reindex.append([cid, hid, i_state, dates])
    def __label_new_infections__(self):
        """create old and new infection timelines"""
        # find infection times
        self.infections = self.timelines.\
            groupby(level=[0,1], sort=False).\
            apply(lambda x : self.__type_labeller__(x)).\
            apply(pd.Series).\
            stack().\
            to_frame('ifx')
        self.infections.index.names = ['cohortid', 'hid', 'ifx_num']

        # split multiple infections to separate observations
        self.infections.groupby(level=[0,1,2], sort=False).\
            apply(lambda x : self.__mark_timeline__(x))

        # split infection dates to separate observations
        self.infections = pd.DataFrame(self.mass_reindex)
        self.infections.columns = ['cohortid', 'hid', 'val', 'date']
        wide_form = self.infections.date.\
            apply(pd.Series).\
            merge(self.infections.drop(columns='date'), left_index=True, right_index=True)

        self.infections = pd.melt(
                wide_form,
                id_vars = ['cohortid', 'hid', 'val']).\
            drop(columns = 'variable').\
            fillna('nope')

        # drop NAs from melted dataframe
        self.infections = self.infections[self.infections.value != 'nope']

        # rename columns
        self.infections.columns = ['cohortid', 'hid', 'val', 'date']

        # convert date column to datetime
        self.infections.date = pd.to_datetime(self.infections.date)
    def __bootstrap_cid__(self):
        """randomly sample with replacement on CID"""
        c = self.original_infections.cohortid.unique().copy()
        rc = np.random.choice(c, c.size)
        self.infections = self.original_infections.copy()

        # calculate index size for each cohortid in random choice
        cid_size = np.array([np.where(self.infections.cohortid == i)[0].size for i in rc])

        # set index to cohortid
        self.infections = self.infections.set_index('cohortid')

        # generate bootstrap
        self.infections = self.infections.loc[rc]

        # create array of new cid_id with expected length
        new_cid = np.concatenate([np.full(cid_size[i], i) for i in range(cid_size.size)]).ravel()

        self.infections = self.infections.reset_index()
        self.infections['cohortid'] = new_cid
    def OldNewSurvival(self, bootstrap=False):
        """plot proportion of haplotypes are old v new in population"""
        def cid_hid_cat_count(x):
            return x[['cohortid','hid']].drop_duplicates().shape[0]
        def calculate_percentages(df):
            chc_counts = df.\
                groupby(['date_bin', 'val']).\
                apply(lambda x : cid_hid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})
            chc_counts['pc'] = chc_counts[['date_bin','counts']].\
                groupby('date_bin').\
                apply(lambda x : x / x.sum())
            chc_counts['val'] = chc_counts.val.apply(lambda x : 'old' if x==1 else 'new')
            return chc_counts
        def plot_ons(odf, boots=pd.DataFrame()):
            if not boots.empty:
                for v in odf.val.unique():
                    sns.lineplot(data=odf[odf.val == v],  x='date_bin', y='pc', label=v, lw=4)
                    plt.fill_between(
                        boots[v].index,
                        [i for i,j in boots[v].values],
                        [j for i,j in boots[v].values],
                        alpha = 0.5)
            else:
                sns.lineplot(data=odf, x='date_bin', y = 'pc', hue='val')
            plt.xlabel('Date')
            plt.ylabel('Percentage')
            plt.title('Fraction of Old Clones In Infected Population')
            plt.savefig("../plots/survival/hid_survival.pdf")
            plt.show()
            plt.close()
        odf = calculate_percentages(self.original_infections)
        if bootstrap:
            boots = []
            for i in tqdm(range(200), desc = 'bootstrapping'):
                self.__bootstrap_cid__()
                df = calculate_percentages(self.infections)
                boots.append(df)

            boots = pd.concat(boots)
            boots = boots.groupby(['val', 'date_bin']).apply(lambda x : np.percentile(x.pc, [2.5, 97.5]))
            plot_ons(odf, boots)
        else:
            plot_ons(odf)
    def CID_oldnewsurvival(self, bootstrap=False):
        """plot proportion of people with old v new v mixed"""
        def cid_cat_count(x, mix=True):
            return x['val'].unique().sum()
        def date_cat_count(x):
            return x.cohortid.drop_duplicates().shape[0]
        def calculate_percentages(df):
            mix_counts = df.\
                groupby(['cohortid', 'date_bin']).\
                apply(lambda x : cid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'c_val'})
            date_counts = mix_counts.groupby(['date_bin', 'c_val']).\
                apply(lambda x : date_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})
            piv = date_counts.\
                pivot(index='date_bin', columns='c_val', values='counts').\
                rename(columns = {1 : 'old', 2 : 'new', 3 : 'mix'}).\
                fillna(0)
            if piv.shape[1] == 3:
                piv['mix_old'] = piv.old + piv.mix
                piv['mix_new'] = piv.new + piv.mix
                piv = piv.drop(columns = 'mix')
            else:
                piv['mix_old'] = piv.old
                piv['mix_new'] = piv.new

            date_sums = piv[['mix_old', 'mix_new']].sum(axis = 1)
            piv = piv.\
                apply(lambda x : x / self.cid_count, axis = 0).\
                reset_index()

            df = pd.melt(piv, id_vars = 'date_bin')
            df['mixed'] = df.c_val.apply(lambda x : 'mix' in x)

            return df
        def plot_cons(odf, boots = pd.DataFrame()):
            if not boots.empty:
                lines = []
                colors = []
                for v in odf.c_val.unique():
                    ls = ':' if 'mix' in v else '-'
                    cl = 'teal' if 'old' in v else 'coral'
                    ax = sns.lineplot(
                        data=odf[odf.c_val == v],
                        x='date_bin',
                        y='value',
                        label=v,
                        # color=cl,
                        lw=4)
                    plt.fill_between(
                        boots[v].index,
                        [i for i,j in boots[v].values],
                        [j for i,j in boots[v].values],
                        alpha = 0.3)
                    lines.append(ls)
                    colors.append(cl)
                [ax.lines[i].set_linestyle(lines[i]) for i in range(len(lines))]
                [ax.lines[i].set_color(colors[i]) for i in range(len(lines))]
            else:
                sns.lineplot(data=odf, x='date_bin', y ='value', hue='c_val', style='mixed')


            plt.xlabel('Date')
            plt.ylabel('Percentage')
            plt.title('Fraction of New and Old Clones by Individual')
            plt.savefig("../plots/survival/CID_survival.pdf")
            plt.show()
            plt.close()

        odf = calculate_percentages(self.original_infections)
        if bootstrap :
            boots = []
            for i in tqdm(range(200), desc = 'bootstrapping'):
                self.__bootstrap_cid__()
                df = calculate_percentages(self.infections)
                boots.append(df)
            boots = pd.concat(boots)
            boots = boots.groupby(['c_val', 'date_bin']).apply(lambda x : np.percentile(x.value, [2.5, 97.5]))
            plot_cons(odf, boots)
        else:
            plot_cons(odf)
    def OldWaning(self, bootstrap=False):
        """calculate fraction of old clones remaining across each month past the burnin"""
        def monthly_kept(x):
            # sys.exit(x)
            return x[x.val == 1][['cohortid', 'hid']].drop_duplicates().shape[0]
        def waning(df):
            monthly_counts = df.groupby('date_bin').apply(lambda x : monthly_kept(x))
            oc_idx = np.where(
                (monthly_counts.index.year == self.burnin.year) &
                (monthly_counts.index.month == self.burnin.month))[0]
            oc = monthly_counts[oc_idx].values
            monthly_counts = monthly_counts/oc
            return monthly_counts[monthly_counts.index > self.burnin]
        def plot_wane(omc, boots=pd.DataFrame()):
            g = sns.lineplot(x = omc.index, y = omc.values, legend=False, lw=5)
            if not boots.empty:
                plt.fill_between(
                    boots.index,
                    y1=[i for i,j in boots.values],
                    y2=[j for i,j in boots.values],
                    alpha=0.3)
            plt.xlabel('Date')
            plt.title('Percentage')
            plt.title('Fraction of Old Clones Remaining')
            g.get_figure().savefig("../plots/survival/oldClones.pdf")
            plt.show()
            plt.close()

        omc = waning(self.original_infections)
        if bootstrap:
            boots = []
            for i in tqdm(range(200), desc='bootstrapping'):
                self.__bootstrap_cid__()
                mc = waning(self.infections)
                boots.append(mc)
            boots = pd.concat(boots)
            boots = boots.groupby(level = 0).apply(lambda x : np.percentile(x, [2.5, 97.5]))
            plot_wane(omc, boots)
        else:
            plot_wane(omc)
    def Durations(self):
        """estimate durations using exponential model"""
        def inf_durations(x):
            cid = x.cohortid.unique()[0]
            burnout = pd.to_datetime(self.cid_dates[cid][-self.allowedSkips:]).min()

            s_obs = ~np.any(x.date <= self.burnin)
            e_obs = ~np.any(x.date >= burnout)


            t_start = x.date.min() - self.minimum_duration if s_obs else self.study_start
            t_end = x.date.max() + self.minimum_duration if e_obs else self.study_end

            t = t_end - t_start

            return t
        def conditional_inf_durations(x):
            """only add minimum_duration if not treated and set hard burnin"""
            cid = x.cohortid.unique()[0]
            burnout = pd.to_datetime(self.cid_dates[cid][-self.allowedSkips:]).min()
            treatment = False


            if x.date.max() <= self.burnin:
                return np.nan
            if x.date.max().to_datetime64() in self.treatments[cid]:
                treatment = True

            s_obs = ~np.any(x.date <= self.burnin)
            e_obs = ~np.any(x.date >= burnout)


            t_start = x.date.min() - self.minimum_duration if s_obs else self.burnin
            t_end = x.date.max() if e_obs else self.study_end
            t_end = t_end + self.minimum_duration if not treatment else t_end

            t = t_end - t_start

            return t
        def exp_likelihood(l, t):
            lik = l * np.exp(-l * t)
            log_lik = np.log(lik).sum()
            return -1 * log_lik



        self.study_start = self.infections.date.min()
        self.study_end = self.infections.date.max()
        t = self.infections.groupby(['cohortid', 'hid']).apply(lambda x : conditional_inf_durations(x)).dt.days.values
        # t = self.infections.groupby(['cohortid', 'hid']).apply(lambda x : inf_durations(x)).dt.days.values
        t = t[~np.isnan(t)]
        l = np.random.random()
        m = minimize(exp_likelihood, l, t, method='SLSQP', bounds=((0, None), ))
        estimated_lambda = m.x
        expected_duration = 1/estimated_lambda
        print("Calculated Expected Duration : {0} days".format(expected_duration[0]))

def main():
    sdo_fn = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta_fn = "../prism2/stata/allVisits.dta"

    sdo = pd.read_csv(sdo_fn, sep='\t')
    meta = pd.read_stata(meta_fn)

    # calculate Expected and Observed for skip vals in range
    s = Survival(sdo, meta, fail_flag=False, qpcr_threshold=0)
    s.OldNewSurvival(bootstrap=True)
    s.CID_oldnewsurvival(bootstrap=True)
    s.OldWaning(bootstrap=True)
    s.Durations()

if __name__ == '__main__':
    main()

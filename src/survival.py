#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, warnings, time, argparse
from seekdeep_modules import SeekDeepUtils
from tqdm import tqdm

from scipy.optimize import minimize
from scipy.special import expit

# np.random.seed(43)
sns.set(rc={'figure.figsize':(30, 30), 'lines.linewidth': 2})
sns.set_style('ticks')

class Survival:
    def __init__(self, infections, meta, n_datebins=18, burnin=3, allowedSkips=3):
        self.infections = infections
        self.meta = meta
        self.n_datebins = n_datebins
        self.burnin = burnin
        self.allowedSkips = allowedSkips
        self.minimum_duration = pd.Timedelta('15 days')


        self.date_bins = pd.DataFrame()
        self.bin_counts = pd.DataFrame()
        self.cid_dates = pd.DataFrame()
        self.treatments = pd.DataFrame()


        self.ValidateInfections()
        self.ValidateMeta()
        self.MakeDateBins()
        self.MakeCohortidDates()
        self.MarkTreatments()
        self.MarkAges()

        self.original_infections = self.infections.copy()
    def ValidateInfections(self):
        """validate labeled infections for information required"""

        # convert infection date to date if not already
        self.infections.date = pd.to_datetime(self.infections.date)

        # convert burnin to date if not already
        self.infections.burnin = pd.to_datetime(self.infections.burnin)
    def ValidateMeta(self):
        """validate meta dataframe for information required"""

        self.meta.date = pd.to_datetime(self.meta.date)
        self.meta.enrolldate = pd.to_datetime(self.meta.enrolldate)

        # add burnin from enrolldate if not already supplied
        self.meta['burnin'] = self.meta.apply(
            lambda x : x['enrolldate'] + pd.DateOffset(months = self.burnin),
            axis=1
            )
    def MakeDateBins(self):
        """bin columns (dates) into evenly spaced windows based on first and last visit"""
        dates = self.meta.date.unique()
        self.date_bins = pd.date_range(
            dates.min(),
            dates.max(),
            periods=self.n_datebins
        )

        date_bins = []
        for d in dates:
            try:
                assigned_bin = self.date_bins[np.where(self.date_bins <= d)[0].max()]
                date_bins.append(assigned_bin)
            except ValueError:
                # condition where date is missing (pd.NaT not working...)
                date_bins.append(0)
                continue

        self.date_bins = pd.DataFrame({'date_bin' : date_bins, 'date' : dates})
        self.date_bins = self.date_bins[self.date_bins.date_bin != 0]

        # apply date bins to infection and meta dataframes
        self.infections = self.infections.\
            merge(self.date_bins, how='left')

        self.meta = self.meta.\
            merge(self.date_bins, how='left')

        self.DateBinCounts()
    def DateBinCounts(self):
        """count number of people within a datebin"""
        cid_enrolldates = self.meta[['cohortid', 'enrolldate']].drop_duplicates()
        bins = self.date_bins.date_bin.unique()

        bin_counts = []
        for i, bin in enumerate(bins[:-1]):
            bin_counts.append(
                cid_enrolldates[cid_enrolldates.enrolldate <= bins[i+1]].cohortid.count()
            )
        # bin_counts.append(cid_enrolldates[cid_enrolldates.enrolldate <= bins[i+1]].cohortid.count())

        self.bin_counts = pd.DataFrame({
            'date_bin' : bins[:-1],
            'bin_counts' : bin_counts
        })

        self.date_bins = self.date_bins.merge(self.bin_counts, how='left')
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
                groupby(['date_bin', 'true_new']).\
                apply(lambda x : cid_hid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})
            chc_counts['pc'] = chc_counts[['date_bin','counts']].\
                groupby('date_bin').\
                apply(lambda x : x / x.sum())
            chc_counts['true_new'] = chc_counts.true_new.apply(lambda x : 'new' if x else 'old')
            return chc_counts
        def plot_ons(odf, boots=pd.DataFrame()):
            if not boots.empty:
                for v in odf.true_new.unique():
                    sns.lineplot(data=odf[odf.true_new == v],  x='date_bin', y='pc', label=v, lw=4)
                    plt.fill_between(
                        boots[v].index,
                        [i for i,j in boots[v].values],
                        [j for i,j in boots[v].values],
                        alpha = 0.5)
            else:
                sns.lineplot(data=odf, x='date_bin', y = 'pc', hue='true_new')
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
            boots = boots.groupby(['true_new', 'date_bin']).apply(lambda x : np.percentile(x.pc, [2.5, 97.5]))
            plot_ons(odf, boots)
        else:
            plot_ons(odf)
    def CID_oldnewsurvival(self, bootstrap=False, n_iter=200):
        """plot proportion of people with old v new v mixed"""
        def cid_cat_count(x, mix=True):
            return (x['true_new'] + 1).unique().sum()
        def date_cat_count(x):
            return x.cohortid.drop_duplicates().shape[0]
        def calculate_percentages(df):
            mix_counts = df.\
                groupby(['cohortid', 'date_bin']).\
                apply(lambda x : cid_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'cid_true_new'})
            date_counts = mix_counts.groupby(['date_bin', 'cid_true_new']).\
                apply(lambda x : date_cat_count(x)).\
                reset_index().\
                rename(columns = {0 : 'counts'})

            piv = date_counts.\
                pivot(index='date_bin', columns='cid_true_new', values='counts').\
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
                lambda x : x.values / self.bin_counts.bin_counts.values,
                axis=0
                ).reset_index()

            df = pd.melt(piv, id_vars = 'date_bin')
            df['mixed'] = df.cid_true_new.apply(lambda x : 'mix' in x)

            return df
        def plot_cons(odf, boots = pd.DataFrame()):
            if not boots.empty:
                lines = []
                colors = []
                for v in odf.cid_true_new.unique():
                    ls = ':' if 'mix' in v else '-'
                    cl = 'teal' if 'old' in v else 'coral'
                    ax = sns.lineplot(
                        data=odf[odf.cid_true_new == v],
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
                sns.lineplot(data=odf, x='date_bin', y ='value', hue='cid_true_new', style='mixed')


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
            boots = boots.groupby(['cid_true_new', 'date_bin']).apply(lambda x : np.percentile(x.value, [2.5, 97.5]))
            plot_cons(odf, boots)
        else:
            plot_cons(odf)
    def OldWaning(self, bootstrap=False, n_iter=200):
        """calculate fraction of old clones remaining across each month past the burnin (use october meta)"""
        def monthly_kept(x):
            """return number of old infections past burnin"""
            monthly_old = x[(~x.true_new) & (x.date >= x.burnin)]
            return monthly_old[['cohortid', 'h_popUID']].drop_duplicates().shape[0]
        def waning(df):
            monthly_counts = df.groupby('date_bin').apply(lambda x : monthly_kept(x))
            return monthly_counts / monthly_counts.values.max()
        def plot_wane(omc, boots=pd.DataFrame()):
            g = sns.lineplot(x = omc.index, y = omc.values, legend=False, lw=5)
            if not boots.empty:
                plt.fill_between(
                    boots.index,
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
    def Durations(self, bootstrap=False, n_iter=200, removeTreatment=False):
        """estimate durations using exponential model"""
        self.in_boot = False
        def inf_durations(x):
            """only add minimum_duration if not treated and set hard burnin"""
            cid = x.cohortid.unique()[0]
            if self.in_boot:
                cid = self.bootstrap_id_dates[cid]


            burnout = pd.to_datetime(self.cid_dates[cid][-self.allowedSkips:]).min()
            treatment = False

            if x.date.max() <= x.burnin.max():
                return np.nan
            if x.date.max().to_datetime64() in self.treatments[cid]:
                treatment = True

            s_obs = ~np.any(x.date <= x.burnin)
            e_obs = ~np.any(x.date >= burnout)


            t_start = x.date.min() - self.minimum_duration if s_obs else x.burnin.min()
            t_end = x.date.max() if e_obs else self.study_end
            t_end = t_end + self.minimum_duration if not treatment else t_end


            t = t_end - t_start

            return t
        def exp_likelihood(l, t):
            lik = l * np.exp(-l * t)
            log_lik = np.log(lik).sum()
            return -1 * log_lik
        def fit_model(df):
            self.study_start = df.date.min()
            self.study_end = df.date.max()
            t = df.\
                groupby(['cohortid', 'h_popUID', 'true_new']).\
                apply(lambda x : inf_durations(x)).\
                dt.days.values
            t = t[~np.isnan(t)]
            while True:
                lam = np.random.random()
                m = minimize(exp_likelihood, lam, t, method='TNC', bounds=((0, 1), ))
                if m.success:
                    break
            return m.x
        def plot_boots(boots, lam):
            sns.distplot(boots, color='teal')
            plt.axvline(lam, linestyle=':', lw=5, color='teal')
            plt.xlabel("Calculated Lambda")
            plt.title("Calculated Lambda and Distribution of Bootstrapped Lambdas")
            plt.savefig("../plots/survival/exponentialSurvival.pdf")
            plt.show()
            plt.close()

        if removeTreatment:
            self.RemoveTreated()

        lam = fit_model(self.original_infections)
        if bootstrap:
            self.in_boot=True
            boots = []
            for i in tqdm(range(n_iter), desc='bootstrapping'):
                self.BootstrapInfections()
                boot_lam = fit_model(self.infections)
                boots.append(boot_lam)
            boots = np.array(boots)
            lower, upper = np.percentile(boots, [2.5, 97.5])
            print("Fit Lambda : {0} ({1}, {2})".format(lam, lower, upper))
            print("Expected Duration : {0} ({1}, {2})".format(1/lam, 1/upper, 1/lower))
            plot_boots(boots, lam)
        else:
            print("Fit Lambda : {}".format(lam))
            print("Expected Duration : {0} days".format(1 / lam))
    def Duration_Age(self, bootstrap=False, n_iter=200, removeTreatment=False):
        """fit exponential model as a function of age"""
        self.in_boot=False
        def inf_durations(x):
            """only add minimum_duration if not treated and set hard burnin"""
            cid = x.cohortid.unique()[0]
            if self.in_boot:
                cid = self.bootstrap_id_dates[cid]


            burnout = pd.to_datetime(self.cid_dates[cid][-self.allowedSkips:]).min()
            treatment = False

            if x.date.max() <= x.burnin.max():
                return np.nan
            if x.date.max().to_datetime64() in self.treatments[cid]:
                treatment = True

            s_obs = ~np.any(x.date <= x.burnin)
            e_obs = ~np.any(x.date >= burnout)


            t_start = x.date.min() - self.minimum_duration if s_obs else x.burnin.min()
            t_end = x.date.max() if e_obs else self.study_end
            t_end = t_end + self.minimum_duration if not treatment else t_end


            t = t_end - t_start

            return t
        def get_age(x):
            """get final age of cohortid"""
            cid = x.cohortid.unique()[0]
            if self.in_boot:
                cid = self.bootstrap_id_dates[cid]
            return self.cid_ages[cid]
        def exp_likelihood_age(lam, vectors):
            """estimate likelihood as a function of age"""
            durations, ages = vectors
            log_lambda = lam[0] + (lam[1] * ages)
            lambdas = np.exp(log_lambda)
            log_lik = log_lambda - (lambdas * durations)
            return -1 * log_lik.sum()
        def fit_model(df):
            """fit model as a a function of age"""
            self.study_start = df.date.min()
            self.study_end = df.date.max()
            t = df.\
                groupby(['cohortid', 'h_popUID', 'true_new']).\
                apply(lambda x : inf_durations(x)).\
                dt.days.values
            age = df.\
                groupby(['cohortid', 'h_popUID', 'true_new']).\
                apply(lambda x : get_age(x)).\
                values
            age = age[~np.isnan(t)]
            t = t[~np.isnan(t)]

            lam = np.random.random(2)
            vectors = [t, age]
            exp_likelihood_age(lam, vectors)
            m = minimize(exp_likelihood_age, lam, vectors, method='Nelder-Mead')
            return m
        def print_coefficients(om, boots):
            boots = np.array(boots)
            CI = np.percentile(boots, [2.5, 97.5], axis=0).T

            print('Calculated Coeffecients : ')
            for i in range(om.x.size):
                print('    l{:d} : {:.5f} ({:.5f} : {:.5f})'.format(i, om.x[i], CI[i][0], CI[i][1]))

        age_space = np.linspace(1,60,200)
        om = fit_model(self.original_infections)
        olam = np.exp(om.x[0] + (om.x[1] * age_space))
        sns.lineplot(age_space , 1/olam)
        if bootstrap:
            self.in_boot=True
            boots=[]
            for i in tqdm(range(n_iter)):
                self.BootstrapInfections()
                m = fit_model(self.infections)
                boots.append(m.x)
                lams = np.exp(m.x[0] + m.x[1] * age_space)
                sns.lineplot(age_space, 1/lams, alpha = 0.3, legend=False, lw=0.5, color='grey')
            print_coefficients(om, boots)

        plt.ylabel("Calculated Duration")
        plt.xlabel("Age")
        plt.title("Calculated Duration as a function of Age")
        plt.savefig("../plots/survival/age_exponentialSurvival.pdf")
        plt.show()
        plt.close()
    def Durations_by_Quarters(self, bootstrap=False, n_iter=200, removeTreatment=False):
        """estimate durations using exponential model by quarter"""
        self.in_boot = False
        def inf_durations(x):
            """only add minimum_duration if not treated and set hard burnin"""
            cid = x.cohortid.unique()[0]
            if self.in_boot:
                cid = self.bootstrap_id_dates[cid]


            burnout = pd.to_datetime(self.cid_dates[cid][-self.allowedSkips:]).min()
            treatment = False

            if x.date.max() <= x.burnin.max():
                return np.nan
            if x.date.max().to_datetime64() in self.treatments[cid]:
                treatment = True

            s_obs = ~np.any(x.date <= x.burnin)
            e_obs = ~np.any(x.date >= burnout)


            t_start = x.date.min() - self.minimum_duration if s_obs else x.burnin.min()
            t_end = x.date.max() if e_obs else self.study_end
            t_end = t_end + self.minimum_duration if not treatment else t_end


            t = t_end - t_start

            return t
        def exp_likelihood(l, t):
            lik = l * np.exp(-l * t)
            log_lik = np.log(lik).sum()
            return -1 * log_lik
        def fit_model(df):
            self.study_start = df.date.min()
            self.study_end = df.date.max()
            t = df.\
                groupby(['cohortid', 'h_popUID', 'true_new']).\
                apply(lambda x : inf_durations(x)).\
                dt.days.values
            t = t[~np.isnan(t)]
            while True:
                lam = np.random.random()
                m = minimize(exp_likelihood, lam, t, method='TNC', bounds=((0, 1), ))
                if m.success:
                    break
            return m.x
        def plot_boots(boots, lam):
            sns.distplot(boots, color='teal')
            plt.axvline(lam, linestyle=':', lw=5, color='teal')
            plt.xlabel("Calculated Lambda")
            plt.title("Calculated Lambda and Distribution of Bootstrapped Lambdas")
            plt.savefig("../plots/survival/exponentialSurvival.pdf")
            plt.show()
            plt.close()

        if removeTreatment:
            self.RemoveTreated()

        dates = self.original_infections.date.unique()
        date_bins = pd.date_range(dates.min(), dates.max(), periods=5)
        for i,_ in enumerate(date_bins[:-1]):
            q = self.original_infections[
                (self.original_infections.date >= date_bins[i]) &
                (self.original_infections.date <= date_bins[i+1])
            ]
            lam = fit_model(q)


            if bootstrap:
                self.in_boot=True
                boots = []
                for i in tqdm(range(n_iter), desc='bootstrapping'):
                    self.BootstrapInfections(frame=q)
                    boot_lam = fit_model(self.infections)
                    boots.append(boot_lam)
                boots = np.array(boots)
                lower, upper = np.percentile(boots, [2.5, 97.5])
                print("Fit Lambda : {0} ({1}, {2})".format(lam, lower, upper))
                print("Expected Duration : {0} ({1}, {2})".format(1/lam, 1/upper, 1/lower))
                plot_boots(boots, lam)
            else:
                print("Fit Lambda : {}".format(lam))
                print("Expected Duration : {0} days".format(1 / lam))
class ExponentialDecay:
    def __init__(self, infections, left_censor='2018-01-01', right_censor='2019-02-01', minimum_duration=15, seed=None):
        if seed:
            np.random.seed(seed)

        # left_censor = pd.to_datetime('2018-01-01')
        # right_censor = pd.to_datetime('2019-02-01')

        self.infections = infections
        self.study_start = pd.to_datetime(left_censor) if left_censor else infections.date.min()
        self.study_end = pd.to_datetime(right_censor) if right_censor else infections.date.max()
        self.minimum_duration = pd.Timedelta('{} Days'.format(minimum_duration))

        self.durations = []
        self.num_classes = np.zeros(5)
        self.optimizers = []
    def BootstrapInfections(self, frame):
        """Bootstrap on Cohortid"""
        cids = frame.cohortid.unique()
        cid_choice = np.random.choice(cids, cids.size)
        bootstrap = pd.concat([frame[frame.cohortid == c] for c in cid_choice])
        return bootstrap
    def ClassifyInfection(self, infection):
        """
        Classify an infection type by whether or not the start date of the
        infection is observed in a given period and return the duration by the
        class
        """
        infection_min, infection_max = infection.date.min(), infection.date.max()

        # infection not active in period
        if (infection_max <= self.study_start) | (infection_min >= self.study_end):
            classification = 0
            duration = None

        # Start and End Observed in Period
        elif (infection_min >= self.study_start) & (infection_max <= self.study_end):
            classification = 1
            duration = infection_max - infection_min

        # Unobserved Start + Observed End in Period
        elif (infection_min <= self.study_start) & (infection_max <= self.study_end):
            classification = 2
            duration = infection_max - self.study_start

        # Observed Start + Unobserved End in Period
        elif (infection_min >= self.study_start) & (infection_max >= self.study_end):
            classification = 3
            duration = self.study_end - infection_min

        # Unobserved Start + Unobserved End in Period
        elif (infection_min <= self.study_start) & (infection_max >= self.study_end):
            classification = 4
            duration = self.study_end - self.study_start


        if duration == pd.to_timedelta(0):
            duration += self.minimum_duration

        if duration:
            duration = duration.days

        self.num_classes[classification] += 1

        return np.array([classification, duration])
    def GetInfectionDurations(self, infection_frame):
        """for each clonal infection calculate duration"""
        durations = infection_frame.\
            groupby(['cohortid', 'h_popUID']).\
            apply(lambda x : self.ClassifyInfection(x)).\
            values
        durations = np.vstack(durations)
        durations = durations[durations[:,0] != 0]
        l1_durations = durations[durations[:,0] <= 2][:,1]
        l2_durations = durations[durations[:,0] > 2][:,1]

        self.durations.append([l1_durations, l2_durations])
        return l1_durations, l2_durations
    def RunDecayFunction(self, lam, l1_durations, l2_durations):
        """
        Exponential Decay Function as Log Likelihood
        """

        l1_llk = (np.log(lam) - (lam * l1_durations)).sum()

        l2_llk = ((-1 * lam) * l2_durations).sum()

        llk = l1_llk + l2_llk

        return -1 * llk
    def GetConfidenceIntervals(self, min = 5, max = 95):
        """
        Return confidence intervals for a bootstrapped array
        """
        ci_min, ci_max = np.percentile(self.bootstrapped_lams, [min, max])
        return ci_min, ci_max
    def fit(self, frame=None, bootstrap=False, n_iter=200):
        """
        Fit Exponential Model
        """
        if type(frame) == type(None):
            frame = self.infections.copy()
        if bootstrap:
            bootstrapped_lams = [self.fit(frame=self.BootstrapInfections(frame)) for _ in tqdm(range(n_iter))]


        # generate durations and initial guess
        l1_durations, l2_durations = self.GetInfectionDurations(frame)
        lam = np.random.random()

        # run minimization of negative log likelihood
        opt = minimize(
            self.RunDecayFunction,
            lam,
            args=(l1_durations, l2_durations),
            method = 'L-BFGS-B',
            bounds = ((1e-6, None),)
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

        sns.distplot(1 / self.bootstrapped_lams, color='teal', bins=30)
        plt.axvline(1 / self.estimated_lam, color='teal', linestyle=':', lw=8)
        plt.xlabel("Calculated Days (1 / lambda)")
        plt.title(
            "Estimated Lambda : {:.4f}e-3 ({:.4f}e-3 -- {:.4f}e-3)\nEstimated Days : {:.1f} ({:.1f} -- {:.1f})".format(
                self.estimated_lam * 1e3, ci_min * 1e3, ci_max * 1e3,
                1/self.estimated_lam, 1/ci_max, 1/ci_min
                )
            )
        plt.show()
        plt.close()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--seekdeep_output',
        required=False,
        default="../prism2/full_prism2/final_filter.tab",
        help="SeekDeep Output to use as input to functions"
        )
    p.add_argument('-m', '--meta',
        required=False,
        default= "../prism2/stata/rolling_enrollment.tab",
        help="Cohort Meta information (tsv) to relate cohortids"
        )
    p.add_argument('-b', '--burnin',
        default=3, type=int,
        help="Number of months to consider a patient in burnin period (default = 3 months)"
        )
    p.add_argument('--bootstrap',
        action='store_true',
        help='run same analyses on simulated data'
        )
    p.add_argument('--num_iter',
        default=200, type=int,
        help='number of bootstraps to run'
        )
    args = p.parse_args()
    return args
def main(args):
    sdo = pd.read_csv(args.seekdeep_output, sep='\t')
    meta = pd.read_csv(args.meta, sep="\t", low_memory=False)

    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep='\t')
    meta = pd.read_csv('../prism2/stata/oct_enroll.tab', sep="\t", low_memory=False)

    sdu = SeekDeepUtils(
        sdo = sdo,
        meta = meta,
        fail_flag=False,
        qpcr_threshold=0,
        burnin=args.burnin
        )

    survival = Survival(
        sdu.Old_New_Infection_Labels(),
        sdu.meta
    )

    survival.OldNewSurvival(bootstrap=True, n_iter=200)
    survival.CID_oldnewsurvival(bootstrap=True, n_iter=200)
    survival.OldWaning(bootstrap=True, n_iter=200)


def DecayByPeriod(infections, n_iter=100, label=None):
    ed_classes = []
    estimated_values = []
    bootstrapped_values = []
    indices = []

    dates = infections.date.unique()
    date_bins = pd.date_range(dates.min(), dates.max(), periods=6)
    for i, bin_label in enumerate(date_bins[1:-1]):
        ed = ExponentialDecay(survival.original_infections, left_censor=date_bins[i], right_censor=date_bins[i+1])
        l, bsl = ed.fit(bootstrap=True, n_iter=n_iter)

        indices.append(bin_label)
        ed_classes.append(ed)
        estimated_values.append(l)
        bootstrapped_values.append(bsl)

    for i, _ in enumerate(ed_classes):
        sns.distplot(1 / bootstrapped_values[i], bins=30)
        plt.axvline(1/ estimated_values[i], label=indices[i], color=sns.color_palette()[i], linestyle=':', lw=5)
        plt.legend(labels = indices)

    if label:
        plt.savefig('../plots/durations/{}.png'.format(label))
    plt.show()

    return ed_classes
def DecayByGroup(infections, n_iter=100, group = ['gender'], label=None):
    ed_classes = []
    estimated_values = []
    bootstrapped_values = []
    indices = []
    for index, frame in infections.groupby(group):
        ed = ExponentialDecay(frame)
        l, bsl = ed.fit(bootstrap=True, n_iter=n_iter)

        indices.append(index)
        ed_classes.append(ed)
        estimated_values.append(l)
        bootstrapped_values.append(bsl)

    for i, _ in enumerate(ed_classes):
        sns.distplot(1 /bootstrapped_values[i], bins=30)
        plt.axvline(1/ estimated_values[i], label=indices[i], color=sns.color_palette()[i], linestyle=':', lw=5)
        plt.legend(labels = indices)

    if label:
        plt.savefig('../plots/durations/{}.png'.format(label))
    plt.show()
def SplitByAgecat(ageyrs_frame, breaks=[9, 20]):
    breaks = np.array(breaks)

    cid_age = ageyrs_frame.groupby(['cohortid']).apply(lambda x : x.ageyrs.min())

    conditions = ['x < {}'.format(i) for i in breaks] + ['x >= {}'.format(breaks[-1])]

    break_points = cid_age.apply(
        lambda x : np.where(x >= breaks)[0]
        )

    cid_ageyrs = break_points.apply(
        lambda x : conditions[x.max() + 1] if len(x) > 0 else conditions[0]
        )

    ageyrs_frame['agecat'] = ageyrs_frame.apply(lambda x : cid_ageyrs[x.cohortid], axis = 1)

    print(ageyrs_frame[['cohortid', 'agecat']].drop_duplicates().groupby('agecat').count())

    return ageyrs_frame

def develop():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t", low_memory=False)
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)
    sdu = SeekDeepUtils(
        sdo, meta
    )

    meta.columns.values
    labels = sdu.Old_New_Infection_Labels()
    labels.cohortid.unique().size

    survival = Survival(
        labels,
        sdu.meta
        )
    survival.original_infections
    survival.Durations(bootstrap=True)


    sys.exit()
    default_decay = ExponentialDecay(survival.original_infections)
    durations = default_decay.GetInfectionDurations(default_decay.infections)
    default_decay.fit(bootstrap=True, n_iter=100)
    default_decay.plot()


    # age categories annotated in meta
    age_frame = survival.original_infections.merge(survival.meta[['cohortid', 'agecat']], how='inner').drop_duplicates()
    DecayByGroup(age_frame[age_frame.cohortid != 3617], n_iter=500, group=['agecat'], label='agecat')

    # age categories from arbitrary break points
    ageyrs_frame = survival.original_infections.merge(survival.meta[['cohortid', 'ageyrs']], how='inner')
    ageyrs_frame = SplitByAgecat(ageyrs_frame, breaks = [5,10,15])
    DecayByGroup(ageyrs_frame[ageyrs_frame.cohortid != 3617], n_iter=300, group=['agecat'], label='arbitrary_agecat')

    # sex differences in durations
    gender_frame = survival.original_infections.merge(survival.meta[['cohortid', 'gender', 'agecat']], how='inner')
    DecayByGroup(gender_frame[gender_frame.cohortid != 3617], n_iter=300, group=['gender'], label='gender')

    # sex and agecat differences in durations
    DecayByGroup(gender_frame[gender_frame.cohortid != 3617], n_iter=300, group=['gender', 'agecat'], label='gender_agecat')

    # durations by quarter
    ed_by_period = DecayByPeriod(survival.original_infections[survival.original_infections.cohortid != 3617], n_iter=300, label='quarterly')


if __name__ == '__main__':
    args = get_args()
    # main(args)
    develop()

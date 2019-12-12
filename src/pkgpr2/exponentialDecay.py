#!/usr/bin/env python3

import numpy as np
import pandas as pd
import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm
from multiprocess import Pool
from scipy.optimize import minimize


@nb.jit(nopython=True)
def decay_function(lam, l1_durations, l2_durations):
    """
    Exponential Decay Function as Log Likelihood
    """

    l1_llk = np.log(lam) - (lam * l1_durations)

    l2_llk = (-1 * lam) * l2_durations

    llk = l1_llk.sum() + l2_llk.sum()

    return -1 * llk


#@nb.jit(nopython=True)
def sample_cohortid(cids):
    return np.random.choice(cids, cids.size)


#@nb.jit(nopython=True)
def generate_indices(original_cids, sampled_cids):
    arr = []
    for i in np.arange(sampled_cids.size):
        cid = sampled_cids[i]
        for j in np.where(original_cids == cid)[0]:
            arr.append(j)

    return np.array(arr)


def bootstrap_frame(frame, num_iter):
    original_cids = frame.iloc[:, 0].values
    unique_cids = np.unique(original_cids)

    for i in range(num_iter):
        sampled_cids = sample_cohortid(unique_cids)

        cid_indices = generate_indices(original_cids, sampled_cids)

        bootstrap = frame.iloc[cid_indices]

        yield bootstrap



class ExponentialDecay(object):

    def __init__(self, infections,
                 left_censor='2018-01-01', right_censor='2019-04-01',
                 minimum_duration=30, seed=None, skips=3, drop_person=True
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
        self.drop_person = drop_person

        self.durations = []
        self.num_classes = np.zeros(5)
        self.optimizers = []

        self.DropMalaria()

    def DropMalaria(self):
        """
        Drop any infections with a malaria event
        """

        if self.drop_person:
            to_drop = self.infections.\
                groupby(['cohortid']).\
                apply(
                    lambda x: np.any(x.malariacat == 'Malaria')
                    ).reset_index()
        else:
            to_drop = self.infections.\
                groupby(['cohortid', 'h_popUID']).\
                apply(
                    lambda x: np.any(x.malariacat == 'Malaria')
                    ).reset_index()

        self.infections = self.infections.\
            merge(to_drop, how='left')

        self.infections = self.infections[~self.infections[0]].\
            drop(columns=0)

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

        if not isinstance(frame, pd.core.frame.DataFrame):
            frame = self.infections.copy()
        if bootstrap:
            p = Pool()
            bootstrapped_lams = p.map(self.fit, bootstrap_frame(frame, n_iter))

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

    def plot(self, save=None):
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

        if not save:
            plt.show()
            plt.close()
        else:
            plt.savefig(save)
            plt.close()

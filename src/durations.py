#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pkgpr2.exponentialDecay as ed
import sys

def load_labels(clone=True):
    fn = "labels.{}.tab"
    if clone:
        fn = fn.format('clone')
    else:
        fn = fn.format('ifx')

    labels = pd.read_csv(fn, sep="\t")
    labels.date = labels.date.astype('datetime64')
    labels.enrolldate = labels.enrolldate.astype('datetime64')
    labels.burnin = labels.burnin.astype('datetime64')

    return labels


def durations_by_cat(sub_labels, label='label'):
    e = ed.ExponentialDecay(sub_labels, seed=42)
    l1, l2, durations = e.GetInfectionDurations(sub_labels)

    frame = pd.DataFrame(durations, columns=['classification', 'duration'])
    frame = frame[frame.classification != 0]

    d = {
        1: 'uncensored',
        2: 'c_left',
        3: 'c_right',
        4: 'c_both'
        }

    frame['classification'] = [d[i] for i in frame.classification]
    frame['label'] = label

    return frame

def get_lam(frame, num_iter=1000):
    e = ed.ExponentialDecay(frame)
    return e.fit(frame, bootstrap=True, n_iter=num_iter)

def durations_by_baseline(labels, save=None):

    baseline = labels[labels.active_baseline_infection]
    new_ifx = labels[~labels.active_baseline_infection]

    baseline_estimate, baseline_boots = get_lam(baseline)
    ifx_estimate, ifx_boots = get_lam(new_ifx)

    palette = sns.color_palette("Set1")

    sns.distplot(1 / baseline_boots, label='Baseline', color = palette[0])
    sns.distplot(1 / ifx_boots, label='New Infection', color = palette[1])

    plt.axvline(1 / baseline_estimate, ls=':', color=palette[0])
    plt.axvline(1 / ifx_estimate, ls=':', color=palette[1])

    plt.legend()
    plt.xlabel("Duration (days)")

    if not save:
        plt.show()
    else:
        print('saving figure : {}'.format(save))
        plt.savefig(save)
        plt.close()

def durations_by_group(frame, group, save=None):

    palette = sns.color_palette("Set1")

    count = 0
    for idx, g_frame in frame.groupby(group):
        estimate, boots = get_lam(g_frame)

        sns.distplot(1 / boots, label=idx, color = palette[count])
        plt.axvline(1 / estimate, ls=':', color = palette[count])
        count += 1

    plt.legend()
    plt.xlabel("Duration (days)")

    if not save:
        plt.show()
    else:
        print('saving figure : {}'.format(save))
        plt.savefig(save)
        plt.close()

def main():

    plot_fn = "../plots/durations/{}.pdf"

    labels_clone = load_labels(clone=True)
    labels_ifx = load_labels(clone=False)


    # by agecat
    durations_by_group(
        labels_clone, group = ['agecat'],
        save=plot_fn.format('agecat_clone')
        )
    durations_by_group(
        labels_ifx, group = ['agecat'],
        save=plot_fn.format('agecat_ifx')
        )

    # by gender
    durations_by_group(
        labels_clone, group = ['gender'],
        save=plot_fn.format('sex_clone')
        )
    durations_by_group(
        labels_ifx, group = ['gender'],
        save=plot_fn.format('sex_ifx')
        )

    # by baseline
    durations_by_baseline(
        labels_clone,
        save=plot_fn.format('baseline_vs_new_clone')
        )
    durations_by_baseline(
        labels_ifx,
        save=plot_fn.format('baseline_vs_new_ifx')
        )

if __name__ == '__main__':
    main()

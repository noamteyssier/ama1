#!/usr/bin/env python3

import numpy as np
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
    return e.fit(bootstrap=True, n_iter=num_iter)


def durations_by_group(original_frame, group, save=None):
    frame = original_frame.copy()

    palette = sns.color_palette("Set2")

    if 'active_baseline_infection' in group:
        frame.active_baseline_infection = frame.active_baseline_infection.\
            apply(
                lambda x: "baseline" if x else "new infection"
                )

    count = 0
    for idx, g_frame in frame.groupby(group):
        print('calculating durations on group : {}'.format(idx))
        if len(group) > 1:
            label = '.'.join(idx)
        else:
            label = idx

        estimate, boots = get_lam(g_frame)

        sns.distplot(1 / boots, label=label, color=palette[count])
        plt.axvline(1 / estimate, ls=':', color=palette[count])
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
        labels_clone, group=['agecat'],
        save=plot_fn.format('agecat_clone')
        )
    durations_by_group(
        labels_ifx, group=['agecat'],
        save=plot_fn.format('agecat_ifx')
        )

    # by gender
    durations_by_group(
        labels_clone, group=['gender'],
        save=plot_fn.format('sex_clone')
        )
    durations_by_group(
        labels_ifx, group=['gender'],
        save=plot_fn.format('sex_ifx')
        )

    # by baseline
    durations_by_group(
        labels_clone, group=['active_baseline_infection'],
        save=plot_fn.format('baseline_vs_new_clone')
        )
    durations_by_group(
        labels_ifx, group=['active_baseline_infection'],
        save=plot_fn.format('baseline_vs_new_ifx')
        )

    # by sex/baseline
    durations_by_group(
        labels_clone, group=['active_baseline_infection', 'gender'],
        save=plot_fn.format('sex_baseline_vs_new_clone')
        )
    durations_by_group(
        labels_ifx, group=['active_baseline_infection', 'gender'],
        save=plot_fn.format('sex_baseline_vs_new_ifx')
        )

    # by age/baseline
    durations_by_group(
        labels_clone, group=['active_baseline_infection', 'agecat'],
        save=plot_fn.format('agecat_baseline_vs_new_clone')
        )
    durations_by_group(
        labels_ifx, group=['active_baseline_infection', 'agecat'],
        save=plot_fn.format('agecat_baseline_vs_new_ifx')
        )

    # by agecat/gender
    durations_by_group(
        labels_clone, group=['gender', 'agecat'],
        save=plot_fn.format('agecat_sex_clone')
        )
    durations_by_group(
        labels_ifx, group=['gender', 'agecat'],
        save=plot_fn.format('agecat_sex_ifx')
        )

if __name__ == '__main__':
    main()

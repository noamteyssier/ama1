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
    try:
        return e.fit(bootstrap=True, n_iter=num_iter)
    except:
        return None


def durations_by_group(original_frame, group, save=None):
    frame = original_frame.copy()

    palette = sns.color_palette("Set2")

    if 'active_baseline_infection' in group:
        frame.active_baseline_infection = frame.active_baseline_infection.\
            apply(
                lambda x: "baseline" if x else "new infection"
                )

    results_dict = {}

    count = 0
    for idx, g_frame in frame.groupby(group):
        print('calculating durations on group : {}'.format(idx))
        if len(group) > 1:
            label = '.'.join(idx)
        else:
            label = idx

        lams = get_lam(g_frame)
        if lams:
            estimate, boots = lams
            results_dict[idx] = lams

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

    return results_dict


def durations(frame, name, num_iter=1000, save=None):
    results_dict = {}
    e = ed.ExponentialDecay(frame)
    lams = e.fit(bootstrap=True, n_iter=num_iter)
    e.plot(save=save)

    results_dict = {name: lams}

    return results_dict


def bin_qpcr(frame):
    vals = frame.\
        groupby(['cohortid', 'h_popUID']).\
        apply(lambda x : x.qpcr.values[0]).\
        reset_index().\
        rename(columns = {0: 'qpcr_val'})

    vals['qpcr_bin'] = np.log10(vals.qpcr_val).round()
    vals.loc[vals.qpcr_bin < 1, 'qpcr_bin'] = 0

    return frame.merge(
        vals[['cohortid', 'h_popUID', 'qpcr_bin']]
        )


def results_to_frame(results_dict):
    frame = []
    for idx in results_dict:
        estimate, boots = results_dict[idx]
        cis = np.percentile(1 / boots, [5, 95])
        frame.append({
            'grouping': idx,
            'estimate': 1 / estimate,
            'ci_05': cis[0],
            'ci_95': cis[1]
            })

    return pd.DataFrame(frame)


def main():

    plot_fn = "../plots/durations/{}.pdf"

    labels_clone = load_labels(clone=True)
    labels_ifx = load_labels(clone=False)

    # overall
    clone_durations = durations(
        labels_clone, name='overall',
        save=plot_fn.format('overall_clone')
        )
    ifx_durations = durations(
        labels_ifx, name='overall',
        save=plot_fn.format('overall_ifx')
        )

    # by agecat
    clone_agecat = durations_by_group(
        labels_clone, group=['agecat'],
        save=plot_fn.format('agecat_clone')
        )
    ifx_agecat = durations_by_group(
        labels_ifx, group=['agecat'],
        save=plot_fn.format('agecat_ifx')
        )

    # by gender
    clone_gender = durations_by_group(
        labels_clone, group=['gender'],
        save=plot_fn.format('sex_clone')
        )
    ifx_gender = durations_by_group(
        labels_ifx, group=['gender'],
        save=plot_fn.format('sex_ifx')
        )

    # by baseline
    clone_baseline = durations_by_group(
        labels_clone, group=['active_baseline_infection'],
        save=plot_fn.format('baseline_vs_new_clone')
        )
    ifx_baseline = durations_by_group(
        labels_ifx, group=['active_baseline_infection'],
        save=plot_fn.format('baseline_vs_new_ifx')
        )

    # by sex/baseline
    clone_sexbaseline = durations_by_group(
        labels_clone, group=['active_baseline_infection', 'gender'],
        save=plot_fn.format('sex_baseline_vs_new_clone')
        )
    ifx_sexbaseline = durations_by_group(
        labels_ifx, group=['active_baseline_infection', 'gender'],
        save=plot_fn.format('sex_baseline_vs_new_ifx')
        )

    # by age/baseline
    clone_agebaseline = durations_by_group(
        labels_clone, group=['active_baseline_infection', 'agecat'],
        save=plot_fn.format('agecat_baseline_vs_new_clone')
        )
    ifx_agebaseline = durations_by_group(
        labels_ifx, group=['active_baseline_infection', 'agecat'],
        save=plot_fn.format('agecat_baseline_vs_new_ifx')
        )

    # by agecat/gender
    clone_agecatgender = durations_by_group(
        labels_clone, group=['gender', 'agecat'],
        save=plot_fn.format('agecat_sex_clone')
        )
    ifx_agecatgender = durations_by_group(
        labels_ifx, group=['gender', 'agecat'],
        save=plot_fn.format('agecat_sex_ifx')
        )


    clone_list = [
        clone_durations, clone_agecat,
        clone_gender, clone_baseline,
        clone_sexbaseline, clone_agebaseline,
        clone_agecatgender
        ]

    ifx_list = [
        ifx_durations, ifx_agecat,
        ifx_gender, ifx_baseline,
        ifx_sexbaseline, ifx_agebaseline,
        ifx_agecatgender
        ]

    clones = pd.concat([
        results_to_frame(i) for i in clone_list
        ])
    ifxs = pd.concat([
        results_to_frame(i) for i in ifx_list
        ])

    clones['ie_type'] = 'clone'
    ifxs['ie_type'] = 'ifx_event'

    frame = pd.concat([clones, ifxs])
    frame.to_csv("durations.tab", sep="\t", index=False)


if __name__ == '__main__':
    main()
    # dev()

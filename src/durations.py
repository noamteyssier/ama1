#!/usr/bin/env python3

import pandas as pd
import pkgpr2.exponentialDecay as ed


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


def main():
    labels_clone = load_labels(clone=True)
    # labels_ifx = pd.read_csv("labels.ifx.tab", sep="\t")

    e = ed.ExponentialDecay(labels_clone, seed=42)
    e.fit(labels_clone, bootstrap=True, n_iter=200)
    e.plot()

if __name__ == '__main__':
    main()

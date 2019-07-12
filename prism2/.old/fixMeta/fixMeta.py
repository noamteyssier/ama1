#!/usr/bin/env python3

"""
Script to generate tab delim file of pattern and substitutions for prism2 data with broken names
"""

import pandas as pd

def date_to_string(row):
    """handles situations where date is shortened between /"""
    return ''.join([i if len(i) != 1 else '0'+i for i in row['goodDate'].split('/')])

def create_regex(row):
    """generates pattern and substition from badDate to goodDate"""
    row['pattern'] = '-'.join([str(i) for i in row[['date', 'cid']].values])
    row['sub'] = '-'.join([str(i) for i in row[['datestr', 'cid']].values])
    return row

def write_subs(row):
    """print to stdout in tab delim"""
    print(
        '\t'.join(row[['sampleName', 'pattern', 'sub']].values)
    )


def main():
    meta = pd.read_csv("prism2_finalMeta.tab", sep='\t')
    sampleNames = pd.read_csv("sampleList.tab", sep='\t', dtype={'date':'str'})

    # merge meta with samples
    merged = pd.merge(meta, sampleNames, left_on='s_Sample', right_on='sampleName').drop_duplicates()

    # convert int to str
    merged['date'] = merged['date'].astype('str')

    # make proper datestring
    merged['datestr'] = merged.apply(date_to_string, axis = 1)

    # identify misplaced dates
    to_replace = merged[merged['date'] != merged['datestr']]

    # create regex
    to_replace = to_replace.apply(create_regex, axis = 1)

    # write to file
    to_replace.apply(write_subs, axis = 1)

if __name__ == '__main__':
    main()

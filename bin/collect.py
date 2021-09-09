#! /usr/bin/env python
import argparse
import string
import numpy as np
from roistats import collect
import json
import logging as log
import os
import os.path as op


desc = 'Collect ROI stats over a set of images and store them in an Excel '\
        'table.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('images', nargs='+', help='Input images')
    parser.add_argument('-o', '--output', required=True,
                        help='Excel file to store the output')
    help = 'Atlas (or any label volume) containing the '\
         'reference regions (supported atlases: JHU-labels, JHU-tracts)'
    parser.add_argument('--atlas', required=True, help=help)
    parser.add_argument('--n_jobs', default=-1,
                        help='Number of parallel jobs')
    parser.add_argument('--function', default='mean',
                        help='numpy function used to get values')
    parser.add_argument('--verbose', action='store_true',
                        help='Be verbose')
    opts = parser.parse_args()

    if opts.verbose:
        log.basicConfig(level=log.INFO)

    n_jobs = int(opts.n_jobs)

    from roistats import atlases
    fsldir = os.environ.get('FSLDIR')
    if fsldir is None:
        raise Exception('FSLDIR not found.')
    atlasdir = op.join(fsldir, 'data', 'atlases')

    if not opts.atlas in ['JHU-labels', 'JHU-tracts']:
        atlas = op.abspath(opts.atlas)
        if not op.isfile(atlas):
            msg = '%s is not an existing image nor a supported atlas (JHU-lab'\
                  'els/JHU-tracts)' % atlas
            raise FileNotError(msg)


    elif opts.atlas == 'JHU-labels':
        labels = atlases.labels('JHU-labels.xml')
        fn = 'JHU-ICBM-labels-1mm.nii.gz'
        atlas = op.join(atlasdir, 'JHU', fn)

    elif opts.atlas == 'JHU-tracts':
        labels = atlases.labels('JHU-tracts.xml')
        fn = 'JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz'
        atlas = op.join(atlasdir, 'JHU', fn)

    table = collect.roistats_from_maps(opts.images, atlas, opts.images,
                                       getattr(np, opts.function),
                                       n_jobs)
    table.to_excel(opts.output)

    # rename columns
    if opts.atlas in ['JHU-labels', 'JHU-tracts']:
        table = table.rename(columns=labels)
        table.to_excel(opts.output)

    d = {'images': opts.images, 'atlas': opts.atlas}
    fp = '%s.json' % op.splitext(opts.output)[0]
    json.dump(d, open(fp, 'w'))

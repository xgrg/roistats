#! /usr/bin/env python
import argparse
import string
import numpy as np
from roistats import collect
import json
import logging as log


desc = 'Collect ROI stats over a set of images and store them in an Excel '\
        'table.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('images', nargs='+', help='Input images')
    parser.add_argument('-o', '--output', required=True,
                        help='Excel file to store the output')
    parser.add_argument('--roi', required=True,
                        help='Atlas (or any label volume) containing the '\
                             'reference regions')
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
    table = collect.roistats_from_maps(opts.images, opts.roi, opts.images,
                                       getattr(np, opts.function),
                                       n_jobs)
    table.to_excel(opts.output)
    d = {'images': opts.images, 'atlas': opts.roi}
    json.dump(d, open(opts.output.replace('.xls', '.json'), 'w'))

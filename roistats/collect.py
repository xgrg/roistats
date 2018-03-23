import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
import logging as log

def roistats_from_map(map_fp, atlas, func=np.mean):
     m = np.array(nib.load(map_fp).dataobj)
     n_labels = list(np.unique(atlas))
     n_labels.remove(0)
     label_values = [func(m[atlas==label]) for label in n_labels]
     return label_values

def roistats_from_maps(maps_fp, atlas_fp, subjects=None, labels=None,
	func=np.mean, n_jobs=7):
     if len(subjects) != len(maps_fp):
         log.error('Images (%s) and subjects (%s) mismatch in size'
                 %(len(maps_fp), len(subjects)))
         return None

     log.info('Collecting ROI stats on %s images (atlas: %s)'
             %(len(maps_fp), atlas_fp))

     # Load atlas and count ROIs
     atlas_im = nib.load(atlas_fp)
     atlas = np.array(atlas_im.dataobj)

     roi_labels = list(np.unique(atlas))
     roi_labels.remove(0)
     if len(roi_labels) != len(labels):
         log.error('%s has %s non-null labels and labels has a size of %s'
                 %(atlas_fp, len(roi_labels), len(labels)))
         log.error('labels found in atlas: %s'%str(roi_labels))
         return None

     # Run it on every image
     df = Parallel(n_jobs=n_jobs, verbose=1)(\
         delayed(roistats_from_map)(maps_fp[i], atlas, func)\
         for i in xrange(len(maps_fp)))

     # Name rows and columns and return the DataFrame
     columns = [int(e) for e in roi_labels]
     res = pd.DataFrame(df, columns=columns).rename(columns=labels)

     res['subject'] = xrange(len(maps_fp)) if subjects is None else subjects
     res = res.set_index('subject')

     return res


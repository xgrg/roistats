import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
import logging as log

def _roistats_from_map(map_fp, atlas, func=np.mean):
     m = np.array(nib.load(map_fp).dataobj)
     assert(m.shape == atlas.shape)
     n_labels = list(np.unique(atlas))
     #n_labels.remove(0)
     print(n_labels)
     label_values = dict([(label, func(m[atlas==label])) for label in n_labels])
     return label_values

def roistats_from_maps(maps_fp, atlas_fp, subjects=None,
	   func=np.mean, n_jobs=7):
     if not subjects is None and len(subjects) != len(maps_fp):
         log.error('Images (%s) and subjects (%s) mismatch in size'
                 %(len(maps_fp), len(subjects)))
         return None

     log.info('Collecting ROI stats on %s images (atlas: %s)'
             %(len(maps_fp), atlas_fp))

     # Load atlas and count ROIs
     atlas_im = nib.load(atlas_fp)
     atlas = np.array(atlas_im.dataobj)

     roi_labels = list(np.unique(atlas))
     #roi_labels.remove(0)
     print(atlas.shape)

     # Run it on every image
     df = Parallel(n_jobs=n_jobs, verbose=1)(\
         delayed(_roistats_from_map)(maps_fp[i], atlas, func)\
         for i in range(0, len(maps_fp)))

     # Name rows and columns and return the DataFrame
     columns = [int(e) for e in roi_labels]
     res = pd.DataFrame(df, columns=columns)

     res['subject'] = range(0, len(maps_fp)) if subjects is None else subjects
     res = res.set_index('subject')

     return res

def roistats_from_map(map_fp, atlas_fp,	func=np.mean):
    return roistats_from_maps([map_fp], atlas_fp, func=func, )

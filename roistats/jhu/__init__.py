import os
import os.path as op
from glob import glob

def labels(fn='JHU-tracts.xml'):
    fsldir = os.environ.get('FSLDIR')
    if not fsldir is None:
        jhudir = op.join(fsldir, 'data', 'atlases')
        x = open(op.join(jhudir, fn)).read()
        d = dict([(int(e.split('index="')[1].split('"')[0]),
            e.split('>')[1].split('<')[0]) \
                for e in x.split('\n') if e.startswith('<label')])
        return d

def atlas(name=None):
    fsldir = os.environ.get('FSLDIR')
    if not fsldir is None:
        jhudir = op.join(fsldir, 'data', 'atlases')
        if name is None:
            print('List of available atlases:')
            files = [op.basename(e) for e in glob(op.join(jhudir, 'JHU', '*'))]
            print(' \n'.join(sorted(files)))
        else:
            print('Loading atlas %s'%name)
            import nibabel as nib
            fn = op.join(jhudir, 'JHU', name)
            return nib.load(name)

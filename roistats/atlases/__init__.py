import os
import os.path as op
from glob import glob

def labels(fn=None): #'JHU-tracts.xml'):
    fsldir = os.environ.get('FSLDIR')
    if fsldir is None:
        raise Exception('FSLDIR not found.')

    atlasdir = op.join(fsldir, 'data', 'atlases')
    if fn is None:
        print('Available atlases (%s):\n'%atlasdir)
        print(' - %s'%'\n - '.join([op.basename(e) for e in \
            glob(op.join(atlasdir, '*.xml'))]))
    else:
        x = open(op.join(atlasdir, fn)).read()
        a = 0
        d = {}
        add_one = ['Thalamus.xml', 'MarsTPJParcellation.xml',
            'Striatum-Connectivity-7sub.xml', 'Striatum-Connectivity-3sub.xml',
            'SalletDorsalFrontalParcellation.xml', 'MNI.xml',
            'Cerebellum_MNIfnirt.xml', 'MarsParietalParcellation.xml',
            'NeubertVentralFrontalParcellation.xml', 'Cerebellum_MNIflirt.xml',
            'HarvardOxford-Subcortical.xml', 'JHU-tracts.xml',
            'STN.xml', 'Juelich.xml', 'HarvardOxford-Cortical.xml']

        if fn in add_one:
            a = 1
            d[0] = 'Background'

        d.update(dict([(int(e.split('index="')[1].split('"')[0]) + a,
            e.split('>')[1].split('<')[0]) \
                for e in x.split('\n') if e.startswith('<label')]))
        return d

def maps(name=None, fn=None):
    fsldir = os.environ.get('FSLDIR')
    if fsldir is None:
        raise Exception('FSLDIR not found.')

    atlasdir = op.join(fsldir, 'data', 'atlases')
    dirs = [op.basename(e) for e in glob(op.join(atlasdir, '*')) \
        if op.isdir(op.join(atlasdir, e)) and op.basename(e) != 'bin']
    if not name is None and not name in dirs:
        print('%s not found'%name)
    if name is None or name not in dirs:
        print('Available atlases (%s):\n'%atlasdir)
        print(' - %s'%'\n - '.join(sorted(dirs)))
    else:
        files = [op.basename(e) for e in glob(op.join(atlasdir, name, '*'))]
        if not fn is None and not fn in files:
            print('%s not found'%fn)
        if fn is None or fn not in files:
            print('Available version in %s:\n'%name)
            print(' - %s'%'\n - '.join(sorted(files)))
        else:
            print('Loading atlas %s/%s'%(name, fn))
            import nibabel as nib
            fp = op.join(atlasdir, name, fn)
            return nib.load(fp)

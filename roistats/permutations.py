import logging as log
log.basicConfig(level=log.INFO)

def run(y, data, by='apoe',
     groups={'NC': [0], 'carriers':[1,2], 'HT': [1], 'HO': [2], 'not HO': [0,1]},
     contrast = ('NC', 'carriers'),
     num_perm=100000):

    import pandas as pd
    import numpy as np

    col = []
    df = pd.DataFrame(data, copy=True)
    grp = {contrast[0]:groups[contrast[0]], contrast[1]:groups[contrast[1]]}
    log.warning('* Contrast: %s'%str(contrast))

    for i, row in df.iterrows():
        for k, group in grp.items():
            if row[by] in group:
                col.append(k)
    df['_group'] = col

    g = df.groupby(by='_group')
    values2 = [list(v[y].tolist()) for e,v in g]

    def run_permutation_test(pooled, sizeZ, sizeY, delta):
        np.random.shuffle(pooled)
        starZ = pooled[:sizeZ]
        starY = pooled[-sizeY:]
        return starZ.mean() - starY.mean()

    z = np.array(values2[1])
    y = np.array(values2[0])

    pooled = np.hstack([z,y])
    delta = z.mean() - y.mean()
    estimates = np.array(list(map(lambda x: run_permutation_test(pooled, z.size, y.size, delta), range(num_perm))))
    diffCount = len(np.where(estimates <=delta)[0])
    hat_asl_perm = 1.0 - (float(diffCount)/float(num_perm))

    return hat_asl_perm

def run_all_contrasts(y, data, by='apoe', num_perm=100000):
    import itertools
    groups={'NC': [0], 'carriers':[1,2], 'HT': [1], 'HO': [2], 'not HO': [0,1]}
    present_groups = []
    apoe = data[by].tolist()
    for e, v in groups.items():
        if set(v).issubset(set(apoe)):
           present_groups.append(e)
    contrasts = []

    for i1, i2 in itertools.combinations(present_groups, 2):
        if len(set(groups[i1]).intersection(set(groups[i2]))) == 0 and \
           len(set(groups[i1])) + len(set(groups[i2])) == len(set(apoe)):
             contrasts.append((i1,i2))
    log.warning(contrasts)

    results = {}
    for contrast in contrasts:
        p = run(y, data, by=by, num_perm=num_perm, contrast=contrast, groups=groups)
        results[contrast] = min(p, 1-p)

    return results

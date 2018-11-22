# roistats

Exploring imaging features across subjects over regions-of-interest (ROI) is
something pretty typical in neuroimaging analysis workflows. `roistats` is a
 collection of basic Python functions based on standard packages which allow
 easier collection and visualization of ROI values.

## Collecting ROI values

In this context a ROI atlas defined as a _label volume_, i.e. an image in which
every voxel is assigned a label from a specific set. Each label is associated to
a given region/structure of the brain. The next figure represents the [AAL](http://neuro.imm.dtu.dk/wiki/Automated_Anatomical_Labeling) ROI atlas overlaid
on the MNI reference template.

![AAL](https://raw.githubusercontent.com/xgrg/roistats/master/doc/aal.png)


From a set of images `maps_fp` (from a list of subjects `subjects`) registered
to the same reference atlas `atlas_fp`, the following command will then generate
 a [**pandas**](https://pandas.pydata.org/) **DataFrame** with mean values from
 every label/region from the atlas:

 ```python
from roistats import collect
collect.roistats_from_maps(maps_fp, atlas_fp, subjects, func=np.mean, n_jobs=7)
 ```

`func` is the function used to aggregate the values from each label, `n_jobs`
allows to parallelize the process in multiple jobs. The `subjects` list will
be taken as index of the produced DataFrame.

Each _label_ will be assigned a different column of the DataFrame. It can then
be transformed using the `_unpivot` function to a more compact format (less
  columns/more rows).

Ex:
```python
from roistats import plotting
plotting._unpivot(data, regions, 'region', 'volume').join(cov)
```
where `cov` may contain covariates.

| ID       | region            | volume | apoe | age | sex |
|----------|-------------------|--------|------|-----|-----|
| subject0 | Precentral_L      | 518    | HT   | 65  | 1   |
| subject0 | Precentral_R      | 520    | HT   | 65  | 1   |
| subject0 | Frontal_Sup_L     | 557    | HT   | 65  | 1   |
| subject0 | Frontal_Sup_R     | 801    | HT   | 65  | 1   |
| subject0 | Frontal_Sup_Orb_L | 946    | HT   | 65  | 1   |

And then back to the initial format.

```python
from roistats import plotting
plotting._pivot(data, regions=['Frontal_Mid_L'],
  covariates=['apoe', 'age', 'sex'], value_colname='volume')
```

| ID        | Frontal_Mid_L | apoe | age | sex |
|-----------|---------------|------|-----|-----|
| subject0  | 917           | HT   | 65  | 1   |
| subject1  | 191           | HO   | 57  | 0   |
| subject10 | 920           | HO   | 63  | 1   |
| subject11 | 699           | HT   | 50  | 1   |
| subject12 | 215           | NC   | 70  | 0   |


## Generating plots: a few examples

```python
from roistats import plotting

roi_name = 'Superior longitudinal fasciculus L'
groups = {'not HO': ('NC', 'HT'),
          'HO': ('HO',)}
plotting.boxplot(roi_name, groups=groups, by='apoe', data=df, covariates=['age','sex'])
```

The boxplot displays the p-value from a two-sample t-test.

![example1](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example1.png)

```python
roi_name = 'Superior longitudinal fasciculus L'
groups = ('male', 'female')
plotting.boxplot(roi_name, groups=groups, by='sex', data=df, covariates=['age'])
```

![example2](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example2.png)

```python
roi_name = 'Superior longitudinal fasciculus L'
plotting.lmplot(roi_name, 'age', hue='apoe', covariates=['gender'], data=df)
```

![example3](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example3.png)

```python
regions = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Sup_Orb_L']
plotting.hist(data, regions[:5], by='apoe', region_colname='region',
    value_colname='volume', covariates=['age','sex'], ylim=[350, 600],
    hue_order=['NC','HT','HO'], size=6, aspect=2)
```

The function will compute pairwise t-tests across all groups and will print
them out.

![example4](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example4.png)


## Permutation tests

```python
from roistats import permutations as per
pval = per.run(regions[1], data, by='apoe', contrast=('NC','carriers'),
    groups={'NC':['NC'], 'carriers':['HT','HO']})

per.run_all_contrasts(regions[1], data)
```

These two functions `permutations.run()` and `permutations.run_all_contrasts()`
allow to perform permutation tests between two groups (or all possible pairs,
  respectively) and return the corresponding p-value.

## More t-tests

Similar to what is done by `boxplot`, extensive "_pairwise_" t-tests can be done
by a single function `genotypes.estimate()`.

```python
from roistats.contrasts import genotypes as g
g.estimate(data, regions[1], covariates=['age','sex'],
    contrasts={'dominant':('NC','carriers')},
    groups={'NC':['NC'], 'carriers':['HT','HO']})
```

This function takes a parameter `interaction` for assessing interaction effect
between the dependent variable and  a given factor (e.g. age).

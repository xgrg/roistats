# roistats

### Examples

```python
from roistats import plotting

roi_name = 'Superior longitudinal fasciculus L'
groups = {'not HO': ('NC', 'HT'),
          'HO': ('HO')}
plotting.boxplot_region(roi_name, groups=groups, by='apoe', data=df, covariates=['age','sex'])
```

![example1](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example1.png)

```python
roi_name = 'Superior longitudinal fasciculus L'
groups = {'female': (0),
          'male': (1)}
plotting.boxplot_region(roi_name, groups=groups, by='sex', data=df, covariates=['age'])
```

![example2](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example2.png)

```python
roi_name = 'Superior longitudinal fasciculus L'
plotting.lm_plot(roi_name, 'age', hue='apoe', covariates=['gender'], data=df)
```

![example3](https://raw.githubusercontent.com/xgrg/roistats/master/doc/example3.png)
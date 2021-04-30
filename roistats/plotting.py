from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import logging as log
from .__init__ import correct
plt.style.use('classic')
plt.style.use('seaborn')


default_palette = {
    'HO': '#ff9999',
    'HT': '#ffd699',
    'NC': '#99ccff',
    'not HO': '#99ccff',
    'female': '#ff9999',
    'male': '#99ccff',
    'apoe44': '#ff9999',
    'apoe34': '#ffd699',
    'apoe33': '#99ccff',
    'carriers': '#ffd699',
    'AnTn': '#6495ed',
    'ApTn': '#ff962c',
    'ApTp': '#850d04',
    True: '#c24d4a',
    False: '#6495ed'}


def ttest_2samp(y, data, x1, x2, by, covariates, groups=None):
    import logging as log
    from statsmodels.formula.api import ols

    data_dummies = pd.get_dummies(data, columns=[by])
    if groups is None:
        groups = {x1: [x1], x2: [x2]}

    dummy_columns = []
    for x in [x1, x2]:
        for each in groups[x]:
            dummy_columns.append('%s_%s' % (by, each))

    formula = '%s ~ %s%s + 1' % (y, ' + '.join(dummy_columns),
                                 {False: '', True: ' + %s' % ' + '.join(covariates)}[len(covariates) != 0])
    log.warning('Used model for significance estimation: %s' % formula)
    fitted_model = ols(formula, data_dummies).fit()
    s1 = ['%s * %s_%s' % (1.0/len(groups[x1]), by, each) for each in groups[x1]]
    s2 = ['%s * %s_%s' % (1.0/len(groups[x2]), by, each) for each in groups[x2]]

    c = '%s - %s' % (' + '.join(s1), ' - '.join(s2))
    T = fitted_model.t_test(c)
    log.warning('Used contrast: %s - p-value: %s' % (c, T.pvalue))
    return T.pvalue


def _plot_significance(data, x1, x2, by, covariates, groups):
    from matplotlib import pyplot as plt

    ymin, ymax = data['y'].quantile([0.25, 0.75])
    # not good if the quantiles differ among groups
    ymax2 = ymax + (ymax - ymin) * 1.5
    ymin2 = ymin - (ymax - ymin) * 1.5

    y, h, col = ymax2 + (ymax2 - ymin2)*0.07, (ymax2 - ymin2)*0.02, 'k'
    i1 = list(groups.keys()).index(x1)
    i2 = list(groups.keys()).index(x2)

    # shift the bar in y to avoid overlap between tests
    y = y + 4 * (i2 - i1 - 1) * h
    plt.plot([i1, i1, i2, i2], [y, y+h, y+h, y], lw=1.5, c=col)
    opt = {'ha': 'center', 'va': 'bottom', 'color': col}
    pval = ttest_2samp('y', data, x1, x2, by, covariates, groups)

    if pval < 0.05:
        opt['weight'] = 'bold'
    plt.text((i1+i2)*.5, y + h, '%.3f' % (pval), **opt)

    return pval


def sort_groups(data, by, order):
    # Create a column with a group index based on `groups`
    d = pd.DataFrame(data, copy=True)
    d[by] = d[by].replace({group: order.index(group) for group in order})
    # d = d.sort_values(by=by)
    d[by] = d[by].replace({order.index(group): group for group in order})
    return d


def boxplot(y, data, by='apoe', covariates=[], palette=None, groups=None):
    '''`y` should be a variable name, `data` is the data, `covariates` lists
    the various nuisance factors, `by` is the variable setting the different
    groups.'''

    from collections import OrderedDict

    if palette == 'default':
        palette = default_palette

    if groups is None:
        groups = list(set(data[by].tolist()))
    if isinstance(groups, list):
        groups = OrderedDict([(e, [e]) for e in groups])
    all_groups = []
    for k, each in groups.items():
        all_groups.extend(each)
    if set(all_groups).difference(set(data[by].tolist())):
        raise Exception('%s not all found as %s groups' % (all_groups, by))

    # Create a column with a group index based on `groups`
    col = []
    d = pd.DataFrame(data, copy=True)
    for i, row in d.iterrows():
        for k, group in groups.items():
            if row[by] in group:
                col.append(k)
    d['_group'] = col

    roi_name = y
    variables = {'_group', y, by}
    for c in covariates:
        variables.add(c)

    # check that all variables are found in columns
    for e in variables:
        if e not in d.columns.tolist():
            raise Exception('%s not found in columns!' % e)

    df = pd.DataFrame(d, columns=list(variables)).rename(columns={y: 'y'})
    df = df.dropna()
    y = 'y'
    log.info('Dependent variable: %s' % roi_name)
    print(list(groups.keys()))
    df = sort_groups(df, by, list(groups.keys()))

    # Correct depending variable for covariates (if any) (df is modified)
    if len(covariates) != 0:
        df[y] = correct(df, '%s ~ %s  + 1' % (y, '+'.join(covariates)))

    box = sns.boxplot(x='_group', y='y', data=df, showfliers=False,
                      palette=palette, order=list(groups.keys()))
    box = sns.swarmplot(x='_group', y='y', data=df,  # palette=palette,
                        edgecolor='gray', color='0.8', linewidth=1,
                        order=list(groups.keys()))
    box.axes.set_yticklabels(['%.2e' % x for x in box.axes.get_yticks()])

    box.axes.set_xlabel('%s groups' % by, fontsize=15, weight='bold')
    ylabel = '%s' % {False: '',
                     True: ' (corrected for %s)'
                           % (' and '.join(covariates))}[len(covariates) != 0]
    box.axes.set_ylabel(ylabel)
    box.set_title(roi_name)
    #box.axes.set_ylim(1.5,4)

    # Estimate p-values and add them to the figure
    import itertools
    pvals = []

    for i1, i2 in itertools.combinations(groups.keys(), 2):
        pval = _plot_significance(df, i1, i2, by, covariates, groups)
        pvals.append((pval, (i1, i2)))

    return pvals


def lmplot(y, x, data, covariates=['gender', 'age'], hue='apoe', ylim=None,
           savefig=None, facecolor='white', order=1, palette=None, size=None,
           s=20, lwl=2, lws=1, ylabel=''):

    plt.style.use('classic')
    plt.style.use('seaborn')

    if palette == 'default':
        palette = default_palette

    # Build a new table with only needed variables
    # y is renamed to roi to avoid potential issues with strange characters
    roi_name = y
    log.info('Dependent variable: %s' % y)
    variables = {x, y}
    if hue is not None:
        variables.add(hue)
    for c in covariates:
        variables.add(c)

    # check that all variables are found in columns
    for e in variables:
        if e not in data.columns.tolist():
            raise Exception('%s not found in columns!' % e)

    df = pd.DataFrame(data, columns=list(variables)).rename(columns={y: 'y'})
    df = df.dropna()
    y = 'y'

    # Correct depending variable for covariates (if any) (df is modified)
    if len(covariates) != 0:
        df[y] = correct(df, '%s ~ %s  + 1' % (y, '+'.join(covariates)))

    lm = sns.lmplot(x=x, y=y,  data=df, height=6.2, hue=hue, size=size,
                    aspect=1.35, ci=95, truncate=True, sharex=False,
                    sharey=False, order=order, palette=palette,
                    scatter_kws={'linewidths': lws, 's': s,
                                 'edgecolors': '#333333'},
                    line_kws={'linewidth': lwl}, legend=True)

    for patch in lm.axes[0, 0].patches:
        clr = patch.get_facecolor()
        patch.set_edgecolor(palette[clr])
    ax = lm.axes
    if ylim is None:
        ax[0, 0].set_ylim([df[y].min(), df[y].max()])
    else:
        ax[0, 0].set_ylim(ylim)
    ax[0, 0].set_xlim([df[x].min(), df[x].max()])
    # ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    # ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    ax[0, 0].tick_params(labelsize=12)
    ax[0, 0].set_ylabel('%s%s' % (ylabel, {False: '',
                                           True: ' (corrected for %s)'
            % (' and '.join(covariates))}[len(covariates) != 0]), fontsize=13)
    ax[0, 0].set_xlabel(x, fontsize=15, weight='bold')
    lm.fig.suptitle(roi_name, fontsize=15)

    if savefig is not None:
        lm.savefig(savefig, facecolor=facecolor)
    return df


def _pivot(data, covariates=[], regions=None, region_colname='region',
           value_colname='value'):

    #print regions
    index_colname = '_ID'
    if regions is None:
        regions = set(data[region_colname].tolist())
    data2 = data[data[region_colname].isin(regions)]
    piv = pd.pivot_table(data2, values=value_colname, index=data2.index,
                         columns=region_colname)
    subject_list = list(data2.index.tolist())
    data2[index_colname] = subject_list
    columns = [index_colname]
    columns.extend(covariates)
    cov = pd.DataFrame(data2, columns=columns).drop_duplicates().set_index(index_colname)
    piv = piv.join(cov)

    return piv


def _unpivot(piv, regions, region_colname='structure', value_colname='value'):
    table = []
    index_colname = 'ID'
    for i, row in piv.iterrows():
        for region in regions:
            r = [i, region, row[region]]
            table.append(r)
    columns = [index_colname, region_colname, value_colname]
    data = pd.DataFrame(table, columns=columns).set_index(index_colname)
    return data


def hist(data, regions=None, by=None, covariates=[], palette=None,
         zscores=False, ylim=None, region_colname='structure',
         value_colname='value', hue_order=None, **kwargs):

    # check that all variables are found in columns
    # and if so collect covariables
    columns = [] if by is None else [by]
    columns.extend(covariates)
    for e in columns:
        if e not in data.columns.tolist():
            raise Exception('%s not found in columns!' % e)
    cov = pd.DataFrame(data, columns=columns)

    # first pivot region/value entries to individual columns
    if regions is None:
        regions = list(set(data[region_colname].tolist()))
    piv = _pivot(data, columns, regions, region_colname, value_colname)

    # correct these region/value entries in their new individual columns
    import itertools
    pvals = []

    for region in regions:
        if len(covariates) != 0:
            if region not in piv.columns.tolist():
                msg = '%s not found in regions! (%s)' % (region, regions)
                raise Exception(msg)
            piv = piv.rename(columns={region: 'y'})
            piv['y'] = correct(piv, '%s ~ %s  + 1' % ('y', '+'.join(covariates)))

            if by is not None:
                groups = list(set(piv[by].tolist()))
                for i1, i2 in itertools.combinations(groups, 2):
                    pval = ttest_2samp('y', piv, i1, i2, by, covariates)
                    pvals.append((region, (i1, i2), pval))
            piv = piv.rename(columns={'y': region})
        if zscores:  # convert them to zscores if asked
            m = piv[region].mean()
            s = piv[region].std()
            piv[region] = (piv[region] - m)/s

    # unpivot (rebuild the region/value two initial columns)
    data2 = _unpivot(piv, regions, region_colname, value_colname)
    data2 = data2.join(cov).dropna()

    # create a `rank` column to control the order of the bars (following order
    # given by `regions`)
    col = []
    for i, row in data2.iterrows():
        col.append(regions.index(row[region_colname]))
    data2['rank'] = col

    if palette == 'default':
        palette = default_palette
        if by is None:
            data2['_by'] = 'NC'
            by = '_by'

    # plot
    plt.figure(figsize=(30, 12))
    bar = sns.catplot(x='rank', y=value_colname, hue=by, data=data2,
                      palette=palette, errwidth=2, ci=90,
                      legend=not by == '_by', kind='bar',
                      hue_order=hue_order, capsize=.05, **kwargs)

    bar.set_xticklabels(rotation=45)
    bar.set_xticklabels(regions)

    if ylim is not None:
        plt.ylim(ylim)

    ylabel = '%s %s' % (value_colname, {False: '',
                                        True: ' (corrected for %s)'
                                              % (' and '.join(covariates))}[len(covariates) != 0])
    bar.set(xlabel=region_colname, ylabel=ylabel)
    return pvals

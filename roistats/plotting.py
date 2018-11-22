import seaborn as sns
import pandas as pd
import logging as log
from __init__ import correct


default_palette = {
    'HO':'#ff9999',
    'HT':'#ffd699',
    'NC':'#99ccff',
    'not HO':'#99ccff',
    'female':'#ff9999',
    'male':'#99ccff',
    'apoe44':'#ff9999',
    'apoe34':'#ffd699',
    'apoe33':'#99ccff',
    'carriers': '#ffd699'
    }
default_palette2 = {(0.6500000000000001, 0.7999999999999999, 0.95, 1.0):'#0062e9',
    (0.95, 0.8294117647058824, 0.6500000000000001, 1.0):'#f8a570',
    (0.95, 0.6500000000000001, 0.6500000000000001, 1.0):'#eb6e6e'}


def ttest_2samp(y, data, x1, x2, by, covariates, groups=None):
    import logging as log
    from statsmodels.formula.api import ols

    data_dummies = pd.get_dummies(data, columns=[by])
    print data_dummies.head()
    if groups is None:
        groups = {x1:[x1], x2:[x2]}

    dummy_columns = []
    for x in [x1, x2]:
        for each in groups[x]:
            dummy_columns.append('%s_%s'%(by, each))

    formula = '%s ~ %s%s + 1'%(y, ' + '.join(dummy_columns),
          {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    log.warning('Used model for significance estimation: %s'%formula)
    fitted_model = ols(formula, data_dummies).fit()
    s1 = ['%s * %s_%s'%(1.0/len(groups[x1]), by, each) for each in groups[x1]]
    s2 = ['%s * %s_%s'%(1.0/len(groups[x2]), by, each) for each in groups[x2]]

    c = '%s - %s'%(' + '.join(s1), ' - '.join(s2))
    T = fitted_model.t_test(c)
    log.warning('Used contrast: %s - p-value: %s'%(c, T.pvalue))
    return T.pvalue

def _plot_significance(data, x1, x2, by, covariates, groups):
    from matplotlib import pyplot as plt

    ymin, ymax = data['y'].quantile([0.25, 0.75])
    # not good if the quantiles differ among groups
    ymax2 = ymax + (ymax - ymin) * 1.5
    ymin2 = ymin - (ymax - ymin) * 1.5

    y, h, col = ymax2 + (ymax2 - ymin2)*0.07, (ymax2 - ymin2)*0.02, 'k'
    i1 = groups.keys().index(x1)
    i2 = groups.keys().index(x2)

    # shift the bar in y to avoid overlap between tests
    y = y + 4 * (i2 - i1 - 1) * h
    plt.plot([i1, i1, i2, i2], [y, y+h, y+h, y], lw=1.5, c=col)
    opt = {'ha':'center', 'va':'bottom', 'color':col}
    pval = ttest_2samp('y', data, x1, x2, by, covariates, groups)

    if pval < 0.05:
        opt['weight'] = 'bold'
    plt.text((i1+i2)*.5, y + h, '%.3f'%(pval), **opt)

    return pval

def sort_groups(data, by, order):
    # Create a column with a group index based on `groups`
    col = []
    d = pd.DataFrame(data, copy=True)
    d[by] = d[by].replace({group:order.index(group) for group in order})
    d = d.sort_values(by=by)
    d[by] = d[by].replace({order.index(group):group for group in order})
    return d


def boxplot_region(y, data, by='apoe', covariates=[], palette=None, groups=None):
    '''`y` should be a variable name, `data` is the data, `covariates` lists
    the various nuisance factors, `by` is the variable setting the different
    groups.'''

    from collections import OrderedDict

    if palette is None:
        palette = default_palette

    if groups is None:
        groups = list(set(data[by].tolist()))
    if isinstance(groups, list):
        groups = OrderedDict([(e, [e]) for e in groups])

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
        if not e in d.columns.tolist():
            raise Exception('%s not found in columns!'%e)

    df = pd.DataFrame(d, columns=list(variables)).rename(columns={y:'y'})
    df = df.dropna()
    y = 'y'
    log.info('Dependent variable: %s'%roi_name)
    df = sort_groups(df, by, groups.keys())


    # Correct depending variable for covariates (if any) (df is modified)
    if len(covariates) != 0:
        df[y]  = correct(df, '%s ~ %s  + 1'%(y, '+'.join(covariates)))

    box = sns.boxplot(x='_group', y='y', data=df, showfliers=False,
                palette=palette)
    #box = sns.violinplot(x='_group', y='roi', data=df, palette=palette)
    #box.axes.set_yticklabels(['%.2e'%x for x in box.axes.get_yticks()])

    box.axes.set_xlabel('groups', fontsize=15, weight='bold')
    ylabel = '%s'\
                %{False:'',
                  True:' (corrected for %s)'\
                     %(' and '.join(covariates))}[len(covariates)!=0]
    box.axes.set_ylabel(ylabel)
    box.set_title(roi_name)

    # Estimate p-values and add them to the figure
    import itertools
    pvals = []

    for i1, i2 in itertools.combinations(groups.keys(), 2):
        print df.head()
        print i1,i2, by, covariates, groups
        pval = _plot_significance(df, i1, i2, by, covariates, groups)
        pvals.append((pval, (i1,i2)))

    return pvals


def lmplot(y, x, data, covariates=['gender', 'age'], hue='apoe', ylim=None,
        savefig=None, facecolor='white', order=1, palette=None):

    if palette is None:
        palette = default_palette

    # Build a new table with only needed variables
    # y is renamed to roi to avoid potential issues with strange characters
    roi_name = y
    log.info('Dependent variable: %s'%y)
    variables = {x, y}
    if not hue is None:
        variables.add(hue)
    for c in covariates:
        variables.add(c)

    # check that all variables are found in columns
    for e in variables:
        if not e in data.columns.tolist():
            raise Exception('%s not found in columns!'%e)

    df = pd.DataFrame(data, columns=list(variables)).rename(columns={y:'y'})
    df = df.dropna()
    y = 'y'

    # Correct depending variable for covariates (if any) (df is modified)
    if len(covariates) != 0:
        df[y]  = correct(df, '%s ~ %s  + 1'%(y, '+'.join(covariates)))

    lm = sns.lmplot(x=x, y=y,  data=df, size=6.2, hue=hue, aspect=1.35, ci=90,
         truncate=True, sharex=False, sharey=False, order=order, palette=palette,
         scatter_kws={'linewidths':0.5, 's':22, 'edgecolors':'#333333' },line_kws={'linewidth':3})

    for patch in lm.axes[0,0].patches:
        clr = patch.get_facecolor()
        patch.set_edgecolor(palette2[clr])
    ax = lm.axes
    if ylim is None:
        ax[0,0].set_ylim([df[y].min(), df[y].max()])
    else:
        ax[0,0].set_ylim(ylim)
    ax[0,0].set_xlim([df[x].min(), df[x].max()])
    #ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    #ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('%s'%({False:'',
        True:' (corrected for %s)'\
            %(' and '.join(covariates))}[len(covariates)!=0]))
    ax[0,0].set_xlabel(x, fontsize=15, weight='bold')
    lm.fig.suptitle(roi_name)

    if not savefig is None:
        lm.savefig(savefig, facecolor=facecolor)
    return df


def _pivot(data, covariates, regions, region_colname, value_colname, index_colname):

    print regions
    data2 = data[data[region_colname].isin(regions)]
    piv = pd.pivot_table(data2, values=value_colname, index=data2.index, columns=region_colname)
    subject_list = list(data2.index.tolist())
    data2[index_colname] = subject_list
    columns = [index_colname]
    columns.extend(covariates)
    cov = pd.DataFrame(data2, columns=columns).drop_duplicates().set_index('subject')
    piv = piv.join(cov)

    return piv

def _unpivot(piv, regions, cov, index_colname='subject',
        region_colname='structure', value_colname='value'):
    table = []
    for i, row in piv.iterrows():
        for region in regions:
            r = [i, region, row[region]]
            table.append(r)
    columns = [index_colname, region_colname, value_colname]
    data = pd.DataFrame(table, columns=columns).set_index(index_colname).join(cov).dropna()
    return data

def hist_regions(data, regions=None, by='apoe', covariates=[], palette=None,
    zscores=False, ylim=None, region_colname='structure', value_colname='value',
    index_colname='subject'):
    from matplotlib import pyplot as plt

    # check that all variables are found in columns
    # and if so collect covariables
    columns = [by]
    columns.extend(covariates)
    for e in columns:
        if not e in data.columns.tolist():
            raise Exception('%s not found in columns!'%e)
    cov = pd.DataFrame(data, columns=columns)

    # first pivot region/value entries to individual columns
    if regions is None:
        regions = list(set(data[region_colname].tolist()))
    piv = _pivot(data, columns, regions, region_colname, value_colname,
        index_colname)

    # correct these region/value entries in their new individual columns
    import itertools
    pvals = []

    for region in regions:
        if len(covariates) != 0:
            piv = piv.rename(columns={region:'y'})
            piv['y'] = correct(piv, '%s ~ %s  + 1'%('y', '+'.join(covariates)))
            groups = list(set(piv[by].tolist()))
            for i1, i2 in itertools.combinations(groups, 2):
                pval = ttest_2samp('y', piv, i1, i2, by, covariates)
                import permutations as per
                pval = per.run('y', piv, by='apoe', contrast=(i1,i2),
                    groups= {i1:[i1], i2:[i2]})
                pval = min(pval, 1 - pval)
                pvals.append((region, (i1, i2), pval))
            piv = piv.rename(columns={'y':region})
        if zscores: # convert them to zscores if asked
            m = piv[region].mean()
            s = piv[region].std()
            piv[region] = (piv[region] - m)/s

    # unpivot (rebuild the region/value two initial columns)
    data2 = _unpivot(piv, regions, cov, index_colname, region_colname,
                    value_colname)

    # create a `rank` column to control the order of the bars (following order
    # given by `regions`)
    col = []
    for i, row in data2.iterrows():
        col.append(regions.index(row[region_colname]))
    data2['rank'] = col

    palette2 = None
    if palette is None:
        palette = default_palette
        palette2 = default_palette2

    # plot
    bar = sns.barplot(x='rank', y =value_colname, hue=by, data=data2,
        palette=palette, errwidth=2, ci=90)
    plt.setp(bar.patches, linewidth=2)

    # tune colors and figure aesthetics
    if not palette2 is None:
        for patch in bar.patches:
            clr = patch.get_facecolor()
            patch.set_edgecolor(palette2[clr])

    for item in bar.get_xticklabels():
        item.set_rotation(45)
        item.set_fontname('Liberation Sans')
    bar.set_xticklabels([e.replace('_',' ') for e in regions])
    if not ylim is None:
        bar.set_ylim(ylim)
    ylabel = '%s %s'\
                %(value_colname, {False:'',
                  True:' (corrected for %s)'\
                     %(' and '.join(covariates))}[len(covariates)!=0])
    bar.set(xlabel=region_colname, ylabel=ylabel)
    return pvals

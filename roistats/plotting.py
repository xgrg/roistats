
def _prefit_data(y, data, covariates):

    adj_model = 'y ~ %s + 1'%' + '.join(covariates)
    ycorr = pd.DataFrame(correct(data, adj_model), columns=[y])
    del data[y]
    data = data.join(ycorr)
    log.info('Fit model used for correction: %s'%adj_model)
    return data

def _plot_significance(data_dummies, x1, x2, groups, by, dummy_columns, covariates):
    import logging as log
    from matplotlib import pyplot as plt
    from statsmodels.formula.api import ols
    #dummy_columns = []
    #for x in [x1, x2]:
    #    for each in groups[x]:
    #        dummy_columns.append('%s_%s'%(by, each))
    formula = 'y ~ %s%s + 1'%(' + '.join(dummy_columns),
          {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    log.info('plot Used model for significance estimation: %s'%formula)
    fitted_model = ols(formula, data_dummies).fit()
    s1 = ['%s * %s_%s'%(1.0/len(groups[x1]), by, each) for each in groups[x1]]
    s2 = ['%s * %s_%s'%(1.0/len(groups[x2]), by, each) for each in groups[x2]]

    c = '%s - %s'%(' + '.join(s1), ' - '.join(s2))
    T = fitted_model.t_test(c)
    log.info('Used contrast: %s - p-value: %s'%(c, T.pvalue))

    ymin, ymax = data_dummies['y'].quantile([0.25, 0.75])
    # not good if the quantiles differ among groups
    ymax2 = ymax + (ymax - ymin) * 1.5
    ymin2 = ymin - (ymax - ymin) * 1.5

    y, h, col = ymax2 + (ymax2 - ymin2)*0.07, (ymax2 - ymin2)*0.02, 'k'
    i1 = sorted(groups.keys()).index(x1)
    i2 = sorted(groups.keys()).index(x2)
    plt.plot([i1, i1, i2, i2], [y, y+h, y+h, y], lw=1.5, c=col)
    opt = {'ha':'center', 'va':'bottom', 'color':col}

    if T.pvalue < 0.05:
        opt['weight'] = 'bold'
    plt.text((i1+i2)*.5, y + h, '%.3f'%(T.pvalue), **opt)

    return T.pvalue


import seaborn as sns
import pandas as pd
import logging as log
from __init__ import correct

def boxplot_region(y, data, groups, by='apoe', covariates=[], palette={}):
    '''`y` should be a variable name, `data` is the data, `covariates` lists
    the various nuisance factors, `by` is the variable setting the different
    groups and `groups` is a dictionary that sets the various groups
    (i.e. boxplots).

    ex: {'not HO': ['NC', 'HT'], 'HO': ['HO']}

    NB: the group splitting must be complete (all subjects from the given
    dataset must be covered, otherwise drop the non-desired groups first)'''

    # Create a column with a group index based on `groups`
    col = []
    d = pd.DataFrame(data, copy=True)
    for i, row in d.iterrows():
        for k, group in groups.items():
            if row[by] in group:
                col.append(k)
    d['_group'] = col

    # Build a new table `df` with only needed variables
    # y is renamed to y to avoid potential issues with strange characters
    roi_name = y
    variables = {'_group', y, by}
    for c in covariates:
        variables.add(c)
    df = pd.DataFrame(d, columns=list(variables)).rename(columns={y:'y'})
    df = df.dropna().sort_values(by='_group')
    y = 'y'
    log.info('Dependent variable: %s'%roi_name)

    # Create dummy variables which will be used to estimate p-values
    data_dummies = pd.get_dummies(df, columns=[by])
    del data_dummies['_group']
    dummy_columns = set(data_dummies.columns).difference(df.columns)

    # Correct depending variable for covariates (if any) (df is modified)
    if len(covariates) != 0:
        df = _prefit_data(y, df, covariates)

    # Plot for good
    if palette == {}:
        palette = {'HO':'#ff9999', 'HT':'#ffd699','NC':'#99ccff', 'not HO':'#99ccff',
           'female':'#ff9999', 'male':'#99ccff', 'apoe44':'#ff9999', 'apoe34':'#ffd699',
           'apoe33':'#99ccff', 'carriers': '#ffd699'}
    box = sns.boxplot(x='_group', y='y', data=df, showfliers=False,
                palette=palette)
    #box = sns.violinplot(x='_group', y='roi', data=df, palette=palette)
    box.axes.set_yticklabels(['%.2e'%x for x in box.axes.get_yticks()])
    xlabel = 'groups%s'\
                %{False:'',
                  True:' (corrected for %s)'\
                     %(' and '.join(covariates))}[len(covariates)!=0]
    box.axes.set_xlabel(xlabel, fontsize=15, weight='bold')
    box.axes.set_ylabel('')
    box.set_title(roi_name)

    # Estimate p-values and add them to the figure
    import itertools
    pvals, hdr = [], []

    for i1, i2 in itertools.combinations(sorted(groups.keys()), 2):
        pval = _plot_significance(data_dummies, i1, i2, groups, by,
                    dummy_columns, covariates)
        pvals.append(pval)
        hdr.append((i1,i2))

    return pvals, hdr


def lmplot(y, x, data, covariates=['gender', 'age'], hue='apoe', ylim=None,
        savefig=None, facecolor='white', order=1, palette={}):

    # Build a new table with only needed variables
    # y is renamed to roi to avoid potential issues with strange characters
    roi_name = y
    log.info('Dependent variable: %s'%y)
    variables = {x, y}
    if not hue is None:
        variables.add(hue)
    for c in covariates:
        variables.add(c)
    df = pd.DataFrame(data, columns=list(variables)).rename(columns={y:'y'})
    df = df.dropna()
    y = 'y'

    if len(covariates) != 0:
        df = _prefit_data(y, df, covariates)

    # Plotting for good
    if palette == {}:
        palette = {'HO':'#ff9999', 'HT':'#ffd699','NC':'#99ccff', 'not HO':'#99ccff',
           'female':'#ff9999', 'male':'#99ccff', 'apoe44':'#ff9999', 'apoe34':'#ffd699',
           'apoe33':'#99ccff', 'carriers': '#ffd699'}
    lm = sns.lmplot(x=x, y=y,  data=df, size=6.2, hue=hue, aspect=1.35, ci=90,
	 truncate=True, sharex=False, sharey=False, order=order, palette=palette)
    ax = lm.axes
    if ylim is None:
        ax[0,0].set_ylim([df[y].min(), df[y].max()])
    else:
        ax[0,0].set_ylim(ylim)
    ax[0,0].set_xlim([df[x].min(), df[x].max()])
    ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('')
    xlabel = '%s %s'%(x, {False:'',
        True:' (corrected for %s)'\
            %(' and '.join(covariates))}[len(covariates)!=0])
    ax[0,0].set_xlabel(xlabel, fontsize=15, weight='bold')
    lm.fig.suptitle(roi_name)

    if not savefig is None:
        lm.savefig(savefig, facecolor=facecolor)
    return df

import pandas as pd
import logging as log
from statsmodels.formula.api import ols


def estimate(data, dv, by='apoe', interaction=None,
        covariates = ['age', 'education'],
        groups = {'not HO': [0, 1], 'HO': [2], 'NC': [0], 'HT': [1], 'carriers': [1,2]},
        contrasts = {'recessive': ('HO', 'not HO'), 'dominant': ('carriers', 'NC'),
            'additive':('NC', 'HO')}):
        #contrasts = {'HT vs NC' : ('HT', 'NC')}):
    data_dummies = pd.get_dummies(data, columns=[by])
    dummy_columns = set(data_dummies.columns).difference(data.columns)

    #log.info('Dependent variable: %s'%dv)
    # Building the formula
    if not interaction is None:
        interaction_columns = []
        for each in dummy_columns:
            col = '%s_%s'%(each, interaction)
            interaction_columns.append(col)
            data_dummies[col] = data_dummies[each] * data_dummies[interaction]
        formula = '%s ~ %s%s + 1'%(dv, ' + '.join(interaction_columns),
        #    ' + '.join(dummy_columns),
          {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    else:
        formula = '%s ~ %s%s + 1'%(dv, ' + '.join(dummy_columns),
          {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    log.info('Used model for significance estimation: %s'%formula)
    fitted_model = ols(formula, data_dummies).fit()

    # Building the contrasts
    results = {}

    def __build_contrast__(model, x1, x2, groups, by, interaction):

        if interaction is None:
            s1 = ['%s * %s_%s'%(1.0/len(groups[x1]), by, each) \
                    for each in groups[x1]]
            s2 = ['%s * %s_%s'%(1.0/len(groups[x2]), by, each) \
                    for each in groups[x2]]
        else:
            s1 = ['%s * %s_%s_%s'%(1.0/len(groups[x1]), by, each, interaction) \
                    for each in groups[x1]]
            s2 = ['%s * %s_%s_%s'%(1.0/len(groups[x2]), by, each, interaction) \
                    for each in groups[x2]]

        c = '%s - %s'%(' + '.join(s1), ' - '.join(s2))
        T = fitted_model.t_test(c)
        log.info('Used contrast: %s - p-value: %s'\
                %(c, T.pvalue))
        return T

    for c_name, (x1, x2) in contrasts.items():
        T = __build_contrast__(fitted_model, x1, x2, groups, by, interaction)
        results[c_name] = T

    return results

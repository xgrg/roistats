import pandas as pd
import logging as log
from statsmodels.formula.api import ols


def estimate(data, dv, by='apoe', interaction=None,
        covariates = ['age', 'education'], first=True,
        groups = {'not HO': [0, 1], 'HO': [2], 'NC': [0], 'HT': [1], 'carriers': [1,2]},
        contrasts = {'recessive': ('HO', 'not HO'), 'dominant': ('carriers', 'NC'),
            'additive':('NC', 'HO')}):
        #contrasts = {'HT vs NC' : ('HT', 'NC')}):
    if not by is None:
        data_dummies = pd.get_dummies(data, columns=[by])
        dummy_columns = set(data_dummies.columns).difference(data.columns)
    else:
        data_dummies = data
        dummy_columns = []

    #log.info('Dependent variable: %s'%dv)
    # Building the formula
    if not interaction is None:
        interaction_columns = []
        for each in dummy_columns:
            col = '%s_%s'%(each, interaction)
            interaction_columns.append(col)
            data_dummies[col] = data_dummies[each] * data_dummies[interaction]
        if first:
            formula = '%s ~ %s + %s%s + 1'%(dv, ' + '.join(interaction_columns),
              ' + '.join(dummy_columns),
              {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
        else:
            formula = '%s ~ %s%s + 1'%(dv, ' + '.join(interaction_columns),
              #  ' + '.join(dummy_columns),
              {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    else:
        formula = '%s ~ %s%s + 1'%(dv, ' + '.join(dummy_columns),
          {False:'', True: ' + %s'%' + '.join(covariates)}[len(covariates)!=0])
    log.info('Used model for significance estimation: %s'%formula)
    fitted_model = ols(formula, data_dummies).fit()
    #print('%s:%s'%(formula, fitted_model.rsquared))

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
        T = model.t_test(c)
        log.info('Used contrast: %s - p-value: %s'\
                %(c, T.pvalue))
        return T

    def __build_contrast2__(model, x1):

        T = model.t_test('%s'%x1)
        log.info('Used contrast: %s - p-value: %s'\
                %(x1, T.pvalue))
        return T

    for c_name in contrasts:
        if isinstance(contrasts, dict):
            x = contrasts[c_name]
            if len(x) == 2:
                x1, x2 = x
                T = __build_contrast__(fitted_model, x1, x2, groups, by, interaction)
            elif len(x) == 1 or isinstance(x, str):
                T = __build_contrast2__(fitted_model, x)
            else:
                raise Exception('Incorrect contrast %s %s'%(c_name, x))
        else: # isinstance(contrasts, list) or isinstance(contrasts, tuple):
            T = __build_contrast2__(fitted_model, c_name)
        results[c_name] = T

    return results


def correct(df, model='roi ~ 1 + gender + age'): #educyears + apo' ):
    ''' Adjusts a variable according to a given model'''
    import numpy as np
    from statsmodels.formula.api import ols
    model = ols(model, data=df)
    depvar = model.endog_names
    test_scores = model.fit()
    err = test_scores.predict(df) - df[depvar]
    ycorr = np.mean(df[depvar]) - err

    return ycorr


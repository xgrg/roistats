from roistats import plotting
import pandas as pd
from roistats.contrasts import genotypes as g


regions = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
            'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L',
            'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
            'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
           'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
           'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
           'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
           'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L',
           'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R',
           'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L',
           'Cingulum_Mid_R', 'Cingulum_Post_L',
           'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R',
           'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
           'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L',
           'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
           'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
           'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
           'Parietal_Sup_L', 'Parietal_Sup_R',
           'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
           'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L',
           'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
           'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L',
           'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R',
           'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
           'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
           'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
           'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
           'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L',
           'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
           'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R',
           'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R',
           'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3',
           'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9',
           'Vermis_10']

table = []
from random import randrange
for s in xrange(100):
    age = randrange(45, 75)
    apoe = ['NC', 'HT', 'HO'][randrange(3)]
    sex = randrange(2)
    row = ['subject%s'%s, apoe, age, sex]
    row.extend([randrange(1000) for i in regions])
    table.append(row)
columns = ['ID', 'apoe', 'age', 'sex']
columns.extend(regions)
data = pd.DataFrame(table, columns=columns).set_index('ID')
cov = data[['apoe', 'age', 'sex']]
data = plotting._unpivot(data, regions, 'region', 'volume').join(cov)

plotting.hist(data, regions[:5], by='apoe', region_colname='region',
    value_colname='volume', covariates=['age','sex'])

data = plotting._pivot(data, cov, regions[1:2], 'region', 'volume')
plotting.boxplot(regions[1], data, covariates=['sex'], groups=['NC', 'HT', 'HO'])

_ = plotting.lmplot('age', regions[1], data, covariates=['sex'], hue='apoe',
    order=1)

g.estimate(data, regions[1], covariates=['age','sex'],
    contrasts={'dominant':('NC','carriers')},
    groups={'NC':['NC'], 'carriers':['HT','HO']})

from roistats import permutations as per
pval = per.run(regions[1], data, by='apoe', contrast=('NC','carriers'),
    groups={'NC':['NC'], 'carriers':['HT','HO']})
per.run_all_contrasts(regions[1], data)
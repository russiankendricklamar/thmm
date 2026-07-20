"""
Этап 10. Макрозависимые переходы
Reduced-form мультиномиальный логит вероятности перехода от макроковариат. Сигнал раннего предупреждения.
Выход: macro_transition_coef.csv

Исходный артефакт: macro_transition_coef.csv
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
npz = np.load("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/e371024d-c7fd-4090-8464-215bc87dc510/vdc7cd182_thmm_model_tr.npz", allow_pickle=True)
macro = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/3868fc29-c8e1-41f9-b512-e0964cbd547c/v170564c6_macro_panel.parquet")

hidden_full = npz['hidden']
dates_full = pd.to_datetime(npz['dates'])

macro_aligned = macro.reindex(dates_full)

df_trans = pd.DataFrame({
    'date': dates_full[:-1],
    's_t': hidden_full[:-1],
    's_t1': hidden_full[1:],
})
macro_t = macro_aligned.iloc[:-1].reset_index(drop=True)
df_trans = pd.concat([df_trans.reset_index(drop=True), macro_t], axis=1)

def to_family(s):
    if s in (0,2): return 0
    if s==1: return 1
    if s==3: return 2

df_trans['fam_t1'] = df_trans['s_t1'].map(to_family)
df_trans['stress_t1'] = (df_trans['fam_t1']>0).astype(int)

macro_cols = ['keyrate','inflation','usdrub_vol','usdrub_ret20','keyrate_chg60']

Z = df_trans[macro_cols].copy()
mu_z = Z.mean(); sd_z = Z.std()
Zs = (Z - mu_z)/sd_z
Zs.columns = [c+'_z' for c in macro_cols]
df_full = pd.concat([df_trans, Zs], axis=1)

origin_dummies = pd.get_dummies(df_full['s_t'], prefix='origin', drop_first=True).astype(float)
X_pooled = pd.concat([Zs, origin_dummies], axis=1)
X_pooled = sm.add_constant(X_pooled)
y_pooled = df_full['stress_t1']

model_pooled = sm.Logit(y_pooled, X_pooled.astype(float))
res_pooled = model_pooled.fit(disp=0, maxiter=200)

or_pooled = np.exp(res_pooled.params)
ci = res_pooled.conf_int()
or_ci = np.exp(ci)
pooled_table = pd.DataFrame({
    'term': res_pooled.params.index,
    'coef': res_pooled.params.values,
    'odds_ratio': or_pooled.values,
    'p_value': res_pooled.pvalues.values,
    'or_ci_low': or_ci[0].values,
    'or_ci_high': or_ci[1].values
})

per_origin_results = {}
for origin in [0,1,2,3]:
    sub = df_full[df_full['s_t']==origin]
    y = sub['stress_t1']
    if y.nunique()<2:
        continue
    X = sm.add_constant(sub[Zs.columns].astype(float))
    try:
        m = sm.Logit(y, X)
        r = m.fit(disp=0, maxiter=300)
        per_origin_results[origin] = r
    except Exception as e:
        print(f"origin {origin} failed: {e}")

per_origin_coef_rows = []
for origin in [0,1,2,3]:
    sub = df_full[df_full['s_t']==origin]
    y = sub['stress_t1'].values
    X = sub[Zs.columns].values.astype(float)
    if len(np.unique(y))<2:
        continue
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000)
    clf.fit(X, y)
    coefs = clf.coef_[0]
    n_stress = int(y.sum()); n_tot = len(y)
    for factor, c in zip(macro_cols, coefs):
        per_origin_coef_rows.append(dict(origin=f"R{origin+1}", factor=factor, coef=c, odds_ratio=np.exp(c),
                                          n_obs=n_tot, n_stress=n_stress, stress_rate=n_stress/n_tot))
per_origin_df = pd.DataFrame(per_origin_coef_rows)

origin_models = {}
origin_models[0] = ('sm', per_origin_results[0])
origin_models[1] = ('sm', per_origin_results[1])

for origin in [2,3]:
    sub = df_full[df_full['s_t']==origin]
    y = sub['stress_t1'].values
    X = sub[Zs.columns].values.astype(float)
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000)
    clf.fit(X, y)
    origin_models[origin] = ('sk', clf)

pooled_macro = pooled_table[pooled_table['term'].isin(Zs.columns)].copy()
pooled_macro['origin'] = 'ALL (pooled, controlling for origin)'
pooled_macro['factor'] = pooled_macro['term'].str.replace('_z','')
pooled_macro['n_obs'] = len(df_full)
pooled_macro['n_stress'] = int(df_full['stress_t1'].sum())
pooled_macro['stress_rate'] = df_full['stress_t1'].mean()
pooled_macro = pooled_macro[['origin','factor','coef','odds_ratio','p_value','n_obs','n_stress','stress_rate']]

per_origin_full_rows = []
for origin in [0,1]:
    r = per_origin_results[origin]
    sub = df_full[df_full['s_t']==origin]
    n_tot = len(sub); n_stress = int(sub['stress_t1'].sum())
    for factor in macro_cols:
        term = factor+'_z'
        per_origin_full_rows.append(dict(origin=f"R{origin+1}", factor=factor, coef=r.params[term],
                                          odds_ratio=np.exp(r.params[term]), p_value=r.pvalues[term],
                                          n_obs=n_tot, n_stress=n_stress, stress_rate=n_stress/n_tot))
for origin in [2,3]:
    sub_rows = per_origin_df[per_origin_df['origin']==f"R{origin+1}"]
    for _, row in sub_rows.iterrows():
        per_origin_full_rows.append(dict(origin=row['origin'], factor=row['factor'], coef=row['coef'],
                                          odds_ratio=row['odds_ratio'], p_value=np.nan,
                                          n_obs=row['n_obs'], n_stress=row['n_stress'], stress_rate=row['stress_rate']))
per_origin_full = pd.DataFrame(per_origin_full_rows)

macro_transition_coef = pd.concat([pooled_macro, per_origin_full], ignore_index=True)
macro_transition_coef['note'] = np.where(macro_transition_coef['origin'].isin(['R3','R4']),
                                          'L2-regularized (near-separated origin, p-value not valid)', '')
macro_transition_coef.to_csv('macro_transition_coef.csv', index=False)

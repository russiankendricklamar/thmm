import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genpareto
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

npz = np.load("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/e371024d-c7fd-4090-8464-215bc87dc510/vdc7cd182_thmm_model_tr.npz", allow_pickle=True)
bt_hmm = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/f7ee217e-c7d3-45f9-87d3-397138454ca4/vfd2071d4_backtest_hmm.parquet")
bt_bench = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/637ec45b-cc0a-43a1-aabb-c11a0151bdc5/v2df90182_backtest_bench.parquet")

port_ret = npz['port_ret']
r = port_ret

n_oos = len(bt_hmm)
assert bt_hmm['idx'].values[0] == 500

real = bt_hmm['real'].values

def gaussian_es(mu, sigma, alpha):
    z = stats.norm.ppf(alpha)
    return mu - sigma * stats.norm.pdf(z) / alpha

def t_es_raw(nu, alpha):
    tq = stats.t.ppf(alpha, nu)
    return -stats.t.pdf(tq, nu) / alpha * (nu + tq**2) / (nu - 1)

def evt_var_es(sample, alpha, q_thresh=0.90):
    losses = -sample
    u = np.quantile(losses, q_thresh)
    exceed = losses[losses > u] - u
    Nu = len(exceed)
    n = len(losses)
    xi, loc, beta = genpareto.fit(exceed, floc=0)
    var_loss = u + beta / xi * (((n / Nu) * alpha) ** (-xi) - 1)
    if xi < 1:
        es_loss = (var_loss + (beta - xi * u)) / (1 - xi)
    else:
        es_loss = np.nan
    return -var_loss, -es_loss

results = {
    'gauss': {'VaR_5': [], 'ES_5': [], 'VaR_1': [], 'ES_1': []},
    'emp':   {'VaR_5': [], 'ES_5': [], 'VaR_1': [], 'ES_1': []},
    'evt':   {'VaR_5': [], 'ES_5': [], 'VaR_1': [], 'ES_1': []},
}

for i in range(n_oos):
    window = r[:500 + i]
    mu_i = window.mean()
    sd_i = window.std(ddof=0)
    for al, tag in [(0.05, '5'), (0.01, '1')]:
        z = stats.norm.ppf(al)
        var_g = mu_i + sd_i * z
        es_g = gaussian_es(mu_i, sd_i, al)
        results['gauss'][f'VaR_{tag}'].append(var_g)
        results['gauss'][f'ES_{tag}'].append(es_g)

        var_e = np.quantile(window, al)
        tail = window[window <= var_e]
        es_e = tail.mean() if len(tail) > 0 else var_e
        results['emp'][f'VaR_{tag}'].append(var_e)
        results['emp'][f'ES_{tag}'].append(es_e)

        var_v, es_v = evt_var_es(window, al)
        results['evt'][f'VaR_{tag}'].append(var_v)
        results['evt'][f'ES_{tag}'].append(es_v)

for k in results:
    for kk in results[k]:
        results[k][kk] = np.array(results[k][kk])

garch_var5 = np.full(n_oos, np.nan)
garch_var1 = np.full(n_oos, np.nan)
garch_es5 = np.full(n_oos, np.nan)
garch_es1 = np.full(n_oos, np.nan)

refit_every = 5
cur_mu = cur_scale_sd = cur_nu = None

for i in range(n_oos):
    if i % refit_every == 0:
        window = r[:500 + i] * 100
        am = arch_model(window, mean='Constant', vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off')
        fc = res.forecast(horizon=1, reindex=False)
        sigma1 = np.sqrt(fc.variance.values[-1, 0]) / 100
        mu1 = res.params['mu'] / 100
        nu_g = res.params['nu']
        cur_mu, cur_scale_sd, cur_nu = mu1, sigma1, nu_g
    c = cur_scale_sd * np.sqrt((cur_nu - 2) / cur_nu)
    for al, var_arr, es_arr in [(0.05, garch_var5, garch_es5), (0.01, garch_var1, garch_es1)]:
        tq = stats.t.ppf(al, cur_nu)
        var_arr[i] = cur_mu + c * tq
        es_raw = t_es_raw(cur_nu, al)
        es_arr[i] = cur_mu + c * es_raw

es_table = pd.DataFrame({
    'idx': bt_hmm['idx'].values,
    'date': bt_hmm['date'].values,
    'real': real,
    'VaR_tHMM_5': bt_hmm['VaR_5'].values, 'ES_tHMM_5': bt_hmm['ES_5'].values,
    'VaR_tHMM_1': bt_hmm['VaR_1'].values, 'ES_tHMM_1': bt_hmm['ES_1'].values,
    'VaR_gauss_5': results['gauss']['VaR_5'], 'ES_gauss_5': results['gauss']['ES_5'],
    'VaR_gauss_1': results['gauss']['VaR_1'], 'ES_gauss_1': results['gauss']['ES_1'],
    'VaR_emp_5': results['emp']['VaR_5'], 'ES_emp_5': results['emp']['ES_5'],
    'VaR_emp_1': results['emp']['VaR_1'], 'ES_emp_1': results['emp']['ES_1'],
    'VaR_evt_5': results['evt']['VaR_5'], 'ES_evt_5': results['evt']['ES_5'],
    'VaR_evt_1': results['evt']['VaR_1'], 'ES_evt_1': results['evt']['ES_1'],
    'VaR_garch_5': garch_var5, 'ES_garch_5': garch_es5,
    'VaR_garch_1': garch_var1, 'ES_garch_1': garch_es1,
})

models_map = {
    't-HMM':    ('VaR_tHMM_1', 'ES_tHMM_1', 'VaR_tHMM_5', 'ES_tHMM_5'),
    'Gaussian': ('VaR_gauss_1', 'ES_gauss_1', 'VaR_gauss_5', 'ES_gauss_5'),
    'Empirical':('VaR_emp_1', 'ES_emp_1', 'VaR_emp_5', 'ES_emp_5'),
    'EVT-GPD':  ('VaR_evt_1', 'ES_evt_1', 'VaR_evt_5', 'ES_evt_5'),
    'GARCH-t':  ('VaR_garch_1', 'ES_garch_1', 'VaR_garch_5', 'ES_garch_5'),
}

def z2_stat(real, VaR, ES, alpha):
    T = len(real)
    breach = real < VaR
    contrib = np.where(breach, real / ES, 0.0)
    return contrib.sum() / (T * alpha) - 1.0, breach.sum()

def bootstrap_z2_pvalue(real, VaR, ES, alpha, n_boot=20000, seed=42):
    T = len(real)
    rng = np.random.default_rng(seed)
    Z2_obs, n_breach = z2_stat(real, VaR, ES, alpha)
    idxs = rng.integers(0, T, size=(n_boot, T))
    real_b = real[idxs]; VaR_b = VaR[idxs]; ES_b = ES[idxs]
    breach_b = real_b < VaR_b
    contrib_b = np.where(breach_b, real_b / ES_b, 0.0)
    Z2_boot = contrib_b.sum(axis=1) / (T * alpha) - 1.0
    p_two = 2 * min((Z2_boot <= 0).mean(), (Z2_boot >= 0).mean())
    p_two = min(p_two, 1.0)
    ci_lo, ci_hi = np.quantile(Z2_boot, [0.025, 0.975])
    return Z2_obs, p_two, n_breach, ci_lo, ci_hi

rows = []
for model, (v1, e1, v5, e5) in models_map.items():
    for alpha, vcol, ecol, tag in [(0.01, v1, e1, '99%'), (0.05, v5, e5, '95%')]:
        VaR = es_table[vcol].values
        ES = es_table[ecol].values
        Z2, p, nb, lo, hi = bootstrap_z2_pvalue(real, VaR, ES, alpha)
        if p < 0.05 and Z2 > 0:
            verdict = 'занижает риск (ES не покрывает потери)'
        elif p < 0.05 and Z2 < 0:
            verdict = 'завышает риск (консервативен)'
        else:
            verdict = 'калиброван'
        rows.append({'model': model, 'level': tag, 'alpha': alpha, 'n_breach': nb, 'Z2': Z2, 'p_value': p,
                     'CI_lo': lo, 'CI_hi': hi, 'verdict': verdict})

es_backtest = pd.DataFrame(rows)
es_backtest_out = es_backtest[['model', 'level', 'alpha', 'n_breach', 'Z2', 'p_value', 'CI_lo', 'CI_hi', 'verdict']].copy()
es_backtest_out.to_csv('es_backtest.csv', index=False)

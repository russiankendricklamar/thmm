import numpy as np
import pandas as pd
import pickle
from scipy import stats
from scipy.special import digamma, gammaln
from scipy.linalg import solve_triangular

# Load data
npz = np.load("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/e371024d-c7fd-4090-8464-215bc87dc510/vdc7cd182_thmm_model_tr.npz", allow_pickle=True)
regime_summary = pd.read_csv("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/aaee7f13-f132-486b-aeea-e9b06c8d9f8d/vf56d0deb_regime_summary_tr.csv")

segs = npz['segs']
nu = npz['nu']
mu_raw = npz['mu_raw']
Sigma_raw = npz['Sigma_raw']
port_ret = npz['port_ret']
stationary = npz['stationary']
w = npz['w']
w_eq = w.copy()

with open("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/5dea4288-69bc-46c8-95b7-f1bba0245680/va03d6748_boot_results_raw.pkl", 'rb') as f:
    boot_results = pickle.load(f)

ok_results = [r for r in boot_results if r[2]]

def t_es_raw(nu, alpha):
    tq = stats.t.ppf(alpha, nu)
    return -stats.t.pdf(tq, nu)/alpha * (nu+tq**2)/(nu-1)

def sample_mvt(mu, Sigma, nu, n, rng):
    d = len(mu)
    g = stats.chi2.rvs(df=nu, size=n, random_state=rng)
    Z = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    X = mu + Z * np.sqrt(nu/g)[:,None]
    return X

def mixture_var_es(mu_k_arr, Sigma_k_arr, nu_k_arr, stat_probs, w_vec, alpha, n_sim=200000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K_ = len(nu_k_arr)
    comp = rng.choice(K_, size=n_sim, p=stat_probs/stat_probs.sum())
    port = np.empty(n_sim)
    for k in range(K_):
        mask = comp==k
        nk = mask.sum()
        if nk==0: continue
        Xs_ = sample_mvt(mu_k_arr[k], Sigma_k_arr[k], nu_k_arr[k], nk, rng)
        port[mask] = Xs_ @ w_vec
    VaR = np.quantile(port, alpha)
    ES = port[port<=VaR].mean()
    return VaR, ES

nu_boot = np.full((len(ok_results),4), np.nan)
vol_boot = np.full((len(ok_results),4), np.nan)
var95_boot = np.full(len(ok_results), np.nan)
es95_boot = np.full(len(ok_results), np.nan)
var99_boot = np.full(len(ok_results), np.nan)
es99_boot = np.full(len(ok_results), np.nan)
breach99_boot = np.full(len(ok_results), np.nan)

rng_mix = np.random.default_rng(777)
for i, (seed, info_b, ok) in enumerate(ok_results):
    nu_boot[i] = info_b['nu']
    vol_boot[i] = info_b['vols_ann_pct']
    stat_b = info_b['stationary']
    VaR95, ES95 = mixture_var_es(info_b['mu'], info_b['Sigma'], info_b['nu'], stat_b, w_eq, 0.05, n_sim=50000, rng=rng_mix)
    VaR99, ES99 = mixture_var_es(info_b['mu'], info_b['Sigma'], info_b['nu'], stat_b, w_eq, 0.01, n_sim=50000, rng=rng_mix)
    var95_boot[i]=VaR95*100; es95_boot[i]=ES95*100
    var99_boot[i]=VaR99*100; es99_boot[i]=ES99*100
    breach99_boot[i] = (port_ret < VaR99).mean()*100

def ci95(arr):
    return np.percentile(arr, [2.5, 97.5])

rows = []
regime_labels = ['R1','R2','R3','R4']
main_nu = nu
main_vol = regime_summary['port_vol_ann_%'].values

for k in range(4):
    lo, hi = ci95(nu_boot[:,k])
    rows.append({'parameter': f'nu_{regime_labels[k]}', 'point_estimate': main_nu[k], 'CI_lo':lo, 'CI_hi':hi, 'n_boot':len(ok_results)})
for k in range(4):
    lo, hi = ci95(vol_boot[:,k])
    rows.append({'parameter': f'vol_ann_pct_{regime_labels[k]}', 'point_estimate': main_vol[k], 'CI_lo':lo, 'CI_hi':hi, 'n_boot':len(ok_results)})

VaR95_main, ES95_main = mixture_var_es(mu_raw, Sigma_raw, nu, stationary, w_eq, 0.05, n_sim=300000, rng=np.random.default_rng(1))
VaR99_main, ES99_main = mixture_var_es(mu_raw, Sigma_raw, nu, stationary, w_eq, 0.01, n_sim=300000, rng=np.random.default_rng(2))

lo,hi = ci95(var95_boot); rows.append({'parameter':'VaR_95_pct_unconditional','point_estimate':VaR95_main*100,'CI_lo':lo,'CI_hi':hi,'n_boot':len(ok_results)})
lo,hi = ci95(es95_boot); rows.append({'parameter':'ES_95_pct_unconditional','point_estimate':ES95_main*100,'CI_lo':lo,'CI_hi':hi,'n_boot':len(ok_results)})
lo,hi = ci95(var99_boot); rows.append({'parameter':'VaR_99_pct_unconditional','point_estimate':VaR99_main*100,'CI_lo':lo,'CI_hi':hi,'n_boot':len(ok_results)})
lo,hi = ci95(es99_boot); rows.append({'parameter':'ES_99_pct_unconditional','point_estimate':ES99_main*100,'CI_lo':lo,'CI_hi':hi,'n_boot':len(ok_results)})

breach99_main = (port_ret < VaR99_main).mean()*100
lo,hi = ci95(breach99_boot); rows.append({'parameter':'breach_rate_99_pct','point_estimate':breach99_main,'CI_lo':lo,'CI_hi':hi,'n_boot':len(ok_results)})

bootstrap_summary = pd.DataFrame(rows)
bootstrap_summary.to_csv('bootstrap_ci.csv', index=False)

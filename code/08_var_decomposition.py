import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, gammaln
from scipy.linalg import solve_triangular

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
npz = np.load("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/e371024d-c7fd-4090-8464-215bc87dc510/vdc7cd182_thmm_model_tr.npz", allow_pickle=True)

segs = npz['segs']
nu = npz['nu']
mu_raw = npz['mu_raw']
Sigma_raw = npz['Sigma_raw']
hidden = npz['hidden']
port_ret = npz['port_ret']
stationary = npz['stationary']
Xmean = npz['Xmean']
Xstd = npz['Xstd']
w = npz['w']

w_eq = w.copy()
K = 4

def t_es_raw(nu, alpha):
    tq = stats.t.ppf(alpha, nu)
    return -stats.t.pdf(tq, nu)/alpha * (nu+tq**2)/(nu-1)

def var_es_decomposition(mu_vec_w, Sigma_w, nu_w, w_vec, alpha):
    sigma_p = np.sqrt(w_vec @ Sigma_w @ w_vec)
    mu_p = w_vec @ mu_vec_w
    q_alpha = stats.t.ppf(alpha, nu_w)
    es_std = t_es_raw(nu_w, alpha)

    VaR_p = mu_p + q_alpha * sigma_p
    ES_p = mu_p + es_std * sigma_p

    Sigma_w_vec = Sigma_w @ w_vec
    component_VaR = w_vec * mu_vec_w + w_vec * Sigma_w_vec / sigma_p * q_alpha
    component_ES  = w_vec * mu_vec_w + w_vec * Sigma_w_vec / sigma_p * es_std

    assert np.isclose(component_VaR.sum(), VaR_p, atol=1e-10)
    assert np.isclose(component_ES.sum(), ES_p, atol=1e-10)

    return {
        'VaR_p': VaR_p, 'ES_p': ES_p, 'mu_p': mu_p, 'sigma_p': sigma_p,
        'component_VaR': component_VaR, 'component_ES': component_ES,
    }

def fit_single_t(X, nu_init=8.0, n_iter=200, tol=1e-8):
    from scipy.optimize import minimize_scalar
    n, d = X.shape
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    nu = nu_init
    ll_prev = -np.inf
    for it in range(n_iter):
        Sigma_inv = np.linalg.inv(Sigma)
        diff = X - mu
        maha = np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff)
        u = (nu + d) / (nu + maha)
        mu_new = (u[:,None]*X).sum(axis=0) / u.sum()
        diff2 = X - mu_new
        Sigma_new = (u[:,None,None]*np.einsum('ij,ik->ijk', diff2, diff2)).sum(axis=0) / n
        def negloglik_nu(nu_):
            if nu_<=2.01: return 1e10
            ll = (stats.multivariate_t.logpdf(X, loc=mu_new, shape=Sigma_new, df=nu_)).sum()
            return -ll
        opt = minimize_scalar(negloglik_nu, bounds=(2.02,200), method='bounded')
        nu_new = opt.x
        ll = -negloglik_nu(nu_new)
        mu, Sigma, nu = mu_new, Sigma_new, nu_new
        if abs(ll-ll_prev) < tol*abs(ll_prev if ll_prev!=-np.inf else 1):
            break
        ll_prev = ll
    return mu, Sigma, nu, ll

X = panel[list(segs)].values
mu_full, Sigma_full, nu_full, ll_full = fit_single_t(X)

def decomp_table(mu_vec, Sigma, nu_, w_vec, alpha, seg_names):
    r = var_es_decomposition(mu_vec, Sigma, nu_, w_vec, alpha)
    df = pd.DataFrame({
        'segment': seg_names,
        'weight': w_vec,
        'component_VaR_pct': r['component_VaR']*100,
        'component_ES_pct': r['component_ES']*100,
    })
    df['share_VaR_%'] = df['component_VaR_pct']/r['VaR_p']/100*100
    df['share_ES_%'] = df['component_ES_pct']/r['ES_p']/100*100
    return df, r['VaR_p']*100, r['ES_p']*100

all_rows = []

for alpha, tag in [(0.05,'95%'), (0.01,'99%')]:
    df_u, VaRp, ESp = decomp_table(mu_full, Sigma_full, nu_full, w_eq, alpha, segs)
    df_u['scope']='Unconditional'; df_u['alpha']=tag
    df_u['VaR_portfolio_pct']=VaRp; df_u['ES_portfolio_pct']=ESp
    all_rows.append(df_u)

for k in range(K):
    for alpha, tag in [(0.05,'95%'), (0.01,'99%')]:
        df_k, VaRp, ESp = decomp_table(mu_raw[k], Sigma_raw[k], nu[k], w_eq, alpha, segs)
        df_k['scope']=f'R{k+1}'; df_k['alpha']=tag
        df_k['VaR_portfolio_pct']=VaRp; df_k['ES_portfolio_pct']=ESp
        all_rows.append(df_k)

var_decomposition = pd.concat(all_rows, ignore_index=True)
var_decomposition = var_decomposition[['scope','alpha','segment','weight','component_VaR_pct','share_VaR_%',
                                       'component_ES_pct','share_ES_%','VaR_portfolio_pct','ES_portfolio_pct']]

var_decomposition.to_csv('var_decomposition.csv', index=False)

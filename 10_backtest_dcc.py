"""
Этап 9. Многомерный бенчмарк DCC-GARCH
GARCH(1,1)-t по 7 сегментам + DCC-динамика корреляций, портфельный VaR. Скользящий OOS-бэктест.
Выход: backtest_dcc.parquet, dcc_coverage.csv, backtest_all_models_v2.csv

Исходный артефакт: backtest_dcc.parquet
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

import numpy as np
import pandas as pd
import time
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import t as student_t
import warnings
warnings.filterwarnings("ignore")

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
bt_hmm = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/f7ee217e-c7d3-45f9-87d3-397138454ca4/vfd2071d4_backtest_hmm.parquet")

segs = panel.columns[:7].tolist()
Rpct = panel[segs].values * 100.0
w = np.ones(7) / 7.0
T, N = Rpct.shape
n_start = 500
n_total_bt = 903
refit_every = 5


def fit_uni_garch(y):
    am = arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = am.fit(disp='off', show_warning=False)
    p = res.params
    return dict(mu=p['mu'], omega=p['omega'], alpha=p['alpha[1]'], beta=p['beta[1]'], nu=p['nu'],
                sigma2_path=res.conditional_volatility**2, resid_path=res.resid)


def dcc_negloglik(params, E, Qbar):
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    T_, N_ = E.shape
    Q = Qbar.copy()
    ll = 0.0
    for tt in range(T_):
        d = np.sqrt(np.diag(Q))
        Rt = Q / np.outer(d, d)
        try:
            sign, logdet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e10
            Rinv = np.linalg.inv(Rt)
        except np.linalg.LinAlgError:
            return 1e10
        z = E[tt]
        ll += -0.5 * (logdet + z @ Rinv @ z)
        e_prev = E[tt][:, None]
        Q = (1 - a - b) * Qbar + a * (e_prev @ e_prev.T) + b * Q
    return -ll


def fit_portfolio_dof(z):
    try:
        nu_hat, loc_hat, scale_hat = student_t.fit(z, floc=0, fscale=1)
        nu_hat = np.clip(nu_hat, 2.5, 200)
    except Exception:
        nu_hat = 5.0
    return nu_hat


block_starts = list(range(n_start, n_total_bt, refit_every))
oos_records = []
t0_start = time.time()

for bi, t0 in enumerate(block_starts):
    block_end = min(t0 + refit_every, n_total_bt)
    fits = {}
    for i, seg in enumerate(segs):
        y = Rpct[:t0, i]
        fits[seg] = fit_uni_garch(y)
    sig2_state = np.array([fits[seg]['sigma2_path'][-1] for seg in segs])
    res_state = np.array([fits[seg]['resid_path'][-1] for seg in segs])
    omega = np.array([fits[seg]['omega'] for seg in segs])
    alpha = np.array([fits[seg]['alpha'] for seg in segs])
    beta = np.array([fits[seg]['beta'] for seg in segs])
    mu_c = np.array([fits[seg]['mu'] for seg in segs])
    nu_c = np.array([fits[seg]['nu'] for seg in segs])

    E_in = np.column_stack([fits[seg]['resid_path'] / np.sqrt(fits[seg]['sigma2_path']) for seg in segs])
    Qbar = np.corrcoef(E_in.T)
    res_dcc = minimize(dcc_negloglik, x0=[0.03, 0.90], args=(E_in, Qbar),
                       bounds=[(1e-4, 0.3), (1e-4, 0.999 - 1e-4)], method='L-BFGS-B')
    a_dcc, b_dcc = res_dcc.x

    Nseg = len(segs)
    Q_path_in = np.empty((E_in.shape[0], Nseg, Nseg))
    Q_path_in[0] = Qbar
    for tt in range(1, E_in.shape[0]):
        e_prev = E_in[tt - 1][:, None]
        Q_path_in[tt] = (1 - a_dcc - b_dcc) * Qbar + a_dcc * (e_prev @ e_prev.T) + b_dcc * Q_path_in[tt - 1]
    Q_state = Q_path_in[-1]

    sig2_in = np.column_stack([fits[seg]['sigma2_path'] for seg in segs])
    y_in = Rpct[:t0, :]
    port_mu_c = w @ mu_c
    z_port_list = []
    for tt in range(t0):
        Qt = Q_path_in[tt]
        d_t = np.sqrt(np.diag(Qt))
        Rt = Qt / np.outer(d_t, d_t)
        Dt = np.diag(np.sqrt(sig2_in[tt]))
        Ht = Dt @ Rt @ Dt
        pv = w @ Ht @ w
        if pv <= 0:
            continue
        z = (y_in[tt].dot(w) - port_mu_c) / np.sqrt(pv)
        z_port_list.append(z)
    z_port_arr = np.array(z_port_list)
    dof_port = fit_portfolio_dof(z_port_arr)

    for t in range(t0, block_end):
        sig2_t = omega + alpha * res_state ** 2 + beta * sig2_state
        e_prev = (res_state / np.sqrt(sig2_state))[:, None]
        Q_t = (1 - a_dcc - b_dcc) * Qbar + a_dcc * (e_prev @ e_prev.T) + b_dcc * Q_state
        d_t = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(d_t, d_t)
        D_t = np.diag(np.sqrt(sig2_t))
        H_t = D_t @ R_t @ D_t
        port_var_pct2 = w @ H_t @ w
        port_sigma_pct = np.sqrt(port_var_pct2)
        port_mu_pct = w @ mu_c

        oos_records.append(dict(idx=t, port_mu_pct=port_mu_pct, port_sigma_pct=port_sigma_pct,
                                dof=dof_port, a_dcc=a_dcc, b_dcc=b_dcc,
                                mean_corr=float((R_t.sum() - N) / (N * (N - 1)))))

        y_t = Rpct[t, :]
        res_t = y_t - mu_c
        sig2_state = sig2_t
        res_state = res_t
        Q_state = Q_t

print("total time:", time.time() - t0_start)
oos_df = pd.DataFrame(oos_records)


def t_es_standard(alpha, nu):
    tq = student_t.ppf(alpha, nu)
    pdf_q = student_t.pdf(tq, nu)
    es = -(pdf_q / alpha) * (nu + tq ** 2) / (nu - 1)
    return tq, es


oos_df['loc'] = oos_df['port_mu_pct'] / 100.0
oos_df['scale'] = oos_df['port_sigma_pct'] / 100.0

for level, alpha in [(5, 0.05), (1, 0.01)]:
    tq_arr, es_arr = [], []
    for nu in oos_df['dof']:
        tq, es = t_es_standard(alpha, nu)
        tq_arr.append(tq)
        es_arr.append(es)
    oos_df[f'VaR_dcc_{level}'] = oos_df['loc'] + oos_df['scale'] * np.array(tq_arr)
    oos_df[f'ES_dcc_{level}'] = oos_df['loc'] + oos_df['scale'] * np.array(es_arr)

oos_df = oos_df.merge(bt_hmm[['idx', 'real']], on='idx', how='left')
for level in [5, 1]:
    oos_df[f'breach_dcc_{level}'] = (oos_df['real'] < oos_df[f'VaR_dcc_{level}']).astype(int)

backtest_dcc = oos_df[['idx', 'real', 'VaR_dcc_5', 'ES_dcc_5', 'breach_dcc_5', 'VaR_dcc_1', 'ES_dcc_1', 'breach_dcc_1',
                        'port_sigma_pct', 'dof', 'a_dcc', 'b_dcc', 'mean_corr']].copy()
backtest_dcc.to_parquet('backtest_dcc.parquet')

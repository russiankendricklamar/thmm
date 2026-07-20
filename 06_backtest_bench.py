"""
Этап 5. Бенчмарки EVT-GPD и GARCH-t
Пики над порогом (genpareto) и GARCH(1,1)-t (arch) на тех же OOS-днях.
Выход: backtest_bench.parquet, backtest_all_models.csv

Исходный артефакт: backtest_bench.parquet
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto, t as student_t, norm, chi2
from arch import arch_model
import time

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
order = ["OFZ_short", "OFZ_med", "OFZ_long", "Corp_L1", "Corp_L2", "Corp_L3", "MBS"]
X = panel[order].values
T, dime = X.shape
w = np.ones(dime) / dime
port = X @ w

START = 500


def evt_var_es(losses, alpha, u_q=0.90):
    u = np.quantile(losses, u_q)
    exc = losses[losses > u] - u
    if len(exc) < 20:
        return np.quantile(losses, 1 - alpha), np.nan
    xi, loc, beta = genpareto.fit(exc, floc=0)
    n = len(losses)
    Nu = len(exc)
    p = alpha
    VaR_L = u + (beta / xi) * (((n / Nu) * p) ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * np.log((n / Nu) * p)
    if xi < 1:
        ES_L = (VaR_L + beta - xi * u) / (1 - xi)
    else:
        ES_L = np.nan
    return -VaR_L, -ES_L


def garch_var(ret_hist, alpha, refit_model=None):
    r = ret_hist * 100
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=1, reindex=False)
    mu = res.params.get("mu", 0.0)
    sig = np.sqrt(fc.variance.values[-1, 0])
    nu = res.params.get("nu", 8.0)
    q = student_t.ppf(alpha, df=nu) / np.sqrt(nu / (nu - 2))
    VaR = (mu + sig * q) / 100.0
    x = student_t.ppf(alpha, df=nu)
    es_std = -(student_t.pdf(x, df=nu) / alpha) * ((nu + x ** 2) / (nu - 1)) / np.sqrt(nu / (nu - 2))
    ES = (mu + sig * es_std) / 100.0
    return VaR, ES


t0 = time.time()
recs = []
garch_cache = {}
for i in range(START, T - 1):
    hist = port[:i]
    real = port[i]
    rec = {"idx": i, "real": real}
    for a in [0.05, 0.01]:
        lvl = int(a * 100)
        ev_v, ev_es = evt_var_es(-hist, a)
        rec[f"VaR_evt_{lvl}"] = ev_v
        rec[f"breach_evt_{lvl}"] = int(real < ev_v)
        key = (i - START) // 5
        if (a, key) not in garch_cache:
            try:
                garch_cache[(a, key)] = garch_var(hist, a)
            except Exception as e:
                garch_cache[(a, key)] = (np.nan, np.nan)
        gv, ges = garch_cache[(a, key)]
        rec[f"VaR_garch_{lvl}"] = gv
        rec[f"breach_garch_{lvl}"] = int(real < gv) if np.isfinite(gv) else 0
    recs.append(rec)

bench = pd.DataFrame(recs)
print(f"done in {time.time() - t0:.0f}s, {len(bench)} days")
for m in ["evt", "garch"]:
    for lvl in [5, 1]:
        print(f"{m} {lvl}%: breach rate {bench[f'breach_{m}_{lvl}'].mean() * 100:.2f}%")
bench.to_parquet("backtest_bench.parquet")

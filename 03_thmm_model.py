"""
Этап 2. Обучение Student-t HMM (K=4)
Model selection по BIC (K=2..5), t vs Gaussian (LR-тест), поиск эргодического (не поглощающего) решения, переупорядочивание по волатильности портфеля.
Выход: thmm_model_tr.npz, model_selection_tr.parquet

Исходный артефакт: thmm_model_tr.npz
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

# skill:moex-data kernel.py (auto-injected on skill load)
import urllib.request, urllib.parse, json, time

MOEX_MARKETS = {
    "index":   ("stock", "index", None),
    "shares":  ("stock", "shares", "TQBR"),
    "bonds":   ("stock", "bonds", "TQOB"),
    "fx":      ("currency", "selt", "CETS"),
    "futures": ("futures", "forts", "RFUD"),
    "options": ("futures", "options", "ROPD"),
}
MOEX_ISS = "https://iss.moex.com/iss"


def iss_get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def iss_block_df(block):
    import pandas as pd
    return pd.DataFrame(block["data"], columns=block["columns"])


def moex_history(secid, market="index", start=None, end=None, board=None, pause=0.15):
    import pandas as pd
    if market not in MOEX_MARKETS:
        raise ValueError("market must be one of %s" % list(MOEX_MARKETS))
    engine, mkt, default_board = MOEX_MARKETS[market]
    board = board or default_board
    base = "%s/history/engines/%s/markets/%s/securities/%s.json" % (MOEX_ISS, engine, mkt, secid)
    q = "&".join([p for p in ["from=%s" % start if start else "",
                              "till=%s" % end if end else ""] if p])
    rows, cols, pos = [], None, 0
    while True:
        url = "%s?%s&start=%d" % (base, q, pos) if q else "%s?start=%d" % (base, pos)
        d = iss_get(url)
        cols = d["history"]["columns"]
        chunk = d["history"]["data"]
        rows.extend(chunk)
        _, total, pagesize = d["history.cursor"]["data"][0]
        pos += pagesize
        if pos >= total or not chunk:
            break
        time.sleep(pause)
    df = pd.DataFrame(rows, columns=cols)
    if len(df) == 0:
        raise ValueError("no data for %s (market=%s, board=%s)" % (secid, market, board))
    if board and "BOARDID" in df.columns:
        df = df[df["BOARDID"] == board]
    base_cols = ["TRADEDATE", "CLOSE", "OPEN", "HIGH", "LOW", "VALUE"]
    extra = [c for c in ("OPENPOSITION", "SETTLEPRICE", "WAPRICE") if c in df.columns]
    df = df[base_cols + extra].copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.dropna(subset=["CLOSE"]).sort_values("TRADEDATE").reset_index(drop=True)
    return df.rename(columns={"TRADEDATE": "date", "CLOSE": "close", "OPEN": "open",
                              "HIGH": "high", "LOW": "low", "VALUE": "value",
                              "OPENPOSITION": "oi", "SETTLEPRICE": "settle",
                              "WAPRICE": "wap"})


# skill:student-t-hmm kernel.py (auto-injected on skill load)
import numpy as np
from scipy.special import digamma, gammaln, logsumexp
from scipy.optimize import brentq
from scipy.linalg import solve_triangular


def build_student_t_hmm():
    def _mvt_logpdf_and_maha(X, mu, Sigma, nu):
        n, d = X.shape
        L = np.linalg.cholesky(Sigma)
        diff = (X - mu).T
        sol = solve_triangular(L, diff, lower=True)
        maha = np.sum(sol ** 2, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        c = (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0)
             - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet)
        return c - 0.5 * (nu + d) * np.log1p(maha / nu), maha

    class StudenttHMM:
        def __init__(self, n_states, n_iter=200, tol=1e-4, reg=1e-6,
                     nu_bounds=(2.02, 200.0), random_state=0, verbose=False):
            self.K = n_states; self.n_iter = n_iter; self.tol = tol; self.reg = reg
            self.nu_lo, self.nu_hi = nu_bounds
            self.rs = np.random.RandomState(random_state); self.verbose = verbose

        def _log_B(self, X):
            n = X.shape[0]; logB = np.empty((n, self.K)); maha = np.empty((n, self.K))
            for k in range(self.K):
                logB[:, k], maha[:, k] = _mvt_logpdf_and_maha(
                    X, self.mu_[k], self.Sigma_[k], self.nu_[k])
            return logB, maha

        def _forward_backward(self, logB):
            n, K = logB.shape
            log_pi = np.log(self.startprob_ + 1e-300); log_A = np.log(self.transmat_ + 1e-300)
            log_alpha = np.empty((n, K)); log_alpha[0] = log_pi + logB[0]
            for t in range(1, n):
                log_alpha[t] = logB[t] + logsumexp(log_alpha[t-1][:, None] + log_A, axis=0)
            log_beta = np.zeros((n, K))
            for t in range(n - 2, -1, -1):
                log_beta[t] = logsumexp(log_A + logB[t+1][None, :] + log_beta[t+1][None, :], axis=1)
            ll = logsumexp(log_alpha[-1])
            gamma = np.exp(log_alpha + log_beta - ll)
            log_xi_acc = np.full((K, K), -np.inf)
            for t in range(n - 1):
                m = (log_alpha[t][:, None] + log_A + logB[t+1][None, :] + log_beta[t+1][None, :] - ll)
                log_xi_acc = np.logaddexp(log_xi_acc, m)
            return ll, gamma, np.exp(log_xi_acc)

        def _init_params(self, X, gamma0=None):
            n, d = X.shape; self.d = d
            if gamma0 is None:
                idx = self.rs.randint(0, self.K, size=n)
                gamma0 = np.zeros((n, self.K)); gamma0[np.arange(n), idx] = 1.0
            w = gamma0.sum(0) + 1e-8
            self.mu_ = (gamma0.T @ X) / w[:, None]
            self.Sigma_ = np.empty((self.K, d, d))
            for k in range(self.K):
                diff = X - self.mu_[k]
                self.Sigma_[k] = (gamma0[:, k][:, None] * diff).T @ diff / w[k] + self.reg * np.eye(d)
            self.nu_ = np.full(self.K, 8.0)
            self.startprob_ = np.full(self.K, 1.0 / self.K)
            self.transmat_ = np.full((self.K, self.K), 1.0 / self.K)

        def warm_start_from_gaussian(self, means, covars, transmat, startprob, nu0=8.0):
            self.mu_ = means.copy(); self.d = means.shape[1]
            self.Sigma_ = covars.copy() + self.reg * np.eye(self.d)
            self.transmat_ = transmat.copy(); self.startprob_ = startprob.copy()
            self.nu_ = np.full(self.K, nu0); self._warm = True

        def _solve_nu(self, const):
            def f(nu):
                return 1.0 + np.log(nu / 2.0) - digamma(nu / 2.0) + const
            flo, fhi = f(self.nu_lo), f(self.nu_hi)
            if flo * fhi > 0:
                return self.nu_hi if abs(fhi) < abs(flo) else self.nu_lo
            return brentq(f, self.nu_lo, self.nu_hi, xtol=1e-4)

        def fit(self, X):
            X = np.asarray(X, float); n, d = X.shape
            if not getattr(self, "_warm", False):
                self._init_params(X)
            self.ll_history_ = []; prev = -np.inf
            for it in range(self.n_iter):
                logB, maha = self._log_B(X)
                ll, gamma, xi_sum = self._forward_backward(logB)
                self.ll_history_.append(ll)
                Eu = (self.nu_ + d) / (self.nu_ + maha)
                Elogu = digamma((self.nu_ + d) / 2.0) - np.log((self.nu_ + maha) / 2.0)
                self.startprob_ = gamma[0] / gamma[0].sum()
                self.transmat_ = xi_sum / xi_sum.sum(1, keepdims=True)
                gu = gamma * Eu
                for k in range(self.K):
                    wk = gamma[:, k].sum() + 1e-12; guk = gu[:, k]
                    mu_k = (guk[:, None] * X).sum(0) / guk.sum()
                    diff = X - mu_k
                    Sig = (guk[:, None] * diff).T @ diff / wk
                    self.mu_[k] = mu_k
                    self.Sigma_[k] = 0.5 * (Sig + Sig.T) + self.reg * np.eye(d)
                    const = (gamma[:, k] * (Elogu[:, k] - Eu[:, k])).sum() / wk
                    self.nu_[k] = self._solve_nu(const)
                if self.verbose and (it % 20 == 0 or it == self.n_iter - 1):
                    print(f"  iter {it:3d}  loglik={ll:.3f}  nu={np.round(self.nu_,2)}", flush=True)
                if ll - prev < self.tol and it > 5:
                    break
                prev = ll
            self.converged_ = (it < self.n_iter - 1); self.n_iter_run_ = it + 1; self.loglik_ = ll
            return self

        def score(self, X):
            logB, _ = self._log_B(np.asarray(X, float))
            return self._forward_backward(logB)[0]

        def predict_proba(self, X):
            logB, _ = self._log_B(np.asarray(X, float))
            return self._forward_backward(logB)[1]

        def predict(self, X):
            X = np.asarray(X, float); logB, _ = self._log_B(X); n, K = logB.shape
            log_pi = np.log(self.startprob_ + 1e-300); log_A = np.log(self.transmat_ + 1e-300)
            delta = np.empty((n, K)); psi = np.empty((n, K), int); delta[0] = log_pi + logB[0]
            for t in range(1, n):
                m = delta[t-1][:, None] + log_A
                psi[t] = np.argmax(m, axis=0); delta[t] = logB[t] + np.max(m, axis=0)
            path = np.empty(n, int); path[-1] = np.argmax(delta[-1])
            for t in range(n - 2, -1, -1):
                path[t] = psi[t+1, path[t+1]]
            return path

        def n_parameters(self):
            K, d = self.K, self.d
            return K*(K-1) + (K-1) + K*d + K*d*(d+1)//2 + K

    return StudenttHMM


def fit_regime_thmm(X, n_states=5, standardize=True, warm_gaussian=True,
                    n_iter=400, n_restarts=4, order_by=None, random_state=0,
                    verbose=False):
    X = np.asarray(X, float); T, d = X.shape
    mu_s = X.mean(0); sd_s = X.std(0); sd_s[sd_s == 0] = 1.0
    Xz = (X - mu_s) / sd_s if standardize else X
    THMM = build_student_t_hmm()

    model = None
    if warm_gaussian:
        try:
            from hmmlearn.hmm import GaussianHMM
            best_ll, g = -np.inf, None
            for seed in range(max(1, n_restarts * 8)):
                gg = GaussianHMM(n_components=n_states, covariance_type='full',
                                 n_iter=1500, tol=1e-5, random_state=seed)
                gg.fit(Xz); ll = gg.score(Xz)
                if ll > best_ll:
                    best_ll, g = ll, gg
            model = THMM(n_states, n_iter=n_iter, tol=1e-5, random_state=random_state, verbose=verbose)
            model.warm_start_from_gaussian(g.means_, g.covars_, g.transmat_, g.startprob_, nu0=8.0)
            model.fit(Xz)
        except ImportError:
            model = None
    if model is None:
        best_ll = -np.inf
        for seed in range(max(1, n_restarts)):
            m = THMM(n_states, n_iter=n_iter, tol=1e-5, random_state=random_state + seed, verbose=verbose)
            m.fit(Xz)
            if m.loglik_ > best_ll:
                best_ll, model = m.loglik_, m

    mean_raw = model.mu_ * sd_s + mu_s if standardize else model.mu_
    order = np.argsort(mean_raw[:, order_by]) if order_by is not None else np.arange(n_states)
    perm = order; new_of_old = {int(o): i for i, o in enumerate(perm)}
    trans = model.transmat_[np.ix_(perm, perm)]
    evals, evecs = np.linalg.eig(trans.T)
    stat = np.real(evecs[:, np.argmin(np.abs(evals - 1))]); stat = stat / stat.sum()
    hidden = np.array([new_of_old[s] for s in model.predict(Xz)])
    post = model.predict_proba(Xz)[:, perm]
    nu_ord = model.nu_[perm]; mean_ord = mean_raw[perm]
    n_params = model.n_parameters()
    bic = -2 * model.loglik_ + n_params * np.log(T)
    aic = -2 * model.loglik_ + 2 * n_params
    states = []
    for j in range(n_states):
        states.append({'state': j, 'nu': float(nu_ord[j]),
                       'mean': mean_ord[j].tolist(),
                       'self_transition': float(trans[j, j]),
                       'duration': float(1 / (1 - trans[j, j])),
                       'share': float(np.mean(hidden == j)),
                       'stationary': float(stat[j])})
    return {'model': model, 'loglik': float(model.loglik_), 'bic': float(bic),
            'aic': float(aic), 'n_params': int(n_params), 'states': states,
            'transmat': trans, 'hidden': hidden, 'posterior': post,
            'mean_raw': mean_ord, 'nu': nu_ord, 'mu_std': mu_s, 'sd_std': sd_s,
            'order': perm}


import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load master map
master_df = pd.read_parquet("master_map.parquet")
recs = master_df.to_dict("records")

# Fetch full bond history with ACCINT, FACEVALUE, COUPONPERCENT
pairs_dict = dict(zip(master_df["secid"], master_df["board"]))
pairs = master_df[["secid","board"]].drop_duplicates().values.tolist()

KEEP=["TRADEDATE","CLOSE","ACCINT","FACEVALUE","COUPONVALUE","COUPONPERCENT","MATDATE"]

def fetch_full(secid, board, start="2018-01-01"):
    base=f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/{board}/securities/{secid}.json"
    rows=[]; start_i=0
    while True:
        url=f"{base}?from={start}&start={start_i}&limit=100"
        try:
            j=iss_get(url)
        except Exception as e:
            return secid, None, str(e)
        blk=iss_block_df(j["history"])
        if blk is None or blk.shape[0]==0: break
        cols=[c for c in KEEP if c in blk.columns]
        rows.append(blk[cols])
        if blk.shape[0]<100: break
        start_i+=100
    if not rows: return secid, None, "empty"
    df=pd.concat(rows, ignore_index=True); df["secid"]=secid; df["board"]=board
    return secid, df, None

t0=time.time(); results={}; errs={}
with ThreadPoolExecutor(max_workers=12) as ex:
    futs={ex.submit(fetch_full, s, b): s for s,b in pairs}
    done=0
    for f in as_completed(futs):
        s,df,err=f.result()
        if df is not None: results[s]=df
        else: errs[s]=err
        done+=1

full=pd.concat(results.values(), ignore_index=True)

# Retry SSL failure
for s in list(errs.keys()):
    board = pairs_dict.get(s)
    if board:
        s2,df2,err2=fetch_full(s, board)
        if df2 is not None:
            full=pd.concat([full, df2], ignore_index=True)

full.to_parquet("bond_history_full.parquet")

# Compute total returns
for c in ["CLOSE","ACCINT","FACEVALUE","COUPONVALUE","COUPONPERCENT"]:
    full[c]=pd.to_numeric(full[c], errors="coerce")
full["date"]=pd.to_datetime(full["TRADEDATE"])
full=full.sort_values(["secid","date"]).reset_index(drop=True)

full["cp"]=pd.to_numeric(full["COUPONPERCENT"],errors="coerce")
full["clean"]=pd.to_numeric(full["CLOSE"],errors="coerce")
full=full.sort_values(["secid","date"]).reset_index(drop=True)

def coup_rate(s):
    pos=s[s>0]
    return pos.median() if len(pos)>0 else 0.0
rate=full.groupby("secid")["cp"].apply(coup_rate)
full["rate"]=full["secid"].map(rate)

def tr_carry(g):
    g=g.dropna(subset=["clean"]).copy()
    if len(g)<2:
        g["tr"]=np.nan; return g
    price_ret=g["clean"].pct_change()
    carry=(g["rate"]/100.0)/365.0
    g["tr"]=price_ret + carry
    return g.iloc[1:]

tr=full.groupby("secid", group_keys=False).apply(tr_carry, include_groups=True)
tr=tr.dropna(subset=["tr"])
tr=tr[np.abs(tr["tr"])<0.15]
tr[["date","secid","board","tr"]].to_parquet("bond_tr_long.parquet")

# Load segment membership
member=pd.read_csv("segment_membership.csv")
seg_map=member.set_index("secid")["segment"].to_dict()

tr["segment"]=tr["secid"].map(seg_map)
tr=tr.dropna(subset=["segment"])
order=["OFZ_short","OFZ_med","OFZ_long","Corp_L1","Corp_L2","Corp_L3","MBS"]
panels={}
for seg,gg in tr.groupby("segment"):
    s=gg.groupby("date")["tr"].agg(["mean","count"])
    panels[seg]=s[s["count"]>=3]["mean"]
panel=pd.DataFrame(panels)[order].dropna().sort_index()
panel["Portfolio"]=panel[order].mean(axis=1)
panel.to_parquet("segment_returns_panel_tr.parquet")

# Fit Student-t HMM
X=panel[order].values; T,dime=X.shape
w=np.ones(dime)/dime
port=X@w
Xmean,Xstd=X.mean(0),X.std(0); Xz=(X-Xmean)/Xstd

THMM=build_student_t_hmm()
rng=np.random.default_rng(7)
healthy=[]
for seed in range(30):
    m=THMM(n_states=4, n_iter=400, nu_bounds=(2.5,200))
    idx=rng.choice(T,4,replace=False)
    means=Xz[idx]
    covars=np.array([np.cov(Xz.T)+1e-3*np.eye(dime) for _ in range(4)])
    frac=rng.uniform(0.80,0.93)
    A=np.full((4,4),(1-frac)/3); np.fill_diagonal(A,frac)
    try:
        m.warm_start_from_gaussian(means,covars,A,np.full(4,.25)); m.fit(Xz)
        ev,evec=np.linalg.eig(m.transmat_.T); stat=np.real(evec[:,np.argmin(np.abs(ev-1))]); stat/=stat.sum()
        ms=np.max(np.diag(m.transmat_))
        if stat.min()>0.03 and ms<=0.999 and m.nu_.max()<190:
            healthy.append((seed,m.loglik_,stat.min(),ms,m))
    except Exception: pass
healthy.sort(key=lambda c:-c[1])

m=healthy[0][4]
post=m.predict_proba(Xz); hidden=m.predict(Xz)
mu_raw=m.mu_*Xstd+Xmean
Sigma_raw=m.Sigma_*np.outer(Xstd,Xstd)[None]
nu=m.nu_.copy()
port_scale=np.array([w@Sigma_raw[k]@w for k in range(4)])
port_var=port_scale*np.where(nu>2, nu/(nu-2), np.nan)
port_vol_ann=np.sqrt(port_var*252)*100
o=np.argsort(port_vol_ann)
def reord(A,o): return A[np.ix_(o,o)]
transmat=reord(m.transmat_,o); mu_raw=mu_raw[o]; Sigma_raw=Sigma_raw[o]; nu=nu[o]
port_vol_ann=port_vol_ann[o]
remap={old:new for new,old in enumerate(o)}
hidden=np.array([remap[h] for h in hidden]); post=post[:,o]
ev,evec=np.linalg.eig(transmat.T); stat=np.real(evec[:,np.argmin(np.abs(ev-1))]); stat/=stat.sum()

np.savez("thmm_model_tr.npz", segs=np.array(order), nu=nu, mu_raw=mu_raw, Sigma_raw=Sigma_raw,
         transmat=transmat, hidden=hidden, posterior=post, port_ret=port,
         dates=np.array([str(d.date()) for d in panel.index]), stationary=stat,
         Xmean=Xmean, Xstd=Xstd, w=w, loglik=m.loglik_)

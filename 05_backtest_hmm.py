"""
Этап 4. Скользящий OOS-бэктест t-HMM (VaR/ES)
Расширяющееся окно старт=500, переоценка каждые 5 дней, forward-фильтр, прогноз на t+1 из смеси t. Сравнение с Gaussian/Empirical.
Выход: backtest_hmm.parquet, backtest_coverage_tests.csv

Исходный артефакт: backtest_hmm.parquet
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


import pandas as pd
import numpy as np
from scipy.stats import t as student_t, norm
from hmmlearn.hmm import GaussianHMM

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
member = pd.read_csv("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9cd51c7f-9dad-43c9-a3cc-a9d21e747b7e/v78b2bf35_segment_membership.csv")

order = ["OFZ_short", "OFZ_med", "OFZ_long", "Corp_L1", "Corp_L2", "Corp_L3", "MBS"]
X = panel[order].values
T, dime = X.shape
w = np.ones(dime) / dime
port_all = X @ w

THMM = build_student_t_hmm()


def log_emission(model, Xz):
    return model._log_B(Xz)[0]   # (T,K)


def forward_filter(model, Xz_hist):
    lB = log_emission(model, Xz_hist)
    logA = np.log(model.transmat_ + 1e-300); logpi = np.log(model.startprob_ + 1e-300)
    T_, K = lB.shape
    la = logpi + lB[0]
    for t in range(1, T_):
        la = lB[t] + logsumexp(la[:, None] + logA, axis=0)
    la -= logsumexp(la)
    return np.exp(la)


def fit_model(Xtr):
    mu_c = Xtr.mean(0); sd_c = Xtr.std(0); Xz = (Xtr - mu_c) / sd_c
    g = GaussianHMM(n_components=4, covariance_type="full", n_iter=100, random_state=0); g.fit(Xz)
    md = THMM(n_states=4, n_iter=200, nu_bounds=(2.5, 200))
    md.warm_start_from_gaussian(g.means_, g.covars_, g.transmat_, g.startprob_); md.fit(Xz)
    return md, mu_c, sd_c


trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def mixture_var_es(pnext, mu_raw, Sigma_raw, nu, w, alpha):
    loc = mu_raw @ w
    scale = np.sqrt(np.array([w @ Sigma_raw[k] @ w for k in range(len(nu))]))
    lo = (loc - 15 * scale).min(); hi = (loc + 15 * scale).max()
    grid = np.linspace(lo, hi, 20000)
    cdf = np.zeros_like(grid); pdf = np.zeros_like(grid)
    for k in range(len(nu)):
        z = (grid - loc[k]) / scale[k]
        cdf += pnext[k] * student_t.cdf(z, df=nu[k])
        pdf += pnext[k] * student_t.pdf(z, df=nu[k]) / scale[k]
    iv = int(np.clip(np.searchsorted(cdf, alpha), 1, len(grid) - 1))
    VaR = grid[iv]
    mask = grid <= VaR
    ES = trapz(grid[mask] * pdf[mask], grid[mask]) / max(trapz(pdf[mask], grid[mask]), 1e-12) if mask.sum() >= 2 else VaR
    return VaR, ES


START = 500; REFIT = 5; alphas = [0.05, 0.01]
records = []; md = None
for i in range(START, T - 1):
    if md is None or (i - START) % REFIT == 0:
        md, mu_c, sd_c = fit_model(X[:i])
        mu_raw = md.mu_ * sd_c + mu_c; Sig_raw = md.Sigma_ * np.outer(sd_c, sd_c)[None]
    Xz_hist = (X[:i] - mu_c) / sd_c
    pnext = forward_filter(md, Xz_hist) @ md.transmat_
    real = port_all[i]
    rec = {"idx": i, "date": str(panel.index[i].date()), "real": real}
    for a in alphas:
        VaR, ES = mixture_var_es(pnext, mu_raw, Sig_raw, md.nu_, w, a)
        rec[f"VaR_{int(a*100)}"] = VaR; rec[f"ES_{int(a*100)}"] = ES; rec[f"breach_{int(a*100)}"] = int(real < VaR)
    records.append(rec)
bt = pd.DataFrame(records)
print(f"done {len(bt)} OOS days")
print("breach 95%:", round(bt['breach_5'].mean() * 100, 2), "% | 99%:", round(bt['breach_1'].mean() * 100, 2), "%")
bt.to_parquet("backtest_hmm.parquet")

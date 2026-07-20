"""
Этап 1. Полная доходность и сегментная панель
Выгрузка истории MOEX ISS -> полная доходность (чистая цена + купонный carry) -> агрегация в 7 сегментов.
Выход: segment_returns_panel_tr.parquet, portfolio_returns_tr.csv

Исходный артефакт: segment_returns_panel_tr.parquet
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis

member = pd.read_csv("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9cd51c7f-9dad-43c9-a3cc-a9d21e747b7e/v78b2bf35_segment_membership.csv")
seg_map = member.set_index("secid")["segment"].to_dict()

tr = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/07bd76b8-3201-4808-9b86-7af3db75679f/vfa3050b7_bond_tr_long.parquet")
tr["segment"] = tr["secid"].map(seg_map)
tr = tr.dropna(subset=["segment"])

order = ["OFZ_short", "OFZ_med", "OFZ_long", "Corp_L1", "Corp_L2", "Corp_L3", "MBS"]
panels = {}
for seg, gg in tr.groupby("segment"):
    s = gg.groupby("date")["tr"].agg(["mean", "count"])
    panels[seg] = s[s["count"] >= 3]["mean"]
panel = pd.DataFrame(panels)[order].dropna().sort_index()
panel["Portfolio"] = panel[order].mean(axis=1)
panel.to_parquet("segment_returns_panel_tr.parquet")

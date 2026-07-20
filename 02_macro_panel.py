"""
Этап 1b. Макроковариаты
USD/RUB (MOEX), ключевая ставка ЦБ (BIS), инфляция (World Bank) + производные, выравнивание по датам панели.
Выход: macro_panel.parquet

Исходный артефакт: macro_panel.parquet
Среда: python (см. requirements.txt / environment.yml)

ПРИМЕЧАНИЕ: пути к входным артефактам в коде — маркеры {{artifact:...}} из исходной
сессии Claude Science. Для локального запуска замените их на пути к файлам из папки data/.
"""

import pandas as pd
import numpy as np

panel = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/9228e694-9144-4aee-873e-f9aa18f7789a/v833fd342_segment_returns_panel_tr.parquet")
pdates = pd.DatetimeIndex(panel.index)
fx2 = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/8cad689a-007a-482e-beb4-dfaf32b84694/ve8891922__usdrub.parquet").set_index("date").sort_index()
fx2["usdrub"] = fx2["usdrub"].where(fx2["usdrub"] > 1)
rate = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/4e1d06d5-0af8-4267-b890-5125ab7c883f/vc467150e__cbr_rate.parquet").rename(columns={"value": "keyrate"}).set_index("date").sort_index()
wb = pd.read_parquet("/Users/egorgalkin/.claude-science/orgs/068613ee-500c-4ede-b4aa-5e412b74c67d/artifacts/proj_7f326450c710/ef17f64e-3c22-45d5-bc15-5e2b012c1283/v52beb7ad__wb_infl.parquet")[["date", "value"]].rename(columns={"value": "inflation"}).set_index("date").sort_index()

mac = pd.DataFrame(index=pdates)
mac["usdrub"] = fx2["usdrub"].reindex(pdates.union(fx2.index)).ffill().reindex(pdates)
mac["keyrate"] = rate["keyrate"].reindex(pdates.union(rate.index)).ffill().reindex(pdates)
mac["inflation"] = wb["inflation"].reindex(pdates.union(wb.index)).ffill().reindex(pdates)
lg = np.log(mac["usdrub"])
mac["usdrub_vol"] = lg.diff().rolling(21).std() * np.sqrt(252) * 100
mac["usdrub_ret20"] = lg.diff(20) * 100
mac["keyrate_chg60"] = mac["keyrate"].diff(60)
mac = mac.ffill().bfill()
print("shape", mac.shape, "NaN", mac.isna().sum().sum(), "inf", np.isinf(mac.values).sum())
print(mac.describe().round(2).T[["mean", "min", "max"]].to_string())
mac.to_parquet("macro_panel.parquet")

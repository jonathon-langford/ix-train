import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

dy = pd.read_parquet("samples_eval/Csplit_Jsamp_DY_test_EB_renamed_processed_eval.parquet")
data = pd.read_parquet("samples_eval/Csplit_Jsamp_Data_test_EB_processed.parquet")
data['weight'] = 1

x = "probe_pt"

fig, ax = plt.subplots()
w = data['weight']/np.sum(data['weight'])
ax.hist(data[x], bins=50, range=(0,150), label="Data", histtype="step", edgecolor="black", weights=w)

w = dy['weight']/np.sum(dy['weight'])
ax.hist(dy[x], bins=50, range=(0,150), label="DY (nominal)", histtype="step", edgecolor="red", weights=w)

w = (dy['weight']*dy['rwgt_1'])/np.sum(dy['weight'])
ax.hist(dy[x], bins=50, range=(0,150), label="DY (rwgt 1)", histtype="step", edgecolor="orange", weights=w)

w = (dy['weight']*dy['rwgt_1']*dy['rwgt_2'])/np.sum(dy['weight'])
ax.hist(dy[x], bins=50, range=(0,150), label="DY (rwgt 2)", histtype="step", edgecolor="green", weights=w)


ax.legend(loc='best')

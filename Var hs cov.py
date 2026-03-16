#!/usr/bin/env python
# coding: utf-8

# In[13]:


# ==============================
# VaR Historique et Variance-Covariance
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------------
# 1️⃣ Data_set contient les PRIX
# chaque colonne = entreprise

# Lire un fichier Excel
df = pd.read_excel(r"C:\Users\hp\Downloads\Data_set.xlsx")
# Afficher les premières lignes
print(df.head())

# Mettre la colonne Date comme index
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Garder seulement les colonnes numériques
df = df.astype(float)


# In[14]:


import numpy as np
from scipy.stats import norm

# ==============================
# 1️⃣ Calcul des rendements log
# ==============================
returns = np.log(df / df.shift(1)).dropna()

# Niveau de confiance
alpha = 0.01   # 99%

# ==================================
# 📌 1) VaR Historique
# ==================================
VaR_HS = -returns.quantile(alpha)

print("VaR Historique (99%) :")
print(VaR_HS)

# ==================================
# 📌 2) VaR Variance-Covariance
# ==================================
mu = returns.mean()
sigma = returns.std()

VaR_VCov = -(mu + sigma * norm.ppf(alpha))

print("\nVaR Variance-Covariance (99%) :")
print(VaR_VCov)


# In[16]:


plt.figure(figsize=(14,7))

for col in returns.columns:
    plt.plot(returns[col], label=f"Rendements {col}")
    VaR_HS_col = -returns[col].quantile(alpha)
    VaR_VCov_col = -(returns[col].mean() + returns[col].std() * norm.ppf(alpha))
    plt.axhline(-VaR_HS_col, linestyle='--', label=f"VaR HS {col}")
    plt.axhline(-VaR_VCov_col, linestyle=':', label=f"VaR VC {col}")

plt.title("Rendements et VaR 99% pour toutes les colonnes")
plt.xlabel("Date")
plt.ylabel("Rendements log")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





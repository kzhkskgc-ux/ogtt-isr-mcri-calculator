import io
import zipfile
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

# =========================================================
# Streamlit page config
# =========================================================
st.set_page_config(
    page_title="神戸大学臨床糖尿病グループ | OGTT ISR / β-cell Function / apparent MCRi Calculator",
    page_icon="🧪",
    layout="wide",
)

APP_VERSION = "Version 1.7.0"

# =========================================================
# Constants
# =========================================================
TIME_POINTS = np.array([0, 30, 60, 90, 120], dtype=float)
CP_MW_KDA = 3.020
GLUC_FACTOR = 18.016  # mg/dL per mmol/L for glucose
DEFAULT_VD = 141.0    # mL/kg; apparent insulin distribution volume
GLUCAGON_MW_KDA = 3.485  # pg/mL -> pmol/L

# Ferrannini et al. J Clin Invest 2014 supplementary method constants
HPF_L_MIN_M2 = 3.2 * 0.6 * 0.30  # hepatic plasma flow = 0.576 L/min/m2
MCR_GLG_L_MIN_M2 = 0.537

# =========================================================
# Column aliases
# =========================================================
def normalize_colname(x):
    s = str(x).strip()
    s = s.replace(" ", "").replace("　", "")
    s = s.replace("μ", "u")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    s = s.replace("_", "").replace("-", "").lower()
    return s

COLUMN_CANDIDATES = {
    "id": ["ID", "Id", "subject_id", "subjectid", "患者ID", "症例ID"],
    "height": ["Ht", "HT", "height", "Height", "身長"],
    "weight": ["Wt", "WT", "weight", "Weight", "体重", "BW"],
    "age": ["Age", "age", "AGE", "年齢", "age_years", "Age_years", "AgeYears"],
    "bg_0": ["O-BG(0)", "BG(0)", "glucose0", "Glu0", "血糖0"],
    "bg_30": ["O-BG(30)", "BG(30)", "glucose30", "Glu30", "血糖30"],
    "bg_60": ["O-BG(60)", "BG(60)", "glucose60", "Glu60", "血糖60"],
    "bg_90": ["O-BG(90)", "BG(90)", "glucose90", "Glu90", "血糖90"],
    "bg_120": ["O-BG(120)", "BG(120)", "glucose120", "Glu120", "血糖120"],
    "iri_0": ["O-IRI(0)", "IRI(0)", "insulin0", "IRI0", "インスリン0"],
    "iri_30": ["O-IRI(30)", "IRI(30)", "insulin30", "IRI30", "インスリン30"],
    "iri_60": ["O-IRI(60)", "IRI(60)", "insulin60", "IRI60", "インスリン60"],
    "iri_90": ["O-IRI(90)", "IRI(90)", "insulin90", "IRI90", "インスリン90"],
    "iri_120": ["O-IRI(120)", "IRI(120)", "insulin120", "IRI120", "インスリン120"],
    "cpr_0": ["O-CPR(0)", "CPR(0)", "cpeptide0", "CPR0", "C-peptide0", "Cペプチド0"],
    "cpr_30": ["O-CPR(30)", "CPR(30)", "cpeptide30", "CPR30", "C-peptide30", "Cペプチド30"],
    "cpr_60": ["O-CPR(60)", "CPR(60)", "cpeptide60", "CPR60", "C-peptide60", "Cペプチド60"],
    "cpr_90": ["O-CPR(90)", "CPR(90)", "cpeptide90", "CPR90", "C-peptide90", "Cペプチド90"],
    "cpr_120": ["O-CPR(120)", "CPR(120)", "cpeptide120", "CPR120", "C-peptide120", "Cペプチド120"],
    "glg_0": ["O-Glg(0)", "Glg(0)", "glucagon0", "Glg0", "グルカゴン0", "GCG0", "GCG(0)"],
    "glg_30": ["O-Glg(30)", "Glg(30)", "glucagon30", "Glg30", "グルカゴン30", "GCG30", "GCG(30)"],
    "glg_60": ["O-Glg(60)", "Glg(60)", "glucagon60", "Glg60", "グルカゴン60", "GCG60", "GCG(60)"],
    "glg_90": ["O-Glg(90)", "Glg(90)", "glucagon90", "Glg90", "グルカゴン90", "GCG90", "GCG(90)"],
    "glg_120": ["O-Glg(120)", "Glg(120)", "glucagon120", "Glg120", "グルカゴン120", "GCG120", "GCG(120)"],
}

DISPLAY_REQUIRED_COLUMNS = [
    "ID", "Age", "Ht", "Wt",
    "O-BG(0)", "O-BG(30)", "O-BG(60)", "O-BG(90)", "O-BG(120)",
    "O-IRI(0)", "O-IRI(30)", "O-IRI(60)", "O-IRI(90)", "O-IRI(120)",
    "O-CPR(0)", "O-CPR(30)", "O-CPR(60)", "O-CPR(90)", "O-CPR(120)",
]

RESULT_UNITS = {
    "glucose_sensitivity_pmol_min_per_mgdl": "pmol/min per mg/dL",
    "rate_sensitivity_pmol_per_mgdl": "pmol per mg/dL",
    "glucose_sensitivity_BSA": "pmol/min/m² per mmol/L",
    "rate_sensitivity_BSA": "pmol/m² per mmol/L",
    "glucose_sensitivity_kg": "pmol/min/kg per mmol/L",
    "rate_sensitivity_kg": "pmol/kg per mmol/L",
    "potentiation_factor": "dimensionless; mean-normalized",
    "apparent MCRi": "mL/min/kg; non-negative constrained in v1.6.1",
    "MCRI_OGTT_L_min_m2": "L/min/m²; AUC formula, non-negative constrained",
    "MCRI_OGTT_mL_min_per_kg": "mL/min/kg; AUC formula, non-negative constrained",
    "MCRI_timepoint_L_min_m2": "L/min/m²; timepoint values, non-negative constrained",
    "MCRI_timepoint_mL_min_kg": "mL/min/kg; timepoint values (0/30/60/90/120 min), non-negative constrained",
    "RdIns_timepoint_pmol_min_m2": "pmol/min/m²",
    "Age_years": "years",
    "k01_min_inv": "min^-1",
    "k21_min_inv": "min^-1",
    "k12_min_inv": "min^-1",
    "V1_L": "L",
    "ISR": "pmol/min",
    "CP_fit_RMS_pmol_L": "pmol/L",
    "MARI_fit_RMS_pmol_min": "pmol/min",
    "fasting_CPR_0_pmol_L": "pmol/L",
    "fasting_insulin_0_pmol_L": "pmol/L",
    "fasting_CPR_insulin_molar_ratio": "dimensionless",
    "fasting_glucagon_pmol_L": "pmol/L",
    "portal_I_Glg_ratio_0_30": "mol/mol",
    "portal_I_Glg_ratio_30_60": "mol/mol",
    "portal_I_Glg_ratio_60_90": "mol/mol",
    "portal_I_Glg_ratio_90_120": "mol/mol",
    "portal_I_Glg_ratio_mean": "mol/mol",
    "AUC_portal_I_Glg_ratio": "mol/mol × min",
    "AUC_Glu_mg_dL_min": "mg/dL × min",
    "AUC_IRI_uU_mL_min": "μU/mL × min",
    "Matsuda_index": "dimensionless",
    "insulinogenic_index": "dimensionless",
    "oral_DI1": "index",
    "oral_DI2": "index",
}

# =========================================================
# Data preparation
# =========================================================
def find_matching_columns(df):
    normalized_actual = {normalize_colname(c): c for c in df.columns}
    mapping = {}
    missing = []
    for canonical, candidates in COLUMN_CANDIDATES.items():
        found = None
        for cand in candidates:
            key = normalize_colname(cand)
            if key in normalized_actual:
                found = normalized_actual[key]
                break
        if found is None:
            missing.append(canonical)
        else:
            mapping[canonical] = found
    return mapping, missing


def standardize_input_df(df):
    mapping, missing = find_matching_columns(df)
    glg_keys = ["glg_0", "glg_30", "glg_60", "glg_90", "glg_120"]
    required_missing = [m for m in missing if m not in glg_keys]
    if required_missing:
        return df.copy(), mapping, required_missing

    out = pd.DataFrame()
    out["ID"] = df[mapping["id"]]
    out["Age"] = df[mapping["age"]]
    out["Ht"] = df[mapping["height"]]
    out["Wt"] = df[mapping["weight"]]
    for t in [0, 30, 60, 90, 120]:
        out[f"O-BG({t})"] = df[mapping[f"bg_{t}"]]
        out[f"O-IRI({t})"] = df[mapping[f"iri_{t}"]]
        out[f"O-CPR({t})"] = df[mapping[f"cpr_{t}"]]
    for t, key in zip([0, 30, 60, 90, 120], ["glg_0", "glg_30", "glg_60", "glg_90", "glg_120"]):
        out[f"O-Glg({t})"] = df[mapping[key]] if key in mapping else np.nan
    return out, mapping, []


def create_input_template_df():
    return pd.DataFrame([
        {
            "ID": "Case-1", "Age": 60.0, "Ht": 170.0, "Wt": 70.0,
            "O-BG(0)": 90.0, "O-BG(30)": 150.0, "O-BG(60)": 140.0, "O-BG(90)": 120.0, "O-BG(120)": 100.0,
            "O-IRI(0)": 5.0, "O-IRI(30)": 40.0, "O-IRI(60)": 50.0, "O-IRI(90)": 35.0, "O-IRI(120)": 20.0,
            "O-CPR(0)": 1.5, "O-CPR(30)": 5.0, "O-CPR(60)": 7.0, "O-CPR(90)": 6.0, "O-CPR(120)": 4.5,
            "O-Glg(0)": 80.0, "O-Glg(30)": 65.0, "O-Glg(60)": 60.0, "O-Glg(90)": 58.0, "O-Glg(120)": 55.0,
        }
    ])

# =========================================================
# Basic utilities
# =========================================================
def to_float_or_nan(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def np_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def cp_ng_to_pmol(x):
    return np.asarray(x, dtype=float) * 1000.0 / CP_MW_KDA


def iri_uU_to_pmol(x):
    return np.asarray(x, dtype=float) * 6.0


def glucagon_pg_to_pmol_L(x):
    return np.asarray(x, dtype=float) / GLUCAGON_MW_KDA


def calc_bsa(height_cm, weight_kg):
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)


def calc_bmi(height_cm, weight_kg):
    return weight_kg / (height_cm / 100.0) ** 2


def trapezoid_auc(time_min, values):
    t = np.asarray(time_min, dtype=float)
    v = np.asarray(values, dtype=float)
    if np.any(np.isnan(v)):
        raise ValueError("AUC calculation failed because values contain NaN.")
    return float(np_trapezoid(v, t))


def calc_fasting_cpr_insulin_molar_ratio(cpr_ng_mL, insulin_uU_mL):
    cpr_pmol_L = cp_ng_to_pmol(cpr_ng_mL)
    insulin_pmol_L = iri_uU_to_pmol(insulin_uU_mL)
    if np.any(insulin_pmol_L <= 0):
        raise ValueError("Fasting insulin must be positive to calculate CPR/insulin molar ratio.")
    return np.asarray(cpr_pmol_L, dtype=float) / np.asarray(insulin_pmol_L, dtype=float)


def calc_matsuda_index(glucose_mg_dL, insulin_uU_mL):
    g = np.asarray(glucose_mg_dL, dtype=float)
    i = np.asarray(insulin_uU_mL, dtype=float)
    if np.any(np.isnan(g)) or np.any(np.isnan(i)):
        raise ValueError("Matsuda index calculation failed because values contain NaN.")
    denom = (g[0] * i[0] * np.mean(g) * np.mean(i)) ** 0.5
    if denom <= 0:
        return np.nan
    return float(10000.0 / denom)


def calc_insulinogenic_index(glucose_mg_dL, insulin_uU_mL):
    g = np.asarray(glucose_mg_dL, dtype=float)
    i = np.asarray(insulin_uU_mL, dtype=float)
    delta_g = float(g[1] - g[0])
    delta_i = float(i[1] - i[0])
    if delta_g <= 0:
        return np.nan
    return float(delta_i / delta_g)


def calc_oral_disposition_indices(glucose_mg_dL, insulin_uU_mL):
    auc_glu = trapezoid_auc(TIME_POINTS, glucose_mg_dL)
    auc_iri = trapezoid_auc(TIME_POINTS, insulin_uU_mL)
    matsuda = calc_matsuda_index(glucose_mg_dL, insulin_uU_mL)
    insulinogenic = calc_insulinogenic_index(glucose_mg_dL, insulin_uU_mL)
    oral_di1 = (auc_iri / auc_glu) * matsuda if auc_glu > 0 and pd.notna(matsuda) else np.nan
    oral_di2 = insulinogenic * matsuda if pd.notna(insulinogenic) and pd.notna(matsuda) else np.nan
    return {
        "AUC_Glu_mg_dL_min": float(auc_glu),
        "AUC_IRI_uU_mL_min": float(auc_iri),
        "Matsuda_index": to_float_or_nan(matsuda),
        "insulinogenic_index": to_float_or_nan(insulinogenic),
        "oral_DI1": to_float_or_nan(oral_di1),
        "oral_DI2": to_float_or_nan(oral_di2),
    }

# =========================================================
# C-peptide deconvolution
# =========================================================
def get_params(height_cm, weight_kg, age_years):
    bmi = calc_bmi(height_cm, weight_kg)
    bsa = calc_bsa(height_cm, weight_kg)
    age = float(age_years)

    # Van Cauter-type population C-peptide kinetics.
    # In Version 1.6, k01 is calculated from age and BMI using the supplied original-method formula:
    # k01 = 0.0564 + 0.00127 * (age - 60) + 0.00461 * (BMI - 25)
    # k21/k12 and V1_ref are retained from the previous BMI-stratified practical approximation,
    # because their original-method regression equations were not specified in the current revision request.
    k01 = 0.0564 + 0.00127 * (age - 60.0) + 0.00461 * (bmi - 25.0)
    k01 = max(float(k01), 1e-6)
    if bmi < 26:
        k21, k12, V1_ref = 0.0568, 0.0169, 2.60
    else:
        k21, k12, V1_ref = 0.0495, 0.0183, 2.44
    return {"k01": k01, "k21": k21, "k12": k12, "V1": V1_ref * bsa, "bsa": bsa, "bmi": bmi, "age_years": age}


def expm2x2(A, t=1.0):
    a, b = A[0, 0] * t, A[0, 1] * t
    c, d = A[1, 0] * t, A[1, 1] * t
    At = np.array([[a, b], [c, d]])
    tr = a + d
    det = a * d - b * c
    disc = tr ** 2 - 4.0 * det
    if disc > 1e-20:
        sq = np.sqrt(disc)
        lam1, lam2 = (tr + sq) / 2.0, (tr - sq) / 2.0
        e1, e2 = np.exp(lam1), np.exp(lam2)
        beta = (e1 - e2) / (lam1 - lam2)
        alpha = e2 - beta * lam2
    else:
        lam = tr / 2.0
        e = np.exp(lam)
        beta, alpha = e, e * (1.0 - lam)
    return alpha * np.eye(2) + beta * At


def nnls_lawson_hanson(A, b, max_outer=300, max_inner=300, tol=1e-12):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(max_outer):
        active_mask = ~passive
        if (not np.any(active_mask)) or np.all(w[active_mask] <= tol):
            break
        cand = np.where(active_mask)[0]
        t_idx = cand[np.argmax(w[cand])]
        passive[t_idx] = True
        for __ in range(max_inner):
            P = np.where(passive)[0]
            s_P, _, _, _ = np.linalg.lstsq(A[:, P], b, rcond=None)
            if np.all(s_P > tol):
                x[P] = s_P
                break
            neg = P[s_P <= tol]
            alpha = np.clip(np.min(x[neg] / (x[neg] - s_P[s_P <= tol])), 0.0, 1.0)
            x[P] += alpha * (s_P - x[P])
            x = np.maximum(0.0, x)
            x[x <= tol] = 0.0
            passive[P[x[P] <= tol]] = False
        w = A.T @ (b - A @ x)
    return x, float(np.linalg.norm(b - A @ x))


def _ode_matrices(p):
    A = np.array([[-(p["k01"] + p["k21"]), p["k12"]], [p["k21"], -p["k12"]]])
    b = np.array([1.0 / p["V1"], 0.0])
    return A, b


def _initial_state(C1_0, p):
    return np.array([C1_0, (p["k21"] / p["k12"]) * C1_0])


def _basal_isr(C1_0, p):
    return p["k01"] * p["V1"] * C1_0


def estimate_isr(time_min, cp_pmol_L, height_cm, weight_kg, age_years):
    t = np.asarray(time_min, dtype=float)
    cp = np.asarray(cp_pmol_L, dtype=float)
    if np.any(np.isnan(cp)):
        raise ValueError("C-peptide contains NaN.")
    if np.any(cp < 0):
        raise ValueError("C-peptide contains negative values.")

    N = len(t)
    M = N - 1
    p = get_params(height_cm, weight_kg, age_years)
    A, b = _ode_matrices(p)
    Ainv = np.linalg.inv(A)
    e1 = np.array([1.0, 0.0])
    x0 = _initial_state(cp[0], p)

    Phi = [np.eye(2)]
    for j in range(1, N):
        Phi.append(expm2x2(A, t[j] - t[j - 1]) @ Phi[-1])
    y_hom = np.array([e1 @ Phi[j] @ x0 for j in range(1, N)])

    H = np.zeros((M, M))
    for k in range(M):
        dt_k = t[k + 1] - t[k]
        Phi_k = expm2x2(A, dt_k)
        intg_k = Ainv @ (Phi_k - np.eye(2)) @ b
        for j in range(k + 1, N):
            dt_ex = t[j] - t[k + 1]
            Phi_ex = expm2x2(A, dt_ex) if dt_ex > 1e-10 else np.eye(2)
            H[j - 1, k] = e1 @ Phi_ex @ intg_k

    rhs = cp[1:] - y_hom
    isr, _ = nnls_lawson_hanson(H, rhs)
    isr_bas = _basal_isr(cp[0], p)
    cp_fit = np.concatenate([[cp[0]], y_hom + H @ isr])
    dt_arr = np.diff(t)

    return {
        "isr": isr,
        "isr_basal": float(isr_bas),
        "time_points": t,
        "t_intervals": list(zip(t[:-1], t[1:])),
        "t_mids": (t[:-1] + t[1:]) / 2.0,
        "dt": dt_arr,
        "auc_isr": float(np.sum(isr * dt_arr)),
        "cp_obs": cp,
        "cp_fit": cp_fit,
        "rms_fit": float(np.sqrt(np.mean((cp_fit - cp) ** 2))),
        "params": p,
        "weight_kg": weight_kg,
        "H_condition_number": float(np.linalg.cond(H)) if H.size else np.nan,
    }

# =========================================================
# Simplified Mari-type model
# =========================================================
def smooth_vector_1d(v, valid=None, strength=0.35):
    v = np.asarray(v, dtype=float).copy()
    if valid is None:
        valid = np.isfinite(v)
    valid = np.asarray(valid, dtype=bool)
    out = v.copy()
    for k in range(len(v)):
        if not valid[k] or not np.isfinite(v[k]):
            continue
        neigh = [v[k]]
        if k > 0 and valid[k - 1] and np.isfinite(v[k - 1]):
            neigh.append(v[k - 1])
        if k < len(v) - 1 and valid[k + 1] and np.isfinite(v[k + 1]):
            neigh.append(v[k + 1])
        out[k] = (1.0 - strength) * v[k] + strength * float(np.mean(neigh))
    return out


def estimate_mari_constrained(isr_result, glucose_mg_dL, n_iter=40, pot_upper=3.0, epsilon_g=1.0, beta_smooth_strength=0.35):
    G = np.asarray(glucose_mg_dL, dtype=float)
    isr = np.asarray(isr_result["isr"], dtype=float)
    isr_b = float(isr_result["isr_basal"])
    isr_incr = isr - isr_b

    p = isr_result["params"]
    w_kg = isr_result.get("weight_kg", 70.0)
    bsa = p["bsa"]
    t = np.asarray(isr_result.get("time_points", TIME_POINTS), dtype=float)

    G_b = G[0]
    G_mid = (G[:-1] + G[1:]) / 2.0
    dG_mid = G_mid - G_b
    dt_arr = np.diff(t)
    Gdot = np.diff(G) / dt_arr
    Gdot_pos = np.maximum(Gdot, 0.0)

    valid = np.isfinite(dG_mid) & np.isfinite(Gdot_pos) & (dG_mid > epsilon_g)
    if not np.any(valid):
        L = len(isr)
        return {"Phi_s": np.nan, "Phi_d": np.nan, "beta": np.full(L, np.nan),
                "beta_mean_early": np.nan, "beta_mean_late": np.nan, "potentiation_ratio": np.nan,
                "isr_static": np.full(L, np.nan), "isr_dynamic": np.full(L, np.nan),
                "isr_pred": np.full(L, np.nan), "rms_mari": np.nan,
                "Phi_s_bsa": np.nan, "Phi_d_bsa": np.nan, "Phi_s_kg": np.nan, "Phi_d_kg": np.nan,
                "valid": valid, "model_note": "no interval with glucose above basal"}

    beta = np.ones(len(isr), dtype=float)
    beta[~valid] = np.nan
    Phi_s = 0.0
    Phi_d = 0.0

    for _ in range(n_iter):
        wdG = np.where(valid, np.nan_to_num(beta, nan=1.0) * dG_mid, 0.0)
        X = np.column_stack([wdG, Gdot_pos])
        rows = valid | (Gdot_pos > 0)
        params_fit, _ = nnls_lawson_hanson(X[rows, :], isr_incr[rows])
        Phi_s_raw = max(float(params_fit[0]), 1e-12)
        Phi_d_new = max(float(params_fit[1]), 0.0)

        new_beta = np.full(len(isr), np.nan)
        for k in range(len(isr)):
            if valid[k]:
                numer = isr_incr[k] - Phi_d_new * Gdot_pos[k]
                denom = Phi_s_raw * dG_mid[k]
                val = numer / denom if abs(denom) > 1e-12 else 1.0
                new_beta[k] = min(max(val, 0.0), pot_upper)

        new_beta = smooth_vector_1d(new_beta, valid=valid, strength=beta_smooth_strength)
        beta_mean = np.nanmean(new_beta[valid])
        if pd.notna(beta_mean) and beta_mean > 1e-12:
            new_beta[valid] = new_beta[valid] / beta_mean
            Phi_s_new = Phi_s_raw * beta_mean
        else:
            Phi_s_new = Phi_s_raw
        beta = new_beta.copy()
        Phi_s = Phi_s_new
        Phi_d = Phi_d_new

    beta[~valid] = np.nan
    isr_static = Phi_s * np.where(valid, np.nan_to_num(beta, nan=0.0), 0.0) * dG_mid
    isr_dynamic = Phi_d * Gdot_pos
    isr_pred = isr_b + isr_static + isr_dynamic
    rms_mari = float(np.sqrt(np.mean((isr_pred - isr) ** 2)))
    beta_early = np.nanmean(beta[:2])
    beta_late = np.nanmean(beta[2:])
    pot_ratio = beta_late / beta_early if pd.notna(beta_early) and beta_early > 1e-12 else np.nan

    conv_bsa = GLUC_FACTOR / bsa
    conv_kg = GLUC_FACTOR / w_kg
    return {
        "Phi_s": float(Phi_s), "Phi_d": float(Phi_d), "beta": beta,
        "beta_mean_early": to_float_or_nan(beta_early), "beta_mean_late": to_float_or_nan(beta_late),
        "potentiation_ratio": to_float_or_nan(pot_ratio),
        "isr_static": isr_static, "isr_dynamic": isr_dynamic, "isr_pred": isr_pred,
        "rms_mari": rms_mari,
        "Phi_s_bsa": float(Phi_s * conv_bsa), "Phi_d_bsa": float(Phi_d * conv_bsa),
        "Phi_s_kg": float(Phi_s * conv_kg), "Phi_d_kg": float(Phi_d * conv_kg),
        "valid": valid,
        "model_note": "simplified Mari-type model with non-negative mean-normalized smoothed potentiation factor",
    }

# =========================================================
# Apparent MCRi and portal I/Glg ratio
# =========================================================
def estimate_mcr(isr_result, iri_uU_mL, vd_mL_per_kg=DEFAULT_VD):
    """
    Apparent insulin metabolic clearance rate with a non-negative physiological constraint.

    v1.6.1 implementation follows the logic used in the reference notebook:
      RdIns(t) = ISR(t)/BSA - V_Ins * dI(t)/dt
      MCRI(t)  = max(0, RdIns(t) / I(t))

    Timepoint MCRI values are calculated at 0, 30, 60, 90, and 120 min.
    The 0-min derivative is fixed to 0 as a fasting steady-state approximation;
    30/60/90-min derivatives use central differences; 120-min derivative uses
    a backward difference. The AUC-based OGTT MCRI is calculated by the
    Gastaldelli-style formula and is also constrained to be non-negative.
    """
    I = iri_uU_to_pmol(np.asarray(iri_uU_mL, dtype=float))  # pmol/L
    if np.any(np.isnan(I)) or np.any(I <= 0):
        raise ValueError("IRI contains missing, zero, or negative values.")

    isr_interval = np.asarray(isr_result["isr"], dtype=float)       # 4 interval values, pmol/min
    isr_b = float(isr_result["isr_basal"])                          # fasting ISR, pmol/min
    w = float(isr_result.get("weight_kg", 70.0))
    bsa = float(isr_result["params"].get("bsa", np.nan))
    if not np.isfinite(bsa) or bsa <= 0:
        raise ValueError("BSA is invalid for MCRI calculation.")

    t = np.asarray(isr_result.get("time_points", TIME_POINTS), dtype=float)
    if len(t) != len(I):
        raise ValueError("Time points and insulin values have different lengths.")

    # Convert the 4 interval ISR estimates to 5 OGTT timepoint values.
    # t=0 uses basal ISR; 30/60/90/120 min use the preceding interval ISR.
    isr_time = np.concatenate([[isr_b], isr_interval])
    isr_pm2 = isr_time / bsa  # pmol/min/m²

    # Insulin distribution volume per BSA, as in the reference notebook.
    Vd_L_m2 = vd_mL_per_kg * 1e-3 * w / bsa  # L/m²

    # Numerical derivative of peripheral insulin concentration.
    dI_dt = np.zeros(len(I), dtype=float)
    dI_dt[0] = 0.0  # fasting steady-state approximation
    for k in range(1, len(I) - 1):
        dI_dt[k] = (I[k + 1] - I[k - 1]) / (t[k + 1] - t[k - 1])
    dI_dt[-1] = (I[-1] - I[-2]) / (t[-1] - t[-2])

    rdins_time = isr_pm2 - Vd_L_m2 * dI_dt  # pmol/min/m²
    mcri_raw_L_min_m2 = rdins_time / I      # L/min/m²
    mcri_time_L_min_m2 = np.maximum(0.0, mcri_raw_L_min_m2)

    # Legacy 4-interval apparent MCRi, kept for backward-compatible column names.
    # This follows the previous app's interval formula but clamps negative values to 0.
    dt_arr = np.diff(t)
    dI_dt_interval = np.diff(I) / dt_arr
    I_mid = (I[:-1] + I[1:]) / 2.0
    Vd_L = vd_mL_per_kg * 1e-3 * w
    mcr_interval_raw = (isr_interval - Vd_L * dI_dt_interval) / I_mid  # L/min
    mcr_interval_L_min_kg = mcr_interval_raw / w                       # L/min/kg
    mcr_interval_mL_min_kg = np.maximum(0.0, mcr_interval_L_min_kg * 1000.0)
    mcr_fasting_mL_min_kg = max(0.0, (isr_b / I[0]) * 1000.0 / w)

    # AUC-based MCRI over the OGTT period. This avoids direct dependence on noisy dI/dt.
    duration = float(t[-1] - t[0])
    auc_mcri_raw = (float(np_trapezoid(isr_pm2 / I, t))
                    - float((np.log(I[-1]) - np.log(I[0])) * Vd_L_m2))
    mean_mcri_raw_L_min_m2 = auc_mcri_raw / duration if duration > 0 else np.nan
    flag_neg_integral = bool(pd.notna(mean_mcri_raw_L_min_m2) and mean_mcri_raw_L_min_m2 < 0)
    mean_mcri_L_min_m2 = max(0.0, mean_mcri_raw_L_min_m2) if pd.notna(mean_mcri_raw_L_min_m2) else np.nan

    flag_neg = bool(np.any(mcri_raw_L_min_m2 < 0) or np.any(mcr_interval_raw < 0))

    # Convert L/min/m² to mL/min/kg for compatibility with previous outputs.
    conv_to_mL_min_kg = 1000.0 * bsa / w
    mcri_time_mL_min_kg = mcri_time_L_min_m2 * conv_to_mL_min_kg
    mean_mcri_mL_min_kg = mean_mcri_L_min_m2 * conv_to_mL_min_kg if pd.notna(mean_mcri_L_min_m2) else np.nan

    return {
        # Legacy-compatible names; interval semantics are preserved and values are now non-negative constrained.
        "mcr_fasting_mL_min_kg": float(mcr_fasting_mL_min_kg),
        "mcr_interval_mL_min_kg": mcr_interval_mL_min_kg,
        "mcr_integral_mL_min_kg": float(mean_mcri_mL_min_kg) if pd.notna(mean_mcri_mL_min_kg) else np.nan,
        # New explicit timepoint/AUC outputs.
        "mcri_time_L_min_m2": mcri_time_L_min_m2,
        "mcri_time_mL_min_kg": mcri_time_mL_min_kg,
        "mcri_raw_L_min_m2": mcri_raw_L_min_m2,
        "mcr_interval_raw_mL_min_kg": mcr_interval_raw * 1000.0 / w,
        "rdins_time_pmol_min_m2": rdins_time,
        "dI_dt_pmol_L_min": dI_dt,
        "MCRI_OGTT_L_min_m2": float(mean_mcri_L_min_m2) if pd.notna(mean_mcri_L_min_m2) else np.nan,
        "MCRI_OGTT_raw_L_min_m2": float(mean_mcri_raw_L_min_m2) if pd.notna(mean_mcri_raw_L_min_m2) else np.nan,
        "MCRI_OGTT_mL_min_kg": float(mean_mcri_mL_min_kg) if pd.notna(mean_mcri_mL_min_kg) else np.nan,
        "flag_negative_mcr": flag_neg,
        "flag_negative_mcr_integral": flag_neg_integral,
        "mcr_negative_values_were_clamped": bool(flag_neg or flag_neg_integral),
        "vd_mL_per_kg": vd_mL_per_kg,
    }


def calc_portal_insulin_glucagon_ratio(isr_result, glucagon_pg_mL, insulin_uU_mL):
    glg_pg = np.asarray(glucagon_pg_mL, dtype=float)
    ins_uU = np.asarray(insulin_uU_mL, dtype=float)
    if np.any(np.isnan(glg_pg)) or np.any(glg_pg <= 0):
        raise ValueError("Glucagon data is missing or invalid.")
    bsa = isr_result["params"]["bsa"]
    hpf = HPF_L_MIN_M2 * bsa
    mcr_glg = MCR_GLG_L_MIN_M2 * bsa
    isr = isr_result["isr"]
    ins_pmol = iri_uU_to_pmol(ins_uU)
    glg_pmol = glucagon_pg_to_pmol_L(glg_pg)
    ins_mid = (ins_pmol[:-1] + ins_pmol[1:]) / 2.0
    glg_mid = (glg_pmol[:-1] + glg_pmol[1:]) / 2.0
    portal_ins = isr / hpf + ins_mid
    portal_glg = glg_mid * (1.0 + mcr_glg / hpf)
    ratio = portal_ins / portal_glg
    return {"ratio_portal": ratio, "portal_ins_pmol_L": portal_ins, "portal_glg_pmol_L": portal_glg,
            "periph_ins_pmol_L": ins_pmol, "periph_glg_pmol_L": glg_pmol, "t_mids": isr_result["t_mids"]}

# =========================================================
# QC and processing
# =========================================================
def add_data_shape_flags(input_df, result_df):
    df = result_df.copy()
    raw = input_df.set_index("ID")
    flags_60_low, flags_90_low, flags_drop_30_60 = [], [], []
    for sid in df["ID"]:
        row = raw.loc[sid]
        g0, g30, g60, g90 = float(row["O-BG(0)"]), float(row["O-BG(30)"]), float(row["O-BG(60)"]), float(row["O-BG(90)"])
        flags_60_low.append(g60 < g0)
        flags_90_low.append(g90 < g0)
        flags_drop_30_60.append((g30 - g60) > 60)
    df["flag_bg60_below_fasting"] = flags_60_low
    df["flag_bg90_below_fasting"] = flags_90_low
    df["flag_glucose_drop_30_60_gt_60mgdl"] = flags_drop_30_60
    return df


def classify_result_quality_improved(result_df):
    df = result_df.copy()
    pf_cols = ["potentiation_factor_0_30", "potentiation_factor_30_60", "potentiation_factor_60_90", "potentiation_factor_90_120"]
    labels, reasons_list = [], []
    for _, row in df.iterrows():
        reasons_ex, reasons_warn = [], []
        pf = pd.to_numeric(row[pf_cols], errors="coerce")
        pf_non_nan = pf.dropna()
        if bool(row.get("mcr_negative_values_were_clamped", False)):
            reasons_warn.append("raw apparent MCRi was negative before non-negative constraint")
        if pf.isna().sum() >= 2:
            reasons_warn.append("multiple NaN potentiation factors")
        if len(pf_non_nan) and (pf_non_nan >= 2.9).any():
            reasons_warn.append("potentiation factor close to upper bound")
        if bool(row.get("flag_bg60_below_fasting", False)):
            reasons_warn.append("BG at 60 min below fasting")
        if bool(row.get("flag_bg90_below_fasting", False)):
            reasons_warn.append("BG at 90 min below fasting")
        if bool(row.get("flag_glucose_drop_30_60_gt_60mgdl", False)):
            reasons_warn.append("rapid glucose drop from 30 to 60 min")
        if pd.notna(row.get("CP_fit_RMS_pmol_L", np.nan)) and row.get("CP_fit_RMS_pmol_L", 0) > 50:
            reasons_warn.append("high C-peptide deconvolution RMS")
        if pd.notna(row.get("MARI_fit_RMS_pmol_min", np.nan)) and row.get("MARI_fit_RMS_pmol_min", 0) > 200:
            reasons_warn.append("high simplified Mari fit RMS")
        if reasons_ex:
            labels.append("除外候補")
            reasons_list.append("; ".join(reasons_ex + reasons_warn))
        elif reasons_warn:
            labels.append("要注意")
            reasons_list.append("; ".join(reasons_warn))
        else:
            labels.append("採用")
            reasons_list.append("")
    df["QC_label"] = labels
    df["QC_reasons"] = reasons_list
    return df


def process_standardized_ogtt_excel(df, vd_mL_per_kg=DEFAULT_VD):
    records, errors = [], []
    for _, row in df.iterrows():
        sid = row["ID"]
        try:
            h = float(row["Ht"])
            w = float(row["Wt"])
            age = float(row["Age"])
            cp_ng = np.array([row[f"O-CPR({int(t)})"] for t in TIME_POINTS], dtype=float)
            bg_mg = np.array([row[f"O-BG({int(t)})"] for t in TIME_POINTS], dtype=float)
            iri_uU = np.array([row[f"O-IRI({int(t)})"] for t in TIME_POINTS], dtype=float)
            cp_pmol = cp_ng_to_pmol(cp_ng)

            isr_res = estimate_isr(TIME_POINTS, cp_pmol, h, w, age)
            mari_res = estimate_mari_constrained(isr_res, bg_mg)
            mcr_res = estimate_mcr(isr_res, iri_uU, vd_mL_per_kg=vd_mL_per_kg)
            di_res = calc_oral_disposition_indices(bg_mg, iri_uU)

            glg_cols = [f"O-Glg({int(t)})" for t in TIME_POINTS]
            has_glg = all(c in row.index for c in glg_cols)
            glg_pg = np.array([row[c] for c in glg_cols], dtype=float) if has_glg else np.full(5, np.nan)
            glg_valid = has_glg and not np.any(np.isnan(glg_pg)) and np.all(glg_pg > 0)
            glg_ratio_res = None
            if glg_valid:
                try:
                    glg_ratio_res = calc_portal_insulin_glucagon_ratio(isr_res, glg_pg, iri_uU)
                except Exception:
                    glg_ratio_res = None

            b = mari_res["beta"]
            mi = mcr_res["mcr_interval_mL_min_kg"]
            p = isr_res["params"]

            rec = {
                "ID": sid, "model_version": APP_VERSION,
                "method_note": "ISR: two-compartment C-peptide deconvolution; k01 adjusted by age and BMI in v1.6; beta-cell indices: simplified Mari-type model; MCRi: apparent single-compartment estimate with non-negative physiological constraint",
                "Ht_cm": h, "Wt_kg": w, "Age_years": age, "BMI_kg_per_m2": p["bmi"], "BSA_m2": p["bsa"],
                "k01_min_inv": p["k01"], "k21_min_inv": p["k21"], "k12_min_inv": p["k12"], "V1_L": p["V1"],
                "fasting_CPR_0_pmol_L": float(cp_pmol[0]),
                "fasting_insulin_0_pmol_L": float(iri_uU_to_pmol(iri_uU[0])),
                "fasting_CPR_insulin_molar_ratio": float(calc_fasting_cpr_insulin_molar_ratio(cp_ng[0], iri_uU[0])),
                "AUC_Glu_mg_dL_min": di_res["AUC_Glu_mg_dL_min"], "AUC_IRI_uU_mL_min": di_res["AUC_IRI_uU_mL_min"],
                "Matsuda_index": di_res["Matsuda_index"], "insulinogenic_index": di_res["insulinogenic_index"],
                "oral_DI1": di_res["oral_DI1"], "oral_DI2": di_res["oral_DI2"],
                "ISR_basal_pmol_min": isr_res["isr_basal"],
                "ISR_0_30_pmol_min": isr_res["isr"][0], "ISR_30_60_pmol_min": isr_res["isr"][1],
                "ISR_60_90_pmol_min": isr_res["isr"][2], "ISR_90_120_pmol_min": isr_res["isr"][3],
                "AUC_ISR_pmol": isr_res["auc_isr"], "CP_fit_RMS_pmol_L": isr_res["rms_fit"],
                "CP_deconv_condition_number": isr_res["H_condition_number"],
                "glucose_sensitivity_pmol_min_per_mgdl": mari_res["Phi_s"],
                "rate_sensitivity_pmol_per_mgdl": mari_res["Phi_d"],
                "glucose_sensitivity_BSA": mari_res["Phi_s_bsa"], "rate_sensitivity_BSA": mari_res["Phi_d_bsa"],
                "glucose_sensitivity_kg": mari_res["Phi_s_kg"], "rate_sensitivity_kg": mari_res["Phi_d_kg"],
                "potentiation_factor_0_30": b[0], "potentiation_factor_30_60": b[1],
                "potentiation_factor_60_90": b[2], "potentiation_factor_90_120": b[3],
                "potentiation_factor_mean_early": mari_res["beta_mean_early"],
                "potentiation_factor_mean_late": mari_res["beta_mean_late"],
                "potentiation_ratio": mari_res["potentiation_ratio"], "MARI_fit_RMS_pmol_min": mari_res["rms_mari"],
                "MCR_fasting_mL_min_per_kg": mcr_res["mcr_fasting_mL_min_kg"],
                "MCR_0_30_mL_min_per_kg": mi[0], "MCR_30_60_mL_min_per_kg": mi[1],
                "MCR_60_90_mL_min_per_kg": mi[2], "MCR_90_120_mL_min_per_kg": mi[3],
                "MCR_integral_mL_min_per_kg": mcr_res["mcr_integral_mL_min_kg"],
                "MCRI_OGTT_L_min_m2": mcr_res["MCRI_OGTT_L_min_m2"],
                "MCRI_OGTT_mL_min_per_kg": mcr_res["MCRI_OGTT_mL_min_kg"],
                "MCRI_0_L_min_m2": mcr_res["mcri_time_L_min_m2"][0],
                "MCRI_30_L_min_m2": mcr_res["mcri_time_L_min_m2"][1],
                "MCRI_60_L_min_m2": mcr_res["mcri_time_L_min_m2"][2],
                "MCRI_90_L_min_m2": mcr_res["mcri_time_L_min_m2"][3],
                "MCRI_120_L_min_m2": mcr_res["mcri_time_L_min_m2"][4],
                "MCRI_0_mL_min_kg": mcr_res["mcri_time_mL_min_kg"][0],
                "MCRI_30_mL_min_kg": mcr_res["mcri_time_mL_min_kg"][1],
                "MCRI_60_mL_min_kg": mcr_res["mcri_time_mL_min_kg"][2],
                "MCRI_90_mL_min_kg": mcr_res["mcri_time_mL_min_kg"][3],
                "MCRI_120_mL_min_kg": mcr_res["mcri_time_mL_min_kg"][4],
                "RdIns_0_pmol_min_m2": mcr_res["rdins_time_pmol_min_m2"][0],
                "RdIns_30_pmol_min_m2": mcr_res["rdins_time_pmol_min_m2"][1],
                "RdIns_60_pmol_min_m2": mcr_res["rdins_time_pmol_min_m2"][2],
                "RdIns_90_pmol_min_m2": mcr_res["rdins_time_pmol_min_m2"][3],
                "RdIns_120_pmol_min_m2": mcr_res["rdins_time_pmol_min_m2"][4],
                "flag_negative_mcr": mcr_res["flag_negative_mcr"],
                "flag_negative_mcr_integral": mcr_res["flag_negative_mcr_integral"],
                "mcr_negative_values_were_clamped": mcr_res["mcr_negative_values_were_clamped"],
                "fasting_glucagon_pmol_L": float(glucagon_pg_to_pmol_L(glg_pg[0])) if glg_valid else np.nan,
                "portal_I_Glg_ratio_0_30": glg_ratio_res["ratio_portal"][0] if glg_ratio_res else np.nan,
                "portal_I_Glg_ratio_30_60": glg_ratio_res["ratio_portal"][1] if glg_ratio_res else np.nan,
                "portal_I_Glg_ratio_60_90": glg_ratio_res["ratio_portal"][2] if glg_ratio_res else np.nan,
                "portal_I_Glg_ratio_90_120": glg_ratio_res["ratio_portal"][3] if glg_ratio_res else np.nan,
                "portal_I_Glg_ratio_mean": float(np.mean(glg_ratio_res["ratio_portal"])) if glg_ratio_res else np.nan,
                "AUC_portal_I_Glg_ratio": float(np.sum(glg_ratio_res["ratio_portal"] * np.diff(TIME_POINTS))) if glg_ratio_res else np.nan,
            }
            records.append(rec)
        except Exception as e:
            errors.append({"ID": sid, "error": str(e)})
    result_df = pd.DataFrame(records)
    error_df = pd.DataFrame(errors)
    if len(result_df) > 0:
        result_df = add_data_shape_flags(df, result_df)
        result_df = classify_result_quality_improved(result_df)
    return result_df, error_df

# =========================================================
# Plotting and export
# =========================================================
def build_subject_results(row, vd_mL_per_kg=DEFAULT_VD):
    h = float(row["Ht"]); w = float(row["Wt"]); age = float(row["Age"])
    cp_ng = np.array([row[f"O-CPR({int(t)})"] for t in TIME_POINTS], dtype=float)
    bg_mg = np.array([row[f"O-BG({int(t)})"] for t in TIME_POINTS], dtype=float)
    iri_uU = np.array([row[f"O-IRI({int(t)})"] for t in TIME_POINTS], dtype=float)
    cp_pmol = cp_ng_to_pmol(cp_ng)
    isr_res = estimate_isr(TIME_POINTS, cp_pmol, h, w, age)
    mari_res = estimate_mari_constrained(isr_res, bg_mg)
    mcr_res = estimate_mcr(isr_res, iri_uU, vd_mL_per_kg=vd_mL_per_kg)
    glg_cols = [f"O-Glg({int(t)})" for t in TIME_POINTS]
    has_glg = all(c in row.index for c in glg_cols)
    glg_pg = np.array([row[c] for c in glg_cols], dtype=float) if has_glg else np.full(5, np.nan)
    glg_valid = has_glg and not np.any(np.isnan(glg_pg)) and np.all(glg_pg > 0)
    glg_ratio_res = calc_portal_insulin_glucagon_ratio(isr_res, glg_pg, iri_uU) if glg_valid else None
    return isr_res, mari_res, mcr_res, bg_mg, iri_uU, glg_pg, glg_valid, glg_ratio_res


def make_subject_figure(row, vd_mL_per_kg=DEFAULT_VD):
    sid = str(row["ID"])
    isr_res, mari_res, mcr_res, bg_mg, iri_uU, glg_pg, glg_valid, glg_ratio_res = build_subject_results(row, vd_mL_per_kg)
    mids = isr_res["t_mids"]; widths = isr_res["dt"] * 0.72
    ncols = 4 if glg_valid else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5.3 * ncols, 8.5))
    fig.suptitle(f"{sid} | Age {isr_res['params']['age_years']:.0f} y | BMI {isr_res['params']['bmi']:.1f} kg/m² | BSA {isr_res['params']['bsa']:.2f} m²", fontsize=14)

    ax = axes[0, 0]; ax.plot(TIME_POINTS, bg_mg, "o-", lw=2); ax.set_title("Glucose"); ax.set_xlabel("Time (min)"); ax.set_ylabel("mg/dL"); ax.grid(alpha=0.3)
    ax = axes[0, 1]; ax.plot(TIME_POINTS, iri_uU, "o-", lw=2); ax.set_title("Insulin"); ax.set_xlabel("Time (min)"); ax.set_ylabel("μU/mL"); ax.grid(alpha=0.3)
    ax = axes[0, 2]; ax.plot(TIME_POINTS, isr_res["cp_obs"], "o-", label="Observed"); ax.plot(TIME_POINTS, isr_res["cp_fit"], "s--", label="Fitted"); ax.set_title("C-peptide fit"); ax.set_xlabel("Time (min)"); ax.set_ylabel("pmol/L"); ax.legend(); ax.grid(alpha=0.3)
    if glg_valid and ncols == 4:
        ax = axes[0, 3]; ax.plot(TIME_POINTS, glg_pg, "o-"); ax.set_title("Glucagon"); ax.set_xlabel("Time (min)"); ax.set_ylabel("pg/mL"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    basal = np.full(4, isr_res["isr_basal"])
    ax.bar(mids, basal, width=widths, alpha=0.6, label="Basal")
    ax.bar(mids, mari_res["isr_static"], bottom=basal, width=widths, alpha=0.75, label="Static")
    ax.bar(mids, mari_res["isr_dynamic"], bottom=basal + mari_res["isr_static"], width=widths, alpha=0.75, label="Dynamic")
    ax.plot(mids, isr_res["isr"], "^", ms=7, label="Deconvolved ISR")
    ax.set_title("ISR decomposition"); ax.set_xticks(mids); ax.set_xticklabels(["0–30", "30–60", "60–90", "90–120"]); ax.set_ylabel("pmol/min"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]; ax.plot(mids, mari_res["beta"], "o-", lw=2); ax.axhline(1.0, ls="--"); ax.set_title("Potentiation factor"); ax.set_xticks(mids); ax.set_xticklabels(["0–30", "30–60", "60–90", "90–120"]); ax.set_ylabel("dimensionless"); ax.grid(alpha=0.3)
    ax = axes[1, 2]
    mcri_tp = mcr_res["mcri_time_mL_min_kg"]
    ax.plot(TIME_POINTS, mcri_tp, "o-", lw=2, color="steelblue", label="Timepoint MCRi")
    ax.bar(mids, mcr_res["mcr_interval_mL_min_kg"], width=widths, alpha=0.35, color="steelblue", label="Interval MCRi")
    ax.axhline(mcr_res["mcr_fasting_mL_min_kg"], ls="--", color="gray", label="Fasting")
    ax.axhline(mcr_res["MCRI_OGTT_mL_min_kg"], ls=":", color="tomato", label="AUC mean")
    ax.set_title("Apparent MCRi — timepoint (non-negative)")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("mL/min/kg")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if glg_valid and ncols == 4 and glg_ratio_res is not None:
        ax = axes[1, 3]; ratio = glg_ratio_res["ratio_portal"]; ax.plot(mids, ratio, "o-", lw=2); ax.set_title("Portal I/Glg ratio"); ax.set_xticks(mids); ax.set_xticklabels(["0–30", "30–60", "60–90", "90–120"]); ax.set_ylabel("mol/mol"); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def dataframe_to_excel_single_sheet_bytes(df, sheet_name="template"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def dataframe_to_excel_bytes(result_df, error_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="results", index=False)
        error_df.to_excel(writer, sheet_name="errors", index=False)
        units_df = pd.DataFrame({"column": list(RESULT_UNITS.keys()), "unit": list(RESULT_UNITS.values())})
        units_df.to_excel(writer, sheet_name="units", index=False)
        if "QC_label" in result_df.columns:
            result_df[result_df["QC_label"] == "採用"].to_excel(writer, sheet_name="accepted", index=False)
            result_df[result_df["QC_label"] == "要注意"].to_excel(writer, sheet_name="warning", index=False)
            result_df[result_df["QC_label"] == "除外候補"].to_excel(writer, sheet_name="exclude_candidates", index=False)
    output.seek(0)
    return output.getvalue()


def figures_to_pdf_bytes(df, vd_mL_per_kg=DEFAULT_VD):
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        for _, row in df.iterrows():
            try:
                fig = make_subject_figure(row, vd_mL_per_kg)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                plt.close("all")
                continue
    output.seek(0)
    return output.getvalue()


def single_case_to_standard_df(sid, age, ht, wt, bg0, bg30, bg60, bg90, bg120, iri0, iri30, iri60, iri90, iri120, cpr0, cpr30, cpr60, cpr90, cpr120, glg0=None, glg30=None, glg60=None, glg90=None, glg120=None):
    return pd.DataFrame([{
        "ID": sid, "Age": age, "Ht": ht, "Wt": wt,
        "O-BG(0)": bg0, "O-BG(30)": bg30, "O-BG(60)": bg60, "O-BG(90)": bg90, "O-BG(120)": bg120,
        "O-IRI(0)": iri0, "O-IRI(30)": iri30, "O-IRI(60)": iri60, "O-IRI(90)": iri90, "O-IRI(120)": iri120,
        "O-CPR(0)": cpr0, "O-CPR(30)": cpr30, "O-CPR(60)": cpr60, "O-CPR(90)": cpr90, "O-CPR(120)": cpr120,
        "O-Glg(0)": glg0 if glg0 is not None else np.nan,
        "O-Glg(30)": glg30 if glg30 is not None else np.nan,
        "O-Glg(60)": glg60 if glg60 is not None else np.nan,
        "O-Glg(90)": glg90 if glg90 is not None else np.nan,
        "O-Glg(120)": glg120 if glg120 is not None else np.nan,
    }])

# =========================================================
# UI
# =========================================================
st.title("神戸大学臨床糖尿病グループ | OGTT ISR / β-cell Function / apparent MCRi Calculator")
st.markdown("**{}**".format(APP_VERSION))
st.caption("ISR推定、簡略化Mari型β細胞機能指標、apparent MCRi、任意のportal I/Glg ratioを算出します。主要出力は丸めずに保存します。")

with st.expander("方法と解釈", expanded=False):
    st.markdown("""
- **ISR**: C-peptide 2コンパートメントモデルに基づくdeconvolutionで推定します。v1.6では k01 を年齢とBMIで補正します。
- **β細胞機能指標**: Mariモデルの考え方に基づく**簡略化Mari型モデル**です。glucose sensitivity、rate sensitivity、potentiation factorを出力します。
- **potentiation factor**: 30分間隔OGTTでは不安定になりやすいため、非負・平均1・隣接区間の軽度平滑化制約を加えています。
- **MCRi**: ISRと末梢インスリン濃度から単一コンパートメント近似で算出する**apparent MCRi**です。
- **portal I/Glg ratio**: グルカゴン値が全時点で入力された場合のみ算出します。単位はmol/molです。

重要: 本アプリのβ細胞機能指標は研究用のmodel-derived exploratory indicesです。
""")

with st.sidebar:
    st.header("Settings")
    vd_mL_per_kg = st.number_input("Insulin distribution volume Vd (mL/kg)", min_value=50.0, max_value=300.0, value=float(DEFAULT_VD), step=1.0)
    show_preview_n = st.slider("Preview rows", 1, 20, 5)

tab_upload, tab_single = st.tabs(["Batch upload", "Single-case input"])

with tab_upload:
    st.subheader("Input template for batch upload")
    template_df = create_input_template_df()
    st.caption("必須単位: Age = years, Glucose = mg/dL, Insulin = μU/mL, C-peptide = ng/mL, Height = cm, Weight = kg | Glucagon = pg/mL（任意）")
    st.dataframe(template_df, use_container_width=True)
    st.download_button("Download Excel template", data=dataframe_to_excel_single_sheet_bytes(template_df), file_name="OGTT_input_template_v1_6_1.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with st.expander("Accepted standard columns"):
        st.code("\n".join(DISPLAY_REQUIRED_COLUMNS))

    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], accept_multiple_files=False, key="batch")
    if uploaded_file is not None:
        try:
            raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error("Failed to read Excel file: {}".format(e)); st.stop()
        st.subheader("Raw input preview")
        st.dataframe(raw_df.head(show_preview_n), use_container_width=True)
        std_df, mapping, missing = standardize_input_df(raw_df)
        st.subheader("Matched columns")
        if mapping:
            st.dataframe(pd.DataFrame({"canonical_name": list(mapping.keys()), "matched_input_column": list(mapping.values())}), use_container_width=True)
        if missing:
            st.error("Missing required inputs after flexible matching:")
            st.code("\n".join(missing))
        else:
            st.subheader("Standardized input preview")
            st.dataframe(std_df.head(show_preview_n), use_container_width=True)
            if st.button("Run batch analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result_df, error_df = process_standardized_ogtt_excel(std_df, vd_mL_per_kg=vd_mL_per_kg)
                    excel_bytes = dataframe_to_excel_bytes(result_df, error_df)
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                    pdf_bytes = figures_to_pdf_bytes(std_df, vd_mL_per_kg=vd_mL_per_kg)
                st.success("Analysis completed.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processed", len(result_df)); c2.metric("Errors", len(error_df))
                c3.metric("Accepted", int((result_df["QC_label"] == "採用").sum()) if len(result_df) else 0)
                c4.metric("Warnings/Exclusions", int((result_df["QC_label"] != "採用").sum()) if len(result_df) else 0)
                st.subheader("Results preview")
                st.dataframe(result_df.head(20), use_container_width=True)
                st.download_button("Download results Excel", data=excel_bytes, file_name="OGTT_ISR_betaFunction_MCRi_results_v1_6_1.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                st.download_button("Download results CSV", data=csv_bytes, file_name="OGTT_ISR_betaFunction_MCRi_results_v1_6_1.csv", mime="text/csv", use_container_width=True)
                st.download_button("Download plots PDF", data=pdf_bytes, file_name="OGTT_all_subjects_plots_v1_6_1.pdf", mime="application/pdf", use_container_width=True)

with tab_single:
    st.subheader("Manual single-case entry")
    with st.form("single_case_form"):
        c1, c2, c3, c4 = st.columns(4)
        sid = c1.text_input("ID", value="Case-1")
        age = c2.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
        ht = c3.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        wt = c4.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        st.markdown("**Glucose (mg/dL)**")
        g = st.columns(5)
        bg0 = g[0].number_input("0 min", value=90.0, step=1.0, key="bg0")
        bg30 = g[1].number_input("30 min", value=150.0, step=1.0, key="bg30")
        bg60 = g[2].number_input("60 min", value=140.0, step=1.0, key="bg60")
        bg90 = g[3].number_input("90 min", value=120.0, step=1.0, key="bg90")
        bg120 = g[4].number_input("120 min", value=100.0, step=1.0, key="bg120")
        st.markdown("**Insulin (μU/mL)**")
        ii = st.columns(5)
        iri0 = ii[0].number_input("0 min ", value=5.0, step=0.1, key="iri0")
        iri30 = ii[1].number_input("30 min ", value=40.0, step=0.1, key="iri30")
        iri60 = ii[2].number_input("60 min ", value=50.0, step=0.1, key="iri60")
        iri90 = ii[3].number_input("90 min ", value=35.0, step=0.1, key="iri90")
        iri120 = ii[4].number_input("120 min ", value=20.0, step=0.1, key="iri120")
        st.markdown("**C-peptide (ng/mL)**")
        cc = st.columns(5)
        cpr0 = cc[0].number_input("0 min  ", value=1.5, step=0.1, key="cpr0")
        cpr30 = cc[1].number_input("30 min  ", value=5.0, step=0.1, key="cpr30")
        cpr60 = cc[2].number_input("60 min  ", value=7.0, step=0.1, key="cpr60")
        cpr90 = cc[3].number_input("90 min  ", value=6.0, step=0.1, key="cpr90")
        cpr120 = cc[4].number_input("120 min  ", value=4.5, step=0.1, key="cpr120")
        st.markdown("**Glucagon (pg/mL) — Optional; 0 means missing**")
        gg = st.columns(5)
        glg0 = gg[0].number_input("0 min   ", value=0.0, min_value=0.0, step=1.0, key="glg0")
        glg30 = gg[1].number_input("30 min   ", value=0.0, min_value=0.0, step=1.0, key="glg30")
        glg60 = gg[2].number_input("60 min   ", value=0.0, min_value=0.0, step=1.0, key="glg60")
        glg90 = gg[3].number_input("90 min   ", value=0.0, min_value=0.0, step=1.0, key="glg90")
        glg120 = gg[4].number_input("120 min   ", value=0.0, min_value=0.0, step=1.0, key="glg120")
        submitted = st.form_submit_button("Run single-case analysis", use_container_width=True)
    if submitted:
        glg_values = [glg0, glg30, glg60, glg90, glg120]
        glg_entered = all(v > 0 for v in glg_values)
        single_df = single_case_to_standard_df(sid, age, ht, wt, bg0, bg30, bg60, bg90, bg120, iri0, iri30, iri60, iri90, iri120, cpr0, cpr30, cpr60, cpr90, cpr120,
                                                glg0 if glg_entered else None, glg30 if glg_entered else None, glg60 if glg_entered else None, glg90 if glg_entered else None, glg120 if glg_entered else None)
        result_df, error_df = process_standardized_ogtt_excel(single_df, vd_mL_per_kg=vd_mL_per_kg)
        st.success("Single-case analysis completed.")
        if len(error_df):
            st.error(error_df.iloc[0]["error"])
        else:
            st.dataframe(result_df, use_container_width=True)
            fig = make_subject_figure(single_df.iloc[0], vd_mL_per_kg=vd_mL_per_kg)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.download_button("Download single-case Excel", data=dataframe_to_excel_bytes(result_df, error_df), file_name="{}_OGTT_ISR_betaFunction_MCRi_v1_6_1.xlsx".format(sid), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            st.download_button("Download single-case CSV", data=result_df.to_csv(index=False).encode("utf-8-sig"), file_name="{}_OGTT_ISR_betaFunction_MCRi_v1_6_1.csv".format(sid), mime="text/csv", use_container_width=True)

import numpy as np

def _model_binary(mu0, mu, f, **kw):
    # Binary (simple shadowing)
    return f

def _model_lambert(mu0, mu, f, **kw):
    # Lambert
    return f * mu0

def _model_ls(mu0, mu, f, eps=1e-8, k=0.0, **kw):
    # Lommel–Seeliger
    return f * (mu * mu0 / (mu0 + mu + eps))

def _model_ls_lambert(mu0, mu, f, k=0.4, eps=1e-8, **kw):
    # Blend of LS and Lambert; k in [0..1]
    return f * ((1 - k) * (mu * mu0 / (mu0 + mu + eps)) + k * mu0)

def _opposition_bump(s_vec, v_vec=(0, 0, 1), B0=0.5, sigma_deg=5.0):
    # Simple global opposition surge (shadow hiding / coherent backscatter)
    s = np.asarray(s_vec, float)
    s /= np.linalg.norm(s)
    v = np.asarray(v_vec, float)
    v /= np.linalg.norm(v_vec)
    g = np.arccos(np.clip(np.dot(s, v), -1.0, 1.0))  # phase angle (rad)
    return 1.0 + B0 * np.exp(-((np.degrees(g) / sigma_deg) ** 2))

def _model_ls_opposition(mu0, mu, f, k=0.4, eps=1e-3, s_vec=None, **kw):
    I = _model_ls_lambert(mu0, mu, f, k=k, eps=eps)
    if s_vec is not None:
        I *= _opposition_bump(
            s_vec, B0=kw.get("B0", 0.45), sigma_deg=kw.get("sigma_deg", 5.0)
        )
    return I

_MODEL_FUNCS = {
    "binary": _model_binary,
    "lambert": _model_lambert,
    "ls": _model_ls,
    "ls_lambert": _model_ls_lambert,
    "ls_opposition": _model_ls_opposition,  
}



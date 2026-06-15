"""
Helper module for loading and applying custom electron MVA model.
Loads pre-trained XGBoost model and applies it to electron events.
"""

import os
import pickle
import joblib


# Model paths (relative to this module or absolute)
_MODEL_DIR = "/eos/user/m/mkumari/Lepton-MVA-Run3/models_old"
_MODEL_PATH = os.path.join(_MODEL_DIR, "ele_xgb_clf.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "ele_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "ele_features.pkl")

# Default feature list (fallback if loading fails)
_DEFAULT_ELECTRON_FEATURES = [
    'pt', 'eta',
    'Irel_neutral', 'Irel_charged',
    'pratio', 'prel_T', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'hoe', 'sieie', 'deltaEtaSC', 'eInvMinusPInv', 'mvaNoIso',
]

# Singleton cache for model and scaler (loaded once)
_model = None
_scaler = None
_features = None


def _load_model():
    """Load and cache the trained model and scaler with robust fallbacks."""
    global _model, _scaler, _features

    if _model is None:
        # Try to load model
        model_error = None
        if os.path.exists(_MODEL_PATH):
            try:
                with open(_MODEL_PATH, "rb") as f:
                    _model = pickle.load(f, encoding='latin1')
            except Exception as e:
                model_error = e
                _model = None
        else:
            model_error = FileNotFoundError(f"Model not found at {_MODEL_PATH}")
        
        # Try to load scaler (saved via joblib.dump in training)
        scaler_error = None
        if os.path.exists(_SCALER_PATH):
            try:
                _scaler = joblib.load(_SCALER_PATH)
            except Exception as e:
                scaler_error = e
                _scaler = None
        else:
            scaler_error = FileNotFoundError(f"Scaler not found at {_SCALER_PATH}")
        
        # Try to load features
        features_error = None
        if os.path.exists(_FEATURES_PATH):
            try:
                with open(_FEATURES_PATH, "rb") as f:
                    _features = pickle.load(f, encoding='latin1')
            except Exception as e:
                features_error = e
                _features = None
        else:
            features_error = FileNotFoundError(f"Features not found at {_FEATURES_PATH}")
        
        # If model or scaler failed, raise an exception
        # (features failing is non-critical, we have defaults)
        if model_error is not None:
            raise RuntimeError(f"Failed to load model from {_MODEL_PATH}: {model_error}")
        if scaler_error is not None:
            raise RuntimeError(f"Failed to load scaler from {_SCALER_PATH}: {scaler_error}")

    return _model, _scaler, _features


def compute_electron_mva_score(events) -> "ak.Array":
    """
    Compute custom electron MVA scores using trained XGBoost model.

    Expected electron features (16 total):
    'pt', 'eta', 'Irel_neutral', 'Irel_charged',
    'pratio', 'prel_T', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'hoe', 'sieie', 'deltaEtaSC', 'eInvMinusPInv', 'mvaNoIso'

    Features computed from NanoAOD following same recipe as training in Lepton-MVA-Run3/src/lepton_producer.py

    Args:
        events: NanoAOD-like awkward array with Electron collection

    Returns:
        Awkward array with electron MVA scores (per-electron, same structure as events.Electron.pt)
    """
    # Lazy imports - only load when function is called
    import numpy as np
    import awkward as ak

    model, scaler, features = _load_model()

    electron = events.Electron
    jet = events.Jet

    # Flatten electrons first to avoid shape mismatch issues
    def _flat(branch):
        return ak.to_numpy(ak.flatten(branch)).astype(np.float32)

    # Extract basic electron properties (flatten to 1D)
    el_pt    = _flat(electron.pt)
    el_eta   = _flat(electron.eta)
    el_phi   = _flat(electron.phi)
    el_dxy   = _flat(electron.dxy)
    el_dz    = _flat(electron.dz)
    el_sip3d = _flat(electron.sip3d)

    # Handle optional branches - check if they exist and are not None
    try:
        el_iso_all = _flat(electron.miniPFRelIso_all)
    except (AttributeError, ValueError):
        el_iso_all = np.zeros_like(el_pt)

    try:
        el_iso_chg = _flat(electron.miniPFRelIso_chg)
    except (AttributeError, ValueError):
        el_iso_chg = np.zeros_like(el_pt)

    try:
        el_hoe = _flat(electron.hoe)
    except (AttributeError, ValueError):
        el_hoe = np.zeros_like(el_pt)

    try:
        el_sieie = _flat(electron.sieie)
    except (AttributeError, ValueError):
        el_sieie = np.zeros_like(el_pt)

    try:
        el_deltaEtaSC = _flat(electron.deltaEtaSC)
    except (AttributeError, ValueError):
        el_deltaEtaSC = np.zeros_like(el_pt)

    try:
        el_eInvMinusPInv = _flat(electron.eInvMinusPInv)
    except (AttributeError, ValueError):
        el_eInvMinusPInv = np.zeros_like(el_pt)

    try:
        el_mvaNoIso = _flat(electron.mvaNoIso)
    except (AttributeError, ValueError):
        el_mvaNoIso = np.zeros_like(el_pt)

    # -------------------------------------------------------------------------
    # Jet matching: fill None in jetIdx BEFORE any boolean operations.
    # electron.jetIdx comes as an option-type (?int32) in awkward when events have
    # no matched jet, causing bitwise_and to fail on None-typed arrays.
    # Filling None -> -1 converts it to a plain integer array first.
    # -------------------------------------------------------------------------
    el_jetidx_ak = ak.fill_none(electron.jetIdx, -1)

    # Cast to int32 explicitly to guarantee a plain (non-option) integer type
    el_jetidx_ak = ak.values_astype(el_jetidx_ak, np.int32)

    n_jets_ak = ak.num(jet)  # Per-event number of jets

    # Broadcast n_jets to match electron structure (per-electron) and cast to int32
    n_jets_per_electron = ak.values_astype(
        ak.broadcast_arrays(n_jets_ak, el_jetidx_ak)[0],
        np.int32,
    )

    # Check validity: jetIdx >= 0 and jetIdx < num_jets (both per-electron)
    # Both sides are now plain int32 arrays — bitwise_and is safe
    valid_ak     = (el_jetidx_ak >= 0) & (el_jetidx_ak < n_jets_per_electron)
    jidx_safe_ak = ak.where(valid_ak, el_jetidx_ak, 0)  # 0 as safe fallback

    # Pad jets to avoid index out of bounds
    max_jets = int(ak.max(n_jets_ak)) + 1 if ak.max(n_jets_ak) >= 0 else 1

    def _gather_jet(branch):
        """Safely gather jet properties matched to electrons."""
        if branch is None:
            return np.zeros_like(el_pt)
        try:
            padded   = ak.pad_none(branch, max_jets, clip=True)
            gathered = padded[jidx_safe_ak]
            filled   = ak.fill_none(gathered, 0.0)
            return _flat(filled)
        except (AttributeError, ValueError, TypeError):
            return np.zeros_like(el_pt)

    # Get matched jet properties
    matched_jpt  = _gather_jet(jet.pt)
    matched_jphi = _gather_jet(jet.phi)

    try:
        bpnet_branch = jet.btagPNetB
    except (AttributeError, ValueError):
        bpnet_branch = None
    matched_bpnet = _gather_jet(bpnet_branch)

    try:
        ncon_branch = jet.nConstituents
    except (AttributeError, ValueError):
        ncon_branch = None
    matched_ncon = _gather_jet(ncon_branch)

    # Flatten valid mask to 1D numpy bool
    valid_flat = ak.to_numpy(ak.flatten(valid_ak)).astype(bool)

    # Compute features
    def _delta_phi(phi1, phi2):
        """Compute delta-phi in [-pi, pi]."""
        dphi = np.abs(phi1 - phi2)
        return np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)

    # pratio = electron_pt / jet_pt (0 if no matched jet)
    with np.errstate(divide="ignore", invalid="ignore"):
        pratio = np.where(
            valid_flat & (matched_jpt > 0),
            el_pt / matched_jpt,
            0.0,
        ).astype(np.float32)

    # prel_T = abs(electron_pt * sin(delta_phi))
    dphi   = _delta_phi(el_phi, matched_jphi)
    prel_T = np.where(
        valid_flat,
        np.abs(el_pt * np.sin(dphi)),
        0.0,
    ).astype(np.float32)

    # Isolation features
    Irel_charged = el_iso_chg
    Irel_neutral = (el_iso_all - el_iso_chg).astype(np.float32)

    # log-transformed IP variables
    log_dxy = np.log(np.abs(el_dxy) + 1e-10).astype(np.float32)
    log_dz  = np.log(np.abs(el_dz)  + 1e-10).astype(np.float32)

    # B-tagging and nTracks (0 if no matched jet)
    btagPNetB = np.where(valid_flat, matched_bpnet, 0.0).astype(np.float32)
    ntracks   = np.where(valid_flat, matched_ncon,  0.0).astype(np.float32)

    # Build feature dictionary with all computed features
    computed = {
        "pt":             el_pt,
        "eta":            el_eta,
        "Irel_neutral":   Irel_neutral,
        "Irel_charged":   Irel_charged,
        "pratio":         pratio,
        "prel_T":         prel_T,
        "ntracks":        ntracks,
        "btagPNetB":      btagPNetB,
        "log_dxy":        log_dxy,
        "log_dz":         log_dz,
        "sip3d":          el_sip3d,
        "hoe":            el_hoe,
        "sieie":          el_sieie,
        "deltaEtaSC":     el_deltaEtaSC,
        "eInvMinusPInv":  el_eInvMinusPInv,
        "mvaNoIso":       el_mvaNoIso,
    }

    # Build feature matrix in correct order
    # Use saved features if available, otherwise use defaults
    feat_order = _features if _features is not None else _DEFAULT_ELECTRON_FEATURES
    
    X_list = []
    for feat in feat_order:
        if feat in computed:
            X_list.append(computed[feat])
        else:
            # Missing feature - fill with zeros
            X_list.append(np.zeros_like(el_pt))

    X = np.column_stack(X_list).astype(np.float32)

    # Apply scaler (trained on same features)
    X_scaled = scaler.transform(X)

    # Get predictions (probabilities for positive class = prompt electron)
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X_scaled)
            scores = model.predict(dmatrix)
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions from model: {e}")

    # Reshape back to awkward structure (per-electron)
    num_electrons = ak.num(electron.pt)
    scores = ak.unflatten(scores, num_electrons)

    return scores

"""
Helper module for loading and applying custom muon MVA model.
Loads pre-trained XGBoost model and applies it to muon events.
"""

import os
import pickle
import joblib


# Model paths (relative to this module or absolute)
_MODEL_DIR = "/eos/user/m/mkumari/hhmultilepton2/multilepton/mva_model"
_MODEL_PATH = os.path.join(_MODEL_DIR, "mu_xgb_clf.pkl")
_SCALER_PATH = os.path.join(_MODEL_DIR, "mu_scaler.pkl")
_FEATURES_PATH = os.path.join(_MODEL_DIR, "mu_features.pkl")

# Default feature list (fallback if loading fails)
_DEFAULT_MUON_FEATURES = [
    'pt', 'eta',
    'Irel_neutral', 'Irel_charged',
    'pratio', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'segmentComp', 'nTrackerLayers',
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


def compute_muon_mva_score(events) -> "ak.Array":
    """
    Compute custom muon MVA scores using trained XGBoost model.

    Expected muon features (12 total):
    'pt', 'eta', 'Irel_neutral', 'Irel_charged', 
    'pratio', 'ntracks', 'btagPNetB',
    'log_dxy', 'log_dz', 'sip3d',
    'segmentComp', 'nTrackerLayers'

    Features computed from NanoAOD following same recipe as training in Lepton-MVA-Run3/src/lepton_producer.py

    Args:
        events: NanoAOD-like awkward array with Muon collection

    Returns:
        Awkward array with muon MVA scores (per-muon, same structure as events.Muon.pt)
    """
    # Lazy imports - only load when function is called
    import numpy as np
    import awkward as ak

    model, scaler, features = _load_model()

    muon = events.Muon
    jet = events.Jet

    # Flatten muons first to avoid shape mismatch issues
    def _flat(branch):
        return ak.to_numpy(ak.flatten(branch)).astype(np.float32)

    # Extract basic muon properties (flatten to 1D)
    mu_pt    = _flat(muon.pt)
    mu_eta   = _flat(muon.eta)
    mu_phi   = _flat(muon.phi)
    mu_dxy   = _flat(muon.dxy)
    mu_dz    = _flat(muon.dz)
    mu_sip3d = _flat(muon.sip3d)

    # Handle optional branches - check if they exist and are not None
    try:
        mu_iso_all = _flat(muon.miniPFRelIso_all)
    except (AttributeError, ValueError):
        mu_iso_all = np.zeros_like(mu_pt)

    try:
        mu_iso_chg = _flat(muon.miniPFRelIso_chg)
    except (AttributeError, ValueError):
        mu_iso_chg = np.zeros_like(mu_pt)

    try:
        mu_seg = _flat(muon.segmentComp)
    except (AttributeError, ValueError):
        mu_seg = np.zeros_like(mu_pt)

    try:
        mu_nlayers = _flat(muon.nTrackerLayers)
    except (AttributeError, ValueError):
        mu_nlayers = np.zeros_like(mu_pt)

    # -------------------------------------------------------------------------
    # Jet matching: fill None in jetIdx BEFORE any boolean operations.
    # muon.jetIdx comes as an option-type (?int32) in awkward when events have
    # no matched jet, causing bitwise_and to fail on None-typed arrays.
    # Filling None -> -1 converts it to a plain integer array first.
    # -------------------------------------------------------------------------
    mu_jetidx_ak = ak.fill_none(muon.jetIdx, -1)

    # Cast to int32 explicitly to guarantee a plain (non-option) integer type
    mu_jetidx_ak = ak.values_astype(mu_jetidx_ak, np.int32)

    n_jets_ak = ak.num(jet)  # Per-event number of jets

    # Broadcast n_jets to match muon structure (per-muon) and cast to int32
    n_jets_per_muon = ak.values_astype(
        ak.broadcast_arrays(n_jets_ak, mu_jetidx_ak)[0],
        np.int32,
    )

    # Check validity: jetIdx >= 0 and jetIdx < num_jets (both per-muon)
    # Both sides are now plain int32 arrays — bitwise_and is safe
    valid_ak     = (mu_jetidx_ak >= 0) & (mu_jetidx_ak < n_jets_per_muon)
    jidx_safe_ak = ak.where(valid_ak, mu_jetidx_ak, 0)  # 0 as safe fallback

    # Pad jets to avoid index out of bounds
    max_jets = int(ak.max(n_jets_ak)) + 1 if ak.max(n_jets_ak) >= 0 else 1

    def _gather_jet(branch):
        """Safely gather jet properties matched to muons."""
        if branch is None:
            return np.zeros_like(mu_pt)
        try:
            padded   = ak.pad_none(branch, max_jets, clip=True)
            gathered = padded[jidx_safe_ak]
            filled   = ak.fill_none(gathered, 0.0)
            return _flat(filled)
        except (AttributeError, ValueError, TypeError):
            return np.zeros_like(mu_pt)

    # Get matched jet properties
    matched_jpt  = _gather_jet(jet.pt)

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

    # pratio = muon_pt / jet_pt (0 if no matched jet)
    with np.errstate(divide="ignore", invalid="ignore"):
        pratio = np.where(
            valid_flat & (matched_jpt > 0),
            mu_pt / matched_jpt,
            0.0,
        ).astype(np.float32)

    # Isolation features
    Irel_charged = mu_iso_chg
    Irel_neutral = (mu_iso_all - mu_iso_chg).astype(np.float32)

    # log-transformed IP variables
    log_dxy = np.log(np.abs(mu_dxy) + 1e-10).astype(np.float32)
    log_dz  = np.log(np.abs(mu_dz)  + 1e-10).astype(np.float32)

    # B-tagging and nTracks (0 if no matched jet)
    btagPNetB = np.where(valid_flat, matched_bpnet, 0.0).astype(np.float32)
    ntracks   = np.where(valid_flat, matched_ncon,  0.0).astype(np.float32)

    # Build feature dictionary with all computed features
    computed = {
        "pt":             mu_pt,
        "eta":            mu_eta,
        "Irel_neutral":   Irel_neutral,
        "Irel_charged":   Irel_charged,
        "pratio":         pratio,
        "ntracks":        ntracks,
        "btagPNetB":      btagPNetB,
        "log_dxy":        log_dxy,
        "log_dz":         log_dz,
        "sip3d":          mu_sip3d,
        "segmentComp":    mu_seg,
        "nTrackerLayers": mu_nlayers,
    }

    # Build feature matrix in correct order
    # Use saved features if available, otherwise use defaults
    feat_order = _features if _features is not None else _DEFAULT_MUON_FEATURES
    
    X_list = []
    for feat in feat_order:
        if feat in computed:
            X_list.append(computed[feat])
        else:
            # Missing feature - fill with zeros
            X_list.append(np.zeros_like(mu_pt))

    X = np.column_stack(X_list).astype(np.float32)

    # Apply scaler (trained on same features)
    X_scaled = scaler.transform(X)

    # Get predictions (probabilities for positive class = prompt muon)
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X_scaled)
            scores = model.predict(dmatrix)
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions from model: {e}")

    # Reshape back to awkward structure (per-muon)
    num_muons = ak.num(muon.pt)
    scores = ak.unflatten(scores, num_muons)

    return scores

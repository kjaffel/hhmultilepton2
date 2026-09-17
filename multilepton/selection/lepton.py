# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import os
import law

from operator import or_
from functools import reduce
from collections import defaultdict
from types import SimpleNamespace
from typing import NamedTuple

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import (
    set_ak_column, sorted_indices_from_mask, flat_np_view, full_like,
)
from columnflow.util import maybe_import
from columnflow.production.cms.jet import jet_id

from multilepton.util import (
    IF_NANO_V9, IF_NANO_GE_V10, IF_NANO_V12, IF_NANO_V14, IF_NANO_V15, IF_NOT_NANO_V15,
)
from multilepton.selection.muon_mva import compute_muon_mva_score
from multilepton.selection.electron_mva import compute_electron_mva_score
from multilepton.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")
logger = law.logger.get_logger(__name__)


class TIDGroups:
    def __init__(self, tid_tags):
        self._groups = defaultdict(list)
        for tid, tags in tid_tags.items():
            for tag in tags:
                self._groups[tag].append(tid)
            if {"cross_tau_tau", "cross_tau_tau_jet", "cross_tau_tau_vbf"} & tags:
                self._groups["cross_tau_tau_any"].append(tid)

    def __getattr__(self, name):
        # Always return empty list if missing
        return self._groups.get(name, [])

    def __getitem__(self, name):
        return self._groups.get(name, [])


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    /,
    *,
    threshold: float = 0.5,
    axis: int = 2,
    event_mask: ak.Array | type(Ellipsis) | None = None,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
        leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. If an *event_mask* is given, the
    the matching is performed only for those events, but a full object mask with the same shape as
    that of *vectors1* is returned, which all objects set to *False* where not matching was done.
    """
    # handle event masks
    used_event_mask = event_mask is not None and event_mask is not Ellipsis
    event_mask = Ellipsis if event_mask is None else event_mask
    # delta_r for all combinations
    dr = vectors1[event_mask].metric_table(vectors2[event_mask])
    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)
    # expand to original shape if an event mask was given
    if used_event_mask:
        full_any_match = full_like(vectors1.pt, False, dtype=bool)
        flat_full_any_match = flat_np_view(full_any_match)
        flat_full_any_match[flat_np_view(full_any_match | event_mask)] = flat_np_view(any_match)
        any_match = full_any_match
    return any_match


def update_channel_ids(
    events: ak.Array,
    previous_channel_ids: ak.Array,
    correct_channel_id: int,
    channel_mask: ak.Array,
) -> ak.Array:
    """
    Check if the events in the is_mask can be inside the given channel
    or have already been sorted in another channel before.
    """
    events_not_in_channel = (previous_channel_ids != 0) & (previous_channel_ids != correct_channel_id)
    channel_id_overwrite = events_not_in_channel & channel_mask
    if ak.any(channel_id_overwrite):
        raise ValueError(
            "The channel_ids of some events are being set to two different values. "
            "The first event of this chunk concerned has index",
            ak.where(channel_id_overwrite)[0],
        )
    return ak.where(channel_mask, correct_channel_id, previous_channel_ids)


def get_cone_pt_from_jetidx(
    lepton_pt: ak.Array,
    lepton_eta: ak.Array,
    lepton_phi: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
    closestjet_indicies: ak.Array,
    tight_mask: ak.Array,
    pfRelIso_03_or_04_all: ak.Array,
) -> ak.Array:
    """
    - if the lepton is tight:
        cone_pt = lepton_pt
    - else, if the associated jet exists and DeltaR(lepton, jet) < 0.4:
        cone_pt = 0.9 * jet_pt
    - else:
        cone_pt = 0.9 * lepton_pt * (1 + pfRelIso_03_or_04_all)
    """

    good_indicies = closestjet_indicies >= 0
    n_jets = ak.to_numpy(ak.num(jet_pt, axis=1))

    # Defining a global index for the closest jet in each event
    jet_offsets = np.cumsum(n_jets) - n_jets
    global_closestjet_indicies = closestjet_indicies + jet_offsets[:, np.newaxis]
    safe_global_closestjet_indicies = ak.where(
        good_indicies,
        global_closestjet_indicies,
        0,
    )
    flat_safe_global_closestjet_indicies = ak.to_numpy(ak.flatten(safe_global_closestjet_indicies, axis=1))
    # Flatenning closest jet associated quantities and associating them with the global indices
    flat_jet_pt = ak.flatten(jet_pt, axis=1)
    flat_jet_eta = ak.flatten(jet_eta, axis=1)
    flat_jet_phi = ak.flatten(jet_phi, axis=1)
    selected_flat_jet_pt = flat_jet_pt[flat_safe_global_closestjet_indicies]
    selected_flat_jet_eta = flat_jet_eta[flat_safe_global_closestjet_indicies]
    selected_flat_jet_phi = flat_jet_phi[flat_safe_global_closestjet_indicies]
    # Associating each lepton with the corresponding nearest jet
    # We need to unflatten the arrays, as the cone-pT need to have the same shape as lepton.pT
    lepton_counts = ak.num(lepton_pt, axis=1)
    closest_jet_pt = ak.unflatten(
        selected_flat_jet_pt,
        lepton_counts,
    )
    closest_jet_eta = ak.unflatten(
        selected_flat_jet_eta,
        lepton_counts,
    )
    closest_jet_phi = ak.unflatten(
        selected_flat_jet_phi,
        lepton_counts,
    )
    # Now we define the lepton's nearby jet kinematics in the jagged structure
    closest_jet_pt = ak.where(
        good_indicies,
        closest_jet_pt,
        lepton_pt,
    )
    closest_jet_eta = ak.where(
        good_indicies,
        closest_jet_eta,
        lepton_eta,
    )
    closest_jet_phi = ak.where(
        good_indicies,
        closest_jet_phi,
        lepton_phi,
    )
    # Computing DeltaR (metric table does not work here)
    delta_phi = lepton_phi - closest_jet_phi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    delta_eta = lepton_eta - closest_jet_eta
    closest_jet_DR = np.sqrt(delta_eta**2 + delta_phi**2)
    # See if the nearby jet is within a 0.4 cone with respect to the lepton
    has_nearby_jet = good_indicies & (closest_jet_DR < 0.4)
    # If lepton is tight or there are no nearby jets, cone-pT resolves to lepton pT
    cone_pt = ak.where(
        tight_mask,
        lepton_pt,
        ak.where(
            has_nearby_jet,
            0.9 * closest_jet_pt,
            0.9 * lepton_pt * (1 + pfRelIso_03_or_04_all),
        ),
    )

    return cone_pt


def hzz_iso_wp(electron, cuts=None):
    """
    Rebuilds Electron_mvaIso_WPHZZ from Electron_mvaHZZIso, for nano versions that contains the
    score but not the WP flag (e.g. v12).
    """
    # mvaEleID-Winter22-HZZ-V1 working point, from cms-sw/cmssw
    # RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Winter22_HZZ_V1_cff.py
    # nano stores tanh(raw) in Electron_mvaHZZIso (see MVAValueMapProducer.h in CMSSW)
    if cuts is None:
        cuts = np.tanh([
            1.633973689084034,    # EB1, 5 < pt < 10
            1.5499076306249353,   # EB2, 5 < pt < 10
            2.0629564440753247,   # EE,  5 < pt < 10
            0.3685228146685872,   # EB1, pt >= 10
            0.2662407818935475,   # EB2, pt >= 10
            -0.5444837363886459,  # EE,  pt >= 10
        ])

    # nano stores deltaEtaSC = superCluster().eta() - eta(), so superCluster().eta()
    abs_sc_eta = abs(electron.eta + electron.deltaEtaSC)

    categories = [
        (electron.pt < 10) & (abs_sc_eta < 0.800),
        (electron.pt < 10) & (abs_sc_eta >= 0.800) & (abs_sc_eta < 1.479),
        (electron.pt < 10) & (abs_sc_eta >= 1.479),
        (electron.pt >= 10) & (abs_sc_eta < 0.800),
        (electron.pt >= 10) & (abs_sc_eta >= 0.800) & (abs_sc_eta < 1.479),
        (electron.pt >= 10) & (abs_sc_eta >= 1.479),
    ]

    passed = ak.zeros_like(electron.pt, dtype=bool)
    for category, cut in zip(categories, cuts):
        passed = passed | (category & (electron.mvaHZZIso > cut))

    return passed


@selector(
    uses={
        "Electron.{pt,eta,phi,dxy,dz}",
        "Electron.{pfRelIso03_all,seediEtaOriX,seediPhiOriY,sip3d,miniPFRelIso_all,sieie}",
        "Electron.{hoe,eInvMinusPInv,convVeto,lostHits,jetPtRelv2,jetIdx}",
        # custom electron LeptonMVA input branches: without these declared here columnflow does
        # not load them, so compute_electron_mva_score silently fed zeros -> degraded score.
        "Electron.{miniPFRelIso_chg,deltaEtaSC,mvaNoIso}", "Jet.nConstituents",
        # v2 model inputs; btagDeepFlavB is read off the matched jet (Electron.jetIdx), not a
        # per-lepton branch
        "Electron.{pfRelIso03_all,jetNDauCharged,jetPtRelv2}",
        "Jet.{pt,eta,phi,btagDeepFlavB}",
        IF_NANO_V12("Electron.{mvaTTH,mvaHZZIso}", "Jet.btagPNetB"),
        IF_NANO_V14("Electron.{promptMVA,mvaIso_WPHZZ}", "Jet.btagPNetB"),
        IF_NANO_V15("Electron.{promptMVA,mvaIso_WPHZZ}", "Jet.{btagPNetB,btagUParTAK4B}"),
        IF_NANO_V9("Electron.mvaFall17V2{Iso_WP80,Iso_WP90}"),
        IF_NANO_GE_V10("Electron.{mvaIso_WP80,mvaIso_WP90}"),
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Electron selection returning three sets of masks and the cone-pT.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    # ch_key = kwargs.get("ch_key", None)
    # is_2016 = self.config_inst.campaign.x.year == 2016
    is_2022_post = (
        self.config_inst.campaign.x.year == 2022 and
        self.config_inst.campaign.has_tag("postEE")
    )
    is_single = trigger.has_tag("single_e") or trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_e_tau")

    if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    # btagcut_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    # btagcut_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]
    btagcut_tight = self.config_inst.x.btag_working_points[btag_tagger]["tight"]

    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        # beware that the available Iso should be mvaFall17V2 for run2 files, not Winter22V1,
        # check this in original root files if necessary
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
        if "mvaIso_WPHZZ" in events.Electron.fields:
            mva_iso_wphzz = events.Electron.mvaIso_WPHZZ
        elif "mvaHZZIso" in events.Electron.fields:
            # v12 carry the score but not the WPHZZ flag, so we apply the WP by hand
            mva_iso_wphzz = hzz_iso_wp(events.Electron)
        else:
            mva_iso_wphzz = None
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        mva_iso_wphzz = None

    # Get electron MVA source from config (default: "custom")
    # Options:
    #   "custom"  - XGBoost trained model from Lepton-MVA-Run3/models
    #   "nanoaod" - Default NanoAOD MVA (promptMVA for v14+, mvaTTH for v<14)
    electron_mva_source = getattr(self.config_inst.x, "electron_mva_source", "nanoaod")

    # Select electron MVA based on configured source
    if electron_mva_source == "custom":
        # Try to use custom trained XGBoost model
        try:
            promptMVA = compute_electron_mva_score(events)

        except Exception as e:
            # Fallback to NanoAOD MVA if custom model fails
            logger.warning_once(f"Failed to load custom electron MVA model ({e}), falling back to NanoAOD MVA")
            if "promptMVA" in events.Electron.fields:
                promptMVA = events.Electron.promptMVA
                logger.warning_once("Using NanoAOD promptMVA (v14+) as fallback for electron selection")
            else:
                promptMVA = events.Electron.mvaTTH
                logger.warning_once("Using NanoAOD mvaTTH (v<14) as fallback for electron selection")

    elif electron_mva_source == "nanoaod":
        # Use NanoAOD default MVA based on version
        if "promptMVA" in events.Electron.fields:
            # >= nano v14
            promptMVA = events.Electron.promptMVA
            logger.warning_once("Using NanoAOD promptMVA (v14+) for electron selection")
        else:
            # nano <v14
            promptMVA = events.Electron.mvaTTH
            logger.warning_once("Using NanoAOD mvaTTH (v<14) for electron selection")

    else:
        raise ValueError(f"Invalid electron_mva_source '{electron_mva_source}'. "
                       f"Choose from: 'custom' (XGBoost model), 'nanoaod' (version-based default)")
    # =========================================
    # MVA comparison debugging (gen-matched ROC check)
    # =========================================
    # NOTE: counting how many electrons pass a fixed *numeric* cut (e.g. 0.3) on both
    # scores is NOT a valid comparison: the custom and nanoAOD scores live on different
    # scales, so 0.3 is a different working point for each, and the sample here mixes
    # prompt (signal) and fake (background) electrons. A ROC gain only shows up when you
    # (a) separate signal/background via truth and (b) compare at a MATCHED efficiency.
    if os.environ.get("MVA_FEATURE_DEBUG") and self.dataset_inst.is_mc and "genPartFlav" in events.Electron.fields:
        try:
            custom = ak.to_numpy(ak.flatten(compute_electron_mva_score(events))).astype(np.float64)
        except Exception as cmp_e:
            logger.warning_once(f"[Comparison electron] custom electron MVA failed ({cmp_e})")
            custom = None

        # same nanoAOD score the selection actually uses
        if "promptMVA" in events.Electron.fields:
            nano = ak.to_numpy(ak.flatten(events.Electron.promptMVA)).astype(np.float64)
        else:
            nano = ak.to_numpy(ak.flatten(events.Electron.mvaTTH)).astype(np.float64)

        flav = ak.to_numpy(ak.flatten(events.Electron.genPartFlav))
        is_sig = (flav == 1)          # prompt electron -> signal
        is_bkg = (flav == 0)          # jet fake        -> background
        keep = is_sig | is_bkg        # drop tau/photon-conversion electrons for a clean 2-class ROC

        if custom is not None and is_sig.sum() > 0 and is_bkg.sum() > 0:
            y = is_sig[keep].astype(int)
            xc, xn = custom[keep], nano[keep]
            n_sig, n_bkg = int(y.sum()), int((y == 0).sum())

            # threshold-free discrimination: AUC (this is what the ROC "gain" reflects)
            from sklearn.metrics import roc_auc_score
            auc_c, auc_n = roc_auc_score(y, xc), roc_auc_score(y, xn)

            # apples-to-apples working point: take nano's signal eff at its 0.3 cut, find the
            # custom threshold giving the SAME signal eff, then compare background efficiency
            sig_eff_nano = float((xn[y == 1] > 0.3).mean())
            bkg_eff_nano = float((xn[y == 0] > 0.3).mean())
            thr_c = float(np.quantile(xc[y == 1], 1.0 - sig_eff_nano))
            bkg_eff_custom = float((xc[y == 0] > thr_c).mean())

            logger.info_once(
                f"[Comparison electron] gen-matched n_sig={n_sig} n_bkg={n_bkg} | \n"
                f"AUC custom={auc_c:.4f} nano={auc_n:.4f} | \n"
                f"@ matched sig-eff={sig_eff_nano:.3f}: bkg-eff nano={bkg_eff_nano:.4f} \n"
                f"custom={bkg_eff_custom:.4f} (custom thr={thr_c:.3f})",
            )

            # ---- event-level yield comparison (nano vs custom) ----
            # Count events keeping >=1 electron passing each MVA. For a FAIR comparison we use
            # nano at its analysis cut (0.3) and custom at the threshold matched to the SAME
            # signal efficiency (thr_c), so any yield difference reflects extra fake rejection,
            # not just a looser/tighter numeric cut. (Stats here are one file only.)
            n_per_evt = ak.to_numpy(ak.num(events.Electron.pt, axis=1))
            custom_jag = ak.unflatten(custom, n_per_evt)
            nano_jag = ak.unflatten(nano, n_per_evt)
            nano_thr, custom_thr = 0.3, thr_c

            n_evt = len(events)
            evt_nano = int(ak.sum(ak.any(nano_jag > nano_thr, axis=1)))
            evt_custom = int(ak.sum(ak.any(custom_jag > custom_thr, axis=1)))
            sig_keep_nano = int(((nano > nano_thr) & is_sig).sum())
            sig_keep_custom = int(((custom > custom_thr) & is_sig).sum())
            fake_keep_nano = int(((nano > nano_thr) & is_bkg).sum())
            fake_keep_custom = int(((custom > custom_thr) & is_bkg).sum())

            # S/sqrt(B) significance-like figure: at matched signal eff S is ~equal, so the gain
            # is driven by fake (B) reduction. Reported as a single number + relative gain.
            z_nano = sig_keep_nano / np.sqrt(fake_keep_nano) if fake_keep_nano > 0 else float("inf")
            z_custom = sig_keep_custom / np.sqrt(fake_keep_custom) if fake_keep_custom > 0 else float("inf")
            z_gain = (z_custom / z_nano - 1.0) * 100.0 if np.isfinite(z_nano) and z_nano > 0 else float("nan")

            logger.info_once(
                f"[Yield electron] nano>{nano_thr:.3f} vs custom>{custom_thr:.3f} (matched sig-eff) | \n"
                f"events(>=1 sel e)/{n_evt}: nano={evt_nano} custom={evt_custom} \n"
                f"(delta={evt_custom - evt_nano:+d}) | \n"
                f"prompt-e kept: nano={sig_keep_nano} custom={sig_keep_custom} | \n"
                f"fake-e kept: nano={fake_keep_nano} custom={fake_keep_custom} \n"
                f"(fakes removed by custom: {fake_keep_nano - fake_keep_custom:+d}) | \n"
                f"S/sqrt(B): nano={z_nano:.2f} custom={z_custom:.2f} (gain={z_gain:+.1f}%)",
            )
    # =========================================

    # default electron mask
    tight_mask = None
    fakeable_mask = None
    if is_single or is_cross or True:  # investigate why trigger dependence on providing masks
        # min_pt = 26.0 if is_2016 else (31.0 if is_single else 25.0)
        # max_eta = 2.5 if is_single else 2.1

        closestjet_indicies = events.Electron.jetIdx[:, :]
        bad_indicies = (closestjet_indicies == -1)  # set btag to 0 if no closest jet
        btag_pad = ak.fill_none(ak.pad_none(events.Jet[btag_discriminator], 1, axis=1), 0.0)
        btag_values = ak.where(
            bad_indicies, 0.0, btag_pad[ak.where(bad_indicies, 0, closestjet_indicies)],
        )
        abs_sc_eta = abs(events.Electron.eta + events.Electron.deltaEtaSC)
        sieie_max = ak.where(abs_sc_eta > 1.479, 0.030, 0.011)  # endcap, barrel

        atleast_loose = ((mva_iso_wp80 == 1) | (mva_iso_wp90 == 1))
        if mva_iso_wphzz is not None:
            atleast_loose = atleast_loose | (mva_iso_wphzz == 1)
        tight_mask = (
            (events.Electron.pt > 10) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.sieie < sieie_max) &
            (events.Electron.hoe < 0.1) &
            (events.Electron.eInvMinusPInv > -0.04) &
            (events.Electron.convVeto == 1) &
            (events.Electron.lostHits == 0) &
            atleast_loose &
            (promptMVA > 0.3) &
            (btag_values < btagcut_tight)
        )

        cone_pt = get_cone_pt_from_jetidx(
            events.Electron.pt,
            events.Electron.eta,
            events.Electron.phi,
            events.Jet.pt,
            events.Jet.eta,
            events.Jet.phi,
            closestjet_indicies,
            tight_mask,
            events.Electron.pfRelIso03_all,
        )

        loose_mask = (
            (events.Electron.pt > 7.0) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.lostHits <= 1) &
            atleast_loose
        )
        idlepmvapassed = (atleast_loose & (promptMVA > 0.3))
        idlepmvafailed = ((mva_iso_wp90 == 1) & (promptMVA <= 0.3))
        jetisolepmvapassed = (promptMVA > 0.3)
        jetisolepmvafailed = ((promptMVA <= 0.3) & (events.Electron.jetPtRelv2 < (1. / 1.7)))
        fakeable_mask = (
            (events.Electron.pt > 10) &
            (cone_pt > 10.0) &
            (abs(events.Electron.eta) < 2.5) &
            (abs(events.Electron.dxy) < 0.5) &
            (abs(events.Electron.dz) < 1) &
            (events.Electron.sip3d < 8) &
            (events.Electron.miniPFRelIso_all < 0.4) &
            (events.Electron.sieie < sieie_max) &
            (events.Electron.hoe < 0.1) &
            (events.Electron.eInvMinusPInv > -0.04) &
            (events.Electron.convVeto == 1) &
            (events.Electron.lostHits == 0) &
            (idlepmvapassed | idlepmvafailed) &
            (btag_values < btagcut_tight) &
            (jetisolepmvapassed | jetisolepmvafailed)
        )
        if is_2022_post:
            tight_mask = tight_mask & ~(
                (events.Electron.eta > 1.556) &
                (events.Electron.seediEtaOriX < 45) &
                (events.Electron.seediPhiOriY > 72)
            )
            fakeable_mask = fakeable_mask & ~(
                (events.Electron.eta > 1.556) &
                (events.Electron.seediEtaOriX < 45) &
                (events.Electron.seediPhiOriY > 72)
            )

    return tight_mask, fakeable_mask, loose_mask, cone_pt


@electron_selection.init
def electron_selection_init(self) -> None:
    if self.config_inst.campaign.x.run == 3 and self.config_inst.campaign.x.year == 2022:
        self.shifts |= {
            shift_inst.name for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag(("ees", "eer"))
        }


@selector(
    uses={"{Electron,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def electron_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Electron trigger matching.
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")

    # catch config errors
    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["e"].pdg_id) == 11
    return trigger_object_matching(
        events.Electron,
        events.TrigObj[leg_masks["e"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={
        "Muon.{pt,eta,phi,looseId,mediumId,tightId}",
        "Muon.{pfRelIso04_all,dxy,dz,sip3d,miniPFRelIso_all,jetPtRelv2,jetIdx}",
        # custom muon LeptonMVA input branches: without these declared here columnflow does not
        # load them, so compute_muon_mva_score silently fed zeros -> degraded score.
        "Muon.{miniPFRelIso_chg,nTrackerLayers,segmentComp,isTracker,nStations,isGlobal}",
        "Jet.nConstituents",
        # v2 model inputs; btagDeepFlavB is read off the matched jet (Muon.jetIdx), not a
        # per-lepton branch
        "Muon.{pfRelIso03_all,jetNDauCharged,jetPtRelv2}",
        "Jet.{pt,eta,phi,btagDeepFlavB}",
        IF_NANO_V12("Muon.mvaTTH", "Jet.btagPNetB"),
        IF_NANO_V14("Muon.promptMVA", "Jet.btagPNetB"),
        IF_NANO_V15("Muon.promptMVA", "Jet.{btagPNetB,btagUParTAK4B}"),
    },
    exposed=False,
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Muon selection returning three sets of masks and the cone-pT.
    References:
    - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
    - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15
    relaxed for multilepton, to be replaced with lepMVA later on
    """
    # ch_key = kwargs.get("ch_key", None)
    # is_2016 = self.config_inst.campaign.x.year == 2016
    is_single = trigger.has_tag("single_mu") or trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_mu_tau")

    if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    # btagcut_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    # btagcut_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]
    btagcut_tight = self.config_inst.x.btag_working_points[btag_tagger]["tight"]

    # Get muon MVA source from config (default: "custom")
    # Options:
    #   "custom"  - XGBoost trained model from Lepton-MVA-Run3/models
    #   "nanoaod" - Default NanoAOD MVA (promptMVA for v14+, mvaTTH for v<14)
    muon_mva_source = getattr(self.config_inst.x, "muon_mva_source", "nanoaod")

    # default muon mask
    tight_mask = None
    fakeable_mask = None
    if is_single or is_cross or True:  # investigate why trigger dependence on providing masks at all
        # if is_2016:
        #    min_pt = 23.0 if is_single else 20.0
        # else:
        #    min_pt = 26.0 if is_single else 22.0

        # Select muon MVA based on configured source
        if muon_mva_source == "custom":
            # Try to use custom trained XGBoost model
            try:
                promptMVA = compute_muon_mva_score(events)

            except Exception as e:
                # Fallback to NanoAOD MVA if custom model fails
                logger.warning(f"Failed to load custom muon MVA model ({e}), falling back to NanoAOD MVA")
                if "promptMVA" in events.Muon.fields:
                    promptMVA = events.Muon.promptMVA
                    logger.info_once("Using NanoAOD promptMVA (v14+) as fallback for muon selection")
                else:
                    promptMVA = events.Muon.mvaTTH
                    logger.info_once("Using NanoAOD mvaTTH (v<14) as fallback for muon selection")

        elif muon_mva_source == "nanoaod":
            # Use NanoAOD default MVA based on version
            if "promptMVA" in events.Muon.fields:
                # >= nano v14
                promptMVA = events.Muon.promptMVA
                logger.info_once("Using NanoAOD promptMVA (v14+) for muon selection")
            else:
                # nano <v14
                promptMVA = events.Muon.mvaTTH
                logger.info_once("Using NanoAOD mvaTTH (v<14) for muon selection")

        else:
            raise ValueError(f"Invalid muon_mva_source '{muon_mva_source}'. "
                           f"Choose from: 'custom' (XGBoost model), 'nanoaod' (version-based default)")

        # =========================================
        # MVA comparison debugging (gen-matched ROC check) -- mirrors the electron block.
        # Judge success by AUC / fake-rate at matched signal efficiency, NOT by counts at a
        # fixed numeric cut (the two scores are on different scales). Muon analysis cut is 0.5.
        # =========================================
        debug = True
        if debug and self.dataset_inst.is_mc and "genPartFlav" in events.Muon.fields:
            try:
                mu_custom = ak.to_numpy(ak.flatten(compute_muon_mva_score(events))).astype(np.float64)
            except Exception as cmp_e:
                logger.warning_once(f"[Comparison muon] custom muon MVA failed ({cmp_e})")
                mu_custom = None

            # same nanoAOD score the selection actually uses
            if "promptMVA" in events.Muon.fields:
                mu_nano = ak.to_numpy(ak.flatten(events.Muon.promptMVA)).astype(np.float64)
            else:
                mu_nano = ak.to_numpy(ak.flatten(events.Muon.mvaTTH)).astype(np.float64)

            mu_flav = ak.to_numpy(ak.flatten(events.Muon.genPartFlav))
            mu_is_sig = (mu_flav == 1)          # prompt muon -> signal
            mu_is_bkg = (mu_flav == 0)          # jet fake    -> background
            mu_keep = mu_is_sig | mu_is_bkg

            if mu_custom is not None and mu_is_sig.sum() > 0 and mu_is_bkg.sum() > 0:
                y = mu_is_sig[mu_keep].astype(int)
                xc, xn = mu_custom[mu_keep], mu_nano[mu_keep]
                n_sig, n_bkg = int(y.sum()), int((y == 0).sum())

                from sklearn.metrics import roc_auc_score
                auc_c, auc_n = roc_auc_score(y, xc), roc_auc_score(y, xn)

                # nano's signal eff at its 0.5 cut, then the custom threshold giving the SAME
                # signal eff -> compare background efficiency
                sig_eff_nano = float((xn[y == 1] > 0.5).mean())
                bkg_eff_nano = float((xn[y == 0] > 0.5).mean())
                thr_c = float(np.quantile(xc[y == 1], 1.0 - sig_eff_nano))
                bkg_eff_custom = float((xc[y == 0] > thr_c).mean())

                logger.info_once(
                    f"[Comparison muon] gen-matched n_sig={n_sig} n_bkg={n_bkg} | \n"
                    f"AUC custom={auc_c:.4f} nano={auc_n:.4f} | \n"
                    f"@ matched sig-eff={sig_eff_nano:.3f}: bkg-eff nano={bkg_eff_nano:.4f} \n"
                    f"custom={bkg_eff_custom:.4f} (custom thr={thr_c:.3f})",
                )

                # ---- event-level yield comparison (nano vs custom) at matched signal eff ----
                n_per_evt = ak.to_numpy(ak.num(events.Muon.pt, axis=1))
                custom_jag = ak.unflatten(mu_custom, n_per_evt)
                nano_jag = ak.unflatten(mu_nano, n_per_evt)
                nano_thr, custom_thr = 0.5, thr_c

                n_evt = len(events)
                evt_nano = int(ak.sum(ak.any(nano_jag > nano_thr, axis=1)))
                evt_custom = int(ak.sum(ak.any(custom_jag > custom_thr, axis=1)))
                sig_keep_nano = int(((mu_nano > nano_thr) & mu_is_sig).sum())
                sig_keep_custom = int(((mu_custom > custom_thr) & mu_is_sig).sum())
                fake_keep_nano = int(((mu_nano > nano_thr) & mu_is_bkg).sum())
                fake_keep_custom = int(((mu_custom > custom_thr) & mu_is_bkg).sum())

                # S/sqrt(B) significance-like figure (see electron block for rationale)
                z_nano = sig_keep_nano / np.sqrt(fake_keep_nano) if fake_keep_nano > 0 else float("inf")
                z_custom = sig_keep_custom / np.sqrt(fake_keep_custom) if fake_keep_custom > 0 else float("inf")
                z_gain = (z_custom / z_nano - 1.0) * 100.0 if np.isfinite(z_nano) and z_nano > 0 else float("nan")

                logger.info_once(
                    f"[Yield muon] nano>{nano_thr:.3f} vs custom>{custom_thr:.3f} (matched sig-eff) | \n"
                    f"events(>=1 sel mu)/{n_evt}: nano={evt_nano} custom={evt_custom} \n"
                    f"(delta={evt_custom - evt_nano:+d}) | \n"
                    f"prompt-mu kept: nano={sig_keep_nano} custom={sig_keep_custom} | \n"
                    f"fake-mu kept: nano={fake_keep_nano} custom={fake_keep_custom} \n"
                    f"(fakes removed by custom: {fake_keep_nano - fake_keep_custom:+d}) | \n"
                    f"S/sqrt(B): nano={z_nano:.2f} custom={z_custom:.2f} (gain={z_gain:+.1f}%)",
                )
        # =========================================

        closestjet_indicies = events.Muon.jetIdx[:, :]
        bad_indicies = (closestjet_indicies == -1)  # set btag to 0 if no closest jet
        btag_pad = ak.fill_none(ak.pad_none(events.Jet[btag_discriminator], 1, axis=1), 0.0)
        btag_values = ak.where(
            bad_indicies, 0.0, btag_pad[ak.where(bad_indicies, 0, closestjet_indicies)],
        )
        atleast_medium = ((events.Muon.mediumId == 1) | (events.Muon.tightId == 1))
        atleast_loose = ((events.Muon.looseId == 1) | (events.Muon.mediumId == 1) | (events.Muon.tightId == 1))
        tight_mask = (
            (events.Muon.pt > 10) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_medium &
            (btag_values < btagcut_tight) &
            (promptMVA > 0.5)
        )

        cone_pt = get_cone_pt_from_jetidx(
            events.Muon.pt,
            events.Muon.eta,
            events.Muon.phi,
            events.Jet.pt,
            events.Jet.eta,
            events.Jet.phi,
            closestjet_indicies,
            tight_mask,
            events.Muon.pfRelIso04_all,
        )

        loose_mask = (
            (events.Muon.pt > 5) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_loose
        )
        fakeable_mask = (
            (events.Muon.pt > 10) &
            (cone_pt > 10) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.05) &
            (abs(events.Muon.dz) < 0.1) &
            (events.Muon.sip3d < 8) &
            (events.Muon.miniPFRelIso_all < 0.4) &
            atleast_loose &
            (btag_values < btagcut_tight) &
            ((promptMVA > 0.5) | ((promptMVA <= 0.5) & (events.Muon.jetPtRelv2 < (1. / 1.8))))
        )

    return tight_mask, fakeable_mask, loose_mask, cone_pt


@selector(
    uses={"{Muon,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def muon_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Muon trigger matching.
    """
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")

    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["mu"].pdg_id) == 13
    return trigger_object_matching(
        events.Muon,
        events.TrigObj[leg_masks["mu"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={
        "Tau.{pt,eta,phi,dz,decayMode}",
        "{Electron,Muon,TrigObj}.{pt,eta,phi}",
    },
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    electron_mask: ak.Array | None,
    muon_mask: ak.Array | None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a masks for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select isolated ones, eventually to separate normal and iso inverted taus
    for QCD estimations.
    """
    # return empty mask if no tagged taus exists in the chunk
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        false_mask = full_like(events.Tau.pt, False, dtype=bool)
        return false_mask, false_mask

    # is_single_e = trigger.has_tag("single_e")
    # is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_run3 = self.config_inst.campaign.x.run == 3
    tagger = self.config_inst.x.tau_tagger
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    get_tau_tagger = lambda tag: f"{col_prefix}{tagger}VS{tag}"
    wp_config = self.config_inst.x.tau_id_working_points

    # determine minimum pt and maximum eta
    max_eta = 2.5
    base_pt = 20.0
    # if is_single_e or is_single_mu:
    if is_cross_e:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 35.0
    elif is_cross_mu:
        min_pt = 25.0 if is_2016 else 32.0
    elif is_cross_tau:
        min_pt = 40.0
    elif is_cross_tau_vbf:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 25.0
    elif is_cross_tau_jet:
        min_pt = None if not is_run3 else 35.0
    else:
        min_pt = 20.0

    # no_id mask for tagge rindependent tests
    noid_mask = (
        (abs(events.Tau.eta) < max_eta) &
        (events.Tau.pt > base_pt) &
        (abs(events.Tau.dz) < 0.2)
    )

    # Decay modes: Run 3 PNet includes DM=2, Run 2 HPS does not
    dm_modes = (0, 1, 2, 10, 11) if is_run3 else (0, 1, 10, 11)

    # base tau mask for default and qcd sideband tau (Fakeable selection)
    base_mask = noid_mask & (
        reduce(or_, [events.Tau.decayMode == mode for mode in dm_modes]) &
        (events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.vvvloose)
        # vs e and mu cuts are channel dependent and thus applied in the overall lepton selection
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_mask]) > 0.3, axis=2)
    if muon_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_mask]) > 0.3, axis=2)

    # trigger dependent cuts
    trigger_specific_mask = base_mask & (events.Tau.pt > min_pt)
    # compute the isolation mask separately as it is used to defined (qcd) categories later on
    iso_mask = events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.tight

    return base_mask, trigger_specific_mask, iso_mask, noid_mask


@tau_selection.init
def tau_selection_init(self: Selector) -> None:
    # register tec shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("tec")
    }
    # Add columns for the right tau tagger (id prefix for DeepTau, raw prefix for PNet)
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    tagger = self.config_inst.x.tau_tagger
    self.uses |= {
        f"Tau.{col_prefix}{tagger}VS{tag}"
        for tag in ("e", "mu", "jet")
    }


@selector(
    uses={"{Tau,TrigObj}.{pt,eta,phi}"},
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Tau trigger matching.
    """
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        return full_like(events.Tau.pt, False, dtype=bool)

    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_any_cross_tau = is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet
    assert is_cross_e or is_cross_mu or is_any_cross_tau

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs["tau"].pdg_id) == 15
        # match leg 1
        return trigger_object_matching(
            events.Tau,
            events.TrigObj[leg_masks["tau"]],
            event_mask=trigger_fired,
        )

    # is_any_cross_tau
    assert trigger.n_legs == len(leg_masks) >= 2
    assert abs(trigger.legs["tau1"].pdg_id) == 15
    assert abs(trigger.legs["tau2"].pdg_id) == 15

    # match both legs
    matches_leg0 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau1"]],
        event_mask=trigger_fired,
    )
    matches_leg1 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau2"]],
        event_mask=trigger_fired,
    )

    # taus need to be matched to at least one leg, but as a side condition
    # each leg has to have at least one match to a tau
    matches = (
        (matches_leg0 | matches_leg1) &
        ak.any(matches_leg0, axis=1) &
        ak.any(matches_leg1, axis=1)
    )
    return matches


# ────────────────────────────────────────────────────────────────
# channel definitions
# ────────────────────────────────────────────────────────────────

# evaluation order of the trigger groups (TIDGroups names), which matters for columns that are
# overwritten per trigger (e.g. leptons_os) and for the order of matched_trigger_ids
TRIGGER_GROUP_ORDER = (
    "single_e", "single_mu",
    "double_e", "double_mu", "double_emu",
    "triple_e", "triple_mu", "triple_eemu", "triple_emumu",
    "cross_e_tau", "cross_mu_tau", "cross_tau_tau_any",
)


def _add_mc_groups(streams: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    # unless given explicitly, simulation uses the triggers of all data streams
    if "mc" in streams:
        return streams
    groups = {group for stream_groups in streams.values() for group in stream_groups}
    unknown = groups - set(TRIGGER_GROUP_ORDER)
    if unknown:
        raise ValueError(f"trigger groups {unknown} missing in TRIGGER_GROUP_ORDER")
    return {**streams, "mc": tuple(group for group in TRIGGER_GROUP_ORDER if group in groups)}


# trigger groups evaluated per channel and data stream (second token of the dataset name, e.g.
# data_mu_d -> "mu"). a channel is skipped for data streams it has no entry for. simulation ("mc")
# evaluates the union of all data stream groups, unless an explicit "mc" entry is given
CHANNEL_TRIGGER_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {ch: _add_mc_groups(streams) for ch, streams in {
    # loose selection for the lepton BDT training, simulation only
    "ceormu": {"mc": ("single_e",)},
    # 3l0th, 3l1th, 4l
    "c3e": {"e": ("single_e", "double_e", "triple_e")},
    "c4e": {"e": ("single_e", "double_e", "triple_e")},
    "c3etau": {"e": ("single_e", "double_e", "triple_e", "cross_e_tau")},
    "c3mu": {"mu": ("single_mu", "double_mu", "triple_mu")},
    "c4mu": {"mu": ("single_mu", "double_mu", "triple_mu")},
    "c3mutau": {"mu": ("single_mu", "double_mu", "triple_mu", "cross_mu_tau")},
    "c2e2mu": {
        "muoneg": ("double_emu", "triple_emumu", "triple_eemu"),
        "mu": ("single_mu", "double_mu"),
        "e": ("single_e", "double_e"),
    },
    "c3emu": {
        "muoneg": ("double_emu", "triple_eemu"),
        "e": ("single_e", "double_e", "triple_e"),
        "mu": ("single_mu",),
    },
    "ce3mu": {
        "muoneg": ("double_emu", "triple_emumu"),
        "mu": ("single_mu", "double_mu", "triple_mu"),
        "e": ("single_e",),
    },
    "c2emu": {
        "muoneg": ("double_emu", "triple_eemu"),
        "e": ("single_e", "double_e"),
        "mu": ("single_mu",),
    },
    "c2emutau": {
        "muoneg": ("double_emu", "triple_eemu"),
        "e": ("single_e", "double_e", "cross_e_tau"),
        "mu": ("single_mu", "cross_mu_tau"),
    },
    "ce2mu": {
        "muoneg": ("double_emu", "triple_emumu"),
        "mu": ("single_mu", "double_mu"),
        "e": ("single_e",),
    },
    "ce2mutau": {
        "muoneg": ("double_emu", "triple_emumu"),
        "mu": ("single_mu", "double_mu", "cross_mu_tau"),
        "e": ("single_e", "cross_e_tau"),
    },
    # 2l0or1tau, 2l2th
    "c2eSS": {"e": ("single_e", "double_e")},
    "c2eSS1tau": {"e": ("single_e", "double_e", "cross_e_tau")},
    "c2e2tau": {
        "e": ("single_e", "double_e", "cross_e_tau"),
        "tau": ("cross_tau_tau_any",),
    },
    "c2muSS": {"mu": ("single_mu", "double_mu")},
    "c2muSS1tau": {"mu": ("single_mu", "double_mu", "cross_mu_tau")},
    "c2mu2tau": {
        "mu": ("single_mu", "double_mu", "cross_mu_tau"),
        "tau": ("cross_tau_tau_any",),
    },
    "cemu2tau": {
        "muoneg": ("double_emu",),
        "e": ("single_e", "cross_e_tau"),
        "mu": ("single_mu", "cross_mu_tau"),
        "tau": ("cross_tau_tau_any",),
    },
    "cemuSS": {
        "muoneg": ("double_emu",),
        "e": ("single_e",),
        "mu": ("single_mu",),
    },
    "cemuSS1tau": {
        "muoneg": ("double_emu",),
        "e": ("single_e", "cross_e_tau"),
        "mu": ("single_mu", "cross_mu_tau"),
    },
    # 1l3th
    "ce3tau": {
        "tau": ("cross_tau_tau_any",),
        "e": ("single_e", "cross_e_tau"),
    },
    "cmu3tau": {
        "tau": ("cross_tau_tau_any",),
        "mu": ("single_mu", "cross_mu_tau"),
    },
    # 1l2th, 4tauh
    "c4tau": {"tau": ("cross_tau_tau_any",)},
    "ce2tau": {
        "e": ("single_e", "cross_e_tau"),
        "tau": ("cross_tau_tau_any",),
    },
    "cmu2tau": {
        "mu": ("single_mu", "cross_mu_tau"),
        "tau": ("cross_tau_tau_any",),
    },
    # measurement regions
    "cttbarMR": {
        "muoneg": ("double_emu",),
        "e": ("single_e", "double_e", "cross_e_tau"),
        "mu": ("single_mu", "double_mu", "cross_mu_tau"),
    },
    "cwzMR": {
        "muoneg": ("double_emu", "triple_eemu", "triple_emumu"),
        "e": ("single_e", "double_e", "triple_e", "cross_e_tau"),
        "mu": ("single_mu", "double_mu", "triple_mu", "cross_mu_tau"),
    },
    "cdyMR": {"mu": ("single_mu", "double_mu", "cross_mu_tau")},
}.items()}


def get_channel_trigger_ids(
    tids: TIDGroups,
    ch_key: str,
    is_mc: bool,
    data_stream: str | None,
) -> list[int] | None:
    """
    Returns the ids of the fired triggers to evaluate for channel *ch_key*, or *None* if the channel
    is not evaluated at all for this dataset.
    """
    groups = CHANNEL_TRIGGER_GROUPS.get(ch_key, {}).get("mc" if is_mc else data_stream)
    if groups is None:
        return None
    return [tid for group in groups for tid in tids[group]]


class ChannelSpec(NamedTuple):
    """
    Object multiplicities and requirements of a physics channel.

    *light_charge* and *total_charge* are the required absolute charge sums of the light leptons
    (e + mu) and of all leptons (e + mu + tau), respectively, both unchecked when *None*. The result
    is stored in the leptons_os column. *trig_match* is the rule of :py:func:`trigger_match_ok`.
    """
    n_e: int
    n_mu: int
    n_tau: int
    light_charge: int | None = None
    total_charge: int | None = None
    trig_match: str = ""


PHYSICS_CHANNELS: dict[str, ChannelSpec] = {
    # 3l0th, 4l
    "c3e": ChannelSpec(3, 0, 0, light_charge=1, trig_match="e"),
    "c3mu": ChannelSpec(0, 3, 0, light_charge=1, trig_match="mu"),
    "c2emu": ChannelSpec(2, 1, 0, light_charge=1, trig_match="emu"),
    "ce2mu": ChannelSpec(1, 2, 0, light_charge=1, trig_match="emu"),
    "c4e": ChannelSpec(4, 0, 0, light_charge=0, trig_match="e"),
    "c4mu": ChannelSpec(0, 4, 0, light_charge=0, trig_match="mu"),
    "c3emu": ChannelSpec(3, 1, 0, light_charge=0, trig_match="emu"),
    "c2e2mu": ChannelSpec(2, 2, 0, light_charge=0, trig_match="emu"),
    "ce3mu": ChannelSpec(1, 3, 0, light_charge=0, trig_match="emu"),
    # 3l1th, 2l2th, 1l3th
    "c3etau": ChannelSpec(3, 0, 1, light_charge=1, total_charge=0, trig_match="etau"),
    "c2e2tau": ChannelSpec(2, 0, 2, total_charge=0, trig_match="etau_tt"),
    "ce3tau": ChannelSpec(1, 0, 3, total_charge=0, trig_match="etau"),
    "c3mutau": ChannelSpec(0, 3, 1, light_charge=1, total_charge=0, trig_match="mutau"),
    "c2mu2tau": ChannelSpec(0, 2, 2, total_charge=0, trig_match="mutau_tt"),
    "cmu3tau": ChannelSpec(0, 1, 3, total_charge=0, trig_match="mutau"),
    "c2emutau": ChannelSpec(2, 1, 1, light_charge=1, total_charge=0, trig_match="emutau"),
    "ce2mutau": ChannelSpec(1, 2, 1, light_charge=1, total_charge=0, trig_match="emutau"),
    "cemu2tau": ChannelSpec(1, 1, 2, total_charge=0, trig_match="emutau_tt"),
    # 2lSS1th, 2l0th
    "c2eSS1tau": ChannelSpec(2, 0, 1, light_charge=2, total_charge=1, trig_match="e"),
    "c2muSS1tau": ChannelSpec(0, 2, 1, light_charge=2, total_charge=1, trig_match="mu"),
    "cemuSS1tau": ChannelSpec(1, 1, 1, light_charge=2, total_charge=1, trig_match="emu"),
    "c2eSS": ChannelSpec(2, 0, 0, light_charge=0, trig_match="e"),
    "c2muSS": ChannelSpec(0, 2, 0, light_charge=0, trig_match="mu"),
    "cemuSS": ChannelSpec(1, 1, 0, light_charge=0, trig_match="emu"),
    # 1l2th, 4th
    "ce2tau": ChannelSpec(1, 0, 2, total_charge=1, trig_match="e"),
    "cmu2tau": ChannelSpec(0, 1, 2, total_charge=1, trig_match="mu"),
    "c4tau": ChannelSpec(0, 0, 4, total_charge=0, trig_match="tautau"),
}


# ────────────────────────────────────────────────────────────────
# channel helpers
# ────────────────────────────────────────────────────────────────

class _LeptonSelectionState:
    """
    Per-event flags and per-object masks accumulated over all channels and fired triggers.
    """

    def __init__(self, events: ak.Array) -> None:
        false_mask = (abs(events.event) < 0)
        self.channel_id = np.uint32(1) * false_mask
        self.ok_bdt_eormu = false_mask
        self.leptons_os = false_mask
        self.single_triggered = false_mask
        self.tight_sel = false_mask
        self.trig_match = false_mask
        self.tight_sel_bdt = false_mask
        self.trig_match_bdt = false_mask
        self.matched_trigger_ids = []

        e_false = full_like(events.Electron.pt, False, dtype=bool)
        mu_false = full_like(events.Muon.pt, False, dtype=bool)
        tau_false = full_like(events.Tau.pt, False, dtype=bool)
        self.sel_electron_mask = self.sel_looseelectron_mask = self.sel_tightelectron_mask = e_false
        self.sel_muon_mask = self.sel_loosemuon_mask = self.sel_tightmuon_mask = mu_false
        self.sel_tau_mask = self.sel_isotau_mask = self.sel_noid_tau_mask = tau_false

    def add_electrons(self, ok: ak.Array, ctrl: ak.Array, veto: ak.Array, tight: ak.Array) -> None:
        self.sel_electron_mask = self.sel_electron_mask | (ok & ctrl)
        self.sel_looseelectron_mask = self.sel_looseelectron_mask | (ok & veto)
        self.sel_tightelectron_mask = self.sel_tightelectron_mask | (ok & tight)

    def add_muons(self, ok: ak.Array, ctrl: ak.Array, veto: ak.Array, tight: ak.Array) -> None:
        self.sel_muon_mask = self.sel_muon_mask | (ok & ctrl)
        self.sel_loosemuon_mask = self.sel_loosemuon_mask | (ok & veto)
        self.sel_tightmuon_mask = self.sel_tightmuon_mask | (ok & tight)

    def add_taus(self, ok: ak.Array, ch_tau: ak.Array, iso: ak.Array) -> None:
        self.sel_tau_mask = self.sel_tau_mask | (ok & ch_tau)
        self.sel_isotau_mask = self.sel_isotau_mask | (ok & (ch_tau & iso))

    def add_matched_trigger(self, trig_match_ok: ak.Array, tid: int) -> None:
        self.single_triggered = ak.where(trig_match_ok, True, self.single_triggered)
        ids = ak.where(trig_match_ok, np.float32(tid), np.float32(np.nan))
        self.matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

    def add_channel_result(
        self,
        ok: ak.Array,
        chargeok: ak.Array,
        tight_ok: ak.Array,
        trig_match_ok: ak.Array,
        tid: int,
    ) -> None:
        self.leptons_os = ak.where(ok, chargeok, self.leptons_os)
        self.tight_sel = self.tight_sel | tight_ok
        self.trig_match = self.trig_match | trig_match_ok
        self.add_matched_trigger(trig_match_ok, tid)


def _n(mask: ak.Array) -> ak.Array:
    return ak.sum(mask, axis=1)


def trigger_match_ok(
    rule: str,
    tid: int,
    base_ok: ak.Array,
    t: SimpleNamespace,
    fam: SimpleNamespace,
    tids: TIDGroups,
) -> ak.Array:
    """
    Applies the trigger object matching *rule* of a channel for trigger *tid* on top of *base_ok*.
    *t* holds the object masks of trigger *tid*, *fam* the masks combined over all fired triggers.
    Triggers not covered by a rule keep *base_ok* as is.
    """
    e_matched = lambda n=1: _n(t.e_match & t.e_ctrl) >= n
    mu_matched = lambda n=1: _n(t.mu_match & t.mu_ctrl) >= n
    tau_matched = lambda: _n(t.tau_match & t.ch_tau_mask) >= 1

    # when a single_e (single_mu) trigger fired as well, an electron (muon) must be matched to one of them
    def and_e_side(ok):
        if_e_fired = base_ok & fam.e_trig_any & (_n(fam.e_match_any & t.e_ctrl) >= 1)
        return ak.where(fam.e_trig_any, ok & if_e_fired, ok)

    def and_mu_side(ok):
        if_mu_fired = base_ok & fam.mu_trig_any & (_n(fam.mu_match_any & t.mu_ctrl) >= 1)
        return ak.where(fam.mu_trig_any, ok & if_mu_fired, ok)

    with_tautau = rule.endswith("_tt")
    rule = rule.removesuffix("_tt")

    if rule == "e":
        return base_ok & e_matched()
    if rule == "mu":
        return base_ok & mu_matched()

    if rule == "emu":
        if tid in tids.single_e:
            # accept only events where no single_mu trigger fired (anti-overlap)
            return base_ok & fam.e_only & e_matched()
        if tid in tids.single_mu:
            return and_e_side(base_ok & mu_matched())
        return base_ok

    if rule in {"etau", "mutau"}:
        lep_trig, lep_matched, lep_only = {
            "etau": (tids.single_e, e_matched, fam.e_only_emutau),
            "mutau": (tids.single_mu, mu_matched, fam.mu_only_emutau),
        }[rule]
        cross_trig = tids.cross_e_tau if rule == "etau" else tids.cross_mu_tau
        if tid in lep_trig:
            return base_ok & lep_only & lep_matched()
        if tid in cross_trig:
            return base_ok & tau_matched() & lep_matched()
        if with_tautau and tid in tids.cross_tau_tau_any:
            return base_ok & tau_matched()
        return base_ok

    if rule == "emutau":
        if tid in tids.single_e:
            return and_mu_side(base_ok & e_matched())
        if tid in tids.single_mu:
            return and_e_side(base_ok & mu_matched())
        if tid in tids.cross_e_tau:
            return and_mu_side(base_ok & tau_matched() & e_matched())
        if tid in tids.cross_mu_tau:
            return and_e_side(base_ok & tau_matched() & mu_matched())
        if with_tautau and tid in tids.cross_tau_tau_any:
            return base_ok & tau_matched()
        return base_ok

    if rule == "tautau":
        if tid in tids.cross_tau_tau_any:
            return base_ok & tau_matched()
        return base_ok

    if rule == "ttbar_mr":
        if tid in tids.single_e:
            return and_mu_side(base_ok & e_matched())
        if tid in tids.single_mu:
            return and_e_side(base_ok & mu_matched())
        if tid in tids.double_e:
            return base_ok & e_matched(2)
        if tid in tids.double_mu:
            return base_ok & mu_matched(2)
        if tid in tids.double_emu:
            return base_ok & e_matched() & mu_matched()
        if tid in tids.cross_e_tau:
            return base_ok & tau_matched() & e_matched()
        if tid in tids.cross_mu_tau:
            return base_ok & tau_matched() & mu_matched()
        return base_ok

    if rule == "dy_mr":
        if tid in tids.single_mu:
            return base_ok & fam.mu_only_emutau & mu_matched()
        if tid in tids.double_mu:
            return base_ok & mu_matched()
        return base_ok

    raise ValueError(f"unknown trigger matching rule '{rule}'")


def physics_channel_ok(
    events: ak.Array,
    spec: ChannelSpec,
    t: SimpleNamespace,
) -> tuple[ak.Array, ak.Array, ak.Array]:
    """
    Returns the base selection, the charge requirement and the tight requirement of a physics channel.
    """
    base_ok = _n(t.ch_tau_mask) == spec.n_tau
    tight_ok = (_n(t.ch_tau_mask & t.tau_iso_mask) == spec.n_tau) if spec.n_tau else True
    light_charge = 0
    for n, ctrl, veto, tight, charge in [
        (spec.n_e, t.e_ctrl, t.e_veto, t.e_mask, events.Electron.charge),
        (spec.n_mu, t.mu_ctrl, t.mu_veto, t.mu_mask, events.Muon.charge),
    ]:
        if n:
            base_ok = base_ok & (_n(ctrl) == n) & (_n(veto) == n)
            tight_ok = tight_ok & (_n(tight) == n)
            light_charge = light_charge + ak.sum(charge[ctrl], axis=1)
        else:
            base_ok = base_ok & (_n(veto) == 0)

    chargeok = True
    if spec.light_charge is not None:
        chargeok = chargeok & (np.abs(light_charge) == spec.light_charge)
    if spec.total_charge is not None:
        total_charge = light_charge + ak.sum(events.Tau.charge[t.ch_tau_mask], axis=1)
        chargeok = chargeok & (np.abs(total_charge) == spec.total_charge)

    return base_ok, chargeok, tight_ok


def _has_z_pair(objs: ak.Array, z_mass: float, z_window: float) -> ak.Array:
    pairs = ak.combinations(objs, 2, axis=1, fields=["l1", "l2"])
    os_pairs = (pairs.l1.charge * pairs.l2.charge) < 0
    masses = (pairs.l1 * 1 + pairs.l2 * 1).mass
    return ak.any(os_pairs & (abs(masses - z_mass) < z_window), axis=1)


MR_Z_MASS = 91.18
MR_Z_WINDOW = 10.0


def ttbar_mr_ok(self: Selector, events: ak.Array, t: SimpleNamespace) -> tuple[ak.Array, ak.Array]:
    """
    Base selection and charge requirement of the ttbar fake factor measurement region.
    """
    if self.config_inst.campaign.x.year in {2024, 2025, 2026}:
        btag_tagger, btag_discriminator = "UParTAK4", "btagUParTAK4B"
    else:
        btag_tagger, btag_discriminator = "particleNet", "btagPNetB"
    btag_wps = self.config_inst.x.btag_working_points[btag_tagger]

    jet_mask = (
        (events.Jet.jetId == 6) &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.5)
    )
    jet_mask = jet_mask & ak.all(events.Jet.metric_table(events.Electron[t.e_ctrl]) > 0.5, axis=2)
    jet_mask = jet_mask & ak.all(events.Jet.metric_table(events.Muon[t.mu_ctrl]) > 0.5, axis=2)
    jet_notau = jet_mask & ak.all(events.Jet.metric_table(events.Tau[t.noid_tau_mask]) > 0.5, axis=2)
    jet_btag = events.Jet[btag_discriminator]
    jet_ok = (
        (_n(jet_mask) >= 2) &
        (_n(jet_notau & (jet_btag > btag_wps["medium"])) >= 1) &
        (_n(jet_mask & (jet_btag > btag_wps["loose"])) >= 2) &
        (_n(jet_notau & (jet_btag > btag_wps["tight"])) < 1)
    )

    n_e = _n(t.e_mask)
    n_mu = _n(t.mu_mask)
    es = ak.pad_none(events.Electron[t.e_mask], 2, axis=1)
    mus = ak.pad_none(events.Muon[t.mu_mask], 2, axis=1)

    # invariant mass and charge product of the two tight leptons: ee, mumu or emu
    mll = ak.where(
        n_e == 2,
        (es[:, 0] * 1 + es[:, 1] * 1).mass,
        ak.where(n_mu == 2, (mus[:, 0] * 1 + mus[:, 1] * 1).mass, (es[:, 0] * 1 + mus[:, 0] * 1).mass),
    )
    mll = ak.fill_none(mll, 0.0)
    charge_prod = ak.where(
        n_e == 2,
        es[:, 0].charge * es[:, 1].charge,
        ak.where(n_mu == 2, mus[:, 0].charge * mus[:, 1].charge, es[:, 0].charge * mus[:, 0].charge),
    )
    os_ok = ak.fill_none(charge_prod < 0, False)
    same_flavour = (n_e == 2) | (n_mu == 2)

    base_ok = (
        (n_e + n_mu == 2) &
        (_n(t.e_veto) + _n(t.mu_veto) == 2) &
        os_ok &
        (mll > 12.0) &
        (~same_flavour | (np.abs(mll - MR_Z_MASS) > MR_Z_WINDOW)) &
        (_n(t.ch_tau_mask) >= 1) &
        jet_ok
    )

    total_charge = (
        ak.sum(events.Electron.charge[t.e_ctrl], axis=1) +
        ak.sum(events.Muon.charge[t.mu_ctrl], axis=1) +
        ak.sum(events.Tau.charge[t.ch_tau_mask], axis=1)
    )
    base_ok = base_ok & ((_n(t.ch_tau_mask) != 2) | (np.abs(total_charge) != 0))

    return base_ok, os_ok


def wz_mr_ok(self: Selector, events: ak.Array, t: SimpleNamespace) -> tuple[ak.Array, ak.Array]:
    """
    Base selection and charge requirement of the WZ fake factor measurement region.
    """
    z_from_e = _has_z_pair(events.Electron[t.e_mask], MR_Z_MASS, MR_Z_WINDOW)
    z_from_mu = _has_z_pair(events.Muon[t.mu_mask], MR_Z_MASS, MR_Z_WINDOW)
    e_charge = ak.sum(events.Electron.charge[t.e_mask], axis=1)
    mu_charge = ak.sum(events.Muon.charge[t.mu_mask], axis=1)

    # (n_e, n_mu) -> Z-pair and charge requirement
    flavour_charge = {
        (0, 3): z_from_mu & (np.abs(mu_charge) == 1),
        (1, 2): z_from_mu & (mu_charge == 0),
        (2, 1): z_from_e & (e_charge == 0),
        (3, 0): z_from_e & (np.abs(e_charge) == 1),
    }
    base_ok = False
    chargeok = False
    for (n_e, n_mu), charge_ok in flavour_charge.items():
        counts_ok = _n(t.ch_tau_mask) >= 1
        for n, ctrl, veto, tight in [
            (n_e, t.e_ctrl, t.e_veto, t.e_mask),
            (n_mu, t.mu_ctrl, t.mu_veto, t.mu_mask),
        ]:
            counts_ok = counts_ok & (
                ((_n(ctrl) == n) & (_n(veto) == n) & (_n(tight) == n)) if n else (_n(veto) == 0)
            )
        base_ok = base_ok | (counts_ok & charge_ok)
        chargeok = chargeok | charge_ok

    # the charge criteria ensure orthogonality with the 3l1th SR
    all_iso = _n(t.ch_tau_mask & t.tau_iso_mask) == _n(t.ch_tau_mask)
    total_charge = (
        ak.sum(events.Electron.charge[t.e_ctrl], axis=1) +
        ak.sum(events.Muon.charge[t.mu_ctrl], axis=1) +
        ak.sum(events.Tau.charge[t.ch_tau_mask], axis=1)
    )
    base_ok = base_ok & ((_n(t.ch_tau_mask) != 1) | ~all_iso | (np.abs(total_charge) != 0))

    return base_ok, chargeok


def dy_mr_ok(self: Selector, events: ak.Array, t: SimpleNamespace) -> tuple[ak.Array, ak.Array]:
    """
    Base selection and charge requirement of the DY fake factor measurement region.
    """
    mus = ak.pad_none(events.Muon[t.mu_mask], 2, axis=1)
    mumu_mass = (mus[:, 0] * 1 + mus[:, 1] * 1).mass
    z_ok = ak.fill_none(abs(mumu_mass - MR_Z_MASS) < MR_Z_WINDOW, False)

    base_ok = (
        (_n(t.mu_ctrl) == 2) &
        (_n(t.mu_veto) == 2) &
        (_n(t.mu_mask) == 2) &
        (_n(t.e_veto) == 0) &
        (_n(t.ch_tau_mask) >= 1) &
        z_ok
    )

    # the charge criteria ensure orthogonality with the 2lSS1th and 2l2th SRs
    all_iso = _n(t.ch_tau_mask & t.tau_iso_mask) == _n(t.ch_tau_mask)
    ctrl_mu_charge = ak.sum(events.Muon.charge[t.mu_ctrl], axis=1)
    total_charge = ctrl_mu_charge + ak.sum(events.Tau.charge[t.ch_tau_mask], axis=1)
    base_ok = base_ok & (
        (np.abs(ctrl_mu_charge) == 0) &
        ((_n(t.ch_tau_mask) != 2) | ~all_iso | (np.abs(total_charge) != 0))
    )

    chargeok = np.abs(ak.sum(events.Muon.charge[t.mu_mask], axis=1)) == 0
    return base_ok, chargeok


# measurement region -> (selection function, trigger matching rule, (with e, with mu))
MEASUREMENT_REGIONS = {
    "cttbarMR": (ttbar_mr_ok, "ttbar_mr", (True, True)),
    "cwzMR": (wz_mr_ok, "emutau", (True, True)),
    "cdyMR": (dy_mr_ok, "dy_mr", (False, True)),
}


# ────────────────────────────────────────────────────────────────
# combined lepton selection
# ────────────────────────────────────────────────────────────────

@selector(
    uses={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
        "event", "{Electron,Muon,Tau}.{charge,mass}",
        # jets are needed for the ttbarMR region
        "Jet.{pt,eta,phi}",
        IF_NOT_NANO_V15("Jet.jetId"),
        IF_NANO_V12("Jet.btagPNetB"),
        IF_NANO_V14("Jet.btagPNetB"),
        IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    },
    produces={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
        # new columns
        "channel_id", "leptons_os", "tau2_isolated",
        "single_triggered", "cross_triggered",
        "trig_match", "trig_match_bdt", "matched_trigger_ids",
        "tight_sel", "tight_sel_bdt",
        "ok_bdt_eormu",
        "TauIso", "TauNoID",
        "MuonLoose", "MuonTight", "Muon.cone_pt", "Muon.muonLeptoMVA_hh",
        "ElectronLoose", "ElectronTight", "Electron.cone_pt", "Electron.electronLeptoMVA_hh",
    },
    # when True, evaluate the measurement regions instead of the physics channels
    ffmr=False,
    mr_channels=set(MEASUREMENT_REGIONS),
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.

    First, the object selections and trigger matching are run once per fired trigger. Then, every
    channel is evaluated for the triggers listed in :py:attr:`CHANNEL_TRIGGER_GROUPS`, with the
    requirements of :py:attr:`PHYSICS_CHANNELS` or :py:attr:`MEASUREMENT_REGIONS`.
    """
    wp_config = self.config_inst.x.tau_id_working_points
    disable_triggers = getattr(self.config_inst.x, "disable_triggers", False)
    tagger = self.config_inst.x.tau_tagger
    col_prefix = getattr(self.config_inst.x, "tau_tagger_column_prefix", "id")
    get_tau_tagger = lambda tag: f"{col_prefix}{tagger}VS{tag}"

    # get channels from the config
    channels = {
        name: self.config_inst.get_channel(name)
        for name in self.config_inst.x.channel_names
    }

    # custom lepton MVA scores, falling back to zeros when the evaluation fails
    for coll, col, compute_score in [
        ("Muon", "muonLeptoMVA_hh", compute_muon_mva_score),
        ("Electron", "electronLeptoMVA_hh", compute_electron_mva_score),
    ]:
        try:
            scores = compute_score(events)
        except Exception as e:
            logger.warning(f"failed to compute custom {coll} MVA ({e}), creating dummy column with zeros")
            scores = ak.zeros_like(events[coll].pt)
        events = set_ak_column(events, (coll, col), scores)

    state = _LeptonSelectionState(events)
    electron_cone_pt = full_like(events.Electron.pt, np.float32(-999.0), dtype=np.float32)
    muon_cone_pt = full_like(events.Muon.pt, np.float32(-999.0), dtype=np.float32)

    # indices for sorting taus first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 1
    if len(ak.flatten(events.Tau.pt)) > 0:
        f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt))) + 2)
    tau_sorting_key = events.Tau[f"raw{self.config_inst.x.tau_tagger}VSjet"] * f + events.Tau.pt

    # ────────────────────────────────────────────────────────────────
    # 1 FIRST LOOP – build and cache masks once per fired trigger
    # ────────────────────────────────────────────────────────────────

    trig_masks: dict[int, SimpleNamespace] = {}
    tid_tags = {}
    e_trig_any = full_like(events.event, False, dtype=bool)
    mu_trig_any = full_like(events.event, False, dtype=bool)
    tau_trig_any = full_like(events.event, False, dtype=bool)
    e_match_any = full_like(events.Electron.pt, False, dtype=bool)
    mu_match_any = full_like(events.Muon.pt, False, dtype=bool)
    tau_trigger_tags = {"cross_tau_tau", "cross_tau_tau_vbf", "cross_tau_tau_jet", "cross_e_tau", "cross_mu_tau"}

    for trigger, fired, leg_masks in trigger_results.x.trigger_data:
        if not ak.any(fired):
            continue

        e_mask, e_ctrl, e_veto, e_cone_pt = self[electron_selection](events, trigger, **kwargs)
        mu_mask, mu_ctrl, mu_veto, mu_cone_pt = self[muon_selection](events, trigger, **kwargs)
        e_mask_bdt, e_ctrl_bdt, e_veto_bdt, _ = self[electron_selection](events, trigger, ch_key="eormu", **kwargs)
        mu_mask_bdt, mu_ctrl_bdt, mu_veto_bdt, _ = self[muon_selection](events, trigger, ch_key="eormu", **kwargs)
        tau_mask, _, tau_iso_mask, noid_tau_mask = self[tau_selection](events, trigger, e_veto, mu_veto, **kwargs)

        electron_cone_pt = ak.where(e_ctrl, e_cone_pt, electron_cone_pt)
        muon_cone_pt = ak.where(mu_ctrl, mu_cone_pt, muon_cone_pt)
        # early study of tagger independent taus
        state.sel_noid_tau_mask = state.sel_noid_tau_mask | noid_tau_mask

        # electron matching: only for single_e triggers
        if trigger.has_tag({"single_e"}):
            e_match = self[electron_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            e_trig_any = e_trig_any | fired
            e_match_any = e_match_any | e_match
        else:
            e_match = full_like(events.Electron.pt, False, dtype=bool)

        # muon matching: only for single_mu triggers
        if trigger.has_tag({"single_mu"}):
            mu_match = self[muon_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            mu_trig_any = mu_trig_any | fired
            mu_match_any = mu_match_any | mu_match
        else:
            mu_match = full_like(events.Muon.pt, False, dtype=bool)

        # tau matching: only for triggers with a tau leg
        if trigger.has_tag(tau_trigger_tags, mode=any):
            tau_match = self[tau_trigger_matching](events, trigger, fired, leg_masks, **kwargs)
            tau_trig_any = tau_trig_any | fired
        else:
            tau_match = full_like(events.Tau.pt, False, dtype=bool)

        # channel independent tau ID cuts vs e and mu (PNet: unified VLoose vs_e + Tight vs_mu)
        ch_tau_mask = (
            tau_mask &
            (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose) &
            (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
        )

        tid_tags[trigger.id] = set(trigger.tags)
        trig_masks[trigger.id] = SimpleNamespace(
            fired=fired,
            e_mask=e_mask, e_ctrl=e_ctrl, e_veto=e_veto, e_match=e_match,
            mu_mask=mu_mask, mu_ctrl=mu_ctrl, mu_veto=mu_veto, mu_match=mu_match,
            e_mask_bdt=e_mask_bdt, e_ctrl_bdt=e_ctrl_bdt, e_veto_bdt=e_veto_bdt,
            mu_mask_bdt=mu_mask_bdt, mu_ctrl_bdt=mu_ctrl_bdt, mu_veto_bdt=mu_veto_bdt,
            tau_mask=tau_mask, ch_tau_mask=ch_tau_mask, tau_iso_mask=tau_iso_mask,
            noid_tau_mask=noid_tau_mask, tau_match=tau_match,
        )

    # masks combined over all fired triggers, e.g. to define orthogonal single lepton trigger families
    fam = SimpleNamespace(
        e_trig_any=e_trig_any,
        mu_trig_any=mu_trig_any,
        e_match_any=e_match_any,
        mu_match_any=mu_match_any,
        # only single_e / only single_mu fired
        e_only=e_trig_any & ~mu_trig_any,
        # only single_e / only single_mu fired, and no tau trigger, for channels with all flavours
        e_only_emutau=e_trig_any & ~mu_trig_any & ~tau_trig_any,
        mu_only_emutau=mu_trig_any & ~e_trig_any & ~tau_trig_any,
    )
    tids = TIDGroups(tid_tags)

    data_stream = self.dataset_inst.name.split("_")[1] if self.dataset_inst.is_data else None

    # ensuring the v15 jetID fix required for the ttbar mr jet selection
    if self.ffmr and self.config_inst.x.jet_id_has_multiplicity:
        events = self[jet_id](events, **kwargs)

    # ────────────────────────────────────────────────────────────────
    # 2 SECOND LOOP – evaluate every channel once per trigger
    # ────────────────────────────────────────────────────────────────

    for ch_key, channel in channels.items():
        # the measurement regions overlap several physics channels, so the two cannot run in the
        # same pass, they would collide in channel_id and in the leptons_os / tight_sel columns
        if (ch_key in self.mr_channels) != self.ffmr:
            continue

        trig_ids = get_channel_trigger_ids(tids, ch_key, self.dataset_inst.is_mc, data_stream)
        if trig_ids is None:
            continue

        # loose e-or-mu selection for the lepton BDT training, does not define a channel_id
        if ch_key == "ceormu":
            for tid in trig_ids:
                t = trig_masks[tid]
                e_base = _n(t.e_veto_bdt) >= 1
                mu_base = _n(t.mu_veto_bdt) >= 1
                base_ok = e_base | mu_base

                state.ok_bdt_eormu = state.ok_bdt_eormu | base_ok
                state.add_electrons(e_base, t.e_ctrl_bdt, t.e_veto_bdt, t.e_mask_bdt)
                state.add_muons(mu_base, t.mu_ctrl_bdt, t.mu_veto_bdt, t.mu_mask_bdt)
                state.tight_sel_bdt = state.tight_sel_bdt | (
                    (e_base & (_n(t.e_mask_bdt) >= 1)) | (mu_base & (_n(t.mu_mask_bdt) >= 1))
                )
                state.trig_match_bdt = state.trig_match_bdt | base_ok
                state.add_matched_trigger(base_ok, tid)
            continue

        if ch_key in MEASUREMENT_REGIONS:
            region_ok, trig_rule, (with_e, with_mu) = MEASUREMENT_REGIONS[ch_key]
            spec = None
        elif ch_key in PHYSICS_CHANNELS:
            spec = PHYSICS_CHANNELS[ch_key]
            trig_rule, with_e, with_mu = spec.trig_match, spec.n_e > 0, spec.n_mu > 0
        else:
            continue

        good_evt = ak.zeros_like(events.event, dtype=bool)
        for tid in trig_ids:
            t = trig_masks[tid]

            if spec is None:
                base_ok, chargeok = region_ok(self, events, t)
                tight_ok = _n(t.ch_tau_mask & t.tau_iso_mask) >= 1
                with_tau = True
            else:
                base_ok, chargeok, tight_ok = physics_channel_ok(events, spec, t)
                with_tau = spec.n_tau > 0

            if not disable_triggers:
                base_ok = base_ok & t.fired
            ok = base_ok

            if with_e:
                state.add_electrons(ok, t.e_ctrl, t.e_veto, t.e_mask)
            if with_mu:
                state.add_muons(ok, t.mu_ctrl, t.mu_veto, t.mu_mask)
            if with_tau:
                state.add_taus(ok, t.ch_tau_mask, t.tau_iso_mask)

            trig_match_ok = trigger_match_ok(trig_rule, tid, base_ok, t, fam, tids)
            state.add_channel_result(ok, chargeok, ok & tight_ok, trig_match_ok, tid)
            good_evt = good_evt | ok

        state.channel_id = update_channel_ids(events, state.channel_id, channel.id, good_evt)

    # some final type conversions
    channel_id = ak.values_astype(state.channel_id, np.uint32)
    leptons_os = ak.fill_none(state.leptons_os, False)
    tight_sel = ak.fill_none(state.tight_sel, False)
    tight_sel_bdt = ak.fill_none(state.tight_sel_bdt, False)
    trig_match = ak.fill_none(state.trig_match, False)
    trig_match_bdt = ak.fill_none(state.trig_match_bdt, False)
    ok_bdt_eormu = ak.fill_none(state.ok_bdt_eormu, False)

    # concatenate matched trigger ids
    empty_ids = ak.singletons(full_like(events.event, 0, dtype=np.int32), axis=0)[:, :0]
    merge_ids = lambda ids: ak.values_astype(ak.concatenate(ids, axis=1), np.int32) if ids else empty_ids
    matched_trigger_ids = merge_ids(state.matched_trigger_ids)
    # trigger ids with jet legs, updated in the jet selection (no lepton trigger has one yet)
    lepton_part_trigger_ids = merge_ids([])

    # save new columns
    false_mask = (abs(events.event) < 0)
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "tau2_isolated", false_mask)
    events = set_ak_column(events, "single_triggered", state.single_triggered)
    events = set_ak_column(events, "cross_triggered", false_mask)
    events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids)
    events = set_ak_column(events, "tight_sel", tight_sel)
    events = set_ak_column(events, "trig_match", trig_match)
    # columns for the lepton bdt
    events = set_ak_column(events, "ok_bdt_eormu", ok_bdt_eormu)
    events = set_ak_column(events, "tight_sel_bdt", tight_sel_bdt)
    events = set_ak_column(events, "trig_match_bdt", trig_match_bdt)
    # cone-pt for fakeable leptons
    events = set_ak_column(events, "Electron.cone_pt", electron_cone_pt)
    events = set_ak_column(events, "Muon.cone_pt", muon_cone_pt)

    # convert lepton masks to sorted indices (pt for e/mu, iso for tau)
    by_pt = lambda mask, coll: sorted_indices_from_mask(mask, events[coll].pt, ascending=False)
    by_iso = lambda mask: sorted_indices_from_mask(mask, tau_sorting_key, ascending=False)
    sel_electron_indices = by_pt(state.sel_electron_mask, "Electron")
    sel_looseelectron_indices = by_pt(state.sel_looseelectron_mask, "Electron")
    sel_tightelectron_indices = by_pt(state.sel_tightelectron_mask, "Electron")
    sel_muon_indices = by_pt(state.sel_muon_mask, "Muon")
    sel_loosemuon_indices = by_pt(state.sel_loosemuon_mask, "Muon")
    sel_tightmuon_indices = by_pt(state.sel_tightmuon_mask, "Muon")
    sel_tau_indices = by_iso(state.sel_tau_mask)
    sel_isotau_indices = by_iso(state.sel_isotau_mask)
    sel_noid_tau_indices = by_pt(state.sel_noid_tau_mask, "Tau")

    events = set_ak_column(events, "ElectronLoose", events.Electron[sel_looseelectron_indices])
    events = set_ak_column(events, "ElectronTight", events.Electron[sel_tightelectron_indices])
    events = set_ak_column(events, "MuonLoose", events.Muon[sel_loosemuon_indices])
    events = set_ak_column(events, "MuonTight", events.Muon[sel_tightmuon_indices])
    events = set_ak_column(events, "TauIso", events.Tau[sel_isotau_indices])
    events = set_ak_column(events, "TauNoID", events.Tau[sel_noid_tau_indices])

    return events, SelectionResult(
        steps={
            "lepton": (channel_id != 0) | ok_bdt_eormu,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
                "ElectronLoose": sel_looseelectron_indices,
                "ElectronTight": sel_tightelectron_indices,
            },
            "Muon": {
                "Muon": sel_muon_indices,
                "MuonLoose": sel_loosemuon_indices,
                "MuonTight": sel_tightmuon_indices,
            },
            "Tau": {
                "Tau": sel_tau_indices,
                "TauIso": sel_isotau_indices,
                "TauNoID": sel_noid_tau_indices,
            },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    events.Muon[sel_muon_indices] * 1,
                    events.Tau[sel_tau_indices] * 1,
                ],
                axis=1,
            )[:, :2],
            # save the matched trigger ids of the trigger with jet legs for the duration of the selection
            # these will be updated in the jet selection and then stored in the matched_trigger_ids column
            "lepton_part_trigger_ids": lepton_part_trigger_ids,
            # leading taus, not filled by any channel at the moment
            "leading_taus": events.Tau[:, :0],
            "eles": sel_electron_indices,
            "mus": sel_muon_indices,
            "taus": sel_tau_indices,
            "eles_loose": sel_looseelectron_indices,
            "mus_loose": sel_loosemuon_indices,
            "eles_tight": sel_tightelectron_indices,
            "mus_tight": sel_tightmuon_indices,
            "taus_iso": sel_isotau_indices,
            "taus_noid": sel_noid_tau_indices,
        },
    )


@lepton_selection.init
def lepton_selection_init(self: Selector, **kwargs) -> None:
    # add column to load the raw tau tagger score for tau sorting
    tagger = self.config_inst.x.tau_tagger
    # For sorting, always use raw scores; for PNet this is "rawPNetVSjet",
    # for DeepTau this is "rawDeepTau...VSjet"
    self.uses.add(f"Tau.raw{tagger}VSjet")

    # ensuring the v15 jetID fix required for the ttbar mr jet selection
    if self.ffmr and self.config_inst.x.jet_id_has_multiplicity:
        self.uses.add(jet_id)


# fills the measurement regions instead of the physics channels, used by the default_ffmr selector.
# the two never run together, so all output columns keep their usual names
lepton_selection_ffmr = lepton_selection.derive("lepton_selection_ffmr", cls_dict={"ffmr": True})

# coding: utf-8

from __future__ import annotations

import law
import functools

import order as od

from columnflow.util import DotDict

from multilepton.config.util import Trigger, TriggerLeg, TriggerBits as Bits

logger = law.logger.get_logger(__name__)

trigger_bits = DotDict.wrap({
    # for v12:
    # - checked with https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py # noqa: E501
    # - and in particular https://github.com/cms-sw/cmssw/blob/2defd844e96613d2438b690d10b79c773e02ab57/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa: E501
    # for v14:
    # - from https://github.com/cms-sw/cmssw/tree/f50cf84669608dbe67fd8430660abe651d5b46fd/PhysicsTools/NanoAOD/python/triggerObjects_cff.py  # noqa: E501
    # - last update in https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py # noqa: E501
    "e": {
        "CaloIdLTrackIdLIsoVL": Bits(v12=1, v14="v12", v15="v14"),
        "WPTightTrackIso": Bits(v12=2, v14="v12", v15="v14"),
        "WPLooseTrackIso": Bits(v12=4, v14="v12", v15="v14"),
        "OverlapFilterPFTau": Bits(v12=8, v14="v12", v15="v14"),
        "DiElectron": Bits(v12=2**4, v14="v12", v15="v14"),
        "DiElectronLeg1": Bits(v12=2**4, v14="v12", v15="v14"),
        "DiElectronLeg2": Bits(v12=2**5, v14="v12", v15="v14"),
        "MuEle": Bits(v12=32, v14=64, v15="v14"),
        "EleTau": Bits(v12=64, v14=128, v15="v14"),
        "TripleElectron": Bits(v12=128, v14=256, v15="v14"),
        "SingleMuonDiEle": Bits(v12=256, v14=512, v15="v14"),
        "DiMuonSingleEle": Bits(v12=512, v14=1024, v15="v14"),
        "SingleEle_L1DoubleAndSingleEle": Bits(v12=1024, v14=2048, v15="v14"),
        "SingleEle_CaloIdVT_GsfTrkIdT": Bits(v12=2048, v14=4096, v15="v14"),
        "SingleEle_PFJet": Bits(v12=4096, v14=8192, v15="v14"),
        "Photon175_Photon200": Bits(v12=8192, v14=16384, v15="v14"),
        "DoubleEle_CaloIdL_MW_seeded": Bits(v14=32768, v15="v14"),
        "DoubleEle_CaloIdL_MW_unseeded": Bits(v14=65536, v15="v14"),
        "EleTauPNet": Bits(v14=131072, v15="v14"),
    },
    "mu": {
        "TrkIsoVVL": Bits(v12=1, v14="v12", v15="v14"),
        "Iso": Bits(v12=2, v14="v12", v15="v14"),
        "OverlapFilterPFTau": Bits(v12=4, v14="v12", v15="v14"),
        "SingleMuon": Bits(v12=8, v14="v12", v15="v14"),
        "DiMuon": Bits(v12=16, v14="v12", v15="v14"),
        "MuEle": Bits(v12=32, v14="v12", v15="v14"),
        "MuTau": Bits(v12=64, v14="v12", v15="v14"),
        "TripleMuon": Bits(v12=128, v14="v12", v15="v14"),
        "DiMuonSingleEle": Bits(v12=256, v14="v12", v15="v14"),
        "SingleMuonDiEle": Bits(v12=512, v14="v12", v15="v14"),
        "Mu50": Bits(v12=1024, v14="v12", v15="v14"),
        "Mu100": Bits(v12=2048, v14="v12", v15="v14"),
        "SingleMuonSinglePhoton": Bits(v12=4096, v14="v12", v15="v14"),
        "MuTauPNet": Bits(v14=8192, v15="v14"),
    },
    "tau": {  # general comment: lot of v14 paths contain PNet paths, not available in v12, e.g. OverlapFilterIsoEle
        "LooseChargedIso": Bits(v12=1),
        "Loose": Bits(v14=1, v15="v14"),
        "MediumChargedIso": Bits(v12=2),
        "Medium": Bits(v14=2, v15="v14"),
        "TightChargedIso": Bits(v12=4),
        "Tight": Bits(v14=4, v15="v14"),
        "DeepTau": Bits(v12=8, v14="v12", v15="v14"),
        "PNet": Bits(v14=16, v15="v14"),
        "TightOOSCPhotons": Bits(v12=16, v14="v12", v15="v14"),
        "HPS": Bits(v12=32, v14=268435456, v15="v14"),
        "ChargedIso": Bits(v14=32, v15="v14"),
        "ChargedIsoDiTau": Bits(v12=2**5, v14="v12", v15="v14"),
        "Dxy": Bits(v14=2**6, v15="v14"),
        "DeepTauDiTau": Bits(v12=128, v14=2048 + 8, v15="v14"),  # manually created bit combinations for v14
        "ETauFilter": Bits(v14=128, v15="v14"),
        "MuTauFilter": Bits(v14=256, v15="v14"),
        "OverlapFilterIsoEle": Bits(v12=256, v14=4096, v15="v14"),  # contains HPS in v14, not in v12
        "OverlapFilterIsoMu": Bits(v12=512, v14=8192, v15="v14"),  # contains HPS in v14, not in v12
        "SingleTau": Bits(v14=512, v15="v14"),
        "SingleTauOrTauMet": Bits(v12=1024, v14="v12", v15="v14"),  # more general paths than SingleTau in v14
        "VBFDiTau": Bits(v14=1024, v15="v14"),
        # "VBFpDoublePFTau_run2": Bits(v12=2048, v15="v14"),
        # "VBFpDoublePFTau_run3": Bits(v12=4096, v15="v14"),  # warning: this trigger bit expects "ChargedIso" in the filter name, this does not correspond to our actual VBF filter name  # noqa
        "DiTau": Bits(v14=2048, v15="v14"),
        "DiPFJetAndDiTau": Bits(v12=8192, v15="v14"),
        "DiTauAndPFJet": Bits(v12=16384, v14="v12", v15="v14"),
        "DisplacedTau": Bits(v12=32768, v15="v14"),
        "ETauDisplaced": Bits(v14=32768, v15="v14"),
        "MuTauDisplaced": Bits(v14=65536, v15="v14"),
        "DiTauDisplaced": Bits(v14=131072, v15="v14"),
        "Monitoring": Bits(v12=65536, v14=262144, v15="v14"),
        "MonitoringForVBFIsoTau": Bits(v14=524288, v15="v14"),
        "MonitoringDiTauAndPFJet": Bits(v14=1048576, v15="v14"),
        "MonitoringMuTauDisplaced": Bits(v14=2097152, v15="v14"),
        "MonitoringDiTau": Bits(v14=8388608, v15="v14"),
        "VBFDoubleTauMonitoring": Bits(v14=33554432, v15="v14"),
        "OverlapFilter": Bits(v14=16777216, v15="v14"),
        "RegionalPaths": Bits(v12=131072, v15="v14"),
        "L1SeededPaths": Bits(v12=262144, v15="v14"),
        "MatchL1HLT": Bits(v12=262144, v14=134217728, v15="v14"),  # for v12: alias for v12-v14 compatibility
        "1Prong": Bits(v12=524288, v15="v14"),
        "OneProng": Bits(v14=4194304, v15="v14"),  # just changed "1" to "One" for v14, still means different filters
        "SinglePFTauFilter": Bits(v14=536870912, v15="v14"),
        "VBFSingleTau": Bits(v14=1073741824, v15="v14"),
    },
    "jet": {
        "4PixelOnlyPFCentralJetTightIDPt20": Bits(v12=1, v14="v12", v15="v14"),
        "3PixelOnlyPFCentralJetTightIDPt30": Bits(v12=2, v14="v12", v15="v14"),
        "PFJetFilterTwoC30": Bits(v12=4, v14="v12", v15="v14"),
        "4PFCentralJetTightIDPt30": Bits(v12=8, v14="v12", v15="v14"),
        "4PFCentralJetTightIDPt35": Bits(v12=16, v14="v12", v15="v14"),
        "QuadCentralJet30": Bits(v12=32, v14="v12", v15="v14"),
        "2PixelOnlyPFCentralJetTightIDPt40": Bits(v12=64, v14="v12", v15="v14"),
        "L1sTripleJetVBF_orHTT_orDoubleJet_orSingleJet": Bits(v12=128, v14="v12", v15="v14"),
        "3PFCentralJetTightIDPt40": Bits(v12=256, v14="v12", v15="v14"),
        "3PFCentralJetTightIDPt45": Bits(v12=512, v14="v12", v15="v14"),
        "L1sQuadJetsHT": Bits(v12=1024, v14="v12", v15="v14"),
        "BTagCaloDeepCSVp17Double": Bits(v12=2048, v14="v12", v15="v14"),
        "PFCentralJetLooseIDQuad30": Bits(v12=4096, v14="v12", v15="v14"),
        "1PFCentralJetLooseID75": Bits(v12=8192, v14="v12", v15="v14"),
        "2PFCentralJetLooseID60": Bits(v12=16384, v14="v12", v15="v14"),
        "3PFCentralJetLooseID45": Bits(v12=32768, v14="v12", v15="v14"),
        "4PFCentralJetLooseID40": Bits(v12=65536, v14="v12", v15="v14"),
        "DoubleTau+Jet": Bits(v12=131072, v14="v12", v15="v14"),  # v14 also contains PNet paths
        "VBFcrossCleanedDeepTauPFTau": Bits(v12=262144, v14="v12", v15="v14"),  # more general VBFDiTauJets in v14  TODO: change name?  # noqa
        "VBFcrossCleanedUsingDijetCorr": Bits(v12=524288, v14="v12", v15="v14"),  # more general VBFSingleTauJets in v14  TODO: change name?  # noqa
        "MonitoringMuon+Tau+Jet": Bits(v12=1048576, v14="v12", v15="v14"),
        "2PFCentralJetTightIDPt50": Bits(v12=2097152, v14="v12", v15="v14"),
        "1PixelOnlyPFCentralJetTightIDPt60": Bits(v12=4194304, v14="v12", v15="v14"),
        "1PFCentralJetTightIDPt70": Bits(v12=8388608, v14="v12", v15="v14"),
        "BTagPFDeepJet1p5Single": Bits(v12=16777216, v14="v12", v15="v14"),
        "BTagPFDeepJet4p5Triple": Bits(v12=33554432, v14="v12", v15="v14"),
        "2BTagSumOR2BTagMeanPaths": Bits(v12=67108864, v14="v12", v15="v14"),
        "2/1PixelOnlyPFCentralJetTightIDPt20/50": Bits(v12=134217728, v14="v12", v15="v14"),
        "2PFCentralJetTightIDPt30": Bits(v12=2**28, v14="v12", v15="v14"),
        "1PFCentralJetTightIDPt60": Bits(v12=2**29, v14="v12", v15="v14"),
        "PF2CentralJetPt30PNet2BTagMean0p50": Bits(v12=2**30, v14="v12", v15="v14"),
    },
})


def get_triggerID(name):
    """
    General requirement from the lepton selection:
    For cross triggers, the lepton leg (lepton= {"e", "mu"}) must be defined before the tau leg.
    An error here would be caught in the lepton selection, but it is better to avoid it.
    Convention for Ids:
    - 1xx: single muon triggers
    - 1xxx: double muon triggers
    - 1xxxx: triple muon triggers
    - 2xx: single electron triggers
    - 2xxx: double electron triggers
    - 2xxxx: triple electron triggers
    - 3xx: mu-tau triggers
    - 4xx: e-tau triggers
    - 5xx: tau-tau triggers
    - 6xx: vbf triggers
    - 7xx: tau tau jet triggers
    - 8xx: quadjet triggers
    - 9xx: cross-e-mu-double/triple triggers
    - x: MET triggers
    Starting from xx = 01 and with a unique name for each path across all years.
    """
    ids = {
        # single muon triggers
        "HLT_IsoMu22": 101,
        "HLT_IsoMu22_eta2p1": 102,
        "HLT_IsoTkMu22": 103,
        "HLT_IsoTkMu22_eta2p1": 104,
        "HLT_IsoMu24": 105,
        "HLT_IsoMu27": 106,
        # double muon triggers
        "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8": 1001,
        "HLT_DoubleMu3_DZ_PFMET50_PFMHT60": 1002,
        "HLT_Mu18_Mu9_SameSign": 1003,
        # triple muon triggers
        "HLT_TripleMu_5_3_3_Mass3p8_DZ": 10001,
        # single electron triggers
        "HLT_Ele25_eta2p1_WPTight_Gsf": 201,
        "HLT_Ele32_WPTight_Gsf": 202,
        "HLT_Ele32_WPTight_Gsf_L1DoubleEG": 203,
        "HLT_Ele35_WPTight_Gsf": 204,
        "HLT_Ele30_WPTight_Gsf": 205,
        "HLT_Ele28_eta2p1_WPTight_Gsf_HT150": 206,
        # double electron triggers
        "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL": 2001,
        "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350": 2002,
        # triple electron triggers
        "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL": 20001,
        # mu–tau triggers
        "HLT_IsoMu19_eta2p1_LooseIsoPFTau20": 301,
        "HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1": 302,
        "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1": 303,
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1": 304,
        # new pnet trigger
        "HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Loose_eta2p3_CrossL1": 305,
        # e–tau triggers
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1": 401,
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20": 402,
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30": 403,
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1": 404,
        "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1": 405,
        # new pnet trigger
        "HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1": 406,
        # tau-tau triggers
        "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg": 501,
        "HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg": 502,
        "HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg": 503,
        "HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg": 504,
        "HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg": 505,
        "HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg": 506,
        "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1": 507,
        "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1": 508,
        "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1": 509,
        # new pnet triggers
        "HLT_DoublePNetTauhPFJet30_Medium_L2NN_eta2p3": 510,
        # VBF di-tau triggers
        "HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg": 601,
        "HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1": 602,
        "HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1": 603,
        "HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1": 604,
        # new pnet triggers
        "HLT_VBF_DoublePNetTauhPFJet20_eta2p2": 605,
        "HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3": 606,
        # tau+jet triggers
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60": 701,
        "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75": 702,
        # new pnet triggers
        "HLT_DoublePNetTauhPFJet26_L2NN_eta2p3_PFJet60": 703,
        # cross-e-mu-double/triple triggers
        "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ": 901,
        "HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8": 902,
        "HLT_Mu8_DiEle12_CaloIdL_TrackIdL": 903,
        # MET triggers
        "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight": 1,
    }
    if name not in ids:
        raise KeyError(f"Trigger name '{name}' not found in trigger ID list.")
    return ids[name]


def get_bit_sum(nano_version: int, obj_name: str, names: list[str | None]) -> int | None:
    total = 0
    for name in names:
        if not name:
            continue
        try:
            val = trigger_bits[obj_name][name].get(nano_version) or 0
            total += val
        except KeyError:
            logger.warning(f"missing trigger bit for {obj_name}.{name} at nano_version={nano_version}")
    return total or None


def add_triggers(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    - Tau Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/Tau
    - Electron Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIIISummary
    - Muon Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT
    """
    # get trigger bits for the requested nano version
    nano_version = config.campaign.x.version
    year = config.campaign.x.year
    get_bit_sum_v = functools.partial(get_bit_sum, nano_version)
    config.x.triggers = od.UniqueObjectIndex(Trigger)
    multileptons_triggers = {}

    if year in [2017, 2018, 2022, 2023, 2024]:
        multileptons_triggers.update({
            # single muon
            "HLT_IsoMu24": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["SingleMuon"]))),
                "filters": ["hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08",
                            "hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07"],  # (1mu + Iso)
                "on_datasets": ["mutau", "emu_from_e", "emu_from_mu", "mumu"],
                "tags": ["single_trigger", "single_mu"],
            },
        })

    if year in [2022, 2023]:
        multileptons_triggers.update({
            # single electron
            "HLT_Ele28_eta2p1_WPTight_Gsf_HT150": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["WPTightTrackIso"]))),
                "filters": [],
                "on_datasets": ["etau", "ee", "emu_from_e", "emu_from_mu"],
                "tags": ["single_trigger", "single_e"],
            },
        })

    if year == 2024:
        multileptons_triggers.update({
            # e tauh
            "HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_Loose_eta2p3_CrossL1": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e",
                                ["OverlapFilterPFTau", "EleTau"])),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau",
                                 ["PNet", "OverlapFilterIsoEle", "ETauFilter" if nano_version == 14 else None]))),   # noqa: E501
                "filters": ["hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30"],  # (OverlapFilter)
                "on_datasets": ["etau"],
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            # mu tauh
            "HLT_IsoMu20_eta2p1_PNetTauhPFJet27_Loose_eta2p3_CrossL1": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu",
                                ["OverlapFilterPFTau", "MuTau"])),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau",
                                 ["PNet", "OverlapFilterIsoMu", "MatchL1HLT", "MuTauFilter" if nano_version == 14 else None]))),  # noqa: E501
                "filters": ["hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched"],
                "on_datasets": ["mutau"],
                "tags": ["cross_trigger", "cross_mu_tau"],
            },

            # tauh tauh
            "HLT_DoublePNetTauhPFJet30_Medium_L2NN_eta2p3": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTau", "PNet", "Medium" if nano_version == 14 else None])),  # noqa: E501
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTau", "PNet", "Medium" if nano_version == 14 else None]))),  # noqa: E501
                "filters": ["hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched"],  # (Deeptau + HPS),
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau"],
            },

            # vbf
            "HLT_VBF_DoublePNetTauhPFJet20_eta2p2": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["VBFDiTau", "PNet" if nano_version == 14 else None])),  # noqa: E501
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["VBFDiTau", "PNet" if nano_version == 14 else None])),  # noqa: E501
                             vbf1=TriggerLeg(pdg_id=1, trigger_bits=None),
                             vbf2=TriggerLeg(pdg_id=1, trigger_bits=None)),
                "filters": ["hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau",
                            "hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?"],
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau_vbf"],
            },

            # tau tau jet
            "HLT_DoublePNetTauhPFJet26_L2NN_eta2p3_PFJet60": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTauAndPFJet"])),
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTauAndPFJet"])),
                             jet=TriggerLeg(pdg_id=1, trigger_bits=get_bit_sum_v("jet", ["DoubleTau+Jet"]))),
                "filters": ["hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet",
                            "hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60"],
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau_jet"],
            },
        })

    if year in [2022, 2023, 2024]:
        multileptons_triggers.update({
            # single electron
            "HLT_Ele30_WPTight_Gsf": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["WPTightTrackIso"]))),
                "filters": [],
                "on_datasets": ["etau", "ee", "emu_from_e", "emu_from_mu"],
                "tags": ["single_trigger", "single_e"],
            },

            # double electron
            "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL": {
                "legs": dict(e1=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectron", "DiElectronLeg1", "CaloIdLTrackIdLIsoVL"])),  # noqa: E501
                             e2=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectron", "DiElectronLeg2", "CaloIdLTrackIdLIsoVL"]))),  # noqa: E501
                "filters": [],
                "on_datasets": ["ee", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_e"],
            },

            "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350": {
                "legs": dict(e1=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectron", "DiElectronLeg1"])),  # noqa: E501
                             e2=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectron", "DiElectronLeg2"]))),  # noqa: E501
                "filters": [],
                "on_datasets": ["ee", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_e"],
            },

            # triple electron
            "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["TripleElectron"]))),
                "filters": [],
                "on_datasets": ["ee", "emu_from_e", "emu_from_mu"],
                "tags": ["triple_trigger", "triple_e"],
            },

            # double muon
            "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["DiMuon", "TrkIsoVVL"]))),
                "filters": [],
                "on_datasets": ["mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_mu"],
            },

            "HLT_DoubleMu3_DZ_PFMET50_PFMHT60": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["DiMuon"]))),
                "filters": [],
                "on_datasets": ["mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_mu"],
            },

            "HLT_Mu18_Mu9_SameSign": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["DiMuon"]))),
                "filters": [],
                "on_datasets": ["mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_mu"],
            },

            # triple muon
            "HLT_TripleMu_5_3_3_Mass3p8_DZ": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["TripleMuon"]))),
                "filters": [],
                "on_datasets": ["mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["triple_trigger", "triple_mu"],
            },

            # e tauh
            "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e",
                                ["OverlapFilterPFTau", "EleTau"])),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau",
                                 ["DeepTau", "HPS", "OverlapFilterIsoEle", "ETauFilter" if nano_version == 14 else None]))),  # noqa: E501
                "filters": ["hltHpsOverlapFilterIsoEle24WPTightGsfLooseETauWPDeepTauPFTau30"],
                "on_datasets": ["etau"],
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            # mu tauh
            "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu",
                                ["OverlapFilterPFTau", "MuTau"])),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau",
                                 ["DeepTau", "HPS", "OverlapFilterIsoMu", "MatchL1HLT", "MuTauFilter" if nano_version == 14 else None]))),   # noqa: E501
                "filters": ["hltHpsSelectedPFTau27LooseMuTauWPDeepTauVsJetsAgainstMuonL1HLTMatched"],
                "on_datasets": ["mutau"],
                "tags": ["cross_trigger", "cross_mu_tau"],
            },

            # tauh tauh
            "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DeepTauDiTau", "HPS", "Medium" if nano_version == 14 else None])),  # noqa: E501
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DeepTauDiTau", "HPS", "Medium" if nano_version == 14 else None]))),  # noqa: E501
                "filters": ["hltHpsDoublePFTau35MediumDitauWPDeepTauL1HLTMatched"],  # (Deeptau + HPS),
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau"],
            },

            # vbf
            "HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["VBFDiTau", "HPS", "DeepTau" if nano_version == 14 else None])),  # noqa: E501
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["VBFDiTau", "HPS", "DeepTau" if nano_version == 14 else None])),  # noqa: E501
                             vbf1=TriggerLeg(pdg_id=1, trigger_bits=get_bit_sum_v("jet", ["VBFcrossCleanedDeepTauPFTau" if nano_version == 14 else None])),  # noqa: E501
                             vbf2=TriggerLeg(pdg_id=1, trigger_bits=get_bit_sum_v("jet", ["VBFcrossCleanedDeepTauPFTau" if nano_version == 14 else None]))),  # noqa: E501
                "filters": ["hltHpsDoublePFTau20TrackDeepTauDitauWPForVBFIsoTau",
                            "hltMatchedVBFTwoPFJets2CrossCleanedFromDoubleMediumDeepTauDitauWPPFTauHPS20?"],
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau_vbf"],
            },

            # tau tau jet
            "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTauAndPFJet"])),
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=get_bit_sum_v("tau", ["DiTauAndPFJet"])),
                             jet=TriggerLeg(pdg_id=1, trigger_bits=get_bit_sum_v("jet", ["DoubleTau+Jet"]))),
                "filters": ["hltHpsDoublePFTau30MediumDitauWPDeepTauL1HLTMatchedDoubleTauJet",
                            "hltHpsOverlapFilterDeepTauDoublePFTau30PFJet60"],
                "on_datasets": ["tautau"],
                "tags": ["cross_trigger", "cross_tau_tau_jet"],
            },

            # cross-e-mu-double/triple triggers
            "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["TrkIsoVVL"])),
                             e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["CaloIdLTrackIdLIsoVL"]))),
                "filters": [],
                "on_datasets": ["ee", "mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["double_trigger", "double_emu"],
            },

            "HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["DiMuon"])),
                             e=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["CaloIdLTrackIdLIsoVL"]))),
                "filters": [],
                "on_datasets": ["mumu", "emu_from_e", "emu_from_mu"],
                "tags": ["triple_trigger", "triple_emumu"],
            },

            "HLT_Mu8_DiEle12_CaloIdL_TrackIdL": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=None),
                             e1=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectronLeg1", "CaloIdLTrackIdLIsoVL"])),  # noqa: E501
                             e2=TriggerLeg(pdg_id=11, trigger_bits=get_bit_sum_v("e", ["DiElectronLeg2", "CaloIdLTrackIdLIsoVL"]))),  # noqa: E501
                "filters": [],
                "on_datasets": ["ee", "emu_from_e", "emu_from_mu"],
                "tags": ["triple_trigger", "triple_eemu"],
            },
        })

    if year == 2016:
        multileptons_triggers.update({
            # e tauh
            "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=None),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=None)),
                # does not exist for run F on but should only be used until run 276215 -> which era?
                "on_datasets": (lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era <= "E"),
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=None),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=None)),
                # does not exist for run F on but should only be used between run 276215 and 278270 -> which eras?
                "on_datasets": (lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era <= "E"),
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=None),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=None)),
                # does not exist until run E but should only be used after run 278270 -> which era?
                "on_datasets": (lambda dataset_inst: dataset_inst.is_data and dataset_inst.x.era >= "E"),
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            # mu tauh
            "HLT_IsoMu19_eta2p1_LooseIsoPFTau20": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=None),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=None)),
                "tags": ["cross_trigger", "cross_mu_tau"],
            },

            "HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=None),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=None)),
                "tags": ["cross_trigger", "cross_mu_tau"],
            },

            # tauh tauh
            "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=None),
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=None)),
                "on_datasets": (lambda dataset_inst: dataset_inst.is_mc or ("B" <= dataset_inst.x.era <= "F")),
                "tags": ["cross_trigger", "cross_tau_tau"],
            },

            "HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=None),
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=None)),
                "on_datasets": (lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "H"),
                "tags": ["cross_trigger", "cross_tau_tau"],
            },
        })

        if config.campaign.has_tag("preVFP"):
            multileptons_triggers.update({
                # single electron
                "HLT_Ele25_eta2p1_WPTight_Gsf": {
                    "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=None)),
                    "tags": ["single_trigger", "single_e"],
                },
            })
            # single muon
            for trig in ["HLT_IsoMu22", "HLT_IsoMu22_eta2p1", "HLT_IsoTkMu22", "HLT_IsoTkMu22_eta2p1"]:
                multileptons_triggers.update({
                    trig: {
                        "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=None)),
                        "tags": ["single_trigger", "single_mu"],
                    },
                })

    if year in [2017, 2018]:
        multileptons_triggers.update({
            # single muon
            "HLT_IsoMu27": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=get_bit_sum_v("mu", ["SingleMuon"]))),
                "filters": ["hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07"],
                "on_datasets": ["mutau", "emu_from_e", "emu_from_mu", "mumu"],
                "tags": ["single_trigger", "single_mu"],
            },

            # single electron
            "HLT_Ele32_WPTight_Gsf": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=2)),
                "filters": ["hltEle32WPTightGsfTrackIsoFilter"],
                "on_datasets": (lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
                "tags": ["single_trigger", "single_e"],
            },

            "HLT_Ele32_WPTight_Gsf_L1DoubleEG": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=2 + 1024)),
                "filters": ["hltEle32L1DoubleEGWPTightGsfTrackIsoFilter", "hltEGL1SingleEGOrFilter"],
                "tags": ["single_trigger", "single_e"],
            },

            "HLT_Ele35_WPTight_Gsf": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=2)),
                "filters": ["hltEle35noerWPTightGsfTrackIsoFilter"],
                "tags": ["single_trigger", "single_e"],
            },

            # e tauh
            "HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1": {
                "legs": dict(e=TriggerLeg(pdg_id=11, trigger_bits=2 + 64),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=1024 + 256)),
                "filters": ["hltEle24erWPTightGsfTrackIsoFilterForTau",
                            "hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30",
                            "hltSelectedPFTau30LooseChargedIsolationL1HLTMatched",
                            "hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30"],
                "tags": ["cross_trigger", "cross_e_tau"],
            },

            # mu tauh
            "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1": {
                "legs": dict(mu=TriggerLeg(pdg_id=13, trigger_bits=2 + 64),
                             tau=TriggerLeg(pdg_id=15, trigger_bits=1024 + 512)),
                "filters": ["hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07",
                            "hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded",
                            "hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched",
                            "hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded"],
                "tags": ["cross_trigger", "cross_mu_tau"],
            },

            # vbf
            "HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg": {
                "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=2048),
                             tau2=TriggerLeg(pdg_id=15, trigger_bits=2048),
                             vbf1=TriggerLeg(pdg_id=1, trigger_bits=1),
                             vbf2=TriggerLeg(pdg_id=1, trigger_bits=1)),
                "filters": ["hltDoublePFTau20TrackPt1LooseChargedIsolation",
                            "hltMatchedVBFOnePFJet2CrossCleanedFromDoubleLooseChargedIsoPFTau20"],
                "on_datasets": (lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
                "tags": ["cross_trigger", "cross_tau_tau_vbf"],
            },
        })

        # tauh tauh
        for trig in [
                "HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg",
                "HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
                "HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
                "HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
        ]:
            multileptons_triggers.update({
                trig: {
                    "legs": dict(tau1=TriggerLeg(pdg_id=15, trigger_bits=64),
                                 tau2=TriggerLeg(pdg_id=15, trigger_bits=64)),
                    "on_datasets": (lambda dataset_inst: dataset_inst.is_data),
                    "tags": ["cross_trigger", "cross_tau_tau"],
                },
            })

    for name, triginfo in multileptons_triggers.items():
        on_datasets = triginfo.get("on_datasets", None)
        kwargs = dict(
            name=name,
            id=get_triggerID(name),
            legs=triginfo["legs"],
            tags=triginfo["tags"],
        )

        # Handle applies_to_dataset only if on_datasets is provided
        if on_datasets is not None:
            if callable(on_datasets):
                kwargs["applies_to_dataset"] = on_datasets
            else:
                kwargs["applies_to_dataset"] = (
                    lambda dataset_inst, tags=on_datasets:
                        dataset_inst.is_mc or any(dataset_inst.has_tag(tag) for tag in tags)
                )

        config.x.triggers.add(**kwargs)

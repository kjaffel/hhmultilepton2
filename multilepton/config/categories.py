# coding: utf-8

"""
Definition of categories.
"""

import functools

import order as od

from columnflow.config_util import add_category


multileptons_categories = {
    # 3l/4l inclusive
    "cat3l0tau_SR": {"id": 1001, "selection": "cat_3l0tau_SR", "label": r"$3\ell 0\tau_{h}$ SR"},
    "cat3l0tau_SB": {"id": 1002, "selection": "cat_3l0tau_SB", "label": r"$3\ell 0\tau_{h}$ SB"},
    "cat4l_SR": {"id": 1003, "selection": "cat_4l_SR", "label": r"$4\ell$ SR"},
    "cat4l_SB": {"id": 1004, "selection": "cat_4l_SB", "label": r"$4\ell$ SB"},
    "cat3l1tau_SR": {"id": 1005, "selection": "cat_3l1tau_SR", "label": r"$3\ell 1\tau_{h}$ SR"},
    "cat3l1tau_SB": {"id": 1006, "selection": "cat_3l1tau_SB", "label": r"$3\ell 1\tau_{h}$ SB"},
    "cat2l2tau_SR": {"id": 1007, "selection": "cat_2l2tau_SR", "label": r"$2\ell 2\tau_{h}$ SR"},
    "cat2l2tau_SB": {"id": 1008, "selection": "cat_2l2tau_SB", "label": r"$2\ell 2\tau_{h}$ SB"},
    "cat1l3tau_SR": {"id": 1009, "selection": "cat_1l3tau_SR", "label": r"$1\ell 3\tau_{h}$ SR"},
    "cat1l3tau_SB": {"id": 1010, "selection": "cat_1l3tau_SB", "label": r"$1\ell 3\tau_{h}$ SB"},
    "cat4tau_SR": {"id": 1011, "selection": "cat_4tau_SR", "label": r"$4\tau_{h}$ SR"},
    "cat4tau_SB": {"id": 1012, "selection": "cat_4tau_SB", "label": r"$4\tau_{h}$ SB"},
    "cat2l0or1tau_SR_SS": {"id": 1013, "selection": "cat_2l0or1tau_SR_SS", "label": r"$2\ell\  \leq 1\,\tau_{h}$ SR, SS"},  # noqa: E501
    "cat2l0or1tau_SR_OS": {"id": 1014, "selection": "cat_2l0or1tau_SR_OS", "label": r"$2\ell\  \leq 1\,\tau_{h}$ SR, OS"},  # noqa: E501
    "cat2l0or1tau_SB_SS": {"id": 1015, "selection": "cat_2l0or1tau_SB_SS", "label": r"$2\ell\  \leq 1\,\tau_{h}$ SB, SS"},  # noqa: E501
    "cat2l0or1tau_SB_OS": {"id": 1016, "selection": "cat_2l0or1tau_SB_OS", "label": r"$2\ell\  \leq 1\,\tau_{h}$ SB, OS"},  # noqa: E501
    # Loose category for BDT trainning + tight + trigmatch
    "ceormu_bveto": {"id": 15000, "selection": "cat_e_or_mu_bveto", "label": r"e or $\mu$ bveto on", "tags": {"ceormu_bveto"}},  # noqa: E501
    # bveto
    "bveto_on": {"id": 30001, "selection": "cat_bveto_on", "label": "bveto on"},
    "bveto_off": {"id": 30002, "selection": "cat_bveto_off", "label": "bveto off"},
    # tight/nontight
    "tight_bdt": {"id": 11000, "selection": "cat_tight_bdt", "label": "tight", "tags": {"tight_bdt"}},
    "nontight_bdt": {"id": 12000, "selection": "cat_nontight_bdt", "label": "fakeable", "tags": {"nontight_bdt"}},
    # trigmatch
    "trigmatch_bdt": {"id": 13000, "selection": "cat_trigmatch_bdt", "label": "trigger matched", "tags": {"trigmatch_bdt"}},  # noqa: E501
    "nontrigmatch_bdt": {"id": 14000, "selection": "cat_nontrigmatch_bdt", "label": "trigger unmatched", "tags": {"nontrigmatch_bdt"}},  # noqa: E501
    # tight/nontight
    "tight": {"id": 10001, "selection": "cat_tight", "label": "tight", "tags": {"tight"}},
    "nontight": {"id": 10002, "selection": "cat_nontight", "label": "fakeable", "tags": {"nontight"}},
    # trigmatch
    "trigmatch": {"id": 10003, "selection": "cat_trigmatch", "label": "trigger matched", "tags": {"trigmatch"}},
    "nontrigmatch": {"id": 10004, "selection": "cat_nontrigmatch", "label": "trigger unmatched", "tags": {"nontrigmatch"}},  # noqa: E501
    # qcd regions
    "os": {"id": 10, "selection": "cat_os", "label": "OS", "tags": {"os"}},
    "ss": {"id": 11, "selection": "cat_ss", "label": "SS", "tags": {"ss"}},
    "iso": {"id": 12, "selection": "cat_iso", "label": r"iso", "tags": {"iso"}},
    "noniso": {"id": 13, "selection": "cat_noniso", "label": r"non-iso", "tags": {"noniso"}},  # noqa: E501
    # kinematic categories
    "incl": {"id": 100, "selection": "cat_incl", "label": "inclusive"},
    "2j": {"id": 110, "selection": "cat_2j", "label": "2 jets"},
    "dy": {"id": 210, "selection": "cat_dy", "label": "DY enriched"},
    "tt": {"id": 220, "selection": "cat_tt", "label": r"$t\bar{t}$ enriched"},
    "res1b": {"id": 300, "selection": "cat_res1b", "label": "res1b"},
    "res2b": {"id": 301, "selection": "cat_res2b", "label": "res2b"},
    "boosted": {"id": 310, "selection": "cat_boosted", "label": "boosted"},
}


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # root category (-1 has special meaning in cutflow)
    root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")
    _add_category = functools.partial(add_category, parent=root_cat)

    # One category per existing channel
    for ch in config.channels:
        _add_category(config, name=ch.name, id=ch.id, selection=f"cat_{ch.name[1:]}", label=ch.label, tags=ch.name)
    # Analysis-specific multilepton categories
    for name, cat in multileptons_categories.items():
        _add_category(
            config,
            name=name,
            id=cat["id"],
            selection=cat["selection"],
            label=cat["label"],
            tags=cat.get("tags"),
        )

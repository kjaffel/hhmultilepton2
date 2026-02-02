# coding: utf-8

"""
Custom base tasks for HH -> Multileptons.
"""

import law

from columnflow.tasks.framework.base import BaseTask
from columnflow.tasks.external import GetDatasetLFNs as CFGetDatasetLFNs
from columnflow.tasks.plotting import PlotVariables1D as CFPlotVariables1D

from multilepton.config.analysis_multilepton import analysis_multilepton


class MultileptonTask(BaseTask):
    task_namespace = "ml"
    limit_dataset_files = law.Parameter(
        default=-1,
        significant=False,
        description="Limit number of dataset files to process (-1 = all)",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pass the limit to BOTH analysis and config
        limit = int(self.limit_dataset_files)
        if limit > 0:
            # set at ANALYSIS level (affects all future configs)
            analysis_multilepton.x.limit_dataset_files = limit
            # set at CONFIG level (affects this config)
            self.config_inst.x.limit_dataset_files = limit
            # apply limit
            self.apply_file_limit_to_config(limit)

    def apply_file_limit_to_config(self, limit):
        """Immediately modify dataset files in the current config."""
        self.logger.warning(f"Applying limit {limit} to config {self.config_inst.name}")

        modified = False
        for dataset in self.config_inst.datasets:
            for info in dataset.info.values():
                original = info.n_files
                if original > limit:
                    info.n_files = limit
                    modified = True
                    # print(f"  Limited {dataset.name}: {original} events → {limit} events")

        if not modified:
            self.logger.warning(f"  No datasets needed limiting (all ≤ {limit})")


class GetDatasetLFNs(MultileptonTask, CFGetDatasetLFNs):
    """
    Task to get dataset LFNs with optional file limiting.
    """
    task_namespace = "ml"
    # Track how many files we've already collected
    _files_collected = 0
    _limit = -1

    def get_dataset_lfns_dasgoclient(self, *args, **kwargs):
        """
        Override the dasgoclient method that actually fetches LFNs.
        Use *args, **kwargs to handle any signature.
        """
        # Call parent method
        lfns = super().get_dataset_lfns_dasgoclient(*args, **kwargs)

        # Extract dataset_key from args or kwargs for logging
        dataset_key = None
        if len(args) >= 3:
            dataset_key = args[2]  # Third positional arg based on error
        elif "dataset_key" in kwargs:
            dataset_key = kwargs["dataset_key"]

        # Initialize limit on first call
        if self._limit == -1:
            self._limit = int(self.limit_dataset_files)

        # Apply the file limit globally
        if self._limit > 0:
            remaining = self._limit - self._files_collected
            if remaining <= 0:
                return []  # Already got enough files
            elif len(lfns) > remaining:
                self.logger.warning(
                    f"Limiting dataset key {dataset_key} to {remaining} files "
                    f"(already collected {self._files_collected}, total limit {self._limit})",
                )
                lfns = lfns[:remaining]
                self._files_collected += remaining
            else:
                self.logger.warning(
                    f"Taking all {len(lfns)} files from dataset key {dataset_key} "
                    f"(already collected {self._files_collected}, total limit {self._limit})",
                )
                self._files_collected += len(lfns)
        return lfns

    def run(self):
        self.logger.warning(f"Running with limit_dataset_files = {self.limit_dataset_files}")
        # Reset counters
        self._files_collected = 0
        self._limit = -1
        # Call parent's run
        return super().run()


class PlotVariables1D(MultileptonTask, CFPlotVariables1D):
    task_namespace = "ml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.warning(f"PlotVariables1D params: limit={self.limit_dataset_files}, dataset={getattr(self, 'dataset', 'N/A')}")  # noqa: E501

    def requires(self):
        reqs = super().requires()
        if isinstance(reqs, dict) and "lfns" in reqs:
            lfns_task = reqs["lfns"]

            if int(self.limit_dataset_files) > 0:
                self.logger.warning(f"Creating ml.GetDatasetLFNs for dataset: {getattr(lfns_task, 'dataset', 'unknown')}")  # noqa: E501
                # Minimal parameters - let law figure out the rest
                reqs["lfns"] = GetDatasetLFNs.req(
                    task=self,  # Pass self so it inherits context
                    limit_dataset_files=self.limit_dataset_files,
                )
        return reqs

    def run(self):
        self.logger.warning(f"Running PlotVariables1D with limit={self.limit_dataset_files}")
        return super().run()

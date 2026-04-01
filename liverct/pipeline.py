"""
High-level processing pipelines for CT BIDS datasets.

Coordinates multiple processing steps and handles subject/session iteration.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
from .segmentation import CTSegmentationPipeline
from .figures import create_vertebrae_slice_report, create_vertebrae_cross_subject_montage, generate_montages_from_bids
from .stats import consolidate_group_statistics
from .scheduler import (
    build_segmentation_jobs,
    build_processing_jobs,
    run_segmentation_jobs,
    run_job_graph,
)

logger = logging.getLogger(__name__)


class BIDSProcessingPipeline:
    """Orchestrate processing of BIDS CT datasets."""

    def __init__(self, bids_root: Path):
        """
        Initialize processing pipeline.

        Parameters
        ----------
        bids_root : Path
            Root directory of BIDS dataset
        """
        self.bids_root = Path(bids_root)
        if not self.bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {self.bids_root}")

        self.segmentation = CTSegmentationPipeline()

    def process_all_subjects(
        self,
        process_func: Callable,
        log_summary: bool = True,
        subjects: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply processing function to all subjects in BIDS dataset.

        Parameters
        ----------
        process_func : callable
            Function to call for each subject/session.
            Signature: process_func(subject_label, session_label=None) -> bool
        log_summary : bool
            Whether to log summary statistics
        subjects : str or list of str, optional
            Single subject ID or list of subject IDs to process.
            If None, processes all subjects. IDs can be with or without 'sub-' prefix.

        Returns
        -------
        dict
            Summary with keys: 'successful', 'failed', 'skipped'
        """
        subject_dirs = sorted([d for d in self.bids_root.glob("sub-*") if d.is_dir()])
        
        # Filter subjects if specified
        if subjects is not None:
            if isinstance(subjects, str):
                subjects = [subjects]
            # Normalize subject IDs (ensure sub- prefix)
            subjects_normalized = [s if s.startswith('sub-') else f'sub-{s}' for s in subjects]
            subject_dirs = [d for d in subject_dirs if d.name in subjects_normalized]
            if not subject_dirs:
                logger.warning(f"No matching subjects found for: {subjects}")
                return {"successful": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Found {len(subject_dirs)} subjects")

        results = {"successful": 0, "failed": 0, "skipped": 0}

        for subject_dir in subject_dirs:
            subject_label = subject_dir.name

            # Check for sessions
            session_dirs = sorted([d for d in subject_dir.glob("ses-*") if d.is_dir()])

            if session_dirs:
                # Process each session
                for session_dir in session_dirs:
                    session_label = session_dir.name
                    result = process_func(subject_label, session_label=session_label)
                    results[result] += 1
            else:
                # Process without session
                result = process_func(subject_label, session_label=None)
                results[result] += 1

        if log_summary:
            self._log_summary(results)

        return results

    def _resolve_subject_output_dir(
        self,
        subject_label: str,
        session_label: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Resolve derivatives output directory for a subject/session run."""
        if output_dir is None:
            output_path = (
                self.bids_root
                / "derivatives"
                / "totalsegmentator"
                / subject_label
            )
            if session_label:
                output_path = output_path / session_label
            return output_path

        return Path(output_dir)

    def segment_ct_series(
        self,
        subject_label: str,
        session_label: Optional[str] = None,
        series_description_pattern: Optional[str] = None,
        tasks: Union[str, List[str]] = "total",
        statistics: bool = True,
        license_number: Optional[str] = None,
        device: str = "gpu",
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        overwrite: bool = False,
        generate_vertebrae_report: bool = True,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        """
        Find, validate, and segment CT series for a single subject/session.

        Parameters
        ----------
        subject_label : str
            Subject identifier (with or without "sub-" prefix)
        session_label : str, optional
            Session identifier (with or without "ses-" prefix)
        series_description_pattern : str, optional
            Regex pattern to match series description
        tasks : str or list of str
            TotalSegmentator task(s) to run.
            Single task: "total" or
            Multiple tasks: ["total", "tissue_types", "liver_segments", "abdominal_muscles"]
        statistics : bool
            Whether to generate volume and intensity statistics
        license_number : str, optional
            License key for premium models (e.g., tissue_types)
        device : str
            Device to use: "gpu" or "cpu"
        nr_thr_resamp : int
            Number of threads for resampling (lower to reduce memory)
        nr_thr_saving : int
            Number of threads for saving (lower to reduce memory)
        overwrite : bool
            If True, re-run segmentation even if output exists. Default: False
        generate_vertebrae_report : bool
            If True and task list includes ``"total"``, generate vertebrae
            slice report outputs after successful segmentation.
        output_dir : Path, optional
            Where to save segmentation derivatives. If None, uses
            bids_root/derivatives/totalsegmentator/
        **kwargs
            Additional arguments forwarded directly to the
            ``totalsegmentator()`` Python API (for example: ``preview=True``,
            ``radiomics=True``).

        Returns
        -------
        bool
            True if all segmentations succeeded, False otherwise
        """
        subject_id = subject_label.replace("sub-", "")
        display_id = f"{subject_id}"
        if session_label:
            session_id = session_label.replace("ses-", "")
            display_id += f" {session_id}"

        logger.info(f"\nProcessing: {display_id}")

        # Find matching CT series
        ct_file = self.segmentation.find_ct_series(
            self.bids_root,
            subject_label,
            session_label=session_label,
            series_description_pattern=series_description_pattern,
        )

        if not ct_file:
            logger.warning(f"  ✗ No matching CT series found")
            return False

        logger.info(f"  ✓ Found CT image: {ct_file.name}")

        # Check if in Hounsfield units
        is_hu, reason = self.segmentation.is_in_hounsfield_units(ct_file)
        logger.info(f"  HU check: {reason}")

        if not is_hu:
            logger.error(f"  ✗ CT image not in Hounsfield units - cannot segment")
            return False

        # Set up output directory
        output_dir = self._resolve_subject_output_dir(
            subject_label=subject_label,
            session_label=session_label,
            output_dir=output_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run segmentation(s)
        if isinstance(tasks, str):
            # Single task
            success = self.segmentation.run_segmentation(
                ct_file,
                output_dir,
                task=tasks,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
                **kwargs,
            )
        else:
            # Multiple tasks
            results = self.segmentation.run_multiple_segmentations(
                ct_file,
                output_dir,
                tasks=tasks,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
                **kwargs,
            )
            success = all(results.values())

        if success:
            logger.info(f"  ✓ All segmentations completed successfully")
            
            # Generate vertebrae slice reports if "total" task was run
            if generate_vertebrae_report and ((isinstance(tasks, str) and tasks == "total") or (isinstance(tasks, list) and "total" in tasks)):
                try:
                    # Check if total directory exists
                    total_dir = output_dir / "total"
                    if total_dir.exists():
                        # Check if reports already exist
                        summary_file = output_dir / f"{subject_label}_vertebrae_slices.tsv"
                        lookup_file = output_dir / f"{subject_label}_slice_vertebrae.tsv"
                        
                        if not (summary_file.exists() and lookup_file.exists()):
                            logger.info(f"  Generating vertebrae slice reports...")
                            create_vertebrae_slice_report(
                                self.bids_root,
                                subject_label,
                                session_label=session_label,
                                output_dir=output_dir,
                            )
                            logger.info(f"  ✓ Vertebrae reports generated")
                        else:
                            logger.debug(f"  Vertebrae reports already exist, skipping")
                    else:
                        logger.debug(f"  Total segmentation directory not found, skipping vertebrae reports")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to generate vertebrae reports: {e}")
            
            return True
        else:
            logger.error(f"  ✗ One or more segmentations failed")
            return False

    def run_pipeline(
        self,
        tasks: Optional[Union[str, List[str]]] = "total",
        subjects: Optional[Union[str, List[str]]] = None,
        session_label: Optional[str] = None,
        series_description_pattern: Optional[str] = None,
        statistics: bool = True,
        license_number: Optional[str] = None,
        device: str = "gpu",
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        overwrite: bool = False,
        output_dir: Optional[Path] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        gpu_workers: Optional[int] = None,
        cpu_workers: int = 1,
        parallel_tasks: bool = False,
        split_postprocessing: bool = False,
        run_group_statistics: bool = False,
        group_statistics_output_file: Optional[Path] = None,
        run_figures: bool = False,
        figures_montage_types: Union[str, List[str], None] = "individual",
        figures_organs_to_montage: Union[str, List[str], None] = None,
        figures_generate_liver_segments: bool = False,
        figures_include_skeletal: bool = True,
        figures_include_organs: Union[str, List[str], None] = None,
        figures_num_slices: int = 12,
        figures_window: tuple = (-200, 250),
        figures_alpha: float = 0.5,
        figures_axis: int = 2,
        figures_dpi: int = 200,
        figures_superior_limit: Optional[str] = None,
        figures_inferior_limit: Optional[str] = None,
        run_cross_subject_montage: bool = False,
        cross_subject_vertebrae: Optional[List[str]] = None,
        cross_subject_num_vertebrae: int = 7,
        cross_subject_output_dir: Optional[Path] = None,
        cross_subject_include_skeletal: bool = True,
        cross_subject_include_organs: Union[List[str], str, None] = None,
        cross_subject_window: tuple = (-200, 250),
        cross_subject_alpha: float = 0.35,
        cross_subject_dpi: int = 200,
        max_retries: int = 0,
        manifest_path: Optional[Path] = None,
        timeline_figure_path: Optional[Path] = None,
        continue_on_error: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the full processing pipeline for all subjects in the BIDS dataset.

        Automatically discovers all ``sub-*`` folders and processes each one.
        Sessions (``ses-*``) are detected and iterated automatically when present.
        Pass ``tasks=None`` to skip segmentation and run only cohort-level jobs
        against existing outputs.

        Parameters
        ----------
        tasks : str, list of str, or None
            TotalSegmentator task(s) to run, e.g. ``"total"`` or
            ``["total", "tissue_types", "liver_segments"]``.
            ``None`` skips segmentation (cohort-only mode).
        subjects : str or list of str, optional
            Limit processing to specific subject IDs. If None, all subjects are
            processed. IDs can be with or without the ``sub-`` prefix.
        session_label : str, optional
            Limit processing to a specific session (e.g. ``"ses-01"`` or ``"01"``).
            If None, all sessions are iterated automatically.
        series_description_pattern : str, optional
            Regex pattern to match CT series description.
        statistics : bool
            Whether to generate volume and intensity statistics for each task.
        license_number : str, optional
            TotalSegmentator license key for premium tasks
            (e.g., ``tissue_types``, ``liver_segments``).
        device : str
            Device for segmentation: ``"gpu"`` or ``"cpu"``.
        nr_thr_resamp : int
            Number of threads for resampling (lower to reduce memory pressure).
        nr_thr_saving : int
            Number of threads for saving output files.
        overwrite : bool
            If True, re-run segmentation even if output already exists.
        output_dir : Path, optional
            Base directory for segmentation derivatives. If None, uses
            ``bids_root/derivatives/totalsegmentator/``.
        parallel : bool
            If True, execute jobs in parallel using the scheduler.
        max_workers : int, optional
            Maximum concurrent workers when ``parallel=True``. Defaults to 1.
        gpu_workers : int, optional
            Maximum concurrent GPU jobs when ``parallel=True``.
        cpu_workers : int
            Maximum concurrent CPU jobs. Defaults to 1.
        parallel_tasks : bool
            If True and ``tasks`` is a list, split each task into separate jobs.
        split_postprocessing : bool
            If True, run stats/reporting as explicit CPU jobs with dependencies.
            Auto-enabled when ``run_figures=True`` or ``tasks=None``.
        run_group_statistics : bool
            If True, add a cohort-level statistics consolidation CPU job.
        group_statistics_output_file : Path, optional
            Output TSV path for cohort statistics consolidation.
        run_figures : bool
            If True, add a per-subject figure generation CPU job after segmentation.
        figures_montage_types : str, list, or None
            Tissue-type montage families: ``"individual"``, ``"cross-subject"``,
            or both. Default ``"individual"``.
        figures_organs_to_montage : str, list, or None
            Organs for dedicated organ montages. ``None`` skips organ montages;
            ``"all"`` includes every supported organ.
        figures_generate_liver_segments : bool
            If True, generate liver subsegment montages from ``liver_segments`` outputs.
        figures_include_skeletal : bool
            Include skeletal composite overlay on tissue montages.
        figures_include_organs : str, list, or None
            Organ overlays to draw on top of tissue montages.
        figures_num_slices : int
            Number of slices in individual montages.
        figures_window : tuple
            HU display window ``(min, max)``.
        figures_alpha : float
            Overlay transparency.
        figures_axis : int
            Slice axis for individual montages (0=sagittal, 1=coronal, 2=axial).
        figures_dpi : int
            Output image resolution.
        figures_superior_limit : str, optional
            Superior anatomical limit for montage range, e.g. ``"C1"``.
        figures_inferior_limit : str, optional
            Inferior anatomical limit, e.g. ``"sacrum"``.
        run_cross_subject_montage : bool
            If True, add a cohort-level cross-subject montage CPU job.
            Requires ``"total"`` task outputs.
        cross_subject_vertebrae : list of str, optional
            Explicit vertebrae labels for cross-subject montage.
        cross_subject_num_vertebrae : int
            Number of vertebrae to include when labels are auto-selected.
        cross_subject_output_dir : Path, optional
            Output directory for cross-subject montage image.
        cross_subject_include_skeletal : bool
            Include skeletal overlay in cross-subject montage.
        cross_subject_include_organs : str, list, or None
            Organ overlays for cross-subject montage.
        cross_subject_window : tuple
            HU window for cross-subject montage rendering.
        cross_subject_alpha : float
            Overlay alpha for cross-subject montage rendering.
        cross_subject_dpi : int
            Output resolution for cross-subject montage.
        max_retries : int
            Maximum retry attempts per scheduled job after first failure.
        manifest_path : Path, optional
            Output path for structured scheduler run manifest JSON.
        timeline_figure_path : Path, optional
            Output path for scheduler timeline figure.
        continue_on_error : bool
            If False, raise ``RuntimeError`` if any job fails.
        **kwargs
            Additional arguments forwarded directly to TotalSegmentator
            (e.g. ``preview=True``, ``radiomics=True``).

        Returns
        -------
        dict
            Summary with keys: ``'successful'``, ``'failed'``, ``'skipped'``.
        """
        task_list = [] if tasks is None else ([tasks] if isinstance(tasks, str) else list(tasks))
        has_total_task = "total" in task_list

        # Cohort-only / figures mode: force DAG path.
        if tasks is None or run_figures:
            split_postprocessing = True

        # Resolve default output paths into the BIDS derivatives folder so run
        # artefacts are co-located with segmentation outputs.
        _derivatives_dir = (
            Path(output_dir)
            if output_dir
            else Path(self.bids_root) / "derivatives" / "totalsegmentator"
        )
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if manifest_path is None:
            manifest_path = _derivatives_dir / f"run_manifest_{_ts}.json"
        if timeline_figure_path is None:
            timeline_figure_path = _derivatives_dir / f"run_timeline_{_ts}.png"

        # Only warn about missing 'total' task when actually running segmentation.
        if tasks is not None and run_cross_subject_montage and not has_total_task:
            logger.warning(
                "run_cross_subject_montage=True requested but 'total' task is not included; "
                "cross-subject montage job will be skipped."
            )
            run_cross_subject_montage = False

        if split_postprocessing:
            jobs = build_processing_jobs(
                bids_root=self.bids_root,
                tasks=tasks,
                subjects=subjects,
                parallel_tasks=parallel_tasks,
                include_stats_jobs=statistics,
                include_vertebrae_report_jobs=True,
                include_figures_jobs=run_figures,
                figures_payload={
                    "montage_types": figures_montage_types,
                    "organs_to_montage": figures_organs_to_montage,
                    "generate_liver_segments": figures_generate_liver_segments,
                    "include_skeletal": figures_include_skeletal,
                    "include_organs": figures_include_organs,
                    "num_slices": figures_num_slices,
                    "window": figures_window,
                    "alpha": figures_alpha,
                    "axis": figures_axis,
                    "dpi": figures_dpi,
                    "superior_limit": figures_superior_limit,
                    "inferior_limit": figures_inferior_limit,
                },
                include_group_stats_job=run_group_statistics,
                include_cross_subject_montage_job=run_cross_subject_montage,
                cross_subject_payload={
                    "subjects": subjects,
                    "session_label": session_label,
                    "output_dir": cross_subject_output_dir,
                    "series_description_pattern": series_description_pattern,
                    "include_skeletal": cross_subject_include_skeletal,
                    "include_organs": cross_subject_include_organs,
                    "window": cross_subject_window,
                    "alpha": cross_subject_alpha,
                    "dpi": cross_subject_dpi,
                    "vertebrae": cross_subject_vertebrae,
                    "num_vertebrae": cross_subject_num_vertebrae,
                },
            )
        else:
            jobs = build_segmentation_jobs(
                bids_root=self.bids_root,
                tasks=tasks,
                subjects=subjects,
                parallel_tasks=parallel_tasks,
            )

        if not jobs:
            if tasks is None:
                logger.warning("tasks=None but no cohort jobs were requested; nothing to do.")
            else:
                logger.warning(f"No matching subjects found for: {subjects}")
            return {"successful": 0, "failed": 0, "skipped": 0}

        logger.info(f"Prepared {len(jobs)} segmentation job(s)")

        def _run_segment_job(job):
            return self.segment_ct_series(
                subject_label=job.subject_label,
                session_label=job.session_label,
                series_description_pattern=series_description_pattern,
                tasks=job.tasks,
                statistics=statistics if not split_postprocessing else False,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
                generate_vertebrae_report=not split_postprocessing,
                output_dir=output_dir,
                **kwargs,
            )

        if split_postprocessing:
            def _run_stats_job(job):
                base_output = self._resolve_subject_output_dir(
                    subject_label=job.subject_label,
                    session_label=job.session_label,
                    output_dir=output_dir,
                )
                return self.segmentation.compute_statistics_only(
                    output_dir=base_output,
                    task=str(job.tasks),
                    overwrite=overwrite,
                )

            def _run_vertebrae_report_job(job):
                base_output = self._resolve_subject_output_dir(
                    subject_label=job.subject_label,
                    session_label=job.session_label,
                    output_dir=output_dir,
                )
                try:
                    create_vertebrae_slice_report(
                        self.bids_root,
                        job.subject_label,
                        session_label=job.session_label,
                        output_dir=base_output,
                    )
                    return True
                except Exception as e:
                    logger.warning(
                        f"  ⚠ Failed to generate vertebrae reports for {job.subject_label}: {e}"
                    )
                    return False

            def _run_group_stats_job(job):
                try:
                    consolidate_group_statistics(
                        bids_root=self.bids_root,
                        subjects=subjects if isinstance(subjects, list) else ([subjects] if isinstance(subjects, str) else None),
                        output_file=group_statistics_output_file,
                        overwrite=overwrite,
                    )
                    return True
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to consolidate group statistics: {e}")
                    return False

            def _run_cross_subject_montage_job(job):
                try:
                    payload = job.payload or {}
                    subjects_arg = payload.get("subjects", subjects)
                    if subjects_arg is None:
                        subject_dirs = sorted([d for d in self.bids_root.glob("sub-*") if d.is_dir()])
                        subjects_arg = [d.name for d in subject_dirs]
                    create_vertebrae_cross_subject_montage(
                        bids_root=self.bids_root,
                        subjects=subjects_arg,
                        vertebrae=payload.get("vertebrae", cross_subject_vertebrae),
                        num_vertebrae=payload.get("num_vertebrae", cross_subject_num_vertebrae),
                        session_label=payload.get("session_label", None),
                        output_dir=payload.get("output_dir", cross_subject_output_dir),
                        series_description_pattern=payload.get("series_description_pattern", series_description_pattern),
                        include_skeletal=payload.get("include_skeletal", cross_subject_include_skeletal),
                        include_organs=payload.get("include_organs", cross_subject_include_organs),
                        window=payload.get("window", cross_subject_window),
                        alpha=payload.get("alpha", cross_subject_alpha),
                        dpi=payload.get("dpi", cross_subject_dpi),
                    )
                    return True
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to generate cross-subject montage: {e}")
                    return False

            def _run_figures_job(job):
                try:
                    payload = job.payload or {}
                    generate_montages_from_bids(
                        bids_root=self.bids_root,
                        subjects=[job.subject_label],
                        session_label=job.session_label,
                        series_description_pattern=series_description_pattern,
                        montage_types=payload.get("montage_types", figures_montage_types),
                        organs_to_montage=payload.get("organs_to_montage", figures_organs_to_montage),
                        generate_liver_segments=payload.get("generate_liver_segments", figures_generate_liver_segments),
                        include_skeletal=payload.get("include_skeletal", figures_include_skeletal),
                        include_organs=payload.get("include_organs", figures_include_organs),
                        num_slices=payload.get("num_slices", figures_num_slices),
                        window=payload.get("window", figures_window),
                        alpha=payload.get("alpha", figures_alpha),
                        axis=payload.get("axis", figures_axis),
                        dpi=payload.get("dpi", figures_dpi),
                        superior_limit=payload.get("superior_limit", figures_superior_limit),
                        inferior_limit=payload.get("inferior_limit", figures_inferior_limit),
                    )
                    return True
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to generate figures for {job.subject_label}: {e}")
                    return False

            results = run_job_graph(
                jobs=jobs,
                run_handlers={
                    "segment": _run_segment_job,
                    "stats": _run_stats_job,
                    "vertebrae_report": _run_vertebrae_report_job,
                    "figures": _run_figures_job,
                    "group_stats": _run_group_stats_job,
                    "cross_subject_montage": _run_cross_subject_montage_job,
                },
                parallel=parallel,
                gpu_workers=gpu_workers if gpu_workers is not None else (max_workers or 1),
                cpu_workers=cpu_workers,
                continue_on_error=continue_on_error,
                max_retries=max_retries,
                manifest_path=manifest_path,
                timeline_figure_path=timeline_figure_path,
            )
        else:
            results = run_segmentation_jobs(
                jobs=jobs,
                run_job=_run_segment_job,
                parallel=parallel,
                max_workers=gpu_workers if gpu_workers is not None else max_workers,
                cpu_workers=cpu_workers,
                continue_on_error=continue_on_error,
                max_retries=max_retries,
                manifest_path=manifest_path,
                timeline_figure_path=timeline_figure_path,
            )

        self._log_summary(results)
        return {
            "successful": results["successful"],
            "failed": results["failed"],
            "skipped": results["skipped"],
            "job_results": results.get("job_results", {}),
        }

    @staticmethod
    def _log_summary(results: Dict[str, int]):
        """Log processing results summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete:")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Skipped: {results['skipped']}")
        logger.info(f"{'='*60}")

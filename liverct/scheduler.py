"""Scheduler utilities for segmentation orchestration.

This module centralizes subject/session discovery and job scheduling logic so
pipeline code remains readable and execution policy can evolve independently.
"""

import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Literal

logger = logging.getLogger(__name__)

JobType = Literal[
    "segment",
    "stats",
    "vertebrae_report",
    "figures",
    "group_stats",
    "cross_subject_montage",
]
ResourceType = Literal["gpu", "cpu"]


@dataclass(frozen=True)
class SegmentationJob:
    """A schedulable processing job unit for a subject/session."""

    id: str
    subject_label: str
    session_label: Optional[str]
    tasks: Union[str, List[str]]
    job_type: JobType = "segment"
    resource_type: ResourceType = "gpu"
    depends_on: Tuple[str, ...] = field(default_factory=tuple)
    payload: Dict[str, Any] = field(default_factory=dict)


def _normalize_subject_filters(
    subjects: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
    if subjects is None:
        return None
    if isinstance(subjects, str):
        subjects = [subjects]
    return [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]


def discover_subject_sessions(
    bids_root: Path,
    subjects: Optional[Union[str, List[str]]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Discover subject/session units from a BIDS root.

    Returns
    -------
    list of tuple
        List of ``(subject_label, session_label_or_none)`` units.
    """
    bids_root = Path(bids_root)
    subject_dirs = sorted([d for d in bids_root.glob("sub-*") if d.is_dir()])

    subject_filter = _normalize_subject_filters(subjects)
    if subject_filter is not None:
        subject_dirs = [d for d in subject_dirs if d.name in subject_filter]

    units: List[Tuple[str, Optional[str]]] = []
    for subject_dir in subject_dirs:
        session_dirs = sorted([d for d in subject_dir.glob("ses-*") if d.is_dir()])
        if session_dirs:
            for session_dir in session_dirs:
                units.append((subject_dir.name, session_dir.name))
        else:
            units.append((subject_dir.name, None))

    return units


def build_segmentation_jobs(
    bids_root: Path,
    tasks: Union[str, List[str]],
    subjects: Optional[Union[str, List[str]]] = None,
    parallel_tasks: bool = False,
) -> List[SegmentationJob]:
    """
    Build segmentation jobs from BIDS structure.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory.
    tasks : str or list of str
        Task(s) requested by user.
    subjects : str or list of str, optional
        Subject filters.
    parallel_tasks : bool
        If True and ``tasks`` is a list, creates one job per task per
        subject/session unit. If False, creates one job per subject/session
        containing all requested tasks.

    Returns
    -------
    list of SegmentationJob
        Jobs ready for execution.
    """
    units = discover_subject_sessions(bids_root=bids_root, subjects=subjects)
    jobs: List[SegmentationJob] = []

    for subject_label, session_label in units:
        unit_id = subject_label if session_label is None else f"{subject_label}_{session_label}"

        if parallel_tasks and isinstance(tasks, list):
            for task in tasks:
                job_id = f"{unit_id}_{task}"
                jobs.append(
                    SegmentationJob(
                        id=job_id,
                        subject_label=subject_label,
                        session_label=session_label,
                        tasks=task,
                    )
                )
        else:
            jobs.append(
                SegmentationJob(
                    id=unit_id,
                    subject_label=subject_label,
                    session_label=session_label,
                    tasks=tasks,
                )
            )

    return jobs


def build_processing_jobs(
    bids_root: Path,
    tasks: Optional[Union[str, List[str]]],
    subjects: Optional[Union[str, List[str]]] = None,
    parallel_tasks: bool = False,
    include_stats_jobs: bool = False,
    include_vertebrae_report_jobs: bool = True,
    include_figures_jobs: bool = False,
    figures_payload: Optional[Dict[str, Any]] = None,
    include_group_stats_job: bool = False,
    include_cross_subject_montage_job: bool = False,
    cross_subject_payload: Optional[Dict[str, Any]] = None,
) -> List[SegmentationJob]:
    """
    Build an explicit processing DAG as flat jobs with dependencies.

    Notes
    -----
    - When ``tasks`` is ``None``, no per-subject segmentation jobs are created
      (cohort-only mode).  Cohort jobs (``group_stats``, ``cross_subject_montage``)
      are still built with no dependencies so they run immediately.
    - Segment jobs are always generated when ``tasks`` is not ``None``.
    - Optional CPU jobs (`stats`, `vertebrae_report`) are attached as dependents.
    - Dependencies are represented by job IDs in ``depends_on``.
    """
    if tasks is None:
        # Cohort-only mode: no per-subject work, cohort jobs have no dependencies.
        segment_jobs: List[SegmentationJob] = []
    else:
        segment_jobs = build_segmentation_jobs(
            bids_root=bids_root,
            tasks=tasks,
            subjects=subjects,
            parallel_tasks=parallel_tasks,
        )
        if not segment_jobs:
            return []

    jobs: List[SegmentationJob] = list(segment_jobs)
    stats_job_ids: List[str] = []
    vertebrae_job_ids: List[str] = []

    for seg_job in segment_jobs:
        task_list = [seg_job.tasks] if isinstance(seg_job.tasks, str) else list(seg_job.tasks)

        if include_stats_jobs:
            for task in task_list:
                # Keep behavior aligned with current pipeline where TotalSegmentator
                # handles total statistics internally.
                if task == "total":
                    continue
                jobs.append(
                    SegmentationJob(
                        id=f"{seg_job.id}_stats_{task}",
                        subject_label=seg_job.subject_label,
                        session_label=seg_job.session_label,
                        tasks=task,
                        job_type="stats",
                        resource_type="cpu",
                        depends_on=(seg_job.id,),
                    )
                )
                stats_job_ids.append(f"{seg_job.id}_stats_{task}")

        if include_vertebrae_report_jobs and "total" in task_list:
            jobs.append(
                SegmentationJob(
                    id=f"{seg_job.id}_vertebrae_report",
                    subject_label=seg_job.subject_label,
                    session_label=seg_job.session_label,
                    tasks="total",
                    job_type="vertebrae_report",
                    resource_type="cpu",
                    depends_on=(seg_job.id,),
                )
            )
            vertebrae_job_ids.append(f"{seg_job.id}_vertebrae_report")

    if include_figures_jobs:
        # When segment_jobs exist, create one figures job per subject/session unit
        # that depends on all segment jobs for that unit.  When tasks=None (cohort-
        # only re-run), discover subjects directly so figures can still be produced.
        if segment_jobs:
            figure_units = {(j.subject_label, j.session_label) for j in segment_jobs}
        else:
            figure_units = {
                (subj, sess)
                for subj, sess in discover_subject_sessions(bids_root=bids_root, subjects=subjects)
            }
        for subj, sess in sorted(figure_units):
            unit_id = subj if sess is None else f"{subj}_{sess}"
            dep_ids = tuple(
                j.id for j in segment_jobs
                if j.subject_label == subj and j.session_label == sess
            )
            jobs.append(
                SegmentationJob(
                    id=f"{unit_id}_figures",
                    subject_label=subj,
                    session_label=sess,
                    tasks="figures",
                    job_type="figures",
                    resource_type="cpu",
                    depends_on=dep_ids,
                    payload=figures_payload or {},
                )
            )

    if include_group_stats_job:
        group_dep_ids = [job.id for job in segment_jobs] + stats_job_ids
        jobs.append(
            SegmentationJob(
                id="group_statistics_consolidation",
                subject_label="group",
                session_label=None,
                tasks="group_stats",
                job_type="group_stats",
                resource_type="cpu",
                depends_on=tuple(group_dep_ids),
            )
        )

    if include_cross_subject_montage_job:
        cross_subject_dep_ids = [job.id for job in segment_jobs] + vertebrae_job_ids
        jobs.append(
            SegmentationJob(
                id="cross_subject_montage",
                subject_label="group",
                session_label=None,
                tasks="cross_subject_montage",
                job_type="cross_subject_montage",
                resource_type="cpu",
                depends_on=tuple(cross_subject_dep_ids),
                payload=cross_subject_payload or {},
            )
        )

    return jobs


def run_job_graph(
    jobs: List[SegmentationJob],
    run_handlers: Dict[JobType, Callable[[SegmentationJob], bool]],
    parallel: bool = False,
    gpu_workers: int = 1,
    cpu_workers: int = 1,
    continue_on_error: bool = True,
    max_retries: int = 0,
    manifest_path: Optional[Path] = None,
    timeline_figure_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute dependency-aware jobs with separate GPU and CPU worker pools.

    Parameters
    ----------
    jobs : list of SegmentationJob
        Jobs to execute.
    run_handlers : dict
        Mapping from job type to callable, e.g.
        ``{"segment": fn, "stats": fn, "vertebrae_report": fn}``.
    parallel : bool
        Whether to execute ready jobs in parallel.
    gpu_workers : int
        Worker count for GPU-labeled jobs.
    cpu_workers : int
        Worker count for CPU-labeled jobs.
    continue_on_error : bool
        If False, raise RuntimeError on first failure.
    max_retries : int
        Maximum retry attempts per job after the first failed attempt.
    manifest_path : Path, optional
        If provided, write a structured JSON run manifest to this path.
    timeline_figure_path : Path, optional
        If provided, write a timeline figure of scheduled jobs.

    Returns
    -------
    dict
        Summary with ``successful``, ``failed``, ``skipped``, and
        ``job_results`` (job_id -> status string).
    """
    if not jobs:
        return {"successful": 0, "failed": 0, "skipped": 0, "job_results": {}}

    run_started = time.time()

    def _task_to_str(tasks_val: Union[str, List[str]]) -> str:
        if isinstance(tasks_val, list):
            return ",".join([str(t) for t in tasks_val])
        return str(tasks_val)

    job_meta: Dict[str, Dict[str, Any]] = {
        j.id: {
            "job_id": j.id,
            "job_type": j.job_type,
            "resource_type": j.resource_type,
            "subject_label": j.subject_label,
            "session_label": j.session_label,
            "tasks": _task_to_str(j.tasks),
            "depends_on": list(j.depends_on),
            "status": "pending",
            "attempts": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "attempt_records": [],
            "payload": j.payload,
        }
        for j in jobs
    }

    by_id: Dict[str, SegmentationJob] = {j.id: j for j in jobs}
    unknown_deps = [
        (j.id, dep)
        for j in jobs
        for dep in j.depends_on
        if dep not in by_id
    ]
    if unknown_deps:
        dep_pairs = ", ".join([f"{jid}->{dep}" for jid, dep in unknown_deps[:5]])
        raise ValueError(f"Jobs contain unknown dependencies: {dep_pairs}")

    def _handler_for(job: SegmentationJob) -> Callable[[SegmentationJob], bool]:
        if job.job_type not in run_handlers:
            raise KeyError(f"No run handler configured for job_type={job.job_type}")
        return run_handlers[job.job_type]

    pending: Dict[str, SegmentationJob] = dict(by_id)
    successful: set = set()
    failed: set = set()
    skipped: set = set()
    job_results: Dict[str, str] = {}
    attempts_by_job: Dict[str, int] = {j.id: 0 for j in jobs}

    def _current_summary() -> Dict[str, Any]:
        return {
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "job_results": dict(job_results),
        }

    def _write_manifest_checkpoint() -> None:
        if manifest_path is None:
            return
        try:
            _write_manifest_file(
                summary=_current_summary(),
                job_meta=job_meta,
                run_started=run_started,
                run_ended=time.time(),
                manifest_path=Path(manifest_path),
            )
        except Exception as e:
            logger.warning("Failed to write scheduler manifest checkpoint: %s", e)

    # Emit an initial checkpoint so long runs have an on-disk manifest quickly.
    _write_manifest_checkpoint()

    def _mark_skippable() -> bool:
        changed = False
        for job_id, job in list(pending.items()):
            if any(dep in failed or dep in skipped for dep in job.depends_on):
                skipped.add(job_id)
                job_results[job_id] = "skipped"
                job_meta[job_id]["status"] = "skipped"
                if job_meta[job_id]["end_time"] is None:
                    job_meta[job_id]["end_time"] = time.time()
                del pending[job_id]
                _write_manifest_checkpoint()
                changed = True
        return changed

    if not parallel:
        while pending:
            _mark_skippable()
            ready = [
                job
                for job in pending.values()
                if all(dep in successful for dep in job.depends_on)
            ]

            if not ready:
                # Remaining jobs have unresolved deps due to failures/skips or a cycle.
                for job_id in list(pending.keys()):
                    skipped.add(job_id)
                    job_results[job_id] = "skipped"
                    job_meta[job_id]["status"] = "skipped"
                    if job_meta[job_id]["end_time"] is None:
                        job_meta[job_id]["end_time"] = time.time()
                    del pending[job_id]
                _write_manifest_checkpoint()
                break

            for job in sorted(ready, key=lambda j: j.id):
                del pending[job.id]
                attempt_idx = attempts_by_job[job.id] + 1
                attempts_by_job[job.id] = attempt_idx
                start_ts = time.time()
                if job_meta[job.id]["start_time"] is None:
                    job_meta[job.id]["start_time"] = start_ts
                job_meta[job.id]["status"] = "running"
                _write_manifest_checkpoint()
                try:
                    ok = bool(_handler_for(job)(job))
                except Exception:
                    logger.exception("Job crashed: %s", job.id)
                    ok = False
                end_ts = time.time()

                job_meta[job.id]["attempts"] = attempts_by_job[job.id]
                job_meta[job.id]["attempt_records"].append(
                    {
                        "attempt": attempt_idx,
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "duration_seconds": end_ts - start_ts,
                        "resource_type": job.resource_type,
                        "lane": 0,
                        "status": "successful" if ok else "failed",
                    }
                )

                if ok:
                    successful.add(job.id)
                    job_results[job.id] = "successful"
                    job_meta[job.id]["status"] = "successful"
                    job_meta[job.id]["end_time"] = end_ts
                    job_meta[job.id]["duration_seconds"] = (
                        job_meta[job.id]["end_time"] - job_meta[job.id]["start_time"]
                    )
                    _write_manifest_checkpoint()
                else:
                    if attempts_by_job[job.id] <= max_retries:
                        logger.warning(
                            "Job failed, retrying (%d/%d): %s",
                            attempts_by_job[job.id],
                            max_retries + 1,
                            job.id,
                        )
                        job_meta[job.id]["status"] = "pending"
                        pending[job.id] = job
                        _write_manifest_checkpoint()
                        continue

                    failed.add(job.id)
                    job_results[job.id] = "failed"
                    job_meta[job.id]["status"] = "failed"
                    job_meta[job.id]["end_time"] = end_ts
                    job_meta[job.id]["duration_seconds"] = (
                        job_meta[job.id]["end_time"] - job_meta[job.id]["start_time"]
                    )
                    _write_manifest_checkpoint()
                    if not continue_on_error:
                        raise RuntimeError(f"Job failed: {job.id}")

        summary = {
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "job_results": job_results,
        }
        _finalize_scheduler_outputs(
            summary=summary,
            job_meta=job_meta,
            run_started=run_started,
            run_ended=time.time(),
            manifest_path=manifest_path,
            timeline_figure_path=timeline_figure_path,
        )
        return summary

    gpu_workers = max(1, int(gpu_workers))
    cpu_workers = max(1, int(cpu_workers))

    logger.info(
        "Running %d jobs with GPU workers=%d and CPU workers=%d",
        len(jobs),
        gpu_workers,
        cpu_workers,
    )

    running: Dict[Future, SegmentationJob] = {}
    running_ctx: Dict[Future, Dict[str, Any]] = {}
    submitted: set = set()
    gpu_free_lanes = list(range(gpu_workers))
    cpu_free_lanes = list(range(cpu_workers))
    gpu_occupied_lanes: set = set()
    cpu_occupied_lanes: set = set()
    with ThreadPoolExecutor(max_workers=gpu_workers) as gpu_pool, ThreadPoolExecutor(
        max_workers=cpu_workers
    ) as cpu_pool:
        while pending or running:
            _mark_skippable()

            ready = [
                job
                for job in pending.values()
                if job.id not in submitted and all(dep in successful for dep in job.depends_on)
            ]

            for job in sorted(ready, key=lambda j: j.id):
                try:
                    handler = _handler_for(job)
                except Exception:
                    logger.exception("No handler for job: %s", job.id)
                    failed.add(job.id)
                    job_results[job.id] = "failed"
                    job_meta[job.id]["status"] = "failed"
                    job_meta[job.id]["end_time"] = time.time()
                    if job_meta[job.id]["start_time"] is None:
                        job_meta[job.id]["start_time"] = job_meta[job.id]["end_time"]
                    job_meta[job.id]["duration_seconds"] = (
                        job_meta[job.id]["end_time"] - job_meta[job.id]["start_time"]
                    )
                    del pending[job.id]
                    _write_manifest_checkpoint()
                    if not continue_on_error:
                        raise
                    continue

                if job.resource_type == "gpu":
                    if not gpu_free_lanes:
                        # No GPU lane is currently free; try this job on a later scheduler tick.
                        continue
                    pool = gpu_pool
                    lane = gpu_free_lanes.pop(0)
                    if lane in gpu_occupied_lanes:
                        raise RuntimeError(f"Scheduler lane invariant violated (GPU lane busy): {lane}")
                    gpu_occupied_lanes.add(lane)
                else:
                    if not cpu_free_lanes:
                        # No CPU lane is currently free; try this job on a later scheduler tick.
                        continue
                    pool = cpu_pool
                    lane = cpu_free_lanes.pop(0)
                    if lane in cpu_occupied_lanes:
                        raise RuntimeError(f"Scheduler lane invariant violated (CPU lane busy): {lane}")
                    cpu_occupied_lanes.add(lane)

                attempt_idx = attempts_by_job[job.id] + 1
                attempts_by_job[job.id] = attempt_idx
                start_ts = time.time()
                if job_meta[job.id]["start_time"] is None:
                    job_meta[job.id]["start_time"] = start_ts

                fut = pool.submit(handler, job)
                running[fut] = job
                running_ctx[fut] = {
                    "attempt": attempt_idx,
                    "start_time": start_ts,
                    "lane": lane,
                }
                job_meta[job.id]["status"] = "running"
                submitted.add(job.id)
                del pending[job.id]
                _write_manifest_checkpoint()

            if not running:
                # No running jobs and none newly ready; remaining jobs are unschedulable.
                for job_id in list(pending.keys()):
                    skipped.add(job_id)
                    job_results[job_id] = "skipped"
                    job_meta[job_id]["status"] = "skipped"
                    if job_meta[job_id]["end_time"] is None:
                        job_meta[job_id]["end_time"] = time.time()
                    del pending[job_id]
                _write_manifest_checkpoint()
                break

            done, _ = wait(set(running.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                job = running.pop(fut)
                ctx = running_ctx.pop(fut)
                try:
                    ok = bool(fut.result())
                except Exception:
                    logger.exception("Job crashed: %s", job.id)
                    ok = False

                end_ts = time.time()
                lane = int(ctx["lane"])
                if job.resource_type == "gpu":
                    gpu_occupied_lanes.discard(lane)
                    gpu_free_lanes.append(lane)
                    gpu_free_lanes.sort()
                else:
                    cpu_occupied_lanes.discard(lane)
                    cpu_free_lanes.append(lane)
                    cpu_free_lanes.sort()

                job_meta[job.id]["attempts"] = attempts_by_job[job.id]
                job_meta[job.id]["attempt_records"].append(
                    {
                        "attempt": int(ctx["attempt"]),
                        "start_time": float(ctx["start_time"]),
                        "end_time": end_ts,
                        "duration_seconds": end_ts - float(ctx["start_time"]),
                        "resource_type": job.resource_type,
                        "lane": lane,
                        "status": "successful" if ok else "failed",
                    }
                )

                if ok:
                    successful.add(job.id)
                    job_results[job.id] = "successful"
                    job_meta[job.id]["status"] = "successful"
                    job_meta[job.id]["end_time"] = end_ts
                    job_meta[job.id]["duration_seconds"] = (
                        job_meta[job.id]["end_time"] - job_meta[job.id]["start_time"]
                    )
                    _write_manifest_checkpoint()
                else:
                    if attempts_by_job[job.id] <= max_retries:
                        logger.warning(
                            "Job failed, retrying (%d/%d): %s",
                            attempts_by_job[job.id],
                            max_retries + 1,
                            job.id,
                        )
                        job_meta[job.id]["status"] = "pending"
                        pending[job.id] = job
                        submitted.discard(job.id)
                        _write_manifest_checkpoint()
                        continue

                    failed.add(job.id)
                    job_results[job.id] = "failed"
                    job_meta[job.id]["status"] = "failed"
                    job_meta[job.id]["end_time"] = end_ts
                    job_meta[job.id]["duration_seconds"] = (
                        job_meta[job.id]["end_time"] - job_meta[job.id]["start_time"]
                    )
                    _write_manifest_checkpoint()
                    if not continue_on_error:
                        raise RuntimeError(f"Job failed: {job.id}")

    summary = {
        "successful": len(successful),
        "failed": len(failed),
        "skipped": len(skipped),
        "job_results": job_results,
    }
    _finalize_scheduler_outputs(
        summary=summary,
        job_meta=job_meta,
        run_started=run_started,
        run_ended=time.time(),
        manifest_path=manifest_path,
        timeline_figure_path=timeline_figure_path,
    )
    return summary


def _build_manifest(
    summary: Dict[str, Any],
    job_meta: Dict[str, Dict[str, Any]],
    run_started: float,
    run_ended: float,
) -> Dict[str, Any]:
    """Build a scheduler manifest dictionary from current state."""
    manifest = {
        "run": {
            "started_at": datetime.fromtimestamp(run_started, tz=timezone.utc).isoformat(),
            "ended_at": datetime.fromtimestamp(run_ended, tz=timezone.utc).isoformat(),
            "duration_seconds": run_ended - run_started,
            "summary": {
                "successful": summary.get("successful", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
            },
        },
        "jobs": [],
    }

    for job_id in sorted(job_meta.keys()):
        rec = dict(job_meta[job_id])
        if rec.get("start_time") is not None:
            rec["start_time_iso"] = datetime.fromtimestamp(
                rec["start_time"], tz=timezone.utc
            ).isoformat()
        if rec.get("end_time") is not None:
            rec["end_time_iso"] = datetime.fromtimestamp(
                rec["end_time"], tz=timezone.utc
            ).isoformat()
        manifest["jobs"].append(rec)

    return manifest


def _write_manifest_file(
    summary: Dict[str, Any],
    job_meta: Dict[str, Dict[str, Any]],
    run_started: float,
    run_ended: float,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Write scheduler manifest JSON and return the manifest dict."""
    manifest = _build_manifest(
        summary=summary,
        job_meta=job_meta,
        run_started=run_started,
        run_ended=run_ended,
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _finalize_scheduler_outputs(
    summary: Dict[str, Any],
    job_meta: Dict[str, Dict[str, Any]],
    run_started: float,
    run_ended: float,
    manifest_path: Optional[Path] = None,
    timeline_figure_path: Optional[Path] = None,
) -> None:
    """Write optional run manifest and timeline figure outputs."""
    manifest = _build_manifest(
        summary=summary,
        job_meta=job_meta,
        run_started=run_started,
        run_ended=run_ended,
    )

    if manifest_path is not None:
        _write_manifest_file(
            summary=summary,
            job_meta=job_meta,
            run_started=run_started,
            run_ended=run_ended,
            manifest_path=Path(manifest_path),
        )

    if timeline_figure_path is not None:
        try:
            _write_timeline_figure(manifest, Path(timeline_figure_path))
        except Exception as e:
            logger.warning("Failed to write scheduler timeline figure: %s", e)


def _write_timeline_figure(manifest: Dict[str, Any], output_png: Path) -> None:
    """Render a scheduler timeline figure from manifest attempt records."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise ImportError("matplotlib is required for timeline figure output") from e

    jobs = manifest.get("jobs", [])
    if not jobs:
        return

    segments = []
    min_t = None
    max_t = None
    lane_keys = set()

    for job in jobs:
        attempt_records = job.get("attempt_records", [])
        for rec in attempt_records:
            s = rec.get("start_time")
            e = rec.get("end_time")
            if s is None or e is None:
                continue
            lane_key = f"{rec.get('resource_type', 'cpu').upper()}-{rec.get('lane', 0)}"
            lane_keys.add(lane_key)
            segments.append(
                {
                    "start": float(s),
                    "end": float(e),
                    "lane": lane_key,
                    "job_type": job.get("job_type", "segment"),
                    "subject": job.get("subject_label", ""),
                    "tasks": job.get("tasks", ""),
                    "status": rec.get("status", ""),
                }
            )
            min_t = float(s) if min_t is None else min(min_t, float(s))
            max_t = float(e) if max_t is None else max(max_t, float(e))

    if not segments or min_t is None or max_t is None:
        return

    lane_order = sorted(lane_keys, key=lambda x: (0 if x.startswith("GPU") else 1, x))
    lane_to_y = {lane: i for i, lane in enumerate(lane_order)}

    color_map = {
        "segment": "#4C78A8",
        "stats": "#F58518",
        "vertebrae_report": "#54A24B",
        "group_stats": "#B279A2",
        "cross_subject_montage": "#E45756",
    }

    width = max(10.0, min(24.0, (max_t - min_t) / 120.0))
    height = max(3.0, 0.8 * max(1, len(lane_order)))
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)

    for seg in segments:
        y = lane_to_y[seg["lane"]]
        x0 = seg["start"] - min_t
        w = max(0.05, seg["end"] - seg["start"])
        color = color_map.get(seg["job_type"], "#777777")
        rect = Rectangle((x0, y - 0.35), w, 0.7, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)

        label = f"{seg['subject']}:{seg['tasks']}"
        if len(label) > 30:
            label = label[:27] + "..."
        ax.text(x0 + w / 2.0, y, label, ha="center", va="center", fontsize=7, color="white")

    ax.set_xlim(0, max(1.0, max_t - min_t))
    ax.set_ylim(-1, len(lane_order))
    ax.set_yticks([lane_to_y[l] for l in lane_order])
    ax.set_yticklabels(lane_order)
    ax.set_xlabel("Time (seconds since scheduler start)")
    ax.set_ylabel("Worker lane")
    ax.set_title("Scheduler Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    legend_handles = []
    for job_type, color in color_map.items():
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5, label=job_type))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def timeline_from_manifest(
    manifest_path: Union[str, Path],
    output_png: Union[str, Path],
) -> None:
    """Render a scheduler timeline PNG from an existing manifest JSON file.

    Useful for generating (or regenerating) the timeline figure after a
    crashed run, using the incremental manifest that was written during
    execution.

    Parameters
    ----------
    manifest_path : str or Path
        Path to the manifest JSON file written by ``run_job_graph``.
    output_png : str or Path
        Destination path for the output PNG file.  Parent directories are
        created automatically.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    ImportError
        If matplotlib is not installed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    _write_timeline_figure(manifest, Path(output_png))


def run_segmentation_jobs(
    jobs: List[SegmentationJob],
    run_job: Callable[[SegmentationJob], bool],
    parallel: bool = False,
    max_workers: Optional[int] = None,
    cpu_workers: int = 1,
    continue_on_error: bool = True,
    max_retries: int = 0,
    manifest_path: Optional[Path] = None,
    timeline_figure_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute segmentation jobs sequentially or in parallel.

    Parameters
    ----------
    jobs : list of SegmentationJob
        Jobs to execute.
    run_job : callable
        Function with signature ``run_job(job) -> bool``.
    parallel : bool
        Whether to execute jobs in parallel.
    max_workers : int, optional
        Max workers when ``parallel=True``. Defaults to 1.
    cpu_workers : int
        CPU worker pool size. Included for API compatibility with the
        general job graph runner.
    continue_on_error : bool
        If False, raises RuntimeError when any job fails.
    max_retries : int
        Maximum retry attempts per job after first failure.
    manifest_path : Path, optional
        Optional JSON run manifest output path.
    timeline_figure_path : Path, optional
        Optional timeline figure output path.

    Returns
    -------
    dict
        Summary with keys: ``successful``, ``failed``, ``skipped``,
        and ``job_results`` mapping job IDs to bool outcomes.
    """
    handlers: Dict[JobType, Callable[[SegmentationJob], bool]] = {
        "segment": run_job,
    }
    return run_job_graph(
        jobs=jobs,
        run_handlers=handlers,
        parallel=parallel,
        gpu_workers=max_workers or 1,
        cpu_workers=cpu_workers,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        manifest_path=manifest_path,
        timeline_figure_path=timeline_figure_path,
    )

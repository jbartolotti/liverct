import json
import time
from pathlib import Path

from liverct.scheduler import SegmentationJob, run_job_graph


def _sleep_handler(job: SegmentationJob) -> bool:
    # Keep each job long enough to force concurrent scheduling decisions.
    time.sleep(0.15)
    return True


def _has_same_lane_overlap(segments):
    """Return True if any intervals overlap on the same (resource_type, lane)."""
    by_lane = {}
    for seg in segments:
        by_lane.setdefault((seg["resource_type"], seg["lane"]), []).append(seg)

    for lane_segments in by_lane.values():
        lane_segments = sorted(lane_segments, key=lambda s: s["start_time"])
        for i in range(1, len(lane_segments)):
            prev = lane_segments[i - 1]
            cur = lane_segments[i]
            if cur["start_time"] < prev["end_time"]:
                return True
    return False


def test_parallel_scheduler_does_not_overlap_same_cpu_lane(tmp_path: Path):
    jobs = [
        SegmentationJob(
            id=f"cpu-job-{i}",
            subject_label=f"sub-{i}",
            session_label=None,
            tasks="stats",
            job_type="stats",
            resource_type="cpu",
        )
        for i in range(6)
    ]

    manifest_path = tmp_path / "manifest.json"
    summary = run_job_graph(
        jobs=jobs,
        run_handlers={"stats": _sleep_handler},
        parallel=True,
        gpu_workers=1,
        cpu_workers=2,
        continue_on_error=True,
        max_retries=0,
        manifest_path=manifest_path,
        timeline_figure_path=None,
    )

    assert summary["failed"] == 0
    assert summary["successful"] == len(jobs)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = []
    for job in manifest["jobs"]:
        for rec in job.get("attempt_records", []):
            segments.append(
                {
                    "resource_type": rec["resource_type"],
                    "lane": int(rec["lane"]),
                    "start_time": float(rec["start_time"]),
                    "end_time": float(rec["end_time"]),
                }
            )

    assert not _has_same_lane_overlap(segments)

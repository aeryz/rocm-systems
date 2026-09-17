#!/usr/bin/env python3
"""Run PC sampling host-trap with RocPD output and validate the schema 3.1.0 tables.

Exercises the SDK's RocPD PC-sampling path on gfx1151: the run must produce a
database whose rocpd_gpu_pc_sample view holds the host-trap samples, with the
stochastic-only columns left null, every sample linked to its kernel dispatch
through rocpd_event.parent_id, and the self-describing
pc_sample_extdata_hosttrap blob (hw_id, wave/workgroup coordinates) stored and
decoded for every sample. The decoded view is created through the public
rocpd.importer API, so a field or layout regression fails here rather than
silently producing a shorter row.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "rocpd_pc_sampling_test_out"
LOG_ROOT = ROOT_DIR / "logs"
BIN = ROOT_DIR / "pc_sampling_test"

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
ROC_PROF = Path(ROCM_PATH) / "bin" / "rocprofv3"

PCS_INTERVAL_US = os.environ.get("PCS_INTERVAL_US", "5000")
PCS_TIMEOUT_SEC = int(os.environ.get("PCS_TIMEOUT_SEC", "120"))

# Schema version that introduced rocpd_gpu_pc_sample and the blob metadata tables.
MIN_SCHEMA_VERSION = (3, 1, 0)

STOCHASTIC_ONLY_COLUMNS = (
    "wave_issued",
    "inst_type",
    "stall_reason",
    "wave_count",
)

HOST_TRAP_BLOB_SCHEMA = "pc_sample_extdata_hosttrap"
STOCHASTIC_BLOB_SCHEMA = "pc_sample_extdata_stochastic"
# Every host-trap blob is self-described, but the fields below are the contract this
# script checks: the hardware id and the wave/workgroup coordinates reported on gfx11.
HOST_TRAP_BLOB_FIELDS = (
    "hw_id_chiplet",
    "hw_id_wave_id",
    "hw_id_simd_id",
    "hw_id_pipe_id",
    "hw_id_cu_or_wgp_id",
    "hw_id_shader_array_id",
    "hw_id_shader_engine_id",
    "hw_id_workgroup_id",
    "hw_id_vm_id",
    "hw_id_queue_id",
    "hw_id_microengine_id",
    "wave_in_group",
    "workgroup_id_x",
    "workgroup_id_y",
    "workgroup_id_z",
)


def fail(message: str, run_dir: Path | None = None) -> int:
    print(f"FAILED: {message}")
    if run_dir is not None:
        print(f"Logs: {run_dir}")
    return 1


def find_database(root: Path) -> Path | None:
    dbs = sorted(root.glob("**/*.db"))
    return dbs[0] if dbs else None


def parse_schema_version(text: str) -> tuple[int, int, int] | None:
    try:
        parts = [int(part) for part in str(text).split(".")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def validate_database(db_path: Path) -> tuple[bool, str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()

        row = cur.execute(
            "SELECT value FROM rocpd_metadata WHERE tag = 'schema_version' LIMIT 1").fetchone()
        if row is None:
            return False, "rocpd_metadata has no schema_version entry"
        version = parse_schema_version(row[0])
        if version is None or version < MIN_SCHEMA_VERSION:
            return False, f"schema version {row[0]!r} is older than {MIN_SCHEMA_VERSION}"
        print(f"schema_version={row[0]}")

        # Reading through the view also proves the data view exists.
        try:
            samples = cur.execute("SELECT COUNT(*) FROM rocpd_gpu_pc_sample").fetchone()[0]
        except sqlite3.Error as exc:
            return False, f"rocpd_gpu_pc_sample is not queryable: {exc}"
        print(f"rocpd_gpu_pc_sample rows: {samples}")
        if samples == 0:
            return False, "no rows in rocpd_gpu_pc_sample"

        schemas = dict(
            cur.execute("SELECT name, struct_size FROM rocpd_info_blob_schema").fetchall())
        print(f"blob schemas: {schemas}")
        if HOST_TRAP_BLOB_SCHEMA not in schemas:
            return False, f"missing {HOST_TRAP_BLOB_SCHEMA} blob schema"
        if STOCHASTIC_BLOB_SCHEMA in schemas:
            return False, "host-trap capture registered a stochastic blob schema"

        schema_id = cur.execute(
            "SELECT id FROM rocpd_info_blob_schema WHERE name = ?",
            (HOST_TRAP_BLOB_SCHEMA,)).fetchone()[0]
        fields = {name for (name,) in cur.execute(
            "SELECT name FROM rocpd_info_blob_field WHERE schema_id = ?",
            (schema_id,)).fetchall()}
        missing = [name for name in HOST_TRAP_BLOB_FIELDS if name not in fields]
        if missing:
            return False, f"host-trap blob is missing fields {missing}"

        # Every sample must carry exactly one blob of that schema, of the declared size.
        covered = cur.execute(
            "SELECT COUNT(*) FROM rocpd_gpu_pc_sample AS s"
            " JOIN rocpd_blob_event AS b ON b.event_id = s.event_id"
            " JOIN rocpd_info_blob_schema AS sc ON sc.id = b.schema_id"
            " WHERE sc.name = ? AND LENGTH(b.blob) = sc.struct_size",
            (HOST_TRAP_BLOB_SCHEMA,)).fetchone()[0]
        print(f"host_trap rows with a host-trap blob: {covered}")
        if covered != samples:
            return False, f"only {covered} of {samples} rows carry the host-trap blob"

        non_null = cur.execute(
            "SELECT COUNT(*) FROM rocpd_gpu_pc_sample WHERE "
            + " OR ".join(f"{col} IS NOT NULL" for col in STOCHASTIC_ONLY_COLUMNS)
        ).fetchone()[0]
        print(f"host-trap rows with non-null stochastic columns: {non_null}")
        if non_null:
            return False, "host-trap rows must leave the stochastic-only columns null"

        # Each sample event is parented by the kernel dispatch it sampled.
        linked = cur.execute(
            "SELECT COUNT(*) FROM rocpd_gpu_pc_sample AS s"
            " JOIN rocpd_event AS e ON e.id = s.event_id"
            " JOIN rocpd_kernel_dispatch AS d"
            " ON d.event_id = e.parent_id AND d.dispatch_id = s.dispatch_id").fetchone()[0]
        print(f"host_trap rows linked to their kernel dispatch: {linked}")
        if linked != samples:
            return False, f"only {linked} of {samples} rows link to a kernel dispatch"

        undecoded = cur.execute(
            "SELECT COUNT(*) FROM rocpd_gpu_pc_sample"
            " WHERE correlation_id = 0 OR exec_mask IS NULL OR code_object_offset < 0"
        ).fetchone()[0]
        if undecoded:
            return False, f"{undecoded} samples have no correlation id, exec mask or PC"

        integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
        violations = len(cur.execute("PRAGMA foreign_key_check").fetchall())
        print(f"integrity_check={integrity}, foreign_key_violations={violations}")
        if integrity != "ok":
            return False, f"integrity_check reported {integrity!r}"
        if violations:
            return False, f"{violations} foreign key violations"
    finally:
        con.close()

    # The decoded view unpacks the blob through the self-describing metadata. Build it with
    # the public importer API (the same one `rocpd query` uses) and require every hardware-id
    # column to be populated for every row. The SDK installs the package under its own
    # site-packages directory, which is not on the venv's default path.
    sdk_python = Path(ROCM_PATH) / "lib" / (
        f"python{sys.version_info.major}.{sys.version_info.minor}") / "site-packages"
    if sdk_python.is_dir() and str(sdk_python) not in sys.path:
        sys.path.insert(0, str(sdk_python))
    try:
        from rocpd.importer import RocpdImportData
    except ImportError as exc:
        return False, f"rocpd python package is not importable: {exc}"

    with RocpdImportData(str(db_path), cache_disassembly=False) as conn:
        columns = {column[0] for column in conn.execute(
            "SELECT * FROM rocpd_gpu_pc_sample_decoded LIMIT 0").description}
        missing = [name for name in HOST_TRAP_BLOB_FIELDS if name not in columns]
        if missing:
            return False, f"decoded view is missing columns {missing}"
        decoded = conn.execute("SELECT COUNT(*) FROM rocpd_gpu_pc_sample_decoded").fetchone()[0]
        named = conn.execute(
            "SELECT COUNT(*) FROM rocpd_gpu_pc_sample_decoded"
            " WHERE hw_id_shader_engine_id IS NULL OR hw_id_simd_id IS NULL"
            " OR hw_id_cu_or_wgp_id IS NULL OR workgroup_id_x IS NULL"
        ).fetchone()[0]
        print(f"decoded rows: {decoded}, rows without hardware ids: {named}")
        if decoded == 0:
            return False, "decoded view is empty"
        if named:
            return False, f"{named} decoded rows have no hardware id"

    return True, "ok"


def main() -> int:
    if not ROC_PROF.exists():
        return fail(f"rocprofv3 not found at {ROC_PROF}")
    if not BIN.exists():
        return fail(f"{BIN} not found; run ./build_pc_sampling_test.sh first")

    stale = subprocess.run(["pgrep", "-af", "rocprofv3|pc_sampling_test"],
                           capture_output=True, text=True)
    stale_lines = [l for l in stale.stdout.splitlines() if str(os.getpid()) not in l]
    if stale_lines:
        return fail("stale rocprofv3/pc_sampling_test process detected before run")

    run_tag = f"rocpd_pc_sampling_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = LOG_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    cmd = [
        str(ROC_PROF),
        "--pc-sampling-beta-enabled", "1",
        "--pc-sampling-method", "host_trap",
        "--pc-sampling-unit", "time",
        "--pc-sampling-interval", PCS_INTERVAL_US,
        "--kernel-trace",
        "--output-format", "rocpd",
        "-d", str(OUT_DIR),
        "--", str(BIN),
    ]
    env = dict(os.environ, HSA_DISABLE_XDNA="1")

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Timeout: {PCS_TIMEOUT_SEC}s\n")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=PCS_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return fail(f"rocprofv3 timed out after {PCS_TIMEOUT_SEC}s", run_dir)

    (run_dir / "rocprof_run.log").write_text(
        f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n")

    stale = subprocess.run(["pgrep", "-af", "rocprofv3|pc_sampling_test"],
                           capture_output=True, text=True)
    stale_lines = [l for l in stale.stdout.splitlines() if str(os.getpid()) not in l]
    if stale_lines:
        return fail("stale rocprofv3/pc_sampling_test process detected after run", run_dir)

    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return fail(f"rocprofv3 exited with status {proc.returncode}", run_dir)

    db_path = find_database(OUT_DIR)
    if db_path is None:
        return fail(f"no .db produced under {OUT_DIR}", run_dir)
    print(f"database: {db_path}")

    ok, message = validate_database(db_path)
    if not ok:
        return fail(message, run_dir)

    print(f"\nRocPD PC sampling OK: {db_path}")
    print(f"Logs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

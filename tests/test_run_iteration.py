"""Tests for 12_run_iteration.py — HITL iteration orchestrator."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Load module via importlib (numbered filename)
spec = importlib.util.spec_from_file_location(
    "run_iteration", SCRIPTS / "12_run_iteration.py"
)
run_iter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_iter_mod)

run_iteration = run_iter_mod.run_iteration
_run = run_iter_mod._run


# ---------------------------------------------------------------------------
# Tests: _run helper
# ---------------------------------------------------------------------------


class TestRunHelper:
    def test_run_success(self, tmp_path):
        """_run should not raise when the subprocess exits 0."""
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0)
            _run([sys.executable, "-c", "pass"], "test step")
            mock_sub.assert_called_once()

    def test_run_failure_raises(self):
        """_run should raise RuntimeError when subprocess exits non-zero."""
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1)
            with pytest.raises(RuntimeError, match="failed with exit code 1"):
                _run([sys.executable, "-c", "pass"], "failing step")

    def test_run_passes_correct_command(self):
        """_run should forward the exact command to subprocess.run."""
        cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0)
            _run(cmd, "check cmd")
            called_cmd = mock_sub.call_args[0][0]
            assert called_cmd == cmd


# ---------------------------------------------------------------------------
# Tests: run_iteration orchestration
# ---------------------------------------------------------------------------


class TestRunIteration:
    @pytest.fixture
    def mock_env(self, tmp_path):
        """Set up a minimal environment for run_iteration tests."""
        model = tmp_path / "best.pt"
        model.touch()
        raster = tmp_path / "raster.tif"
        raster.touch()
        config = tmp_path / "pipeline.yaml"
        config.write_text("run_id: test\n")
        output_dir = tmp_path / "output"

        # Pre-create tiles_meta.json that _run would normally produce
        tiles_dir = output_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tiles_meta = {
            "tile_size": 256,
            "stride": 64,
            "inputs": [{"id": "test", "raster": str(raster)}],
        }
        (tiles_dir / "tiles_meta.json").write_text(json.dumps(tiles_meta))

        # Pre-create predictions CSV
        preds_dir = output_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        (preds_dir / "predictions_tiles.csv").write_text(
            "tile,tile_r,tile_c,x1,y1,x2,y2,conf,class_id\n"
        )

        # Pre-create a dummy gpkg in gpkg dir
        gpkg_dir = output_dir / "gpkg"
        gpkg_dir.mkdir(parents=True, exist_ok=True)
        (gpkg_dir / "detections_merged_test.gpkg").touch()

        return {
            "model": model,
            "raster": raster,
            "config": config,
            "output_dir": output_dir,
        }

    def test_run_iteration_calls_four_scripts(self, mock_env):
        """run_iteration must invoke scripts 05, 06, 07, and 09 in order."""
        called_scripts: list[str] = []

        def fake_run(cmd, step):
            # Extract the script filename from the command
            for arg in cmd:
                arg_str = str(arg)
                if arg_str.endswith(".py"):
                    called_scripts.append(Path(arg_str).name)
                    break

        with patch.object(run_iter_mod, "_run", side_effect=fake_run):
            run_iteration(
                model=mock_env["model"],
                raster=mock_env["raster"],
                config=mock_env["config"],
                output_dir=mock_env["output_dir"],
                input_id="test",
            )

        assert len(called_scripts) == 4, f"Expected 4 script calls, got: {called_scripts}"
        assert called_scripts[0] == "05_make_inference_tiles.py"
        assert called_scripts[1] == "06_infer.py"
        assert called_scripts[2] == "07_merge_nms.py"
        assert called_scripts[3] == "09_export_predictions_gpkg.py"

    def test_run_iteration_creates_dataset_meta(self, mock_env):
        """run_iteration must write a dataset_meta_iter.json for 07 and 09."""

        def fake_run(cmd, step):
            pass

        with patch.object(run_iter_mod, "_run", side_effect=fake_run):
            run_iteration(
                model=mock_env["model"],
                raster=mock_env["raster"],
                config=mock_env["config"],
                output_dir=mock_env["output_dir"],
                input_id="test",
            )

        meta_path = mock_env["output_dir"] / "dataset_meta_iter.json"
        assert meta_path.exists(), "dataset_meta_iter.json should be written"
        meta = json.loads(meta_path.read_text())
        assert "tile_size" in meta
        assert "stride" in meta

    def test_run_iteration_returns_gpkg_path(self, mock_env):
        """run_iteration must return a Path pointing to the GeoPackage."""

        def fake_run(cmd, step):
            pass

        with patch.object(run_iter_mod, "_run", side_effect=fake_run):
            result = run_iteration(
                model=mock_env["model"],
                raster=mock_env["raster"],
                config=mock_env["config"],
                output_dir=mock_env["output_dir"],
                input_id="test",
            )

        assert isinstance(result, Path)

    def test_run_iteration_stops_on_step_failure(self, mock_env):
        """If a step fails, run_iteration should propagate the RuntimeError."""
        call_count = 0

        def fail_on_second(cmd, step):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Step failed")

        with patch.object(run_iter_mod, "_run", side_effect=fail_on_second):
            with pytest.raises(RuntimeError, match="Step failed"):
                run_iteration(
                    model=mock_env["model"],
                    raster=mock_env["raster"],
                    config=mock_env["config"],
                    output_dir=mock_env["output_dir"],
                    input_id="test",
                )

        assert call_count == 2, "Should stop after the failing step"

    def test_run_iteration_passes_model_to_infer(self, mock_env):
        """Script 06 must receive the model path as an argument."""
        script_06_cmd: list[str] | None = None

        def capture_06(cmd, step):
            nonlocal script_06_cmd
            if "06_infer.py" in str(cmd):
                script_06_cmd = [str(c) for c in cmd]

        with patch.object(run_iter_mod, "_run", side_effect=capture_06):
            run_iteration(
                model=mock_env["model"],
                raster=mock_env["raster"],
                config=mock_env["config"],
                output_dir=mock_env["output_dir"],
                input_id="test",
            )

        assert script_06_cmd is not None
        assert "--model" in script_06_cmd
        model_idx = script_06_cmd.index("--model")
        assert script_06_cmd[model_idx + 1] == str(mock_env["model"])

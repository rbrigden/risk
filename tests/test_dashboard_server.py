import http.client
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVE_PATH = REPO_ROOT / "web" / "dashboard" / "serve"


def load_serve_module():
    loader = importlib.machinery.SourceFileLoader("dashboard_serve", str(SERVE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


serve = load_serve_module()


class DashboardServerTests(unittest.TestCase):
    def _start_server(self, runs_dir: Path):
        server = serve.create_server(
            host="127.0.0.1",
            port=0,
            runs_dir=runs_dir,
            quiet=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def _request(self, server, path: str):
        host, port = server.server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, dict(resp.getheaders()), body
        finally:
            conn.close()

    def test_empty_and_nonexistent_runs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "does_not_exist"
            server = self._start_server(missing_dir)

            status, headers, body = self._request(server, "/api/runs")
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            self.assertEqual(json.loads(body), {"runs": []})

            # Create empty directory and verify again
            missing_dir.mkdir()
            status, _, body = self._request(server, "/api/runs")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"runs": []})

    def test_root_serves_dashboard_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = self._start_server(Path(tmp))
            status, headers, body = self._request(server, "/")
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers.get("Content-Type", ""))
            text = body.decode("utf-8")
            self.assertIn("Riskbot", text)
            self.assertIn("Elo Progression", text)

    def test_populated_runs_sorted_newest_first_and_partial_jsonl_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)

            older = runs_dir / "run_older"
            older.mkdir()
            (older / "metrics.jsonl").write_text(
                json.dumps({"update": 1, "total_env_steps": 2000, "learner_elo": 1012.5}) + "\n",
                encoding="utf-8",
            )
            os.utime(older / "metrics.jsonl", (1_700_000_000, 1_700_000_000))
            os.utime(older, (1_700_000_000, 1_700_000_000))

            newer = runs_dir / "run_newer"
            newer.mkdir()
            metric_1 = {
                "update": 1,
                "total_env_steps": 2200,
                "elapsed_secs": 4.2,
                "steps_per_sec": 523.8,
                "learning_rate": 0.0003,
                "policy_loss": -0.01,
                "value_loss": 0.02,
                "entropy": 2.9,
                "approx_kl": 0.004,
                "clip_frac": 0.05,
                "episodes_finished": 3,
                "win_frac": 0.67,
                "avg_turns": 48.0,
                "learner_elo": 1025.0,
                "pool_size": 2,
                "eval_random_win_rate": None,
                "eval_heuristic_win_rate": None,
                "eval_league_win_rate": None,
                "eval_random_avg_turns": None,
                "eval_heuristic_avg_turns": None,
            }
            metric_2 = {
                "update": 10,
                "total_env_steps": 22286,
                "elapsed_secs": 41.2,
                "steps_per_sec": 585.4,
                "learning_rate": 0.000273,
                "policy_loss": -0.0080,
                "value_loss": 0.0002,
                "entropy": 2.530,
                "approx_kl": 0.0051,
                "clip_frac": 0.06,
                "episodes_finished": 4,
                "win_frac": 1.0,
                "avg_turns": 42.5,
                "learner_elo": 1142.3,
                "pool_size": 3,
                "eval_random_win_rate": 0.917,
                "eval_heuristic_win_rate": 0.417,
                "eval_league_win_rate": 0.625,
                "eval_random_avg_turns": 26.0,
                "eval_heuristic_avg_turns": 20.0,
            }
            # Include a truncated trailing JSON line simulating a live writer mid-flush
            (newer / "metrics.jsonl").write_text(
                json.dumps(metric_1)
                + "\n"
                + json.dumps(metric_2)
                + '\n{"update": 11, "total_env_steps": 24',
                encoding="utf-8",
            )
            league_payload = {
                "run_name": "run_newer",
                "players": 2,
                "max_turns": 200,
                "backend": "WGPU/Vulkan GPU",
                "pfsp_mode": "hard",
                "pfsp_p": 2.0,
                "current_update": 10,
                "total_updates": 200,
                "learner_elo": 1142.3,
                "anchors": {"random_elo": 800.0, "heuristic_elo": 1200.0},
                "members": [
                    {
                        "id": "random",
                        "kind": "random",
                        "created_update": 0,
                        "elo": 800.0,
                        "games": 32,
                        "wins": 3,
                        "losses": 29,
                        "draws": 0,
                        "learner_win_rate": 0.906,
                        "pfsp_weight": 0.015,
                        "active": True,
                    }
                ],
                "elo_history": [
                    {
                        "update": 1,
                        "steps": 2200,
                        "learner_elo": 1025.0,
                        "best_snapshot_elo": 1000.0,
                        "mean_snapshot_elo": 1000.0,
                    }
                ],
            }
            (newer / "league.json").write_text(json.dumps(league_payload), encoding="utf-8")
            os.utime(newer / "metrics.jsonl", (1_700_005_000, 1_700_005_000))
            os.utime(newer / "league.json", (1_700_005_000, 1_700_005_000))
            os.utime(newer, (1_700_005_000, 1_700_005_000))

            server = self._start_server(runs_dir)

            # Verify /api/runs lists newest-first with expected metadata
            status, headers, body = self._request(server, "/api/runs")
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            runs = json.loads(body)["runs"]
            self.assertEqual([r["name"] for r in runs], ["run_newer", "run_older"])

            first_run, second_run = runs
            self.assertTrue(first_run["has_metrics"])
            self.assertTrue(first_run["has_league"])
            self.assertEqual(first_run["current_update"], 10)
            self.assertEqual(first_run["total_updates"], 200)
            self.assertAlmostEqual(first_run["learner_elo"], 1142.3)

            self.assertTrue(second_run["has_metrics"])
            self.assertFalse(second_run["has_league"])
            self.assertEqual(second_run["current_update"], 1)
            self.assertIsNone(second_run["total_updates"])
            self.assertAlmostEqual(second_run["learner_elo"], 1012.5)

            # Verify /api/runs/run_newer/state skips incomplete trailing line
            status, headers, body = self._request(server, "/api/runs/run_newer/state")
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            state_data = json.loads(body)
            self.assertEqual(state_data["run"], "run_newer")
            self.assertEqual(len(state_data["metrics"]), 2)
            self.assertEqual([m["update"] for m in state_data["metrics"]], [1, 10])
            self.assertEqual(state_data["league"], league_payload)

            # Verify metrics-only run has league == None
            status, _, body = self._request(server, "/api/runs/run_older/state")
            self.assertEqual(status, 200)
            older_state = json.loads(body)
            self.assertEqual(len(older_state["metrics"]), 1)
            self.assertIsNone(older_state["league"])

            # Missing run returns 404
            status, _, _ = self._request(server, "/api/runs/nonexistent/state")
            self.assertEqual(status, 404)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            secret = Path(tmp) / "secret"
            secret.mkdir()
            (secret / "league.json").write_text('{"secret": true}', encoding="utf-8")

            server = self._start_server(runs_dir)

            bad_paths = [
                "/api/runs/../secret/state",
                "/api/runs/..%2Fsecret/state",
                "/api/runs/%2e%2e/state",
                "/api/runs/..%5Csecret/state",
                "/api/runs/foo%2Fbar/state",
                "/api/runs/foo/bar/state",
                "/api/runs/../etc/passwd/state",
            ]
            for path in bad_paths:
                with self.subTest(path=path):
                    status, headers, _ = self._request(server, path)
                    self.assertIn(status, (400, 404))
                    self.assertEqual(headers.get("Cache-Control"), "no-store")

            for bad_name in ("..", ".", "../etc", "a/b", "a\\b", "foo..bar", ""):
                with self.subTest(bad_name=bad_name):
                    with self.assertRaises(ValueError):
                        serve.get_run_state(runs_dir, bad_name)


if __name__ == "__main__":
    unittest.main()

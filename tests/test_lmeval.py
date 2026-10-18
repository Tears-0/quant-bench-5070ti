"""lm-eval bridge: command construction and result collection, no lm-eval install needed."""

from __future__ import annotations

import json

from quantbench.lmeval import build_command, collect_results, model_args_hf, task_groups


def test_tasks_grouped_by_fewshot():
    tasks = [
        {"task": "arc_challenge", "num_fewshot": 0},
        {"task": "gsm8k", "num_fewshot": 8},
        {"task": "hellaswag", "num_fewshot": 0},
        {"task": "mmlu_abstract_algebra", "num_fewshot": 5},
    ]
    assert task_groups(tasks) == {
        0: ["arc_challenge", "hellaswag"],
        5: ["mmlu_abstract_algebra"],
        8: ["gsm8k"],
    }


def test_command_shape(tmp_path):
    cmd = build_command("gguf", "base_url=http://127.0.0.1:9", ["arc_challenge"], 0, tmp_path, 1337)
    assert cmd[1:3] == ["-m", "lm_eval"]
    assert "--batch_size" not in cmd  # the gguf backend is sequential
    assert cmd[cmd.index("--num_fewshot") + 1] == "0"
    hf = build_command("hf", model_args_hf("m", "nf4"), ["x"], 5, tmp_path, 1, limit=10)
    assert "load_in_4bit=True" in hf[hf.index("--model_args") + 1]
    assert hf[hf.index("--limit") + 1] == "10"


def test_results_are_merged_and_mmlu_subset_averaged(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "results_1.json").write_text(
        json.dumps(
            {"results": {"arc_challenge": {"acc_norm,none": 0.5, "acc_norm_stderr,none": 0.01}}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "m" / "results_2.json").write_text(
        json.dumps(
            {
                "results": {
                    "mmlu_abstract_algebra": {"acc,none": 0.3},
                    "mmlu_world_religions": {"acc,none": 0.7},
                }
            }
        ),
        encoding="utf-8",
    )
    scores, files, _ = collect_results(tmp_path)
    assert len(files) == 2
    assert scores["arc_challenge"] == 0.5 and scores["arc_challenge_stderr"] == 0.01
    assert abs(scores["mmlu_subset"] - 0.5) < 1e-12


def test_gguf_artifacts_load_through_transformers(tmp_path):
    gguf = tmp_path / "qwen3-4b-q4_k_m.gguf"
    # no one-time dequantization yet: Transformers dequantizes the GGUF itself (slow path)
    args = model_args_hf(str(gguf), "gguf")
    assert "gguf_file=qwen3-4b-q4_k_m.gguf" in args and "dtype=float16" in args
    # once `quantbench models dequantize` has run, the fp16 checkpoint is preferred
    (tmp_path / "qwen3-4b-q4_k_m-dequant-fp16").mkdir()
    args = model_args_hf(str(gguf), "gguf")
    assert "qwen3-4b-q4_k_m-dequant-fp16" in args and "gguf_file" not in args


def test_choose_prefers_smaller_batch_within_tolerance_and_never_over_budget():
    from quantbench.lmeval_tune import Probe, choose

    probes = [
        Probe(1, 9000, 1.0, 1000.0, True),
        Probe(2, 9500, 1.0, 1900.0, True),
        Probe(4, 10000, 1.0, 2000.0, True),
        Probe(8, 11000, 1.0, 2050.0, True),  # +2.5% over b=4: within tolerance
        Probe(16, 13000, 1.0, 3000.0, False),  # fastest, but over budget
    ]
    assert choose(probes) == 4
    assert choose([Probe(8, 20000, 1.0, 5000.0, False, spilled=True)]) == 1  # nothing valid


def test_tuned_batch_sizes_roundtrip(tmp_path):
    from quantbench.lmeval_tune import tuned_batch_sizes, tuning_path

    path = tuning_path(tmp_path, "qwen3-4b")
    path.parent.mkdir(parents=True)
    path.write_text("tasks:\n  arc_challenge: {batch_size: 64}\n  gsm8k: {batch_size: 16}\n")
    assert tuned_batch_sizes(path) == {"arc_challenge": 64, "gsm8k": 16}
    assert tuned_batch_sizes(tmp_path / "missing.yaml") == {}


def test_run_resumes_completed_tasks_without_spawning(tmp_path):
    from quantbench.lmeval import run

    done = tmp_path / "arc_challenge" / "model"
    done.mkdir(parents=True)
    (done / "results_1.json").write_text(
        json.dumps({"results": {"arc_challenge": {"acc_norm,none": 0.5}}}), encoding="utf-8"
    )
    result = run(
        "hf",
        "pretrained=x",
        [{"task": "arc_challenge", "num_fewshot": 0}],
        tmp_path,
        seed=1,
        timeout_s=10,
        batch_sizes={"arc_challenge": 32},
    )
    assert result.resumed == ["arc_challenge"]
    assert result.commands == []  # nothing was launched
    assert result.scores["arc_challenge"] == 0.5
    assert result.batch_sizes == {"arc_challenge": 32}


def test_job_model_kwargs_parsing():
    from quantbench.lmeval_job import _model_kwargs

    assert _model_kwargs("pretrained=models/x,dtype=float16,attn_implementation=sdpa") == {
        "pretrained": "models/x",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    }

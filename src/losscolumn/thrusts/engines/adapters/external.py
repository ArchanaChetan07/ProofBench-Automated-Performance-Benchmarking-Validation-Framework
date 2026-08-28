"""Adapters for the real engines: vLLM, SGLang and TensorRT-LLM.

These are the paths the funded audit runs. They are written to the same
contract as the simulated engines so the tuning loop, the ledger and the
parity certificate are literally the same code -- which is what makes the
parity claim mean something.

Design decisions worth stating, because each one is a place a comparison could
go quietly wrong:

**One benchmark client for all three.** Each project ships its own benchmark
script, and they differ in how they generate load, whether they count the
prompt in throughput, and how they compute percentiles. Using each project's
own script would compare three measurement methodologies, not three engines.
All three are driven here through their OpenAI-compatible endpoint by one
client, replaying one trace.

**Server startup is part of the trial.** An engine is launched fresh per
configuration, warmed with a fixed number of requests, then measured. Reusing a
warm server across configurations lets earlier trials contaminate later ones
through cache state.

**Failures are recorded, not retried.** A configuration that OOMs is a data
point about that configuration. Silently retrying at a lower memory setting --
which is what a human tuner does -- would put a thumb on the scale for whichever
engine fails more gracefully.

**Nothing is imported at module load.** The audit must be runnable on a laptop
to inspect and validate its own analysis path with no engine installed at all.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from losscolumn.core.parity import SearchSpace
from losscolumn.thrusts.engines.adapters.base import EngineUnavailable, ServeResult
from losscolumn.thrusts.engines.adapters.synthetic import DEFAULTS, SPACES
from losscolumn.thrusts.engines.workloads import Workload


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _installed(module: str, binary: str | None = None) -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec(module) is not None:
        return True, ""
    if binary and shutil.which(binary):
        return True, ""
    return False, f"neither the {module!r} package nor the {binary!r} binary is present"


@dataclass
class ExternalEngine:
    """Drive a real engine through its OpenAI-compatible server."""

    name: str
    model: str = "meta-llama/Llama-2-7b-hf"
    host: str = "127.0.0.1"
    startup_timeout_s: float = 900.0
    warmup_requests: int = 32
    evidence_class: str = "measured"
    extra_args: list[str] = field(default_factory=list)
    _proc: Any = None
    _port: int = 0

    # ---- contract ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        import importlib.metadata as md

        pkg = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}[self.name]
        try:
            version = md.version(pkg)
        except Exception:
            version = None
        return {
            "engine": self.name,
            "package": pkg,
            "version": version,
            "model": self.model,
            "evidence_class": self.evidence_class,
        }

    def available(self) -> tuple[bool, str]:
        pkg = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}[self.name]
        return _installed(pkg, self.name)

    def search_space(self) -> SearchSpace:
        return SearchSpace(name=self.name, dims=SPACES[self.name])

    def defaults(self) -> dict[str, Any]:
        return dict(DEFAULTS[self.name])

    # ---- server lifecycle -------------------------------------------------

    def _launch_argv(self, config: dict[str, Any]) -> list[str]:
        self._port = _free_port()
        base = ["--host", self.host, "--port", str(self._port)]
        if self.name == "vllm":
            argv = ["vllm", "serve", self.model, *base,
                    "--max-num-seqs", str(config["max_num_seqs"]),
                    "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
                    "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
                    "--block-size", str(config["block_size"])]
            if config.get("enable_chunked_prefill"):
                argv.append("--enable-chunked-prefill")
            if config.get("enable_prefix_caching"):
                argv.append("--enable-prefix-caching")
            return argv + self.extra_args
        if self.name == "sglang":
            argv = ["python", "-m", "sglang.launch_server", "--model-path", self.model, *base,
                    "--max-running-requests", str(config["max_running_requests"]),
                    "--chunked-prefill-size", str(config["chunked_prefill_size"]),
                    "--mem-fraction-static", str(config["mem_fraction_static"]),
                    "--schedule-policy", str(config["schedule_policy"])]
            if config.get("disable_radix_cache"):
                argv.append("--disable-radix-cache")
            return argv + self.extra_args
        # TensorRT-LLM needs an engine built for the shape before it can serve;
        # the build is charged to this configuration's wall-clock, which is
        # exactly the asymmetry the parity certificate reports.
        return ["trtllm-serve", self.model, *base,
                "--max_batch_size", str(config["max_batch_size"]),
                "--max_num_tokens", str(config["max_num_tokens"]),
                "--kv_cache_free_gpu_memory_fraction",
                str(config["kv_cache_free_gpu_mem_fraction"])] + self.extra_args

    def start(self, config: dict[str, Any]) -> None:
        ok, why = self.available()
        if not ok:
            raise EngineUnavailable(self.name, why)
        argv = self._launch_argv(config)
        env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"}
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = (self._proc.stdout.read() if self._proc.stdout else "")[-4000:]
                raise RuntimeError(f"{self.name} exited during startup:\n{out}")
            if self._health():
                return
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(f"{self.name} did not become healthy within {self.startup_timeout_s}s")

    def _health(self) -> bool:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"http://{self.host}:{self._port}/health", timeout=2
            ) as r:
                return r.status == 200
        except Exception:
            return False

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    # ---- measurement ------------------------------------------------------

    def serve(
        self, config: dict[str, Any], workload: Workload, *, concurrency: int, seed: int = 0
    ) -> ServeResult:
        try:
            self.start(config)
        except EngineUnavailable as e:
            return ServeResult(*([float("nan")] * 9), error=str(e))
        except Exception as e:
            return ServeResult(*([float("nan")] * 9), error=f"{type(e).__name__}: {e}")
        try:
            return self._replay(workload, concurrency=concurrency)
        finally:
            self.stop()

    def _replay(self, workload: Workload, *, concurrency: int) -> ServeResult:
        """Replay the trace at a fixed concurrency and collect per-request timings."""
        from concurrent.futures import ThreadPoolExecutor

        url = f"http://{self.host}:{self._port}/v1/completions"
        reqs = workload.requests
        for r in reqs[: self.warmup_requests]:
            self._one(url, r)

        ttfts: list[float] = []
        e2es: list[float] = []
        norms: list[float] = []
        failed = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for res in pool.map(lambda r: self._one(url, r), reqs):
                if res is None:
                    failed += 1
                    continue
                ttft, e2e, out_tokens = res
                ttfts.append(ttft)
                e2es.append(e2e)
                norms.append(e2e / max(out_tokens, 1))
        wall = time.perf_counter() - t0

        if not e2es:
            return ServeResult(*([float("nan")] * 9), n_failed=failed,
                               error="every request failed")
        out_tokens_total = sum(r.output_tokens for r in reqs)
        meets = np.ones(len(e2es), dtype=bool)
        if workload.slo_ttft_ms:
            meets &= np.asarray(ttfts) <= workload.slo_ttft_ms
        if workload.slo_tpot_ms:
            meets &= np.asarray(norms) <= workload.slo_tpot_ms
        return ServeResult(
            ttft_ms_p50=float(np.percentile(ttfts, 50)),
            ttft_ms_p99=float(np.percentile(ttfts, 99)),
            tpot_ms_p50=float(np.percentile(norms, 50)),
            tpot_ms_p99=float(np.percentile(norms, 99)),
            e2e_ms_p50=float(np.percentile(e2es, 50)),
            e2e_ms_p99=float(np.percentile(e2es, 99)),
            output_throughput=out_tokens_total / wall,
            total_throughput=(out_tokens_total + workload.total_prompt_tokens) / wall,
            request_throughput=len(e2es) / wall,
            norm_latency_ms_p50=float(np.percentile(norms, 50)),
            norm_latency_ms_p99=float(np.percentile(norms, 99)),
            goodput=len(e2es) / wall * float(meets.mean()),
            n_ok=len(e2es),
            n_failed=failed,
            meta={"wall_s": wall, "concurrency": concurrency, "port": self._port},
        )

    def _one(self, url: str, req: Any) -> tuple[float, float, int] | None:
        """One streaming completion; returns (ttft_ms, e2e_ms, output_tokens)."""
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                # A deterministic synthetic prompt of the right length. Real
                # text would be better and is what the funded run uses; the
                # length is what drives the systems behaviour under test.
                "prompt": "word " * max(req.prompt_tokens // 2, 1),
                "max_tokens": req.output_tokens,
                "min_tokens": req.output_tokens,   # fix the length so timing is comparable
                "temperature": 0.0,
                "stream": True,
                "ignore_eos": True,
            }
        ).encode()
        rq = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        ttft = float("nan")
        tokens = 0
        try:
            with urllib.request.urlopen(rq, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    if line.endswith("[DONE]"):
                        break
                    if tokens == 0:
                        ttft = (time.perf_counter() - start) * 1e3
                    tokens += 1
        except Exception:
            return None
        return ttft, (time.perf_counter() - start) * 1e3, max(tokens, 1)


def external_engines(
    model: str, names: tuple[str, ...] = ("vllm", "sglang", "trtllm")
) -> tuple[dict[str, ExternalEngine], dict[str, str]]:
    """Build adapters for whichever engines are installed.

    Returns ``(available, unavailable_reasons)``. The second half is not
    optional bookkeeping: an audit that silently drops an unavailable engine
    reports a comparison over a different set of systems than the one it
    claims, so the caller receives the reasons and the artifact names them.
    """
    ok_engines: dict[str, ExternalEngine] = {}
    missing: dict[str, str] = {}
    for n in names:
        e = ExternalEngine(name=n, model=model)
        ok, why = e.available()
        if ok:
            ok_engines[n] = e
        else:
            missing[n] = why
    return ok_engines, missing


def availability(names: tuple[str, ...] = ("vllm", "sglang", "trtllm")) -> dict[str, str]:
    """Which engines this machine can actually run, and why not otherwise."""
    out: dict[str, str] = {}
    for n in names:
        ok, why = ExternalEngine(name=n).available()
        out[n] = "available" if ok else why
    return out

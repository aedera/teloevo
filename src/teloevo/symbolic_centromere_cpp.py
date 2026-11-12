"""Compiled C++ backend for the symbolic Dong-style centromere simulator.

This module keeps the Python simulator unchanged and provides a second backend
with the same high-level state/checkpoint interface.

The C++ executable is compiled on first use and cached in the system temporary
directory. It implements the full per-generation loop and mutation operations
in compiled code.

Important
---------
The Python and C++ backends implement the same probability model and event
ordering, but they use different random-number generators. Therefore the same
seed does not produce an identical trajectory across backends. Backends should
be compared statistically and by runtime, not by exact event-for-event identity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np

from .symbolic_centromere import (
    DEFAULT_MONOMER,
    CentromereState,
    CheckpointState,
    EventRecord,
    SequenceLookup,
    SimulationParameters,
    SimulationResult,
    _prepare_snp_model,
)


def _cpp_source_path() -> Path:
    return Path(__file__).resolve().parent / "cpp" / "symbolic_centromere_backend.cpp"


def compile_cpp_backend(
    compiler: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Compile the C++ backend and return the cached executable path."""
    source = _cpp_source_path()

    if not source.exists():
        raise FileNotFoundError(f"C++ backend source not found: {source}")

    compiler = compiler or os.environ.get("CXX") or "g++"

    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(
            f"C++ compiler '{compiler}' was not found. "
            "Install g++/clang++ or set the CXX environment variable."
        )

    flags = [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-march=native",
    ]

    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(compiler_path.encode())
    digest.update("\0".join(flags).encode())

    key = digest.hexdigest()[:16]

    cache_dir = (
        Path(tempfile.gettempdir())
        / "teloevo"
        / "cpp_backend"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    executable = cache_dir / f"symbolic_centromere_{key}"

    if executable.exists() and not force:
        return executable

    temporary_executable = executable.with_suffix(
        f".tmp.{os.getpid()}"
    )

    command = [
        compiler_path,
        *flags,
        str(source),
        "-o",
        str(temporary_executable),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        try:
            temporary_executable.unlink()
        except FileNotFoundError:
            pass

        raise RuntimeError(
            "Failed to compile C++ backend.\n"
            f"Command: {' '.join(command)}\n"
            f"{completed.stderr}"
        )

    os.replace(temporary_executable, executable)
    executable.chmod(0o755)

    return executable


def _write_symbols(path: Path, symbols: list[int]) -> None:
    with path.open("w") as handle:
        handle.write(" ".join(map(str, symbols)))
        handle.write("\n")


def _write_lookup(path: Path, lookup: dict[int, str]) -> None:
    keys = sorted(lookup)

    if keys != list(range(len(keys))):
        raise ValueError(
            "The C++ backend requires contiguous lookup symbols "
            "0..N-1."
        )

    with path.open("w") as handle:
        for symbol in keys:
            handle.write(lookup[symbol])
            handle.write("\n")


def _read_lookup(path: Path) -> dict[int, str]:
    output: dict[int, str] = {}

    with path.open() as handle:
        fallback_symbol = 0

        for line in handle:
            line = line.rstrip("\n\r")

            if not line:
                continue

            if "\t" in line:
                symbol_text, sequence = line.split("\t", 1)
                output[int(symbol_text)] = sequence
            else:
                # Backwards compatibility with older compiled backends that
                # wrote a dense lookup containing sequence-only lines.
                output[fallback_symbol] = line
                fallback_symbol += 1

    return output


def _read_symbol_line(path: Path) -> list[int]:
    text = path.read_text().strip()

    if not text:
        return []

    return np.fromstring(
        text,
        dtype=np.int64,
        sep=" ",
    ).astype(int).tolist()


def _read_checkpoints(path: Path) -> list[CheckpointState]:
    checkpoints: list[CheckpointState] = []

    if not path.exists():
        return checkpoints

    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")

            if not line:
                continue

            generation_text, symbols_text = line.split("\t", 1)

            symbols = np.fromstring(
                symbols_text,
                dtype=np.int64,
                sep=" ",
            ).astype(int).tolist()

            checkpoints.append(
                CheckpointState(
                    generation=int(generation_text),
                    symbols=symbols,
                )
            )

    return checkpoints


def _compact_outputs(
    checkpoints: list[CheckpointState],
    final_symbols: list[int],
    sparse_lookup: dict[int, str],
) -> tuple[list[CheckpointState], list[int], dict[int, str]]:
    required = set(final_symbols)

    for checkpoint in checkpoints:
        required.update(checkpoint.symbols)

    missing = required.difference(sparse_lookup)

    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(
            "C++ backend did not return all sequence states required "
            f"by its outputs. Missing symbols: {preview}"
        )

    old_symbols = sorted(required)
    remap = {
        old_symbol: new_symbol
        for new_symbol, old_symbol in enumerate(old_symbols)
    }

    compact_lookup = {
        remap[old_symbol]: sparse_lookup[old_symbol]
        for old_symbol in old_symbols
    }

    compact_checkpoints = [
        CheckpointState(
            generation=checkpoint.generation,
            symbols=[
                remap[symbol]
                for symbol in checkpoint.symbols
            ],
        )
        for checkpoint in checkpoints
    ]

    compact_final_symbols = [
        remap[symbol]
        for symbol in final_symbols
    ]

    return (
        compact_checkpoints,
        compact_final_symbols,
        compact_lookup,
    )


def _lookup_object(
    lookup_dict: dict[int, str],
) -> SequenceLookup:
    if not lookup_dict:
        raise ValueError("Lookup cannot be empty.")

    monomer_length = len(lookup_dict[0])
    lookup = SequenceLookup(monomer_length)

    for expected_symbol in range(len(lookup_dict)):
        symbol = lookup.intern(
            lookup_dict[expected_symbol]
        )

        if symbol != expected_symbol:
            raise RuntimeError(
                "Lookup symbol order changed while rebuilding state."
            )

    return lookup


class CppSymbolicDongSimulator:
    """Compiled C++ backend for symbolic centromere evolution."""

    backend = "cpp"

    def __init__(
        self,
        state: CentromereState,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
        compiler: Optional[str] = None,
        profile: bool = False,
        profile_interval: int = 100_000,
        symbol_storage: str = "vector",
        symbol_block_size: int = 512,
    ):
        self.state = state
        self.parameters = parameters
        self.seed = (
            int(seed)
            if seed is not None
            else secrets.randbits(63)
        )
        self.record_events = bool(record_events)
        self.compiler = compiler
        self.profile_enabled = bool(profile)
        self.profile_interval = int(profile_interval)
        if self.profile_enabled and self.profile_interval <= 0:
            raise ValueError("profile_interval must be > 0.")

        symbol_storage = str(symbol_storage).lower()
        if symbol_storage not in {"vector", "blocked"}:
            raise ValueError(
                "symbol_storage must be 'vector' or 'blocked'."
            )

        self.symbol_storage = symbol_storage
        self.symbol_block_size = int(symbol_block_size)

        if self.symbol_block_size <= 0:
            raise ValueError("symbol_block_size must be > 0.")

        (
            self._snp_position_rates,
            self._snp_spectrum,
        ) = _prepare_snp_model(
            parameters,
            state.monomer_length,
        )

        self.events: list[EventRecord] = []
        self.profile: list[dict] = []
        self.profile_io: dict[str, float] = {}

    @classmethod
    def homogeneous(
        cls,
        n_monomers: int,
        monomer_sequence: str = DEFAULT_MONOMER,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
        compiler: Optional[str] = None,
        profile: bool = False,
        profile_interval: int = 100_000,
        symbol_storage: str = "vector",
        symbol_block_size: int = 512,
    ) -> "CppSymbolicDongSimulator":
        state = CentromereState.homogeneous(
            n_monomers=n_monomers,
            monomer_sequence=monomer_sequence,
        )

        return cls(
            state=state,
            parameters=parameters,
            seed=seed,
            record_events=record_events,
            compiler=compiler,
            profile=profile,
            profile_interval=profile_interval,
            symbol_storage=symbol_storage,
            symbol_block_size=symbol_block_size,
        )

    def _checkpoint_set(
        self,
        n_generations: int,
        start_generation: int,
        checkpoints: Optional[list[int]],
    ) -> set[int]:
        end_generation = start_generation + n_generations

        if checkpoints is None:
            return {
                start_generation,
                end_generation,
            }

        checkpoint_set = {
            int(generation)
            for generation in checkpoints
        }

        invalid = [
            generation
            for generation in checkpoint_set
            if (
                generation < start_generation
                or generation > end_generation
            )
        ]

        if invalid:
            raise ValueError(
                "All checkpoints must fall between start_generation "
                f"({start_generation}) and final generation "
                f"({end_generation}). Invalid: {sorted(invalid)}"
            )

        return checkpoint_set

    def _write_config(
        self,
        path: Path,
        n_generations: int,
        start_generation: int,
        checkpoints: set[int],
    ) -> None:
        p = self.parameters

        minimum = (
            -1
            if p.min_array_bp is None
            else int(p.min_array_bp)
        )

        maximum = (
            -1
            if p.max_array_bp is None
            else int(p.max_array_bp)
        )

        checkpoint_text = (
            ",".join(map(str, sorted(checkpoints)))
            if checkpoints
            else "-"
        )

        snp_model_code = {
            "dong": 0,
            "cen178": 1,
            "none": 2,
        }[p.snp_model]

        snp_position_rates_text = (
            "-"
            if self._snp_position_rates is None
            else ",".join(
                f"{value:.17g}"
                for value in self._snp_position_rates
            )
        )

        snp_spectrum_text = (
            "-"
            if self._snp_spectrum is None
            else ",".join(
                f"{value:.17g}"
                for value in self._snp_spectrum.ravel()
            )
        )

        values = {
            "n_generations": int(n_generations),
            "start_generation": int(start_generation),
            "seed": int(self.seed),
            "snp_rate": p.snp_rate_per_bp,
            "snp_model": snp_model_code,
            "snp_position_rates": snp_position_rates_text,
            "snp_spectrum": snp_spectrum_text,
            "ins_rate": p.ins_rate_per_bp,
            "del_rate": p.del_rate_per_bp,
            "large_del_rate": p.large_del_rate_per_bp,
            "conversion_rate": p.conversion_rate_per_bp,
            "ins_span_lambda": p.ins_span_lambda,
            "del_span_lambda": p.del_span_lambda,
            "large_del_span_lambda": p.large_del_span_lambda,
            "conversion_p": p.conversion_geometric_p,
            "min_array_bp": minimum,
            "max_array_bp": maximum,
            "ins_attempt_multiplier": p.ins_attempt_multiplier,
            "del_attempt_multiplier": p.del_attempt_multiplier,
            "large_del_attempt_multiplier":
                p.large_del_attempt_multiplier,
            "conversion_attempt_multiplier":
                p.conversion_attempt_multiplier,
            "record_events": int(self.record_events),
            "profile": int(self.profile_enabled),
            "profile_interval": self.profile_interval,
            "symbol_storage": (
                0 if self.symbol_storage == "vector" else 1
            ),
            "symbol_block_size": self.symbol_block_size,
            "checkpoints": checkpoint_text,
        }

        with path.open("w") as handle:
            for key, value in values.items():
                handle.write(f"{key} {value}\n")

    def _read_events(self, path: Path) -> list[EventRecord]:
        if not self.record_events or not path.exists():
            return []

        output: list[EventRecord] = []

        with path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")

                if len(fields) != 6:
                    continue

                output.append(
                    EventRecord(
                        generation=int(fields[0]),
                        event_type=fields[1],
                        start_monomer=int(fields[2]),
                        phase=int(fields[3]),
                        span_monomers=int(fields[4]),
                        size_bp=int(fields[5]),
                    )
                )

        return output

    def _read_profile(self, path: Path) -> list[dict]:
        if not self.profile_enabled or not path.exists():
            return []

        integer_fields = {
            "generation_start",
            "generation_end",
            "n_generations",
            "n_monomers",
            "n_states",
            "n_symbol_blocks",
            "SNP",
            "INS",
            "DEL",
            "Large_DEL",
            "Conversion",
            "conversion_fast_path",
            "conversion_noop",
            "ins_inserted_symbols",
            "ins_shifted_symbols",
            "ins_reallocations",
            "ins_block_splits",
            "intern_calls",
            "new_states",
            "rehashes",
        }

        output: list[dict] = []

        with path.open() as handle:
            header = handle.readline().rstrip("\n").split("\t")

            for line in handle:
                if not line.strip():
                    continue

                values = line.rstrip("\n").split("\t")
                row = {}

                for key, value in zip(header, values):
                    if key in integer_fields:
                        row[key] = int(value)
                    else:
                        row[key] = float(value)

                output.append(row)

        return output

    def run(
        self,
        n_generations: int,
        start_generation: int = 0,
        checkpoints: Optional[list[int]] = None,
    ) -> SimulationResult:
        """Run the simulation using the compiled C++ backend."""
        if n_generations < 0:
            raise ValueError("n_generations must be >= 0.")

        checkpoint_set = self._checkpoint_set(
            n_generations=n_generations,
            start_generation=start_generation,
            checkpoints=checkpoints,
        )

        timing_start = time.perf_counter()

        t0 = time.perf_counter()
        executable = compile_cpp_backend(
            compiler=self.compiler,
        )
        time_compile = time.perf_counter() - t0

        with tempfile.TemporaryDirectory(
            prefix="teloevo_cpp_"
        ) as directory:
            directory = Path(directory)

            config_path = directory / "config.txt"
            symbols_in = directory / "symbols.in.txt"
            lookup_in = directory / "lookup.in.txt"

            checkpoints_out = directory / "checkpoints.out.txt"
            final_symbols_out = directory / "symbols.out.txt"
            lookup_out = directory / "lookup.out.txt"
            events_out = directory / "events.out.txt"
            profile_out = directory / "profile.out.tsv"

            t0 = time.perf_counter()

            self._write_config(
                config_path,
                n_generations=n_generations,
                start_generation=start_generation,
                checkpoints=checkpoint_set,
            )

            _write_symbols(
                symbols_in,
                self.state.symbols,
            )

            _write_lookup(
                lookup_in,
                self.state.lookup.as_dict(),
            )

            time_write_inputs = time.perf_counter() - t0

            command = [
                str(executable),
                str(config_path),
                str(symbols_in),
                str(lookup_in),
                str(checkpoints_out),
                str(final_symbols_out),
                str(lookup_out),
                str(events_out),
                str(profile_out),
            ]

            t0 = time.perf_counter()

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

            time_cpp_process = time.perf_counter() - t0

            if completed.returncode != 0:
                raise RuntimeError(
                    "C++ symbolic simulator failed.\n"
                    f"{completed.stderr}"
                )

            t0 = time.perf_counter()
            stored_checkpoints = _read_checkpoints(
                checkpoints_out
            )
            time_read_checkpoints = time.perf_counter() - t0

            t0 = time.perf_counter()
            final_symbols = _read_symbol_line(
                final_symbols_out
            )
            time_read_symbols = time.perf_counter() - t0

            t0 = time.perf_counter()
            sparse_lookup = _read_lookup(
                lookup_out
            )
            time_read_lookup = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.events = self._read_events(
                events_out
            )
            time_read_events = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.profile = self._read_profile(
                profile_out
            )
            time_read_profile = time.perf_counter() - t0

        t0 = time.perf_counter()

        (
            stored_checkpoints,
            final_symbols,
            final_lookup,
        ) = _compact_outputs(
            checkpoints=stored_checkpoints,
            final_symbols=final_symbols,
            sparse_lookup=sparse_lookup,
        )

        time_compact_outputs = time.perf_counter() - t0

        t0 = time.perf_counter()

        self.state = CentromereState(
            symbols=final_symbols,
            lookup=_lookup_object(final_lookup),
        )

        time_rebuild_state = time.perf_counter() - t0

        profiled_cpp_loop = (
            sum(row["time_total"] for row in self.profile)
            if self.profile
            else float("nan")
        )

        cpp_unprofiled_overhead = (
            time_cpp_process - profiled_cpp_loop
            if self.profile
            else float("nan")
        )

        self.profile_io = {
            "time_compile": time_compile,
            "time_write_inputs": time_write_inputs,
            "time_cpp_process": time_cpp_process,
            "time_cpp_profiled_loop": profiled_cpp_loop,
            "time_cpp_unprofiled_overhead": cpp_unprofiled_overhead,
            "time_read_checkpoints": time_read_checkpoints,
            "time_read_symbols": time_read_symbols,
            "time_read_lookup": time_read_lookup,
            "time_read_events": time_read_events,
            "time_read_profile": time_read_profile,
            "time_compact_outputs": time_compact_outputs,
            "time_rebuild_state": time_rebuild_state,
            "n_output_states": len(final_lookup),
            "time_total_python_call": time.perf_counter() - timing_start,
        }

        return SimulationResult(
            checkpoints=stored_checkpoints,
            lookup=final_lookup,
        )



class CppBlockedSymbolDongSimulator(CppSymbolicDongSimulator):
    """C++ simulator using blocked storage for the symbolic monomer array."""

    backend = "cpp-blocked"

    def __init__(
        self,
        state: CentromereState,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
        compiler: Optional[str] = None,
        profile: bool = False,
        profile_interval: int = 100_000,
        symbol_block_size: int = 512,
    ):
        super().__init__(
            state=state,
            parameters=parameters,
            seed=seed,
            record_events=record_events,
            compiler=compiler,
            profile=profile,
            profile_interval=profile_interval,
            symbol_storage="blocked",
            symbol_block_size=symbol_block_size,
        )

    @classmethod
    def homogeneous(
        cls,
        n_monomers: int,
        monomer_sequence: str = DEFAULT_MONOMER,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
        compiler: Optional[str] = None,
        profile: bool = False,
        profile_interval: int = 100_000,
        symbol_block_size: int = 512,
    ) -> "CppBlockedSymbolDongSimulator":
        state = CentromereState.homogeneous(
            n_monomers=n_monomers,
            monomer_sequence=monomer_sequence,
        )

        return cls(
            state=state,
            parameters=parameters,
            seed=seed,
            record_events=record_events,
            compiler=compiler,
            profile=profile,
            profile_interval=profile_interval,
            symbol_block_size=symbol_block_size,
        )

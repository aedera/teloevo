"""Coarse-grained symbolic implementation of Dong-style centromere evolution.

The centromere is stored as an ordered list of integer symbols. Each symbol
maps to one fixed-length monomer sequence in SequenceLookup. Identical monomer
sequences always reuse the same symbol.

This preserves the main Dong et al. mutation logic while avoiding a full DNA
string:
    SNP -> INS -> DEL -> large DEL -> gene conversion

For INS/DEL, breakpoints use the same repeat phase in two monomers. Because all
monomers have the same length, no pairwise alignment is required.

The implementation is intentionally simple and modular. Each mutation class is
a separate public step_* method, and each event can also be applied directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np


DEFAULT_MONOMER = 'AGTATAAGAACTTAAACCGCAACCCGATCTTAAAAGCCTAAGTAGTGTTTCCTTGTTAGAAGACACAAAGCCAAAGACTCATATGGACTTTGGCTACACCATGAAAGCTTTGAGAAGCAAGAAGAAGGTTGGTTAGTGTTTTGGAGTCGAATATGACTTGATGTCATGTGTATGATTG'


@dataclass(frozen=True)
class SimulationParameters:
    """Rates and size distributions for the symbolic simulation.

    Rates are per bp per generation, as in Dong et al.

    Span parameters use Dong's rule:
        K = max(1, Poisson(span_lambda))

    Set any non-SNP rate to 0 to disable that process. SNP behavior is
    selected by snp_model; snp_rate_per_bp is used only by the Dong SNP
    model.
    """

    snp_rate_per_bp: float = 1.97e-8

    # SNP model:
    #   "dong"   = Dong's uniform per-bp rate and uniform alternative base.
    #   "cen178" = position-specific absolute rates plus a substitution
    #              spectrum loaded from the supplied files.
    #   "none"   = disable SNPs.
    snp_model: str = "dong"
    snp_position_rates_path: Optional[str] = None
    snp_spectrum_path: Optional[str] = None
    snp_background_rate: Optional[float] = 1e-10

    ins_rate_per_bp: float = 1.15e-8
    del_rate_per_bp: float = 6.93e-9
    large_del_rate_per_bp: float = 3.04e-11
    conversion_rate_per_bp: float = 3.39e-8

    ins_span_lambda: float = 10.51
    del_span_lambda: float = 5.19
    large_del_span_lambda: float = 2809.0
    conversion_geometric_p: float = 0.05

    # Optional Dong-like size limits. None disables a limit.
    min_array_bp: Optional[int] = 1_000_000
    max_array_bp: Optional[int] = 100_000_000

    # Attempt multipliers used when proposals are rejected.
    ins_attempt_multiplier: int = 50
    del_attempt_multiplier: int = 100
    large_del_attempt_multiplier: int = 200
    conversion_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        rates = (
            self.snp_rate_per_bp,
            self.ins_rate_per_bp,
            self.del_rate_per_bp,
            self.large_del_rate_per_bp,
            self.conversion_rate_per_bp,
        )
        if any(x < 0 for x in rates):
            raise ValueError("Mutation rates must be >= 0.")

        if self.snp_model not in {"dong", "cen178", "none"}:
            raise ValueError(
                "snp_model must be 'dong', 'cen178', or 'none'."
            )

        if self.snp_model == "cen178":
            if self.snp_position_rates_path is None:
                raise ValueError(
                    "snp_position_rates_path is required when "
                    "snp_model='cen178'."
                )
            if self.snp_spectrum_path is None:
                raise ValueError(
                    "snp_spectrum_path is required when "
                    "snp_model='cen178'."
                )

        if (
            self.snp_background_rate is not None
            and self.snp_background_rate < 0
        ):
            raise ValueError("snp_background_rate must be >= 0 or None.")

        spans = (
            self.ins_span_lambda,
            self.del_span_lambda,
            self.large_del_span_lambda,
        )
        if any(x < 0 for x in spans):
            raise ValueError("Span lambdas must be >= 0.")

        if not 0 < self.conversion_geometric_p <= 1:
            raise ValueError("conversion_geometric_p must be in (0, 1].")


_BASES = "ACGT"
_BASE_TO_INDEX = {base: i for i, base in enumerate(_BASES)}


def _prepare_snp_model(
    parameters: SimulationParameters,
    monomer_length: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and validate the optional CEN178-specific SNP model."""
    if parameters.snp_model != "cen178":
        return None, None

    rates_path = Path(parameters.snp_position_rates_path)
    spectrum_path = Path(parameters.snp_spectrum_path)

    rates = np.load(
        rates_path,
        allow_pickle=False,
    )
    rates = np.asarray(rates, dtype=float).reshape(-1)

    if rates.shape != (monomer_length,):
        raise ValueError(
            "CEN178 position-rate array length must match the monomer "
            f"length ({monomer_length}); got {rates.size} values."
        )

    if not np.all(np.isfinite(rates)):
        raise ValueError(
            "CEN178 position rates must all be finite."
        )

    if np.any(rates < 0):
        raise ValueError(
            "CEN178 position rates must be >= 0."
        )

    if parameters.snp_background_rate is not None:
        rates = rates.copy()
        rates[rates == 0] = parameters.snp_background_rate

    if float(rates.sum()) <= 0:
        raise ValueError(
            "CEN178 position rates must contain at least one positive rate."
        )

    with spectrum_path.open() as handle:
        spectrum = json.load(handle)

    matrix = np.zeros((4, 4), dtype=float)

    def add(ref: str, alt: str, value) -> None:
        if (
            ref in _BASE_TO_INDEX
            and alt in _BASE_TO_INDEX
            and ref != alt
            and float(value) > 0
        ):
            matrix[
                _BASE_TO_INDEX[ref],
                _BASE_TO_INDEX[alt],
            ] += float(value)

    if all(">" in key for key in spectrum):
        for key, value in spectrum.items():
            ref, alt = key.split(">", 1)
            add(ref, alt, value)
    else:
        for ref, row in spectrum.items():
            for alt, value in row.items():
                add(ref, alt, value)

    for ref_index in range(4):
        row_sum = matrix[ref_index].sum()

        if row_sum == 0:
            for alt_index in range(4):
                if alt_index != ref_index:
                    matrix[ref_index, alt_index] = 1.0 / 3.0
        else:
            matrix[ref_index] /= row_sum
            matrix[ref_index, ref_index] = 0.0

    return rates, matrix


class SequenceLookup:
    """Bidirectional lookup between integer symbols and monomer sequences."""

    def __init__(self, monomer_length: int):
        if monomer_length <= 0:
            raise ValueError("monomer_length must be > 0.")
        self.monomer_length = int(monomer_length)
        self._sequence_to_symbol: dict[str, int] = {}
        self._symbol_to_sequence: list[str] = []

    def intern(self, sequence: str) -> int:
        """Return the existing symbol for sequence, or create a new one."""
        sequence = sequence.upper()
        if len(sequence) != self.monomer_length:
            raise ValueError(
                f"Expected monomer length {self.monomer_length}, "
                f"got {len(sequence)}."
            )
        if any(base not in "ACGT" for base in sequence):
            raise ValueError("Monomer sequences must contain only A/C/G/T.")

        symbol = self._sequence_to_symbol.get(sequence)
        if symbol is not None:
            return symbol

        symbol = len(self._symbol_to_sequence)
        self._sequence_to_symbol[sequence] = symbol
        self._symbol_to_sequence.append(sequence)
        return symbol

    def sequence(self, symbol: int) -> str:
        return self._symbol_to_sequence[symbol]

    def __len__(self) -> int:
        return len(self._symbol_to_sequence)

    def as_dict(self) -> dict[int, str]:
        return {i: seq for i, seq in enumerate(self._symbol_to_sequence)}


@dataclass
class CentromereState:
    """Ordered symbolic monomer array plus its sequence lookup."""

    symbols: list[int]
    lookup: SequenceLookup

    @classmethod
    def homogeneous(
        cls,
        n_monomers: int,
        monomer_sequence: str = DEFAULT_MONOMER,
    ) -> "CentromereState":
        if n_monomers <= 0:
            raise ValueError("n_monomers must be > 0.")
        seq = monomer_sequence.upper()
        lookup = SequenceLookup(len(seq))
        symbol = lookup.intern(seq)
        return cls(symbols=[symbol] * int(n_monomers), lookup=lookup)

    @classmethod
    def from_sequences(cls, sequences: list[str]) -> "CentromereState":
        if not sequences:
            raise ValueError("sequences cannot be empty.")
        m = len(sequences[0])
        lookup = SequenceLookup(m)
        symbols = [lookup.intern(seq) for seq in sequences]
        return cls(symbols=symbols, lookup=lookup)

    @property
    def n_monomers(self) -> int:
        return len(self.symbols)

    @property
    def monomer_length(self) -> int:
        return self.lookup.monomer_length

    @property
    def length_bp(self) -> int:
        return self.n_monomers * self.monomer_length

    def monomer_sequence(self, index: int) -> str:
        return self.lookup.sequence(self.symbols[index])


@dataclass(frozen=True)
class EventRecord:
    generation: int
    event_type: str
    start_monomer: int
    phase: int
    span_monomers: int = 0
    size_bp: int = 0


@dataclass
class GenerationSummary:
    generation: int
    snp: int = 0
    ins: int = 0
    deletion: int = 0
    large_deletion: int = 0
    conversion: int = 0
    n_monomers: int = 0
    n_sequence_states: int = 0


@dataclass(frozen=True)
class CheckpointState:
    """Snapshot of the symbolic monomer array at one generation."""

    generation: int
    symbols: list[int]


@dataclass(frozen=True)
class SimulationResult:
    """Stored symbolic checkpoints plus the final append-only lookup table."""

    checkpoints: list[CheckpointState]
    lookup: dict[int, str]


class SymbolicDongSimulator:
    """Forward simulator on a symbolic fixed-length monomer array."""

    backend = "python"

    def __init__(
        self,
        state: CentromereState,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
    ):
        self.state = state
        self.parameters = parameters
        self.rng = np.random.default_rng(seed)
        self.record_events = bool(record_events)
        self.events: list[EventRecord] = []

        (
            self._snp_position_rates,
            self._snp_spectrum,
        ) = _prepare_snp_model(
            parameters,
            state.monomer_length,
        )

        self._snp_position_probabilities = None

        if self._snp_position_rates is not None:
            self._snp_position_probabilities = (
                self._snp_position_rates
                / self._snp_position_rates.sum()
            )

    @classmethod
    def homogeneous(
        cls,
        n_monomers: int,
        monomer_sequence: str = DEFAULT_MONOMER,
        parameters: SimulationParameters = SimulationParameters(),
        seed: Optional[int] = None,
        record_events: bool = False,
    ) -> "SymbolicDongSimulator":
        state = CentromereState.homogeneous(n_monomers, monomer_sequence)
        return cls(state, parameters, seed=seed, record_events=record_events)

    # ------------------------------------------------------------------
    # Random draws. Kept separate so distributions are easy to replace.
    # ------------------------------------------------------------------

    def draw_event_count(self, rate_per_bp: float) -> int:
        """Poisson number of events for the current array length."""
        if rate_per_bp == 0:
            return 0
        return int(self.rng.poisson(self.state.length_bp * rate_per_bp))

    def draw_span(self, span_lambda: float) -> int:
        """Dong-style K = max(1, Poisson(lambda))."""
        return max(1, int(self.rng.poisson(span_lambda)))

    def draw_breakpoint(self) -> tuple[int, int]:
        """Uniform base position returned as (monomer_index, repeat_phase)."""
        bp = int(self.rng.integers(0, self.state.length_bp))
        return divmod(bp, self.state.monomer_length)

    def draw_conversion_length(self) -> int:
        """Positive geometric tract length, matching Dong's +1 convention."""
        return int(self.rng.geometric(self.parameters.conversion_geometric_p))

    # ------------------------------------------------------------------
    # Low-level sequence helpers.
    # ------------------------------------------------------------------

    def _extract_bases(self, start_bp: int, end_bp: int) -> str:
        if not 0 <= start_bp <= end_bp <= self.state.length_bp:
            raise ValueError("Invalid base interval.")

        m = self.state.monomer_length
        pos = start_bp
        pieces: list[str] = []

        while pos < end_bp:
            idx, phase = divmod(pos, m)
            take = min(end_bp - pos, m - phase)
            seq = self.state.monomer_sequence(idx)
            pieces.append(seq[phase : phase + take])
            pos += take

        return "".join(pieces)

    def _replace_bases(self, start_bp: int, end_bp: int, replacement: str) -> None:
        """Replace an equal-length tract and intern any new monomer states."""
        if end_bp - start_bp != len(replacement):
            raise ValueError("Replacement must preserve tract length.")
        if not 0 <= start_bp <= end_bp <= self.state.length_bp:
            raise ValueError("Invalid base interval.")

        m = self.state.monomer_length
        pos = start_bp
        cursor = 0

        while pos < end_bp:
            idx, phase = divmod(pos, m)
            take = min(end_bp - pos, m - phase)

            old = self.state.monomer_sequence(idx)
            new = (
                old[:phase]
                + replacement[cursor : cursor + take]
                + old[phase + take :]
            )
            self.state.symbols[idx] = self.state.lookup.intern(new)

            pos += take
            cursor += take

    def _record(
        self,
        generation: int,
        event_type: str,
        start_monomer: int,
        phase: int,
        span_monomers: int = 0,
        size_bp: int = 0,
    ) -> None:
        if self.record_events:
            self.events.append(
                EventRecord(
                    generation=generation,
                    event_type=event_type,
                    start_monomer=start_monomer,
                    phase=phase,
                    span_monomers=span_monomers,
                    size_bp=size_bp,
                )
            )

    # ------------------------------------------------------------------
    # Individual event operations.
    # ------------------------------------------------------------------

    def apply_snp(self, generation: int = 0) -> bool:
        """Apply one SNP using the configured SNP model."""
        if self.parameters.snp_model == "none":
            return False

        if self.parameters.snp_model == "cen178":
            i = int(
                self.rng.integers(
                    0,
                    self.state.n_monomers,
                )
            )

            phase = int(
                self.rng.choice(
                    self.state.monomer_length,
                    p=self._snp_position_probabilities,
                )
            )

            seq = self.state.monomer_sequence(i)
            ref = seq[phase]
            ref_index = _BASE_TO_INDEX[ref]

            alt_index = int(
                self.rng.choice(
                    4,
                    p=self._snp_spectrum[ref_index],
                )
            )
            alt = _BASES[alt_index]
        else:
            i, phase = self.draw_breakpoint()
            seq = self.state.monomer_sequence(i)
            ref = seq[phase]
            choices = [b for b in "ACGT" if b != ref]
            alt = choices[int(self.rng.integers(0, 3))]

        new = seq[:phase] + alt + seq[phase + 1 :]
        self.state.symbols[i] = self.state.lookup.intern(new)
        self._record(generation, "SNP", i, phase, 0, 1)
        return True

    def apply_phase_insertion(
        self,
        start_monomer: int,
        span_monomers: int,
        phase: int,
        generation: int = 0,
    ) -> bool:
        """Duplicate a phase-matched interval spanning K monomers.

        If the two endpoint monomers differ, the insertion creates one
        fixed-length chimeric monomer at the new junction. Identical resulting
        sequences automatically reuse an existing lookup symbol.
        """
        n = self.state.n_monomers
        m = self.state.monomer_length
        i = int(start_monomer)
        k = int(span_monomers)
        p = int(phase)
        j = i + k

        if k < 1 or not 0 <= p < m or not 0 <= i < n or j >= n:
            return False

        added_bp = k * m
        max_bp = self.parameters.max_array_bp
        if max_bp is not None and self.state.length_bp + added_bp > max_bp:
            return False

        left_symbol = self.state.symbols[i]
        right_symbol = self.state.symbols[j]
        left_seq = self.state.lookup.sequence(left_symbol)
        right_seq = self.state.lookup.sequence(right_symbol)

        # Sequence at the insertion junction: prefix of right endpoint followed
        # by suffix of left endpoint.
        chimera = right_seq[:p] + left_seq[p:]
        chimera_symbol = self.state.lookup.intern(chimera)

        between = self.state.symbols[i + 1 : j]
        replacement = (
            [left_symbol]
            + between
            + [chimera_symbol]
            + between
            + [right_symbol]
        )
        self.state.symbols[i : j + 1] = replacement

        self._record(generation, "INS", i, p, k, added_bp)
        return True

    def apply_phase_deletion(
        self,
        start_monomer: int,
        span_monomers: int,
        phase: int,
        generation: int = 0,
        event_type: str = "DEL",
    ) -> bool:
        """Delete a phase-matched interval spanning K monomers."""
        n = self.state.n_monomers
        m = self.state.monomer_length
        i = int(start_monomer)
        k = int(span_monomers)
        p = int(phase)
        j = i + k

        if k < 1 or not 0 <= p < m or not 0 <= i < n or j >= n:
            return False

        removed_bp = k * m
        min_bp = self.parameters.min_array_bp
        if min_bp is not None and self.state.length_bp - removed_bp < min_bp:
            return False

        left_seq = self.state.monomer_sequence(i)
        right_seq = self.state.monomer_sequence(j)

        # Sequence left after deleting from phase p in i to phase p in j.
        chimera = left_seq[:p] + right_seq[p:]
        chimera_symbol = self.state.lookup.intern(chimera)

        self.state.symbols[i : j + 1] = [chimera_symbol]

        self._record(generation, event_type, i, p, k, removed_bp)
        return True

    def apply_insertion(self, generation: int = 0) -> bool:
        i, phase = self.draw_breakpoint()
        k = self.draw_span(self.parameters.ins_span_lambda)
        return self.apply_phase_insertion(i, k, phase, generation=generation)

    def apply_deletion(self, generation: int = 0) -> bool:
        i, phase = self.draw_breakpoint()
        k = self.draw_span(self.parameters.del_span_lambda)
        return self.apply_phase_deletion(i, k, phase, generation=generation)

    def apply_large_deletion(self, generation: int = 0) -> bool:
        i, phase = self.draw_breakpoint()
        k = self.draw_span(self.parameters.large_del_span_lambda)
        return self.apply_phase_deletion(
            i,
            k,
            phase,
            generation=generation,
            event_type="Large_DEL",
        )

    def apply_conversion(self, generation: int = 0) -> bool:
        """Copy a donor tract to the same phase one monomer downstream."""
        m = self.state.monomer_length
        length_bp = self.state.length_bp

        start = int(self.rng.integers(0, length_bp))
        tract = self.draw_conversion_length()
        end = start + tract

        # Dong rejects if donor end is outside the sequence, or if the
        # downstream receptor unit needed to map the end does not exist.
        if end >= length_bp or end + m >= length_bp:
            return False

        donor = self._extract_bases(start, end)
        receptor_start = start + m
        receptor_end = end + m
        self._replace_bases(receptor_start, receptor_end, donor)

        i, phase = divmod(receptor_start, m)
        self._record(generation, "Conversion", i, phase, 0, tract)
        return True

    # ------------------------------------------------------------------
    # Dong mutation classes. Event counts are drawn once per class.
    # ------------------------------------------------------------------

    def step_snps(self, generation: int) -> int:
        if self.parameters.snp_model == "none":
            return 0

        if self.parameters.snp_model == "cen178":
            total_rate_per_monomer = float(
                self._snp_position_rates.sum()
            )
            requested = int(
                self.rng.poisson(
                    self.state.n_monomers
                    * total_rate_per_monomer
                )
            )
        else:
            requested = self.draw_event_count(
                self.parameters.snp_rate_per_bp
            )

        for _ in range(requested):
            self.apply_snp(generation=generation)

        return requested

    def _run_rejection_step(
        self,
        requested: int,
        attempt_multiplier: int,
        event_function: Callable[[int], bool],
        generation: int,
    ) -> int:
        if requested <= 0:
            return 0

        accepted = 0
        attempts = 0
        max_attempts = requested * attempt_multiplier

        while accepted < requested and attempts < max_attempts:
            attempts += 1
            if event_function(generation):
                accepted += 1

        return accepted

    def step_insertions(self, generation: int) -> int:
        requested = self.draw_event_count(self.parameters.ins_rate_per_bp)
        return self._run_rejection_step(
            requested,
            self.parameters.ins_attempt_multiplier,
            self.apply_insertion,
            generation,
        )

    def step_deletions(self, generation: int) -> int:
        requested = self.draw_event_count(self.parameters.del_rate_per_bp)
        return self._run_rejection_step(
            requested,
            self.parameters.del_attempt_multiplier,
            self.apply_deletion,
            generation,
        )

    def step_large_deletions(self, generation: int) -> int:
        requested = self.draw_event_count(self.parameters.large_del_rate_per_bp)
        return self._run_rejection_step(
            requested,
            self.parameters.large_del_attempt_multiplier,
            self.apply_large_deletion,
            generation,
        )

    def step_conversions(self, generation: int) -> int:
        requested = self.draw_event_count(self.parameters.conversion_rate_per_bp)
        return self._run_rejection_step(
            requested,
            self.parameters.conversion_attempt_multiplier,
            self.apply_conversion,
            generation,
        )

    def step_generation(self, generation: int) -> GenerationSummary:
        """Run one generation in the same mutation-class order as Dong."""
        min_bp = self.parameters.min_array_bp
        max_bp = self.parameters.max_array_bp

        if min_bp is not None and self.state.length_bp < min_bp:
            raise RuntimeError("Array is below min_array_bp.")
        if max_bp is not None and self.state.length_bp > max_bp:
            raise RuntimeError("Array is above max_array_bp.")

        # Order matches Dong:
        # SNP -> INS -> DEL -> large DEL -> conversion
        snp = self.step_snps(generation)
        ins = self.step_insertions(generation)
        deletion = self.step_deletions(generation)
        large_deletion = self.step_large_deletions(generation)
        conversion = self.step_conversions(generation)

        return GenerationSummary(
            generation=generation,
            snp=snp,
            ins=ins,
            deletion=deletion,
            large_deletion=large_deletion,
            conversion=conversion,
            n_monomers=self.state.n_monomers,
            n_sequence_states=len(self.state.lookup),
        )

    def run(
        self,
        n_generations: int,
        start_generation: int = 0,
        checkpoints: Optional[list[int]] = None,
    ) -> SimulationResult:
        """Run forward and store symbolic arrays at requested generations.

        Parameters
        ----------
        n_generations
            Number of generations to evolve.
        start_generation
            Generation number assigned to the input state.
        checkpoints
            Absolute generation numbers to store. By default, only the initial
            state and the final state are stored.

        Returns
        -------
        SimulationResult
            A list of checkpoint snapshots plus one final append-only lookup
            table. The same final lookup reconstructs every stored checkpoint.
        """
        if n_generations < 0:
            raise ValueError("n_generations must be >= 0.")

        end_generation = start_generation + n_generations

        if checkpoints is None:
            checkpoint_set = {start_generation, end_generation}
        else:
            checkpoint_set = {int(g) for g in checkpoints}
            invalid = [
                g for g in checkpoint_set
                if g < start_generation or g > end_generation
            ]
            if invalid:
                raise ValueError(
                    "All checkpoints must fall between start_generation "
                    f"({start_generation}) and final generation "
                    f"({end_generation}). Invalid: {sorted(invalid)}"
                )

        stored: list[CheckpointState] = []

        if start_generation in checkpoint_set:
            stored.append(
                CheckpointState(
                    generation=start_generation,
                    symbols=self.state.symbols.copy(),
                )
            )

        for offset in range(1, n_generations + 1):
            generation = start_generation + offset
            self.step_generation(generation)

            if generation in checkpoint_set:
                stored.append(
                    CheckpointState(
                        generation=generation,
                        symbols=self.state.symbols.copy(),
                    )
                )

        return SimulationResult(
            checkpoints=stored,
            lookup=self.state.lookup.as_dict(),
        )


def dong_parameters(
    include_large_deletions: bool = True,
    include_snps: bool = True,
    include_conversion: bool = True,
    snp_model: str = "dong",
    snp_position_rates: Optional[str] = None,
    snp_spectrum: Optional[str] = None,
    snp_background_rate: Optional[float] = 1e-10,
) -> SimulationParameters:
    """Convenience constructor using Dong structural-mutation parameters.

    snp_model="dong" preserves Dong's original uniform SNP process.

    snp_model="cen178" replaces only the SNP process with the supplied
    absolute per-position mutation rates and substitution spectrum. INS, DEL,
    large DEL, and conversion parameters remain Dong-like.

    include_snps=False disables SNPs regardless of snp_model.
    """
    effective_snp_model = (
        snp_model
        if include_snps
        else "none"
    )

    return SimulationParameters(
        snp_rate_per_bp=1.97e-8 if include_snps else 0.0,
        snp_model=effective_snp_model,
        snp_position_rates_path=snp_position_rates,
        snp_spectrum_path=snp_spectrum,
        snp_background_rate=snp_background_rate,
        ins_rate_per_bp=1.15e-8,
        del_rate_per_bp=6.93e-9,
        large_del_rate_per_bp=3.04e-11 if include_large_deletions else 0.0,
        conversion_rate_per_bp=3.39e-8 if include_conversion else 0.0,
        ins_span_lambda=10.51,
        del_span_lambda=5.19,
        large_del_span_lambda=2809.0,
        conversion_geometric_p=0.05,
        min_array_bp=1_000_000,
        max_array_bp=100_000_000,
    )

"""Run a single centromere trajectory, using bundled CEN1 by default."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from . import __version__
from .example_data import example_data_path
from .initial_state import _read_fasta_record, centromere_state_from_gff_fixed
from .symbolic_centromere import DEFAULT_MONOMER, SymbolicDongSimulator, dong_parameters
from .symbolic_centromere_cpp import CppBlockedSymbolDongSimulator, CppSymbolicDongSimulator

MOTIFS = ("AAACCCT", "AACCCTA", "ACCCTAA", "CCCTAAA", "CCTAAAC", "CTAAACC", "TAAACCC")
MOTIFS += tuple(m.translate(str.maketrans("ACGT", "TGCA"))[::-1] for m in MOTIFS)


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative.")
    return result


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("Must be finite and nonnegative.")
    return result


def size_limit(value):
    return None if value.lower() == "none" else nonnegative_int(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"teloevo {__version__}")
    parser.add_argument("--output", required=True, type=Path, help="New .npz output file.")
    parser.add_argument("--generations", type=nonnegative_int, default=1000)
    parser.add_argument("--seed", type=nonnegative_int, default=846)
    parser.add_argument("--backend", choices=["cpp-blocked", "cpp", "python"], default="cpp-blocked")
    parser.add_argument("--checkpoints", help="Comma-separated generations; default logarithmic. Always includes 0 and the final generation.")
    parser.add_argument("--fasta", type=Path, help="Custom reference FASTA; requires --gff.")
    parser.add_argument("--gff", type=Path, help="Custom satellite GFF/GFF3; requires --fasta.")
    parser.add_argument("--chrom", default="CEN1", help="Matching FASTA/GFF record identifier.")
    parser.add_argument("--start", type=nonnegative_int, default=1, help="1-based inclusive interval start.")
    parser.add_argument("--end", type=nonnegative_int, help="1-based inclusive end; default record length.")
    parser.add_argument("--feature-type", help="Optional exact GFF feature-type filter.")
    parser.add_argument("--consensus-seq", default=DEFAULT_MONOMER, help="Consensus for fixed-length normalization; default 178-bp CEN178.")
    parser.add_argument("--snp-model", choices=["cen178", "dong", "none"], default="cen178")
    parser.add_argument("--snp-rates", type=Path, help="Custom absolute per-position .npy rates.")
    parser.add_argument("--snp-spectrum", type=Path, help="Custom substitution-spectrum JSON.")
    parser.add_argument("--snp-background-rate", type=nonnegative_float, default=1e-10, help="Replacement for zero position rates.")
    for name in ["snp", "ins", "del", "large-del", "conversion"]:
        parser.add_argument(f"--{name}-rate", type=nonnegative_float, help="Rate per bp per generation; --snp-rate applies to the dong SNP model.")
    for name in ["ins-span-lambda", "del-span-lambda", "large-del-span-lambda"]:
        parser.add_argument(f"--{name}", type=nonnegative_float)
    parser.add_argument("--conversion-geometric-p", type=nonnegative_float)
    parser.add_argument("--min-array-bp", type=size_limit, default=None, help="Default none.")
    parser.add_argument("--max-array-bp", type=size_limit, default=100_000_000, help="Default 100000000; use none to disable.")
    return parser


def checkpoint_generations(text, final):
    if text is None:
        checkpoints, value = [0], 1
        while value <= final:
            checkpoints.append(value)
            value *= 10
    else:
        checkpoints = [int(value.strip()) for value in text.split(",")]
    if any(value < 0 or value > final for value in checkpoints):
        raise ValueError("Checkpoints must lie between 0 and --generations.")
    return sorted(set(checkpoints + [0, final]))


def write_result(output, result, metadata):
    sequences = [result.lookup[i] for i in range(len(result.lookup))]
    tagged = {i for i, seq in enumerate(sequences) if any(m in seq for m in MOTIFS)}
    arrays = {
        "generations": np.asarray([cp.generation for cp in result.checkpoints], dtype=np.int64),
        "total_monomers": np.asarray([len(cp.symbols) for cp in result.checkpoints], dtype=np.int64),
        "telobox_monomers": np.asarray([sum(s in tagged for s in cp.symbols) for cp in result.checkpoints], dtype=np.int64),
        "lookup_sequences": np.asarray(sequences, dtype=f"S{len(sequences[0])}"),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update({f"symbols_{i:06d}": np.asarray(cp.symbols, dtype=np.int64)
                   for i, cp in enumerate(result.checkpoints)})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".npz":
            raise ValueError("--output must end in .npz.")
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if args.seed >= 2**64:
            raise ValueError("--seed must fit an unsigned 64-bit integer.")
        if (args.fasta is None) != (args.gff is None):
            raise ValueError("Supply both --fasta and --gff.")
        fasta = args.fasta or example_data_path("CEN1.fasta")
        gff = args.gff or example_data_path("CEN1.gff3")
        rates = args.snp_rates or example_data_path("CEN178_mu_per_position.npy")
        spectrum = args.snp_spectrum or example_data_path("cen_spectrum.json")
        end = args.end if args.end is not None else len(_read_fasta_record(fasta, args.chrom))
        checkpoints = checkpoint_generations(args.checkpoints, args.generations)
        state, actual_start, actual_end = centromere_state_from_gff_fixed(
            fasta, gff, args.chrom, args.start, end, args.consensus_seq,
            target_length=len(args.consensus_seq), feature_type=args.feature_type,
        )
        parameters = dong_parameters(
            snp_model=args.snp_model, snp_position_rates=str(rates),
            snp_spectrum=str(spectrum), snp_background_rate=args.snp_background_rate,
        )
        overrides = {"min_array_bp": args.min_array_bp, "max_array_bp": args.max_array_bp}
        for name in ["snp", "ins", "del", "large_del", "conversion"]:
            value = getattr(args, f"{name}_rate")
            if value is not None:
                overrides[f"{name}_rate_per_bp"] = value
        for name in ["ins_span_lambda", "del_span_lambda", "large_del_span_lambda", "conversion_geometric_p"]:
            value = getattr(args, name)
            if value is not None:
                overrides[name] = value
        parameters = replace(parameters, **overrides)
        backends = {"cpp-blocked": CppBlockedSymbolDongSimulator,
                    "cpp": CppSymbolicDongSimulator, "python": SymbolicDongSimulator}
        simulator = backends[args.backend](state=state, parameters=parameters, seed=args.seed)
        result = simulator.run(n_generations=args.generations, checkpoints=checkpoints)
        inputs = {"fasta": fasta, "gff": gff}
        if args.snp_model == "cen178":
            inputs.update(snp_rates=rates, snp_spectrum=spectrum)
        metadata = {
            "schema_version": 1, "teloevo_version": __version__, "backend": args.backend,
            "seed": args.seed, "n_generations": args.generations, "parameters": asdict(parameters),
            "monomer_length_bp": state.monomer_length, "consensus_sequence": args.consensus_seq.upper(),
            "chrom": args.chrom, "requested_interval_1based_inclusive": [args.start, end],
            "actual_interval_1based_inclusive": [actual_start, actual_end],
            "inputs": {name: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in inputs.items()},
        }
        write_result(args.output, result, metadata)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved {len(result.checkpoints)} checkpoints through generation {args.generations:,} to {args.output}")

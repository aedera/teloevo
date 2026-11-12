"""Export one saved symbolic checkpoint as FASTA and GFF3.

Requires only Python and NumPy. This file can also be run directly, without
installing teloevo or importing its simulation code. The NPZ must contain
generations, lookup_sequences and the selected symbols_NNNNNN array.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def export_checkpoint(input_path, output_prefix, generation=None, seqid="centromere"):
    """Write a checkpoint's ordered DNA and satellite boundaries; return both paths.

    By default, export the highest saved generation. Coordinates are 1-based
    and inclusive within the reconstructed array. SymbolID identifies a DNA
    sequence in the input lookup, not an ancestral satellite or genomic locus.
    Existing output files are never overwritten.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", seqid):
        raise ValueError("Sequence ID may contain only letters, digits, _, ., : and -.")
    fasta = Path(str(output_prefix) + ".fasta")
    gff = Path(str(output_prefix) + ".gff3")
    for path in (fasta, gff):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    data = np.load(input_path, allow_pickle=False)
    if not hasattr(data, "files"):
        raise ValueError("Input must be an NPZ trajectory containing checkpoint sequences.")
    with data:
        if "generations" not in data or "lookup_sequences" not in data:
            raise ValueError("Input must contain generations, lookup_sequences and checkpoint symbols; summary-only files cannot be exported.")
        generations = data["generations"]
        if (generations.ndim != 1 or not generations.size or generations.dtype.kind not in "iu"
                or np.any(generations < 0) or np.unique(generations).size != generations.size):
            raise ValueError("Generations must be a nonempty vector of distinct nonnegative integers.")
        if generation is None:
            generation = int(generations.max())
        indices = np.flatnonzero(generations == generation)
        if indices.size != 1:
            raise ValueError(f"Generation {generation} is not saved in the input.")
        key = f"symbols_{int(indices[0]):06d}"
        if key not in data:
            raise ValueError(f"Missing checkpoint array: {key}")
        symbols = data[key]
        lookup = data["lookup_sequences"]

    if lookup.ndim != 1 or not lookup.size or lookup.dtype.kind not in "SU":
        raise ValueError("lookup_sequences must be a nonempty vector of DNA strings.")
    if symbols.ndim != 1 or not symbols.size or symbols.dtype.kind not in "iu":
        raise ValueError("Checkpoint symbols must be a nonempty integer vector.")
    if np.any(symbols < 0) or np.any(symbols >= lookup.size):
        raise ValueError("Checkpoint contains a symbol outside the sequence lookup.")
    sequences = {}
    for symbol in np.unique(symbols):
        value = lookup[int(symbol)]
        sequence = value.decode("ascii") if isinstance(value, bytes) else str(value)
        if not sequence or set(sequence.upper()) - set("ACGT"):
            raise ValueError("Satellite sequences must be nonempty and contain only A/C/G/T.")
        sequences[int(symbol)] = sequence
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError("Satellite sequences must all have the same length.")
    monomer_length = lengths.pop()
    total_length = int(symbols.size) * monomer_length

    fasta.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with fasta.open("x", encoding="ascii", newline="\n") as fa:
            created.append(fasta)
            with gff.open("x", encoding="ascii", newline="\n") as annotation:
                created.append(gff)
                fa.write(f">{seqid} generation={generation}\n")
                annotation.write(f"##gff-version 3\n##sequence-region {seqid} 1 {total_length}\n")
                annotation.write(f"# Generation: {generation}; coordinates refer to the simulated array.\n")
                buffer = ""
                for index, symbol in enumerate(symbols):
                    buffer += sequences[int(symbol)]
                    while len(buffer) >= 80:
                        fa.write(buffer[:80] + "\n")
                        buffer = buffer[80:]
                    start = index * monomer_length + 1
                    end = (index + 1) * monomer_length
                    attributes = f"ID=satellite_{index + 1};SymbolID={int(symbol)};Length={monomer_length}"
                    annotation.write(f"{seqid}\tteloevo\tsatellite_DNA\t{start}\t{end}\t.\t+\t.\t{attributes}\n")
                if buffer:
                    fa.write(buffer + "\n")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return fasta, gff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved trajectory NPZ.")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Write PREFIX.fasta and PREFIX.gff3.")
    parser.add_argument("--generation", type=int, help="Exact saved generation; default last saved generation.")
    parser.add_argument("--seqid", default="centromere", help="Record name in both outputs; default centromere.")
    args = parser.parse_args(argv)
    try:
        fasta, gff = export_checkpoint(args.input, args.output_prefix, args.generation, args.seqid)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {fasta} and {gff}")


if __name__ == "__main__":
    main()


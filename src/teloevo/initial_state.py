"""Build symbolic centromere states from a reference FASTA and GFF/GFF3.

Coordinates supplied to centromere_state_from_gff follow the GFF convention:
1-based and inclusive. The interval selects annotated monomer features; selected
features are extracted in full rather than clipped to the requested interval.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TextIO

from .symbolic_centromere import CentromereState


def _open_text(path: str | Path) -> TextIO:
    path = Path(path)

    if path.suffix == ".gz":
        return gzip.open(path, "rt")

    return path.open("r")


def _reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return sequence.translate(table)[::-1]


def _read_fasta_record(
    fasta_path: str | Path,
    chrom: str,
) -> str:
    """Read one named FASTA record into memory."""
    sequence_parts: list[str] = []
    found = False

    with _open_text(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                record_name = line[1:].split(None, 1)[0]

                if found:
                    break

                found = record_name == chrom
                continue

            if found:
                sequence_parts.append(line)

    if not found:
        raise ValueError(
            f"Chromosome/record {chrom!r} was not found in FASTA "
            f"{str(fasta_path)!r}."
        )

    sequence = "".join(sequence_parts).upper()

    if not sequence:
        raise ValueError(
            f"FASTA record {chrom!r} is empty."
        )

    return sequence


def centromere_state_from_gff(
    fasta_path: str | Path,
    gff_path: str | Path,
    chrom: str,
    start: int | None = None,
    end: int | None = None,
    feature_type: str | None = None,
) -> CentromereState:
    """Create a CentromereState from annotated monomer features.

    Parameters
    ----------
    fasta_path
        Reference FASTA. Plain text and .gz files are supported.
    gff_path
        GFF/GFF3 containing monomer annotations. Plain text and .gz files
        are supported.
    chrom
        GFF seqid / FASTA record name.
    start, end
        Optional 1-based inclusive genomic interval. Any feature overlapping
        the interval is selected, and the complete feature sequence is used.
    feature_type
        Optional exact match against GFF column 3.

    Notes
    -----
    Features are ordered by ascending genomic coordinate. Minus-strand
    features are reverse-complemented individually. The symbolic simulator
    requires all selected monomers to have exactly the same length; mixed
    lengths therefore raise ValueError.
    """
    if start is not None and start < 1:
        raise ValueError("start must be >= 1 (GFF coordinates are 1-based).")

    if end is not None and end < 1:
        raise ValueError("end must be >= 1 (GFF coordinates are 1-based).")

    if start is not None and end is not None and start > end:
        raise ValueError("start must be <= end.")

    reference = _read_fasta_record(
        fasta_path=fasta_path,
        chrom=chrom,
    )

    interval_start = 1 if start is None else int(start)
    interval_end = len(reference) if end is None else int(end)

    if interval_end > len(reference):
        raise ValueError(
            f"Requested end ({interval_end}) exceeds FASTA record "
            f"{chrom!r} length ({len(reference)})."
        )

    features: list[tuple[int, int, str]] = []

    with _open_text(gff_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip() or raw_line.startswith("#"):
                continue

            fields = raw_line.rstrip("\n\r").split("\t")

            if len(fields) != 9:
                raise ValueError(
                    f"Expected 9 GFF columns at line {line_number}, "
                    f"found {len(fields)}."
                )

            seqid, _, gff_type, start_text, end_text, _, strand, _, _ = fields

            if seqid != chrom:
                continue

            if feature_type is not None and gff_type != feature_type:
                continue

            try:
                feature_start = int(start_text)
                feature_end = int(end_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid GFF coordinates at line {line_number}."
                ) from error

            if feature_start < 1 or feature_end < feature_start:
                raise ValueError(
                    f"Invalid GFF interval at line {line_number}: "
                    f"{feature_start}-{feature_end}."
                )

            if feature_end < interval_start or feature_start > interval_end:
                continue

            if feature_end > len(reference):
                raise ValueError(
                    f"GFF feature at line {line_number} ends at "
                    f"{feature_end}, beyond FASTA record {chrom!r} "
                    f"length {len(reference)}."
                )

            if strand not in {"+", "-", ".", "?"}:
                raise ValueError(
                    f"Unsupported GFF strand {strand!r} at line "
                    f"{line_number}."
                )

            features.append(
                (feature_start, feature_end, strand)
            )

    if not features:
        type_message = (
            ""
            if feature_type is None
            else f" with feature type {feature_type!r}"
        )

        raise ValueError(
            f"No GFF features found for {chrom!r}{type_message} "
            f"overlapping {interval_start}-{interval_end}."
        )

    features.sort(key=lambda item: (item[0], item[1]))

    sequences: list[str] = []

    for feature_start, feature_end, strand in features:
        sequence = reference[feature_start - 1 : feature_end]

        if strand == "-":
            sequence = _reverse_complement(sequence)

        sequences.append(sequence)

    lengths = {len(sequence) for sequence in sequences}

    if len(lengths) != 1:
        counts: dict[int, int] = {}

        for sequence in sequences:
            length = len(sequence)
            counts[length] = counts.get(length, 0) + 1

        detail = ", ".join(
            f"{length} bp: {count}"
            for length, count in sorted(counts.items())
        )

        raise ValueError(
            "The symbolic simulator currently requires equal-length "
            f"monomers, but the selected GFF features contain: {detail}."
        )

    return CentromereState.from_sequences(sequences)


def centromere_state_from_gff_fixed(
    fasta_path: str | Path,
    gff_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    consensus: str,
    target_length: int = 178,
    feature_type: str | None = None,
) -> tuple[CentromereState, int, int]:
    """Create a fixed-length symbolic state from annotated monomers.

    start and end are 1-based inclusive genomic coordinates. They do not
    need to coincide with monomer boundaries. All annotated monomers
    overlapping the requested interval are selected in full, and the returned
    actual_start / actual_end are snapped to the outer boundaries of the
    selected monomers.

    Every selected monomer is first oriented 5'->3' relative to the supplied
    plus-strand consensus. Minus-strand features are therefore reverse-
    complemented before length normalization.

    Monomers longer than target_length are trimmed from their biological
    5' end. Shorter monomers are padded at their biological 3' end with the
    corresponding suffix of the plus-strand consensus.
    """
    if start < 1 or end < 1:
        raise ValueError(
            "start and end must be >= 1 "
            "(GFF coordinates are 1-based)."
        )

    if start > end:
        raise ValueError("start must be <= end.")

    if target_length <= 0:
        raise ValueError("target_length must be > 0.")

    consensus = consensus.upper()

    if len(consensus) != target_length:
        raise ValueError(
            "Consensus length must equal target_length "
            f"({target_length}); got {len(consensus)}."
        )

    if any(base not in "ACGT" for base in consensus):
        raise ValueError(
            "Consensus must contain only A/C/G/T."
        )

    reference = _read_fasta_record(
        fasta_path=fasta_path,
        chrom=chrom,
    )

    if end > len(reference):
        raise ValueError(
            f"Requested end ({end}) exceeds FASTA record "
            f"{chrom!r} length ({len(reference)})."
        )

    features: list[tuple[int, int, str]] = []

    with _open_text(gff_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip() or raw_line.startswith("#"):
                continue

            fields = raw_line.rstrip("\n\r").split("\t")

            if len(fields) != 9:
                raise ValueError(
                    f"Expected 9 GFF columns at line {line_number}, "
                    f"found {len(fields)}."
                )

            (
                seqid,
                _,
                gff_type,
                start_text,
                end_text,
                _,
                strand,
                _,
                _,
            ) = fields

            if seqid != chrom:
                continue

            if feature_type is not None and gff_type != feature_type:
                continue

            try:
                feature_start = int(start_text)
                feature_end = int(end_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid GFF coordinates at line {line_number}."
                ) from error

            if feature_start < 1 or feature_end < feature_start:
                raise ValueError(
                    f"Invalid GFF interval at line {line_number}: "
                    f"{feature_start}-{feature_end}."
                )

            if feature_end < start or feature_start > end:
                continue

            if feature_end > len(reference):
                raise ValueError(
                    f"GFF feature at line {line_number} ends at "
                    f"{feature_end}, beyond FASTA record {chrom!r} "
                    f"length {len(reference)}."
                )

            if strand not in {"+", "-"}:
                raise ValueError(
                    "Fixed-length initialization requires explicit '+' or "
                    f"'-' strand, but found {strand!r} at line {line_number}."
                )

            features.append(
                (feature_start, feature_end, strand)
            )

    if not features:
        type_message = (
            ""
            if feature_type is None
            else f" with feature type {feature_type!r}"
        )

        raise ValueError(
            f"No GFF features found for {chrom!r}{type_message} "
            f"overlapping {start}-{end}."
        )

    features.sort(key=lambda item: (item[0], item[1]))

    actual_start = features[0][0]
    actual_end = features[-1][1]

    sequences: list[str] = []

    for feature_start, feature_end, strand in features:
        sequence = reference[feature_start - 1 : feature_end]

        if strand == "-":
            sequence = _reverse_complement(sequence)

        if len(sequence) > target_length:
            sequence = sequence[-target_length:]
        elif len(sequence) < target_length:
            sequence = (
                sequence
                + consensus[len(sequence) : target_length]
            )

        sequences.append(sequence)

    state = CentromereState.from_sequences(sequences)

    return state, actual_start, actual_end

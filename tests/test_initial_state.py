from teloevo.initial_state import (
    centromere_state_from_gff,
    centromere_state_from_gff_fixed,
)


def _write_example_files(tmp_path):
    fasta = tmp_path / "ref.fa"
    gff = tmp_path / "monomers.gff3"

    fasta.write_text(
        ">Chr1\n"
        "AAAACCCCGGGGTTTT\n"
        ">Chr2\n"
        "ACGT\n"
    )

    gff.write_text(
        "##gff-version 3\n"
        "Chr1\ttest\tmonomer\t1\t4\t.\t+\t.\tID=m1\n"
        "Chr1\ttest\tother\t5\t8\t.\t+\t.\tID=x1\n"
        "Chr1\ttest\tmonomer\t5\t8\t.\t-\t.\tID=m2\n"
        "Chr1\ttest\tmonomer\t9\t12\t.\t+\t.\tID=m3\n"
        "Chr1\ttest\tmonomer\t13\t16\t.\t+\t.\tID=m4\n"
    )

    return fasta, gff


def test_centromere_state_from_gff_extracts_orders_and_orients(tmp_path):
    fasta, gff = _write_example_files(tmp_path)

    state = centromere_state_from_gff(
        fasta_path=fasta,
        gff_path=gff,
        chrom="Chr1",
        feature_type="monomer",
    )

    # m2 is CCCC on the reference but is annotated on the minus strand.
    assert state.lookup.as_dict() == {
        0: "AAAA",
        1: "GGGG",
        2: "TTTT",
    }
    assert state.symbols == [0, 1, 1, 2]
    assert state.n_monomers == 4
    assert state.monomer_length == 4


def test_centromere_state_from_gff_interval_selects_overlapping_full_features(tmp_path):
    fasta, gff = _write_example_files(tmp_path)

    state = centromere_state_from_gff(
        fasta_path=fasta,
        gff_path=gff,
        chrom="Chr1",
        start=6,
        end=10,
        feature_type="monomer",
    )

    # The interval overlaps m2 (5-8) and m3 (9-12); both full units are used.
    assert state.lookup.as_dict() == {0: "GGGG"}
    assert state.symbols == [0, 0]


def test_centromere_state_from_gff_rejects_mixed_lengths(tmp_path):
    fasta = tmp_path / "ref.fa"
    gff = tmp_path / "mixed.gff3"

    fasta.write_text(">Chr1\nAAAACCCCCGGGG\n")
    gff.write_text(
        "Chr1\ttest\tmonomer\t1\t4\t.\t+\t.\tID=m1\n"
        "Chr1\ttest\tmonomer\t5\t9\t.\t+\t.\tID=m2\n"
    )

    try:
        centromere_state_from_gff(
            fasta_path=fasta,
            gff_path=gff,
            chrom="Chr1",
            feature_type="monomer",
        )
    except ValueError as error:
        assert "equal-length monomers" in str(error)
    else:
        raise AssertionError("Expected mixed-length monomers to be rejected")


def _write_fixed_length_example_files(tmp_path):
    fasta = tmp_path / "fixed_ref.fa"
    gff = tmp_path / "fixed_monomers.gff3"

    fasta.write_text(
        ">Chr1\n"
        "GGAAAACCCACGTAATTT\n"
    )

    gff.write_text(
        "##gff-version 3\n"
        "Chr1\ttest\tmonomer\t1\t6\t.\t+\t.\tID=m1\n"
        "Chr1\ttest\tmonomer\t7\t9\t.\t+\t.\tID=m2\n"
        "Chr1\ttest\tmonomer\t10\t15\t.\t-\t.\tID=m3\n"
        "Chr1\ttest\tmonomer\t16\t18\t.\t-\t.\tID=m4\n"
    )

    return fasta, gff


def test_centromere_state_from_gff_fixed_snaps_orients_trims_and_pads(tmp_path):
    fasta, gff = _write_fixed_length_example_files(tmp_path)

    state, actual_start, actual_end = centromere_state_from_gff_fixed(
        fasta_path=fasta,
        gff_path=gff,
        chrom="Chr1",
        start=2,
        end=17,
        consensus="ACGT",
        target_length=4,
        feature_type="monomer",
    )

    assert (actual_start, actual_end) == (1, 18)
    assert state.monomer_length == 4
    assert state.symbols == [0, 1, 2, 3]
    assert state.lookup.as_dict() == {
        0: "AAAA",
        1: "CCCT",
        2: "ACGT",
        3: "AAAT",
    }


def test_centromere_state_from_gff_fixed_reuses_duplicate_symbols(tmp_path):
    fasta = tmp_path / "dup_ref.fa"
    gff = tmp_path / "dup_monomers.gff3"

    fasta.write_text(">Chr1\nGGAAAAAAAA\n")
    gff.write_text(
        "Chr1\ttest\tmonomer\t1\t6\t.\t+\t.\tID=m1\n"
        "Chr1\ttest\tmonomer\t7\t10\t.\t+\t.\tID=m2\n"
    )

    state, actual_start, actual_end = centromere_state_from_gff_fixed(
        fasta_path=fasta,
        gff_path=gff,
        chrom="Chr1",
        start=1,
        end=10,
        consensus="ACGT",
        target_length=4,
        feature_type="monomer",
    )

    assert (actual_start, actual_end) == (1, 10)
    assert state.lookup.as_dict() == {0: "AAAA"}
    assert state.symbols == [0, 0]


def test_centromere_state_from_gff_fixed_validates_consensus_length(tmp_path):
    fasta, gff = _write_fixed_length_example_files(tmp_path)

    try:
        centromere_state_from_gff_fixed(
            fasta_path=fasta,
            gff_path=gff,
            chrom="Chr1",
            start=2,
            end=17,
            consensus="ACG",
            target_length=4,
            feature_type="monomer",
        )
    except ValueError as error:
        assert "Consensus length" in str(error)
    else:
        raise AssertionError("Expected consensus-length validation to fail")


def test_centromere_state_from_gff_fixed_rejects_no_overlap(tmp_path):
    fasta = tmp_path / "gap_ref.fa"
    gff = tmp_path / "gap_monomers.gff3"

    fasta.write_text(">Chr1\nAAAACCCC\n")
    gff.write_text(
        "Chr1\ttest\tmonomer\t1\t4\t.\t+\t.\tID=m1\n"
    )

    try:
        centromere_state_from_gff_fixed(
            fasta_path=fasta,
            gff_path=gff,
            chrom="Chr1",
            start=5,
            end=8,
            consensus="ACGT",
            target_length=4,
            feature_type="monomer",
        )
    except ValueError as error:
        assert "No GFF features found" in str(error)
    else:
        raise AssertionError("Expected no-overlap selection to fail")

"""Access the self-contained real CEN1 example and CEN178 SNP inputs."""

from dataclasses import replace
from pathlib import Path

from .initial_state import _read_fasta_record, centromere_state_from_gff_fixed
from .symbolic_centromere import DEFAULT_MONOMER, dong_parameters


def example_data_path(name: str) -> Path:
    """Return a bundled data file; paths are independent of the working directory."""
    if Path(name).name != name:
        raise ValueError("Supply a bundled filename, without directory components.")
    path = Path(__file__).resolve().parent / "data" / name
    if not path.is_file():
        raise FileNotFoundError(f"Bundled data file not found: {name}")
    return path


def load_example_centromere():
    """Load real CEN1 in annotation order, normalizing every satellite to 178 bp."""
    fasta = example_data_path("CEN1.fasta")
    sequence = _read_fasta_record(fasta, "CEN1")
    state, _, _ = centromere_state_from_gff_fixed(
        fasta_path=fasta,
        gff_path=example_data_path("CEN1.gff3"),
        chrom="CEN1",
        start=1,
        end=len(sequence),
        consensus=DEFAULT_MONOMER,
        target_length=178,
        feature_type="centromere",
    )
    return state


def cen178_parameters(**overrides):
    """Dong structural parameters, bundled CEN178 SNP model and no lower bound."""
    parameters = dong_parameters(
        snp_model="cen178",
        snp_position_rates=str(example_data_path("CEN178_mu_per_position.npy")),
        snp_spectrum=str(example_data_path("cen_spectrum.json")),
    )
    return replace(parameters, **dict({"min_array_bp": None}, **overrides))


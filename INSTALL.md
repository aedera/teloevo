# Installation

Requires Python 3.9 or newer. The default C++ backend also requires a C++17
compiler (`g++`, or `clang++` selected with `CXX=clang++`). On Debian/Ubuntu,
the system packages are `python3-venv` and `g++`. The Python backend does
not need a compiler. NumPy is the only Python runtime dependency.

From the downloaded `teloevo` directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The C++ source, CEN1 FASTA/GFF3, consensus, SNP rate vector and substitution
spectrum are included in the installed package. A compiler is invoked on
first use and its executable is cached in the system temporary directory;
that directory must allow executable files. No genome download, FASTA index,
GNU Make, pandas or SciPy is required.

Run the bundled real CEN1 example from any working directory:

```bash
teloevo --generations 1000 --seed 846 --output cen1_example.npz
```

Alternatively use `python -m teloevo` with the same arguments. To select the
Python backend, add `--backend python`. Run `teloevo --help` for the input,
mutation-rate, size-limit and checkpoint options. Output files are never
overwritten.

The example retains the reference's existing teloboxes. It normalizes each
annotated satellite to 178 bp and uses the bundled position-specific CEN178
SNP rates, the `1e-10` replacement for zero rates, Dong structural mutation
parameters, a 100-Mb maximum and no lower array-size limit. Raw reference
length is 2,287,856 bp; the 12,746 normalized satellites total 2,268,788 bp.
The package data's `provenance.json` records extraction coordinates and hashes.

Inspect the saved summary and available arrays:

```bash
python -c 'import numpy as np; d = np.load("cen1_example.npz", allow_pickle=False); print(d.files); print(d["generations"]); print(d["total_monomers"]); print(d["telobox_monomers"])'
```

The NPZ contains checkpoint generations, satellite counts, telobox-bearing
satellite counts, a shared sequence lookup, ordered `symbols_000000` arrays
for each checkpoint, and a JSON manifest. It contains no pickled objects.

Export a saved checkpoint as DNA FASTA and satellite GFF3:

```bash
teloevo-export cen1_example.npz --generation 1000 --output-prefix cen1_gen1000 --seqid CEN1
```

This writes `cen1_gen1000.fasta` and `cen1_gen1000.gff3`. Omit `--generation`
to export the last saved generation. The GFF3 contains one `satellite_DNA`
feature per satellite, with 1-based inclusive coordinates, satellite ID, sequence SymbolID
and length. SymbolID identifies a lookup sequence, not an ancestral satellite.
Coordinates refer to the reconstructed simulated array; original genomic
coordinates and omitted intervening sequences are not restored. Existing
outputs are never overwritten. Only saved checkpoints can be exported;
summary-only simulation files do not contain the necessary sequences.

The exporter is independent of the simulation code and requires only NumPy.
It also works as `python -m teloevo.export`, or directly as
`python src/teloevo/export.py` with the same arguments, without installation.
It does not compile the C++ backend or rerun the simulation.

For development and verification:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Python and C++ implement the same model but use different random-number
generators; identical seeds are reproducible within the same backend and
software/toolchain environment, not across Python and C++.

No license has been assigned yet.


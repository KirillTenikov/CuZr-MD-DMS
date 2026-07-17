# Data and repository policy

This repository separates source code from scientific data products.

## Source repository

The Git source tree contains:

- Python and shell scripts;
- lightweight configuration files;
- documentation;
- environment descriptions;
- reusable analysis code;
- small metadata files required to understand or reproduce the workflow.

The source tree does not contain complete production trajectories or raw run
archives.

## Local working data

The local directory

```text
results/
```

is the working archive for raw calculations and compressed run packages. It is
intentionally ignored by Git.

This directory may contain:

- `.tar.gz` run archives;
- stress time series;
- LAMMPS trajectories;
- restart and data files;
- logs;
- model outputs;
- checksums and local provenance records.

The local `results/` directory must not be removed or reorganized as part of
routine Git cleanup. Git operations should affect only the repository index
and source tree, never the scientific archive stored on disk.

Generated run directories under `runs/` are also local and are not intended
for source control.

## GitHub Releases

Selected datasets intended for public distribution are uploaded as GitHub
Release assets.

A release archive should include, where applicable:

- the accepted raw stress series;
- trajectories needed for the published analysis;
- LAMMPS inputs and logs;
- metadata and protocol information;
- restart or final-state files needed for reproducibility;
- a manifest;
- SHA-256 checksums.

Release assets are immutable scientific records. A corrected dataset should
normally be published as a new release or clearly versioned replacement,
rather than silently modified.

## Postprocessing outputs

Generated analysis directories are local products and are not committed to
the source repository. Examples include:

```text
analysis_outputs/
postprocessing_outputs/
validation_outputs/
```

They may contain:

- fitted JSON and CSV files;
- diagnostic plots;
- aggregate tables;
- intermediate statistics;
- internal validation reports.

Numerical values and figures used in a publication should be regenerated from
the released raw data with the committed analysis script.

## Manuscripts and figures

Manuscript files and submission packages remain outside version control,
including:

```text
*.doc
*.docx
*.odt
*.rtf
*.pages
submission/
```

Generated local figures are also excluded unless a specific figure is
deliberately chosen as lightweight project documentation.

## Models and potentials

Large trained model files and production potential files are not committed to
the source tree unless there is a specific reproducibility reason and the file
size is appropriate for Git.

Public distribution should use an external archive or GitHub Release asset
with:

- an unambiguous filename;
- model or potential provenance;
- a SHA-256 checksum;
- the exact version used for production calculations.

## Recommended publication workflow

1. Keep active and archived runs in the local `results/` directory.
2. Commit only code, documentation, and lightweight configuration.
3. Run the committed postprocessing script on the accepted raw data.
4. Record the Git commit and input-file hashes used for the final analysis.
5. Package the selected raw data with a manifest and checksums.
6. Upload the package as a versioned GitHub Release asset.
7. Report the release identifier and repository commit in the manuscript.

## Safety rule

Do not use broad cleanup commands on scientific-data directories.

Before any command that changes tracked raw files, verify:

```bash
git status --short
git diff --cached --name-status
```

When an old raw archive was historically committed and must be removed from
the source tree, first make and verify an external backup. Use an index-only
operation such as `git rm --cached` only after confirming that the local file
remains intact.

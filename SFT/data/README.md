# Dataset Layout

This repository does not ship training datasets.

Expected local layout:

- `dataset_info.json`: dataset registry consumed by the SFT scripts
- one subdirectory per dataset, containing local JSON/JSONL manifests and any
  small metadata files you keep in your own workspace

Large datasets, media files, and private annotations should stay out of git.
Point the adaptation scripts at your local data roots via CLI flags or
environment variables.

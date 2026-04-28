#!/usr/bin/env bash
set -euo pipefail


# Use this to copy the data from DRToM to Hadronization

TeV_folder="TeV13p0_110test"

SRC="/hepusers2/fuscomus/DRToM/Generation/LHEF/$TeV_folder"
DEST_BASE="/hepusers2/fuscomus/Hadronization/Generation/Storage/LHEF/"
DEST="$DEST_BASE/$TeV_folder"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "Copying contents of $SRC -> $DEST"

cp -a "$SRC/." "$DEST"

echo "Done."

#!/usr/bin/env bash
# Run a command inside the lpcjobqueue coffea apptainer container, with the
# lpcjobqueue .env active (built on first use).  Mirrors the bootstrap `shell`
# wrapper but runs non-interactively.
#
#   ./lpc_container.sh 'python scripts/run_zjet_skim.py --config configs/zjet_omnifold_skim_lpc.json'
#
# Requires (in this dir): .bashrc, .cmslpc-local-conf  (copied from the
# lpcjobqueue bootstrap area), and a valid grid proxy.

export COFFEA_IMAGE="${COFFEA_IMAGE:-coffeateam/coffea-dask-almalinux9:2025.9.0-py3.12}"
export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/x509up_u25128}"
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/etc/condor/config.d/01_cmslpc_interactive,/usr/local/bin/cmslpc-local-conf.py:/usr/local/bin/cmslpc-local-conf.py.orig,.cmslpc-local-conf:/usr/local/bin/cmslpc-local-conf.py

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

apptainer exec -B "${PWD}:/srv" --pwd /srv \
  "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE}" \
  bash -c "source /srv/.bashrc >/dev/null 2>&1; cd /srv; $*"

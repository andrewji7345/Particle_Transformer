#!/bin/bash

set -euo pipefail

mode="${1:---test}"
if [[ "${mode}" != "--test" && "${mode}" != "--full" ]]; then
    echo "Usage: $0 [--test|--full]" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

cmssw_src="$(readlink -f "${HOME}/nobackup/research/CMSSW_15_0_19/src")"
cmssw_dir="$(dirname "${cmssw_src}")"
cmssw_parent="$(dirname "${cmssw_dir}")"
cmssw_release="$(basename "${cmssw_dir}")"

cfg_relative="SuuAnalysis/ParticleTransformer/test/runParticleTransformerNtuplizer_cfg.py"
input_dir_relative="SuuAnalysis/ParticleTransformer/test/signalMCFiles"

tarball_name="CMSSW_15_0_19_particleTransformer.tgz"
local_tarball="${script_dir}/${tarball_name}"
eos_tarball_dir="/store/user/aji/condor_inputs"
eos_output_dir="/store/user/aji/rootfiles_particleTransformer"

if [[ "${cmssw_release}" != "CMSSW_15_0_19" ]]; then
    echo "ERROR: expected CMSSW_15_0_19, found ${cmssw_release}" >&2
    exit 2
fi

if [[ ! -f "${cmssw_src}/${cfg_relative}" ]]; then
    echo "ERROR: configuration not found:" >&2
    echo "  ${cmssw_src}/${cfg_relative}" >&2
    echo "Adjust cfg_relative in submit_ntuplizer_scan.sh if the path differs." >&2
    exit 3
fi

if [[ ! -d "${cmssw_src}/${input_dir_relative}" ]]; then
    echo "ERROR: input-list directory not found:" >&2
    echo "  ${cmssw_src}/${input_dir_relative}" >&2
    echo "Adjust input_dir_relative in both shell scripts if its actual location differs." >&2
    exit 4
fi

if ! voms-proxy-info --exists --valid 00:30 >/dev/null 2>&1; then
    echo "ERROR: no CMS proxy valid for at least 30 minutes." >&2
    echo "Run: voms-proxy-init --valid 192:00 -voms cms" >&2
    exit 5
fi

channels=("WbWb" "WbZt" "WbHt" "ZtZt" "HtHt" "HtZt")
masses=("4000_1000" "6000_2000" "8000_3000")

# Globally optimal ntuplizer configuration: pt120/ak6/ca4/th85.
pt=120
ak="0.6"
ca="0.4"
th="0.85"
ak_int=6
ca_int=4
th_int=85

if [[ "${mode}" == "--test" ]]; then
    channels=("WbWb")
    masses=("4000_1000")
fi

for channel in "${channels[@]}"; do
    for mass in "${masses[@]}"; do
        sample="${channel}_${mass}"
        input_list="${cmssw_src}/${input_dir_relative}/${sample}.txt"
        if [[ ! -s "${input_list}" ]]; then
            echo "ERROR: missing or empty input list: ${input_list}" >&2
            exit 6
        fi
    done
done

mkdir -p logs
: > scan_parameters.tsv

for channel in "${channels[@]}"; do
    for mass in "${masses[@]}"; do
        sample="${channel}_${mass}"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${sample}" "${pt}" "${ak}" "${ca}" "${th}" \
            "${ak_int}" "${ca_int}" "${th_int}" \
            >> scan_parameters.tsv
    done
done

job_count="$(wc -l < scan_parameters.tsv)"
echo "Prepared ${job_count} parameter points (${mode})."

echo "Building compact CMSSW tarball..."
tar \
    --exclude-vcs \
    --exclude-caches-all \
    --exclude='*.root' \
    --exclude='*/tmp/*' \
    --exclude='*/rootfiles_particleTransformer/*' \
    --exclude="${tarball_name}" \
    -czf "${local_tarball}" \
    -C "${cmssw_parent}" \
    "${cmssw_release}"

echo "Tarball size: $(du -h "${local_tarball}" | cut -f1)"

xrdfs root://cmseos.fnal.gov mkdir -p "${eos_tarball_dir}"
xrdfs root://cmseos.fnal.gov mkdir -p "${eos_output_dir}"

echo "Uploading CMSSW tarball to EOS..."
xrdcp --nopbar --force \
    "${local_tarball}" \
    "root://cmseos.fnal.gov/${eos_tarball_dir}/${tarball_name}"

rm -f "${local_tarball}"

echo "Submitting ${job_count} jobs..."
condor_submit submit_ntuplizer_scan.jdl

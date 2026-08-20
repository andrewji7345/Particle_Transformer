#!/bin/bash

for pt in $(seq 0 20 400); do
    echo "========================================"
    echo "Running jet pT cutoff = ${pt} GeV"
    echo "========================================"

    cmsRun ./SuuAnalysis/ParticleTransformer/test/runParticleTransformerNtuplizer_cfg.py jetPtCut=${pt} \
        outputRootFile=rootfiles_particleTransformer/WbWb_4000_1000_jetPtCut${pt}.root

    if [ $? -ne 0 ]; then
        echo "ERROR: cmsRun failed for pT = ${pt}"
        exit 1
    fi
done
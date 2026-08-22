# ParticleTransformer CMSLPC submission

Copy this directory anywhere under your CMSLPC `~/nobackup` area. Submit from
an AlmaLinux 9 interactive node so the jobs automatically use EL9 containers.

The worker nodes do not mount `~/nobackup` or `/uscms_data`. `submit_ntuplizer_scan.sh`
therefore creates a compact tarball of `CMSSW_15_0_19`, uploads it once to EOS,
and each worker downloads and unpacks that tarball in its local scratch area.

## Before submission

```bash
cd ~/nobackup/research/CMSSW_15_0_19/src
cmsenv
scram b -j 8

voms-proxy-init --valid 192:00 -voms cms
```

Check these two paths in `submit_ntuplizer_scan.sh` if the path in your CMSSW
area differs:

```text
SuuAnalysis/ParticleTransformer/test/runParticleTransformerNtuplizer_cfg.py
SuuAnalysis/ParticleTransformer/test/signalMCFiles
```

The second path follows the location given with this request. Both paths are
validated before the tarball is made or any jobs are submitted.

## Submit one test point first

```bash
chmod +x run_ntuplizer_scan.sh submit_ntuplizer_scan.sh
./submit_ntuplizer_scan.sh --test
```

This submits only `WbWb_4000_1000`, `pT=120`, `AK=0.6`, `CA=0.4`, and
`cos(thrust)=0.85`. Verify it before launching the full scan:

```bash
condor_q
tail -f logs/job_*.out
xrdfs root://cmseos.fnal.gov ls /store/user/aji/rootfiles_particleTransformer
```

If it fails with the error:

```bash
ERROR: Can't read config source /storage/local/data1/condor/config.d/aji.config
```

Then make sure to populate the node-specific config and rerun:

```bash
touch /storage/local/data1/condor/config.d/aji.config
```

## Submit all samples

```bash
./submit_ntuplizer_scan.sh --full
```

The full submission contains 18 jobs: six decay channels (`WbWb`, `WbZt`,
`WbHt`, `ZtZt`, `HtHt`, and `HtZt`) at mass points `(4000,1000)`,
`(6000,2000)`, and `(8000,3000)`. Every job uses the globally optimal
configuration `pt120/ak6/ca4/th85`, corresponding to `pT=120`, `AK=0.6`,
`CA=0.4`, and `cos(thrust)=0.85`. The worker selects the configurable
`algorithmMode=packed`; the `slimmed` mode only permits its fixed
`pt300/ak8/ca8/th85` reference configuration.

Jobs that exit nonzero are held in Condor so failures remain visible in
`condor_q` rather than appearing as normally completed jobs.

## Outputs

Successful ROOT files are written directly to:

```text
/eos/uscms/store/user/aji/rootfiles_particleTransformer/
```

through the XRootD endpoint `root://cmseos.fnal.gov/`. The worker removes its
local ROOT file after a successful copy, preventing Condor from returning a
duplicate through NFS.

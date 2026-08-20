for pt in $(seq 100 20 400); do
    for ak in $(seq 0.4 0.2 1.6); do
        # Compute dynamic lower and upper bounds using bc
        ca_min=$(echo "$ak - 0.2" | bc)
        ca_min=$(echo "if ($ca_min < 0.4) $ca_min else 0.4" | bc)
        ca_max=$(echo "1.6" | bc)

        for ca in $(seq $ca_min 0.2 $ca_max); do
            # Standardize ca format to 1 decimal place to prevent floating point mismatch
            ca_formatted=$(printf "%.1f" $ca)
            
            # Calculate 10 * ak and 10 * ca and 10 * th as integers
            ak_int=$(printf "%.0f" $(echo "$ak * 10" | bc))
            ca_int=$(printf "%.0f" $(echo "$ca * 10" | bc))
            th_int=$(printf "%.0f" $(echo "$th * 100" | bc))
            echo "========================================"
            echo "Running jet pT cutoff = ${pt} GeV, ak radius = ${ak}, ca radius = ${ca}, cos thrust = ${th}"
            echo "========================================"

            cmsRun ./SuuAnalysis/ParticleTransformer/test/runParticleTransformerNtuplizer_cfg.py \
                jetPtCut=${pt} \
                akRadius=${ak} \
                caRadius=${ca} \
                cosThrust=${th} \
                outputRootFile=rootfiles_particleTransformer/WbWb_4000_1000_jetPtCut${pt}_ak${ak_int}_ca${ca_int}_th${th_int}.root

            if [ $? -ne 0 ]; then
                echo "ERROR: cmsRun failed for pT = ${pt}, akRadius = ${ak}, caRadius = ${ca}, cosThrust = ${th}".
                exit 1
            fi
        done
    done
done
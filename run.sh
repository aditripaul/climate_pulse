# 0. Environment and credentials
make install-dev                    # venv, torch, deps, pytest
cp .env.example .env                # set DATA_OPENAQ_API_KEY

# 1. Start clean (the second command also deletes downloaded data; skip it to reuse the cache)
make clean-checkpoints clean-eval-artifacts
make clean-downloaded-datasets

# 2. Data
make download-data                  # NOAA, OpenAQ, GISTEMP, Climate TRACE
make download-nwm-data              # ERA5 subset (large)
make test                           # 13 tests

# 3. El Niño: five seeds, then the main run
make train-elnino-seeds             # writes elnino_seeds.csv and elnino_seed_selection.json
#   Set model.tft_use_year in configs/runtime_conf.yaml to the "tft_use_year"
#   value in runtime/eval_artifacts/run_results/elnino_seed_selection.json (manual step)
make train-elnino                   # main TFT + LSTM

# 4. Other models
make train-pollution
make train-climate                  # optional: not used in any reported result
make train-nwm                      # main weather model (about 2 h)
make export-tft-forecasts           # needs the main TFT from step 3
make nwm-ablation                   # trains the 3 comparison versions in parallel (about 5 h)

# 5. Artifacts, snapshot, documents
make gen-eval-artifacts
make save-run                       # timestamped snapshot in runtime/saved/
make sync-paper-artifacts           # points the paper at that snapshot
make gen-research-paper-pdf
make gen-patent-application-pdf     # copies runtime/eval_artifacts/ into docs/patent/artifacts/

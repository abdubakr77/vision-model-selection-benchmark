# datasets

One folder per scenario. Each contains a README (source, license, how to download, task definition) and a preparation script that builds the fixed split.

| Folder | Scenario | Dataset | License | Status |
|---|---|---|---|---|
| manufacturing_visa | Manufacturing defects | VisA | CC BY 4.0 (authors), confirm | Phase 1 |
| documents | Document understanding | TBD | check | Phase 3 |
| retail | Product recognition | TBD | check | Phase 3 |

Raw data, prepared images and generated split files (`bench/`) are git-ignored. Only scripts and READMEs are versioned; the split is reproducible from the script and seed 42.

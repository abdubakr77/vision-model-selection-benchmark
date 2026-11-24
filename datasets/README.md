# datasets

One folder per scenario. Each contains a README (source, license, where to put the raw files, task definition). The fixed split is built by the scenario's `02_data_prep` notebook.

| Folder | Scenario | Dataset | License | Status |
|---|---|---|---|---|
| manufacturing_visa | Manufacturing defects | VisA | CC BY 4.0 (authors), confirm | Phase 1 |
| documents | Document understanding | TBD | check | Phase 3 |
| retail | Product recognition | TBD | check | Phase 3 |

Raw data, prepared images and generated split files (`bench/`) are git-ignored. Only READMEs are versioned; the split is reproducible from the data prep notebook and seed 42.

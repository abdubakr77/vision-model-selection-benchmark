# Dataset research

## Selection criteria

Public, license allows publication in a company case study, a task shared by all model families, realistic enterprise relevance, and room for edge cases. Medical datasets are out of scope for this project.

## Manufacturing

| Dataset | License | Decision |
|---|---|---|
| VisA (Amazon) | CC BY 4.0 per authors' repo and AWS Open Data; anomalib docs list CC BY-NC-SA 4.0 | Selected. Confirm license with the company. |
| MVTec AD | CC BY-NC-SA 4.0, commercial use not allowed | Rejected for a published company case study |

VisA facts: 12 object types, 10,821 images (9,621 normal, 1,200 anomalous), image and pixel annotations. First subset: `pcb1` (1004 normal, 100 defective, counted on disk).

### pcb1 EDA summary

Full table in [notebooks/manufacturing/01_eda.ipynb](../notebooks/manufacturing/01_eda.ipynb) and [datasets/manufacturing_visa/README.md](../datasets/manufacturing_visa/README.md).

- Four defect types: melt 54, scratch 21, missing 20, bent 15 images (9 images have more than one).
- Masks store the defect type id as the pixel value (1 bent, 2 melt, 3 scratch, 4 missing), so they look black; read them with `> 0`.
- Defects are small: median 0.2% of the image, about a quarter of the defect images have a box under 32 x 32 px at 1024 wide, all of them `melt`.
- Board orientation is tied to the label in the raw data (most normal boards rotated 180 degrees, almost all defective boards upright). Left as is, supervised models can learn orientation instead of defects. Fixed in data prep by turning every used board upright (443 of 650 used images).
- Defects stand out by brightness more than by color (93% of defects), so grayscale input is viable.
- The official VisA split files were not used: `1cls` has no defects in train and the `2cls` files have no validation split.

## Documents

To be researched in Phase 3. Check license first.

## Retail

To be researched in Phase 3. Check license first.

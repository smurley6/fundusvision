# data/

Only `full_df.csv` (ODIR-5K metadata, 1.4 MB) is tracked. Everything else is downloaded:

```bash
fundusvision download odir      # -> data/odir5k/{full_df.csv, preprocessed_images/, ODIR-5K/}
fundusvision download vessel    # -> data/vessel_segmentation/{train,test}/{Original,Ground truth}/
```

Both need a Kaggle API token at `~/.kaggle/kaggle.json`
(Kaggle → Settings → API → *Create new token*).

| Folder | Source | Size |
|---|---|---|
| `odir5k/` | [andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) | ~1.9 GB |
| `vessel_segmentation/` | [nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation) | ~1.6 GB |

If the vessel archive unpacks with an extra top-level folder, the loader finds `train/` and
`test/` one level down automatically.

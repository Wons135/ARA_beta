# Bundled demo data

`reviews.jsonl` contains **1,024 original synthetic movie reviews of eight fictional
titles**, with zero, one, and multiple helpful votes. It is not an excerpt of Amazon
reviews and must not be used to assess model quality or reproduce thesis metrics.
The JSONL file is bundled; the demo never downloads review data.

All review text comes from the original sentence templates in
[`scripts/build_demo_sample.py`](../../scripts/build_demo_sample.py). There are no
real reviewer identities, product identifiers, or copied Amazon review sentences.
You may use, copy, modify, and redistribute these generated records and templates
for any purpose, with or without attribution. This permission applies to this
original synthetic material only, not to Amazon Reviews 2023 or model weights.

Rebuild with `python scripts/build_demo_sample.py` (standard library only, seed 42).
`manifest.json` records the SHA-256 checksum, row count, and vote distribution.
Labels are constructed during preprocessing using the existing thesis quantile
rules; the shared train/validation/test split uses `random_state=42`.

## Source and citation for the full research dataset

The field schema and the full thesis data source are **Amazon Reviews 2023**,
`Movies_and_TV`, McAuley Lab:

- [Dataset documentation and category downloads](https://amazon-reviews-2023.github.io/main.html)
- [Hugging Face dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [Paper](https://arxiv.org/abs/2403.03952)

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An
          and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

The dataset maintainers state that they cannot assign a licence to the Amazon
review text ([maintainer statement](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1)).
The submission therefore bundles synthetic text rather than claiming redistribution
rights to real reviews. Access to the full data remains separate.

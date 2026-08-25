# Public-code conventions used

The release layout follows three recurring MICCAI/CVPR practices:

1. The [MICCAI Reproducibility Checklist](https://github.com/JunMa11/MICCAI-Reproducibility-Checklist)
   separates environment, data preparation, training, testing, post-processing,
   ablation, and expected results, with an explicit path for non-shareable data.
2. The CVPR 2024 [C2P repository](https://github.com/javrtg/C2P) gives one
   environment setup and experiment-specific commands that write to a predictable
   `results` directory.
3. The CVPR 2024 [SnAG repository](https://github.com/fmu2/snag_release) records
   data layout and checksums, preserves experiment configuration, and prints the
   expected paper metrics next to evaluation commands.

Accordingly, this repository keeps the README short, provides one end-to-end
command, freezes dependencies, records expected metrics, separates restricted
data from public artifacts, and moves detailed audit material into `docs/`.

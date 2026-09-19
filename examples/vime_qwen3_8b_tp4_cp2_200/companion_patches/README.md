# CUDA companion source

Apply vime.patch to a separate clean VIME checkout at the base in sources.json, using git apply --check then git apply --index. The patch contains the H100 sampling, reference KL, weight export and train-offload changes used in the two-step validation. The tested commit is a local source identifier, not a claim of upstream availability. CUDA uses vLLM 0.16.0. ROCm has separate companion sources in the ROCm example.

Do not mix these framework revisions or apply the CUDA patch on top of the ROCm companion. Configure paths once in .rlk-profile.json. See docs/usage/h100-configurable.md for validation scope.

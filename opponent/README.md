# Opponent Baselines

This directory hosts every third-party baseline that we compare against. Each
subdirectory vendors the upstream GitHub project so we have a clean separation
between first-party code (`EfficientQAT/`) and baselines. Once network access is
available you can convert any of these folders into git submodules with the
usual `git submodule add <url> opponent/<name>` flow.

| Baseline | Path | Upstream |
| --- | --- | --- |
| BitDistiller | `opponent/BitDistiller` | https://github.com/TencentARC/BitDistiller |
| llm-awq | `opponent/llm_awq` | https://github.com/mit-han-lab/llm-awq |
| QuIP# | `opponent/quip_sharp` | https://github.com/Cornell-RelaxML/quip-sharp |
| QAttune recipe (torchtune) | `opponent/qattune` | https://github.com/facebookresearch/torchtune |
| AQLM | `opponent/llm_aqlm` | https://github.com/Vahe1994/AQLM |
| QTIP | `opponent/qtip` | https://github.com/Cornell-RelaxML/qtip |
| GPTQModel | `opponent/GPTQModel` | https://github.com/ModelCloud/GPTQModel |
| llm-compressor | `opponent/llm-compressor` | https://github.com/vllm-project/llm-compressor |

`opponent/GPTQModel` and `opponent/llm-compressor` already come in as git
submodules because they were part of the original project layout.

## Converting to submodules later

When GitHub access is available you can convert any opponent into a submodule:

```bash
git mv opponent/<name> /tmp/<name>
git submodule add https://github.com/<owner>/<repo>.git opponent/<name>
rsync -a /tmp/<name>/ opponent/<name>/
rm -rf /tmp/<name>
```

This keeps the local tweaks you might have while letting git track the remote
repository for future updates.

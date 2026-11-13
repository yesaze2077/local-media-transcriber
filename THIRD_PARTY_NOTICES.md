# Third-party notices and model licenses

Checked 2026-11-13 against publisher-owned repositories and model cards. [Machine-readable source checksums and exact revisions](docs/model-license-evidence.json) also include inspection of the official FunASR 1.4.26 wheel LICENSE, which is MIT. The project code and Codex Skill are MIT; this does **not** relicense dependency code, model weights, FFmpeg, input recordings or generated content. No model weights, third-party binaries or private media are distributed with this repository.

## Model weights

| Registered component | Publisher / checkpoint | Revision checked | Declared license and primary evidence |
|---|---|---|---|
| Paraformer Chinese | `funasr/paraformer-zh` | `d7811ee3ac581fbcfdeb37c98c6ba674028433dc` | Model card declares [Apache-2.0](https://huggingface.co/funasr/paraformer-zh/blob/d7811ee3ac581fbcfdeb37c98c6ba674028433dc/README.md). Mandarin and applicable Chinese/English mixed speech; no universal-language claim. |
| FSMN-VAD | `funasr/fsmn-vad` | `df20e6b30c653645fa4ff125cacfcabd1020a669` | Model card declares [Apache-2.0](https://huggingface.co/funasr/fsmn-vad/blob/df20e6b30c653645fa4ff125cacfcabd1020a669/README.md). |
| CT-Punc | `funasr/ct-punc` | `d0e55e2b8722a78b63705ff443d09c4f86e5d750` | Model card declares [Apache-2.0](https://huggingface.co/funasr/ct-punc/blob/d0e55e2b8722a78b63705ff443d09c4f86e5d750/README.md). Chinese/English punctuation. |
| CAM++ | `funasr/campplus` | `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` | Model card declares [Apache-2.0](https://huggingface.co/funasr/campplus/blob/e4b6ede7ce16997aff4ae69fbca1f0175e2afede/README.md). Speaker embeddings; clustering is an additional pipeline step. |
| SenseVoiceSmall | `FunAudioLLM/SenseVoiceSmall` | `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` | Model card declares [custom `model-license`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/README.md), linking the [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/6f52414c5730a1c368b4d164853582ee9689249a/MODEL_LICENSE). Applies separately from MIT code. |

The checked Hugging Face repositories do not include standalone LICENSE files. The first four entries reflect the publishers' card metadata; FunASR also publishes a general model agreement. This is an upstream licensing ambiguity rather than proof that every weight is Apache-2.0. Retain publisher, checkpoint and downloaded revision, inspect the downloaded files/card, and clarify applicable terms with the publisher before redistributing weights or making a licensing commitment. This project downloads weights for the user's own local installation and does not bundle them.

The custom SenseVoice weight agreement requires source/author/model attribution, contains termination and conduct provisions, describes use at the user's risk, and allows subsequent license revisions. It is not represented here as an OSI-approved license or as equivalent to MIT/Apache. The [SenseVoice code license](https://github.com/FunAudioLLM/SenseVoice/blob/main/LICENSE) does not replace its checkpoint license.

## Software and external tools

| Dependency | Version / scope | Publisher evidence |
|---|---|---|
| FunASR | `1.4.26` | [PyPI metadata](https://pypi.org/project/funasr/1.4.26/) declares MIT; [publisher LICENSE at checked commit](https://github.com/modelscope/FunASR/blob/6f52414c5730a1c368b4d164853582ee9689249a/LICENSE) is MIT. Preserve upstream copyright when distributing its code. |
| PyTorch / TorchAudio | `2.8.0` | BSD-style licenses in [PyTorch](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) and [TorchAudio](https://github.com/pytorch/audio/blob/v2.8.0/LICENSE). Bundled numerical/audio libraries have their own notices. |
| ModelScope | `1.40.3` | [Apache-2.0](https://github.com/modelscope/modelscope/blob/master/LICENSE). SDK code license does not set every hosted model's terms. |
| Hugging Face Hub | pinned by lockfile | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE). |
| jsonschema | pinned by lockfile | [MIT](https://github.com/python-jsonschema/jsonschema/blob/main/COPYING). |
| platformdirs | pinned by lockfile | [MIT](https://github.com/tox-dev/platformdirs/blob/main/LICENSE). |
| python-docx, optional DOCX export | pinned by lockfile | [MIT](https://github.com/python-openxml/python-docx/blob/master/LICENSE). Its `lxml` dependency includes additional notices. |
| uv, installation/build tooling | installed externally | [MIT OR Apache-2.0](https://github.com/astral-sh/uv#license). |
| pytest / Ruff / build tooling | development only | [pytest MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [Ruff MIT](https://github.com/astral-sh/ruff/blob/main/LICENSE), [build MIT](https://github.com/pypa/build/blob/main/LICENSE). |
| FFmpeg / FFprobe | external, not bundled | [FFmpeg legal page](https://ffmpeg.org/legal.html): LGPL by default; GPL/nonfree build switches change obligations. Inspect `ffmpeg -version` for your binary. |

The lockfiles identify the installed Python dependency graph. Regenerate the installed-package inventory with `.venv/bin/python docs/generate_dependency_licenses.py --checked-date YYYY-MM-DD`. It is recorded in `docs/dependency-licenses.json` when available; inspect distribution-level license files for transitive dependencies before redistributing a packaged runtime. An inventory is disclosure, not a blanket relicensing or full legal clearance. This source release instructs users to install dependencies from their original publishers.

## Excluded dependency: kaldiio and unsupported Kaldi archives

FunASR 1.4.26 declares `kaldiio>=2.17.0`, but the publisher's [2.18.1 LICENSE](https://github.com/nttcslab-sp/kaldiio/blob/v2.18.1/LICENSE) is an NTT evaluation agreement, not an open-source grant. The official PyPI wheel and source archives of both [2.18.0](https://pypi.org/project/kaldiio/2.18.0/) and [2.18.1](https://pypi.org/project/kaldiio/2.18.1/) contain the same LICENSE SHA256 `ef2ebae3d9f21f550c6db96cb91de36050d4a03c37831c5af091c4814b6bf98a`. It restricts use to internal evaluation and limits transfer/distribution. Older official Git tags, including [2.17.1](https://github.com/nttcslab-sp/kaldiio/blob/v2.17.1/LICENSE), [2.17.2](https://github.com/nttcslab-sp/kaldiio/blob/v2.17.2/LICENSE) and [2.16.0](https://github.com/nttcslab-sp/kaldiio/blob/v2.16.0/LICENSE), carry that same agreement. Some old source archives omit the LICENSE and declare UNKNOWN; omission is not evidence of open-source permission. Downgrading does not resolve this finding.

This release excludes `kaldiio` through uv's documented [`exclude-dependencies`](https://docs.astral.sh/uv/reference/settings/#exclude-dependencies) setting and the frozen lockfile. `core/compat.py` is original MIT project code, using only Python's standard library to satisfy the unused import with explicit unsupported-operation handlers. It copies no `kaldiio` implementation and provides no archive functionality. If a real `kaldiio` module is installed or loaded, the adapter refuses that environment. The supported NumPy waveform path never calls the disabled archive operations. Kaldi `.ark` and `.scp` inputs are unsupported.

**Use the repository's frozen uv installation for full ASR.** Standard `pip install .[asr]` does not honor `[tool.uv]`, would install the excluded dependency, and is not a supported full-runtime installation. The published wheel is a base CLI distribution, not a bundled ASR runtime. The third-party dependency metadata remains upstream metadata; this adapter does not grant new rights to upstream packages.

The installed-license scan found no additional evaluation/non-commercial restrictions among the recorded distributions after exclusion. This is a metadata/license-file review, not full legal clearance. Open-source reciprocal licenses still apply: for example `soxr` is LGPL-2.1-or-later, `certifi` is MPL-2.0 and `tqdm` uses MPL/MIT terms. Inspect their exact license files and bundled native-library notices before redistributing a binary runtime.

## Model download transport

Registry entries pin the Hugging Face checkpoint revision and weight LFS SHA256. By default the downloader gets large weight bytes from the publishers' official ModelScope mirrors using resumable ranges, and small configuration/vocabulary files from the pinned Hugging Face snapshot. Full-file hash mismatch refuses activation; resumed blocks are checked before reuse. `LMT_WEIGHT_SOURCE=hf` chooses direct Hugging Face weight transport. Mirror choice does not change the weights' publisher identity, fixed content or applicable license. No media or transcript is uploaded by either route.

## Test recordings

Tests generate disposable tones, silence, noise and locally synthesized speech, or use explicitly documented public samples. Synthetic speech validates execution and recognizable content; it does not establish accuracy on real interviews or independent human voices. Private recordings must stay outside version control. Test media and model caches are excluded by `.gitignore`.

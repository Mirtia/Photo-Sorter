# photo-sorter

[![Release](https://img.shields.io/github/v/release/mirtia/photo-sorter)](https://img.shields.io/github/v/release/mirtia/photo-sorter)
[![Build status](https://img.shields.io/github/actions/workflow/status/mirtia/photo-sorter/main.yml?branch=main)](https://github.com/mirtia/photo-sorter/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/mirtia/photo-sorter/branch/main/graph/badge.svg)](https://codecov.io/gh/mirtia/photo-sorter)
[![Commit activity](https://img.shields.io/github/commit-activity/m/mirtia/photo-sorter)](https://img.shields.io/github/commit-activity/m/mirtia/photo-sorter)
[![License](https://img.shields.io/github/license/mirtia/photo-sorter)](https://img.shields.io/github/license/mirtia/photo-sorter)

Sort, de-duplicate, enhance, and search a personal photo collection (for example an old external disk).

- **Github repository**: <https://github.com/mirtia/photo-sorter/>
- **Documentation** <https://mirtia.github.io/photo-sorter/>

## Install

```bash
make install
# or: uv sync
```

That installs the CLI as `photo-sorter`. For object detection, CLIP embeddings, and Elasticsearch:

```bash
uv sync --extra ml --extra search
# with a GPU, add the high-accuracy backends:
uv sync --extra ml --extra gpu --extra search
```

- `ml` — CLIP (`open-clip-torch`), Real-ESRGAN (`spandrel`), and YOLO (`ultralytics`) for search embeddings, ML upscaling, and fallback object labels.
- `gpu` — the high-accuracy neural backends: RF-DETR (`rfdetr`, Apache-2.0) for objects and InsightFace ArcFace (`insightface` + `onnxruntime-gpu`) for faces. Both run on CUDA when a GPU is present.
- `search` — Elasticsearch Python client.

The pipeline picks the best backend available and degrades gracefully:

- **Objects** — RF-DETR (`gpu` extra) → YOLO (`ml` extra) → colour/layout heuristics.
- **Faces** — InsightFace ArcFace R100 (`gpu` extra) → OpenCV YuNet + SFace (`ml`/base) → none.

Download the shared weights once (CLIP for embeddings, Real-ESRGAN for `--apply-upscaling ml`, YOLO + YuNet/SFace for the fallbacks) before running the pipeline:

```bash
photo-sorter pull-models
# optional: photo-sorter pull-models --yolo-model yolov8s.pt
```

Weights land in `~/.cache/photo-sorter/models/` (or `PHOTO_SORTER_MODEL_DIR`). RF-DETR and the InsightFace `antelopev2` pack download themselves on first use (into `~/.roboflow/models/` and `~/.cache/photo-sorter/insightface/` respectively). `metadata` / `run` / `--apply-upscaling ml` also fetch the shared weights on first use if you skip `pull-models`.

`--apply-color-correction` only lifts exposure and contrast. It does **not** grey-world white-balance, which was turning red photos green.

Without `ml`, metadata still runs using colour/layout heuristics and skips embeddings, and upscaling falls back to Lanczos.

```bash
photo-sorter --help
photo-sorter run --help
```

## Configuration file

Copy [`photo-sorter.example.yaml`](photo-sorter.example.yaml) to `photo-sorter.yaml` and edit the paths. Command-line flags override the file.

```bash
cp photo-sorter.example.yaml photo-sorter.yaml
# edit paths, enhancement, and search settings

photo-sorter run
photo-sorter --config photo-sorter.yaml run
photo-sorter -c ./photo-sorter.yaml sort
photo-sorter run --input /other/disk   # overrides paths.input from YAML
```

If `photo-sorter.yaml` (or `.photo-sorter.yaml`) is in the current directory, it is loaded automatically. Nested keys (`paths`, `dedup`, `enhance`, `search`) and flat keys (`input:`, `archive:`) are both accepted. For denoising/sharpening/upscaling, `true` means `medium`; use `light`, `medium`, `strong`, or `false` to skip. Upscaling also accepts `ml` (Real-ESRGAN).

## Layout

Face samples (required for the metadata / `run` commands). Either a folder per person or a single image named after the person:

```text
samples/
  John_Doe/
    face_1.jpg
    face_2.jpg
  Jane_Doe/
    face_1.jpg
  Myrto.jpg
```

The directory or file stem is the person label (`John_Doe` → `John Doe`, `Myrto.jpg` → `Myrto`). Put one or more clear face photos per person — several varied shots per identity (frontal, ¾, with/without glasses, indoor/outdoor) improve recognition, since a probe matches on the best cosine over all of a person's exemplars. A name is written into metadata only when that face is actually found in the picture. `face_count` / “visible faces” is the number of detected face boxes that did not match a sample; scenery is not counted.

### GPU backends (ArcFace faces + RF-DETR objects)

With the `gpu` extra installed, faces use **InsightFace `antelopev2`** (RetinaFace/SCRFD detection + ArcFace R100 512-d embeddings) and objects use **RF-DETR** (Apache-2.0) instead of OpenCV YuNet/SFace and YOLO. Both run on the CUDA execution provider automatically when a GPU is present, and fall back cleanly when a package or GPU is missing.

```bash
uv sync --extra ml --extra gpu     # onnxruntime-gpu + insightface + rfdetr
```

The CUDA 13 / cuDNN 9 libraries bundled with the `torch` wheel are preloaded automatically, so no system-wide CUDA install is required (measured: all three neural models fit in ~1.9 GB of VRAM together).

Tuning knobs (all optional environment variables):

- `PHOTO_SORTER_FORCE_CPU=1` — pin every model (torch and ONNX) to the CPU.
- `PHOTO_SORTER_RFDETR_MODEL` — RF-DETR size: `nano` | `small` | `medium` (default) | `base` | `large`.

A reference face is matched on the **best cosine over all of a person's exemplars**, so adding several varied photos per identity is the most reliable way to improve recognition. The ArcFace match threshold defaults to `0.35` (vs `0.40` for the SFace fallback).

A full `run` writes:

```text
backup.tar.gz
work/
  sorted/      # season_year/pictures and .../videos
  deduped/     # + manifest.jsonl (group_id, phash, kept, origin per picture)
  enhanced/    # pictures renamed to <md5>.jpg, + manifest.jsonl (hash -> origin)
  metadata/    # <md5>_metadata.json
  discarded/   # near-duplicates and unfixable motion/focus blur
```

`deduped/manifest.jsonl` and `enhanced/manifest.jsonl` are lineage stores. Because Phase 3 renames each picture to the MD5 of its enhanced bytes, the manifests are what let you trace `enhanced_hash → original_path → group_id` and find the near-duplicates dropped in favour of a kept shot (each row is one JSON object).

## Run the whole pipeline

With a config file (see above), this is enough:

```bash
photo-sorter run
```

Minimum without a config file (backup, sort, dedup, enhance with no extra techniques, metadata):

```bash
photo-sorter run \
  --input /path/to/photos \
  --archive ./backup.tar.gz \
  --output ./work \
  --discard ./work/discarded \
  --face-samples ./samples
```

Typical run with enhancement and verbose logs:

```bash
photo-sorter -v run \
  --input /path/to/photos \
  --archive ./backup.tar.gz \
  --output ./work \
  --discard ./work/discarded \
  --face-samples ./samples \
  --quality-threshold 0.75 \
  --apply-color-correction \
  --apply-denoising medium \
  --apply-sharpening light \
  --apply-artifact-removal \
  --apply-upscaling ml
```

Same thing plus indexing into Elasticsearch at the end:

```bash
photo-sorter run \
  --input /path/to/photos \
  --archive ./backup.tar.gz \
  --output ./work \
  --discard ./work/discarded \
  --face-samples ./samples \
  --apply-color-correction \
  --apply-denoising \
  --es-url http://localhost:9200 \
  --es-index photos
```

`--apply-denoising` and `--apply-sharpening` take `light`, `medium`, or `strong`. `--apply-upscaling` also accepts `ml` (Real-ESRGAN; needs `uv sync --extra ml` and `photo-sorter pull-models`). If you pass the flag with no value, the level is `medium` (Lanczos).

Pictures whose quality score is below `--quality-threshold` (default **0.75**) are enhanced. Pictures with motion streak or defocus blur are copied to `--discard` instead — those defects are not worth running the enhancer on.

Interactive duplicate picking opens a small window for each similar group: browse with the arrow keys, check the pictures to keep, and press OK. Closing the window keeps the largest file. If there is no display, a text prompt is used instead.

```bash
photo-sorter run \
  --input /path/to/photos \
  --archive ./backup.tar.gz \
  --output ./work \
  --discard ./work/discarded \
  --face-samples ./samples \
  --enable-manual-selection
```

Resume from a later stage when earlier output already exists (for example after adding a face sample). Phase names or numbers `0`–`5` both work:

```bash
photo-sorter run --from-phase metadata
photo-sorter run --from-phase 4          # same as metadata
photo-sorter run --from-phase enhance    # re-enhance + metadata (+ index if --es-url)
```

`--from-phase metadata` only needs `--output` (to find `work/enhanced`) and `--face-samples`. Earlier flags such as `--input` and `--archive` are not required.

To re-tag people without re-running object detection or CLIP, use `--faces-only`. That reads existing `work/metadata/*.json`, re-runs face detection + recognition (ArcFace when available, else YuNet/SFace), and patches `people`, `face_count`, `tags`, and `description`:

```bash
photo-sorter run --faces-only --output ./work --face-samples ./samples

photo-sorter metadata --faces-only \
  --input ./work/enhanced \
  --output ./work/metadata \
  --face-samples ./samples
```

## Run phases one at a time

Use this when you want to inspect an intermediate directory before the next step.

```bash
# Phase 0 — compress the original tree (not fed into later phases)
photo-sorter backup --input /path/to/photos --archive ./backup.tar.gz
photo-sorter backup --input /path/to/photos --archive ./backup.zip

# Phase 1 — PNG→JPG, flatten folders, bin into spring_2025/pictures and .../videos
photo-sorter sort --input /path/to/photos --output ./work/sorted

# Phase 2 — group similar pictures; keep the largest; copy discards aside
photo-sorter dedup \
  --input ./work/sorted \
  --output ./work/deduped \
  --discard ./work/discarded

photo-sorter dedup \
  --input ./work/sorted \
  --output ./work/deduped \
  --discard ./work/discarded \
  --enable-manual-selection

# Phase 3 — enhance low-quality pictures; rename every image to its MD5 hash
# Motion/focus blur is copied to --discard and skipped
photo-sorter enhance \
  --input ./work/deduped \
  --output ./work/enhanced \
  --discard ./work/discarded \
  --quality-threshold 0.75 \
  --apply-color-correction \
  --apply-denoising strong \
  --apply-sharpening \
  --apply-artifact-removal \
  --apply-upscaling ml

# Phase 4 — EXIF, objects, faces, scene caption, optional CLIP embedding → JSON
photo-sorter pull-models
photo-sorter metadata \
  --input ./work/enhanced \
  --output ./work/metadata \
  --face-samples ./samples

# Phase 5 — Elasticsearch (optional)
photo-sorter index \
  --metadata ./work/metadata \
  --images ./work/enhanced \
  --es-url http://localhost:9200 \
  --es-index photos
```

## Search

JSON on disk (no Elasticsearch). Free-text matches the generated caption (`description`), tags, people, and objects:

```bash
photo-sorter search "boat on the water" \
  --metadata ./work/metadata \
  --images ./work/enhanced

photo-sorter search \
  --metadata ./work/metadata \
  --person "John Doe" \
  --object boat \
  --from 2024-01-01 \
  --to 2025-12-31 \
  --limit 20
```

Elasticsearch (keyword filters, plus kNN when CLIP embeddings exist):

```bash
photo-sorter search "red hat on the beach" \
  --es-url http://localhost:9200 \
  --es-index photos \
  --person "Dimitris" \
  --object boat
```

`--person` and `--object` are exact filters. The free-text query is matched against descriptions, tags, and (with `ml`) CLIP vectors.

## Local Elasticsearch

Example single-node container for development only. `xpack.security.enabled=false` turns off login and TLS so `http://localhost:9200` works with no password. Do not use that on a machine reachable from the network.

```bash
docker run -d --name es -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  elasticsearch:8.17.0
```

With security left on, set `ELASTIC_PASSWORD` and pass credentials in `--es-url`, for example `http://elastic:yourpassword@localhost:9200`.

## Development

```bash
make install
make check
make test
```

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).

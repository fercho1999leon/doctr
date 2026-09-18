# Text detection

The sample training script was made to train text detection model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python references/detection/train.py db_resnet50 --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

### Device selection (CUDA, Apple Silicon MPS, CPU)

`--device` accepts a CUDA index (`0`), `cuda:N`, `mps` (Apple Silicon GPU) or `cpu`. Without it the script picks CUDA, then MPS, then CPU. `--amp` is only supported on CUDA.

```shell
# NVIDIA GPU
python references/detection/train.py db_resnet50 --train_path path/to/train --val_path path/to/val --device 0 --amp
# Apple Silicon (set the fallback so the few ops MPS lacks run on CPU)
PYTORCH_ENABLE_MPS_FALLBACK=1 python references/detection/train.py db_resnet50 --train_path path/to/train --val_path path/to/val --device mps
```

Every checkpoint `<name>.pt` is written together with a `<name>.json` sidecar (architecture, ordered class names, input size, target options, dataset hashes, git revision, versions and the full argument list). The inference and evaluation scripts below read the sidecar, so the class list never has to be typed by hand.

Alternatively, instead of providing local folders you can train directly on one or several built-in datasets, which are downloaded automatically. When several are passed, the first one is loaded and extended with the others:

```shell
python references/detection/train.py db_resnet50 --train_datasets FUNSD SVHN --val_datasets FUNSD --epochs 5
```

The available built-in datasets are `CORD`, `FUNSD`, `IC03`, `IIIT5K`, `SVHN`, `SVT` and `SynthText`. Note that they are single-class (`words`) and therefore cannot be used for multi-class training. For each split use either the local path or the built-in datasets (not both): `--train_path` or `--train_datasets`, and `--val_path` or `--val_datasets`.

### Multi-GPU support

We now use the built-in [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher to spawn your DDP workers. `torchrun` will set all the necessary environment variables (`LOCAL_RANK`, `RANK`, etc.) for you. Arguments are the same than the ones from single GPU, except:

- `--backend`: you can specify another `backend` for `DistributedDataParallel` if the default one is not available on
your operating system. Fastest one is `nccl` according to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

#### Key `torchrun` parameters

- `--nproc_per_node=<N>`
  Spawn `<N>` processes on the local machine (typically equal to the number of GPUs you want to use).
- `--nnodes=<M>`
  (Optional) Total number of nodes in your job. Default is 1.
- `--rdzv_backend`, `--rdzv_endpoint`, `--rdzv_id`
  (Optional) Rendezvous settings for multi-node jobs. See the [torchrun docs](https://pytorch.org/docs/stable/elastic/run.html) for details.

#### GPU selection

By default all visible GPUs will be used. To limit which GPUs participate, set the `CUDA_VISIBLE_DEVICES` environment variable **before** running `torchrun`. For example, to use only CUDA devices 0 and 2:

```shell
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --nproc_per_node=2 references/detection/train.py \
  db_resnet50 \
  --train_path path/to/train \
  --val_path   path/to/val \
  --epochs 5 \
  --backend nccl
  ```

## Data format

To train on your own data you need to provide both `train_path` and `val_path` arguments (or use the built-in datasets shown above).
Each path must lead to folder with 1 subfolder and 1 file:

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png
│   └── ...
└── labels.json
```

Each JSON file must be a dictionary, where the keys are the image file names and the value is a dictionary with 3 entries: `img_dimensions` (spatial shape of the image), `img_hash` (SHA256 of the image file), `polygons` (the set of 2D points forming the localization polygon).
The order of the points does not matter inside a polygon. Points are (x, y) absolutes coordinates.

labels.json

```shell
{
    "sample_img_01.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "theimagedumpmyhash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
     },
     "sample_img_02.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "thisisahash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
     }
     ...
}
```

If you want to train a model with multiple classes, you can use the following format where polygons is a dictionary where each key represents one class and has all the polygons representing that class.

labels.json

```shell
{
    "sample_img_01.png": {
        'img_dimensions': (900, 600),
        'img_hash': "theimagedumpmyhash",
        'polygons': {
            "class_name_1": [[[x10, y10], [x20, y20], [x30, y30], [x40, y40]], ...],
            "class_name_2": [[[x11, y11], [x21, y21], [x31, y31], [x41, y41]], ...]
        }
    },
    "sample_img_02.png": {
        'img_dimensions': (900, 600),
        'img_hash': "thisisahash",
        'polygons': {
            "class_name_1": [[[x12, y12], [x22, y22], [x32, y32], [x42, y42]], ...],
            "class_name_2": [[[x13, y13], [x23, y23], [x33, y33], [x43, y43]], ...]
        }
    },
    ...
}
```

Every class must appear in **every** `labels.json` (train and val) and, ideally, in every image entry: use an empty list for a class that has no box in a given image. The class → channel mapping is derived from the sorted set of class names, and the script aborts if train and val expose different classes.

By default a class without any box in an image is *ignored* by the loss (the image may simply not be annotated for it). When your annotations are exhaustive, i.e. "no box" means "this field is not on the page", pass `--exhaustive-labels` so the absence is learnt as background: this is what you want for semantic fields (KIE).

Two augmentation switches are useful for field detection: `--no-hflip` disables horizontal flips (mirrored text does not help when the classes are the fields of a form) and `--crop-scale-min` (default 0.75) controls how aggressive the random crop is; raise it if the audit shows large fields being cut.

## Field (KIE) detection from Google Document AI exports

The `convert_documentai.py`, `audit_crops.py`, `kie_inference.py` and `evaluate_fields.py` scripts turn a set of Document AI labelled documents (each JSON holding the page image in base64 plus the labelled entities) into a multi-class detection dataset, audit it, and extract/evaluate fields with the trained detector.

### 1. Convert and audit the annotations

```shell
python references/detection/convert_documentai.py \
  --input path/to/export_a path/to/export_b \
  --output data/fields --val-ratio 0.2 --seed 42 \
  --rename facha=fecha --drop numero_cuenta
```

- Decodes the embedded page image (format detected from the bytes), converts the normalised vertices to absolute pixels and writes `train/` and `val/` folders in the format above, with every class listed in every entry.
- Near-duplicate pages (perceptual hash within `--dup-threshold`) are grouped and never split across train/val; exact duplicates are reported.
- `--test-ratio 0.2` additionally carves a held-out `test/` split before train/val, so no near-duplicate of a test page is ever trained on. Always convert all your folders in one call: converting train and test exports separately cannot detect duplicates across them.
- The split is stratified on the presence of each class at the group level (`--folds K --fold i` gives a grouped K-fold instead, useful to estimate the variance on small datasets). With a few dozen documents treat `val/` as a **development** set, not as an independent test set.
- `manifest.json` (per split) keeps the provenance of every page and the annotated text of every box; `audit.json` lists boxes per class, multi-line and long texts, tiny boxes, cross-class overlaps and pages without boxes.

```shell
python references/detection/audit_crops.py --data data/fields/val
```

draws the ground-truth polygons on a few pages (`val/audit_viz/`) and runs the pretrained recognisers on the ground-truth crops, reporting the exact-match rate per class. It tells you, before any training, which fields cannot be read as a single crop (multi-line blocks, texts longer than PARSeq's 32 characters, characters missing from the recogniser vocabulary such as `ñ` with the default French vocab).

### 2. Train

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python references/detection/train.py db_resnet50 --pretrained \
  --train_path data/fields/train --val_path data/fields/val \
  --device mps -b 2 --epochs 60 --lr 1e-3 --sched cosine --no-hflip --exhaustive-labels \
  --early-stop --early-stop-epochs 10 --output_dir runs --name db_resnet50_fields
```

Run a short pilot (5 epochs) and look at the field-level report before launching the full run. To train on Google Colab, open `references/detection/colab_train.ipynb` (`File > Open notebook > GitHub`) and follow its cells; checkpoints are saved to Drive so a disconnected session can be resumed. On an NVIDIA machine use `--device 0 --amp -b 4` (or `torchrun` as above).

### 3. Extract fields

```shell
python references/detection/kie_inference.py --checkpoint runs/db_resnet50_fields.pt page.jpg --json out.json
```

The detector localises the field regions; the text is read by the standard `ocr_predictor` and its words are assigned to each region by their centre (`--reco-mode words`, default). This handles multi-line and long fields, which `kie_predictor` (one recognition per region, `--reco-mode kie`) cannot. The output is `{class: [{"value", "confidence", "detection_score", "geometry", "words"}]}`; a class without detection maps to an empty list. When every field occurs at most once per page, `--top-k-per-class 1` keeps only the best-scored region per class, which removes most false positives; `--bin-thresh` / `--box-thresh` override the detector thresholds (tune them with `evaluate_fields.py`).

### 4. Evaluate per field

```shell
python references/detection/evaluate_fields.py --checkpoint runs/db_resnet50_fields.pt --data data/fields/val \
  --required valor_transferido fecha cuenta_destino numero_comprobante
```

reports, per class and matched by name: detection precision/recall/F1 at `--iou` (default 0.5), the false-positive rate on documents where the field is absent, the exact match of the read text against the annotation (Unicode NFKC, upper-case, whitespace collapsed; currency symbols and separators removed for amounts) the **key match** (both texts reduced to the canonical value of the field: date -> `YYYY-MM-DD [HH:MM]`, amount -> `0.00`, receipt number without a `No.` prefix, masked account -> its digit suffix, name -> letters only; see `extract_key` in `field_utils.py`) and the share of documents where every required field is correct, as exact text and as canonical keys. The key match is the number that matters for a downstream system; the exact match mostly reflects formatting differences between the annotation and the OCR output. Masked accounts match by digit suffix and names by fuzzy similarity (one misread character is tolerated, a different person is not). `kie_inference.py` returns the same canonical value in the `normalized` entry of every field.

The report also sweeps a **minimum detection score per class** on the split and prints the values that maximise F1 (a field detected on a page where it is absent counts as a false positive). Apply them with `--min-score numero_control=0.62 ...` in both `evaluate_fields.py` and `kie_inference.py`; tune on `val`, then run `test` once. The report ends with a diagnosis and a list of recommendations derived from the metrics (train longer, tune thresholds, add data for a class, fine-tune the recogniser, ...). A Markdown and a JSON report are written next to the data.

The pretrained recognition weights use the French vocabulary, which lacks `ñ` and Spanish accents; fine-tuning a recogniser with `VOCABS["spanish"]` (see `references/recognition`) is the natural next step once detection is solid.

## Slack Logging with tqdm

To enable Slack logging using `tqdm`, you need to set the following environment variables:

- `TQDM_SLACK_TOKEN`: the Slack Bot Token
- `TQDM_SLACK_CHANNEL`: you can retrieve it using `Right Click on Channel > Copy > Copy link`. You should get something like `https://xxxxxx.slack.com/archives/yyyyyyyy`. Keep only the `yyyyyyyy` part.

You can follow this page on [how to create a Slack App](https://api.slack.com/quickstart).

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/detection/train.py --help
```

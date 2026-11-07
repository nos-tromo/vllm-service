# Face detection and grouping service (`faces`) — design

**Date:** 2026-11-07
**Status:** Approved
**Consumer:** Nextext (face stage; its own spec lives in the Nextext repo)

## Problem

Nextext is to find the faces in an uploaded video, group them per person within
that one video ("Person 1, 2 …", like its speaker labels) and export the best
picture of each person. Apps run no models in-process ("inference is shared,
never embedded"), so the models need a home in this repo, behind the router like
`/vad`, `/diarize` and `/clip/*`.

## Approach (decided)

- **Models:** YuNet for detection (box + 5 landmarks; `opencv/face_detection_yunet`,
  MIT, 232 KB ONNX) and SFace for the features that drive grouping
  (`opencv/face_recognition_sface`, Apache-2.0, 39 MB ONNX), both run through
  `opencv-python-headless` (`cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF`), CPU
  only. Rejected: InsightFace (SCRFD/ArcFace weights are licensed for
  non-commercial research only), YOLO-face (AGPL), a vision LLM (no reliable
  boxes).
- **A first-party FastAPI server**, `src/faces_server.py`, CPU in every shape
  (like `vad`), with the pure-NumPy clustering in `src/faces_cluster.py` (the
  `topics_fit`/`topics_server` split).
- **Embeddings never leave the process.** `/faces/group` returns one label per
  track and nothing else; embeddings are computed, clustered and dropped inside
  the request — the way `/diarize` returns speaker labels, not voiceprints.
  They are never logged, stored or returned.
- **JSON with base64 images, not multipart.** LiteLLM rebuilds a multipart body
  from a dict keyed by part name, so repeated `images` parts would collapse to
  the last one; JSON already passes the router in production (docint →
  `/clip/embed_image`). LiteLLM drops top-level keys named like its own
  parameters (`metadata`, `tags`, `id`, `timeout`, `headers`, …) and switches
  to streaming on `stream`, so the top-level keys are exactly `images`,
  `min_score` and `tracks`, and the server's models use `extra="ignore"`.
- **Weights from the `huggingface-cache` volume**, offline, at a pinned
  revision and a pinned file name (each repo holds fp32, int8 and int8bq
  variants). Only the `.onnx` files and `LICENSE` are needed; both repos also
  ship `example_outputs/` with photos of people, which the airgap transfer
  should leave behind.
- **Per-route revisions in the router audit.** One backend serves two models, so
  the audit callback learns a route → revision-variable map, checked before the
  host map.

## Wire contract

`POST /faces/detect`

```json
{"images": ["<base64 JPEG/PNG>", "... at most 8"], "min_score": 0.7}
```

```json
{"results": [
  {"faces": [{"box": [0.41, 0.22, 0.12, 0.21], "score": 0.97,
              "landmarks": [[0.44, 0.29], [0.50, 0.29], [0.47, 0.33], [0.45, 0.37], [0.49, 0.37]]}]},
  {"error": "undecodable"}
]}
```

One result per image, in order. Coordinates are normalized to `[0, 1]` of the
submitted image's width and height; landmarks are YuNet's order (right eye, left
eye, nose tip, right mouth corner, left mouth corner). An image whose long side
exceeds 1280 px is downscaled first. Detection runs at the image's own scale and
at half scale, and the two sets are merged by non-maximum suppression at IoU 0.4,
because YuNet finds faces of roughly 10–300 px.

`POST /faces/group`

```json
{"tracks": [
  {"samples": [3, 4, 5], "faces": [{"image": "<base64 JPEG>", "landmarks": [[x, y], "... 5, crop pixels"]}]}
]}
```

```json
{"labels": [0, 0, 1, -1]}
```

At most 500 tracks and 3 faces per track. Per face: SFace `alignCrop` +
`feature`, L2-normalized. Per track: drop faces whose cosine to the track's
medoid is below 0.3, average the rest, normalize. Tracks that share a sample
index were on screen at the same time and are never merged. Average-linkage
agglomerative clustering on cosine similarity, merging while the best pair is
at least `FACE_GROUP_THRESHOLD` (default 0.42 — stricter than OpenCV's 0.363
pair threshold, because merging two people is worse than splitting one).
Labels are numbered by each cluster's first track; a track with no decodable
face gets `-1`.

`GET /health` → `{"status": "ok", "detect_model": ..., "embed_model": ...}`.

Errors: 400 malformed payload or over a count limit; 413 when `Content-Length`
(or the body as read) exceeds 32 MiB; 422 when no image of a detect request
decodes; 503 with `Retry-After: 5` when the model lock is not free within 30 s.

## Configuration

| Variable | Default |
|---|---|
| `FACE_DETECT_MODEL` / `FACE_DETECT_MODEL_FILE` / `FACE_DETECT_MODEL_REVISION` | `opencv/face_detection_yunet` / `face_detection_yunet_2023mar.onnx` / unset |
| `FACE_EMBED_MODEL` / `FACE_EMBED_MODEL_FILE` / `FACE_EMBED_MODEL_REVISION` | `opencv/face_recognition_sface` / `face_recognition_sface_2021dec.onnx` / unset |
| `FACE_GROUP_THRESHOLD` | `0.42` |

## Shapes

Full stack: service `faces` (image `vllm-service-faces`), CPU, router pass-through
`/faces/detect` and `/faces/group` with `auth: true`. CPU shape:
`compose.faces-only.yaml` (image `vllm-service-faces-cpu`, alias `faces-only`,
dev port 8009), the ninth `-only` shape.

## Out of scope

Attributes of any kind (age, sex, emotion, origin), matching against a gallery
or across requests, any state kept between requests, `/metrics`.

# Object detection demo

An end-to-end walkthrough of serving an object detector with ReactantServer: it exports a torchvision
Faster R-CNN (`fasterrcnn_resnet50_fpn`, pretrained on COCO) into StableHLO bundles, serves them on a
single GPU, sends an image with `ReactantServerClient`, and draws the predicted boxes and COCO labels
back onto the image with CairoMakie.

## Run

```
julia --project=examples/object_detection -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/object_detection examples/object_detection/demo.jl
```

The first run exports the bundles into `examples/object_detection/bundles/` (skipped afterward),
downloads a test image, serves the model, runs inference, and writes annotated boxes to
`examples/object_detection/detections.jpg` (a JPEG, convenient to copy off a remote machine). Pass
your own image as the first argument:

```
julia --project=examples/object_detection examples/object_detection/demo.jl path/to/image.jpg
```

## Requirements

- **GPU:** the default uses the Reactant CUDA backend (device 0). For a GPU-free smoke test pass
  `--cpu`, which serves on the CPU backend (much slower).
- **Export stack:** step 1 runs `tools/convert_to_stablehlo.jl`, which needs a Python with
  `torch` / `torchax` / `torchvision` / `triton` wired into PythonCall (the same stack any model
  conversion needs). If this project's PythonCall is not set up for that, point the export step at a
  project that is via `DEMO_CONVERT_PROJECT=/path/to/convert/env`. Once `bundles/object_detector/`
  exists, the export step is skipped and no Python is needed for the serve/infer/draw steps.

## What the model predicts

COCO's 80 everyday object classes (person, bicycle, car, bus, dog, cat, ...). The served model bakes a
0.05 score threshold, 0.5 NMS, and up to 100 detections; the demo additionally only draws detections
scoring at least 0.5. Each output row is `[x1, y1, x2, y2, score, class]` with boxes in the 640×640
input pixel space and `class` a COCO id (1..90), mapped to a name via the table in `demo.jl`.

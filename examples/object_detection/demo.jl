# End-to-end object detection demo for ReactantServer.
#
#   1. Export a torchvision Faster R-CNN (pretrained COCO) into StableHLO bundles (first run only).
#   2. Start a single-GPU ReactantServer hosting the bundles.
#   3. Send an image with ReactantServerClient.
#   4. Draw the predicted boxes + COCO labels onto the image with CairoMakie -> detections.jpg.
#
# Run:
#   julia --project=examples/object_detection examples/object_detection/demo.jl [IMAGE_PATH] [--cpu]
#
# With no IMAGE_PATH the demo downloads a known object-rich photo. `--cpu` serves on the Reactant CPU
# backend (slow, no GPU needed) for a smoke test; the default is the CUDA backend (device 0).
#
# The export step (step 1) shells out to tools/convert_to_stablehlo.jl in a SEPARATE process so torch
# imports before Reactant (the converter's required order); this process never imports torch. The
# export needs a Python with torch/torchax/torchvision/triton wired into PythonCall — set
# DEMO_CONVERT_PROJECT to a Julia project that has that stack if this one does not. If the bundles
# already exist under ./bundles/, export is skipped.

using Downloads
using FileIO
using ImageTransformations: imresize
using Colors
using Sockets
using CairoMakie
using ReactantServer
using ReactantServerClient

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))
const MODEL = "object_detector"
const IMG_SIZE = 640
const DISPLAY_THRESH = 0.5
const DEFAULT_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

# torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"] (91 entries, index = class id;
# "N/A" are unused COCO ids). A detection's class id `c` (1..90) names COCO_LABELS[c + 1].
const COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

label_for(class_id::Integer) =
    (1 <= class_id + 1 <= length(COCO_LABELS)) ? COCO_LABELS[class_id + 1] : "class $class_id"

# --- 1. Export the bundles (skipped if already present) -----------------------------------------
function ensure_bundles(bundles_dir)
    if isdir(joinpath(bundles_dir, MODEL))
        @info "Bundles present, skipping export" dir = joinpath(bundles_dir, MODEL)
        return
    end
    convert_script = joinpath(REPO, "tools", "convert_to_stablehlo.jl")
    config = joinpath(HERE, "detector.convert.yaml")
    project = get(ENV, "DEMO_CONVERT_PROJECT", HERE)
    @info "Exporting bundles (first run; this builds the torchvision model and traces it)" project
    run(`julia --project=$project $convert_script $config --only $MODEL`)
end

# --- 2. Load + letterbox-free resize the image to the network's input ---------------------------
# Returns (input::Array{UInt8,4} in (W,H,C,1) layout for INPUT__0, display::Matrix{RGB} sized IMG_SIZE).
function load_image(path)
    img = RGB.(FileIO.load(path))                 # normalize Gray/RGBA -> RGB, indexed [y, x]
    img = imresize(img, (IMG_SIZE, IMG_SIZE))     # (H, W) = (640, 640); plain resize (no aspect pad)
    arr = Array{UInt8}(undef, IMG_SIZE, IMG_SIZE, 3, 1)   # (W, H, C, N): wire reverses to (1,3,H,W)
    @inbounds for y in 1:IMG_SIZE, x in 1:IMG_SIZE
        px = img[y, x]
        arr[x, y, 1, 1] = round(UInt8, clamp(float(red(px)) * 255, 0, 255))
        arr[x, y, 2, 1] = round(UInt8, clamp(float(green(px)) * 255, 0, 255))
        arr[x, y, 3, 1] = round(UInt8, clamp(float(blue(px)) * 255, 0, 255))
    end
    return arr, img
end

# --- 4. Parse OUTPUT__0 into detections ----------------------------------------------------------
# OUTPUT__0 is [x1, y1, x2, y2, score, class] per detection. The meta emits a (6, Ndet) array; detect
# orientation defensively in case the wire/manifest hands it back transposed.
function parse_detections(out)
    ndims(out) == 2 || (out = reshape(out, length(out) ÷ 6, 6))  # be lenient on a flat result
    dets = size(out, 1) == 6 ? out : permutedims(out)            # -> (6, Ndet)
    rows = NamedTuple[]
    for j in 1:size(dets, 2)
        x1, y1, x2, y2, score, cls = ntuple(i -> Float64(dets[i, j]), 6)
        push!(rows, (; x1, y1, x2, y2, score, class = round(Int, cls)))
    end
    return rows
end

# --- 5. Draw boxes + labels ----------------------------------------------------------------------
function draw(display_img, dets, outfile)
    fig = Figure(size = (IMG_SIZE, IMG_SIZE))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true)  # y down = image coords
    hidedecorations!(ax); hidespines!(ax)
    image!(ax, 0 .. IMG_SIZE, 0 .. IMG_SIZE, permutedims(display_img))  # permute to [x, y]
    shown = 0
    for d in dets
        d.score >= DISPLAY_THRESH || continue
        shown += 1
        lines!(ax, [d.x1, d.x2, d.x2, d.x1, d.x1], [d.y1, d.y1, d.y2, d.y2, d.y1];
               color = :red, linewidth = 2)
        text!(ax, d.x1 + 2, d.y1 + 2; text = "$(label_for(d.class)) $(round(d.score; digits = 2))",
              color = :yellow, fontsize = 14, align = (:left, :top))
    end
    # CairoMakie's save() only writes Cairo formats (png/svg/pdf/eps), so to emit a jpeg (or any other
    # ImageIO format) render to a pixel buffer and write it through FileIO. Works headless.
    if lowercase(splitext(outfile)[2]) in (".png", ".svg", ".pdf", ".eps")
        save(outfile, fig)
    else
        FileIO.save(outfile, RGB.(colorbuffer(fig)))   # RGB. drops alpha (jpeg has no alpha channel)
    end
    return shown
end

function main()
    args = ARGS
    use_cpu = "--cpu" in args
    image_arg = findfirst(a -> !startswith(a, "--"), args)
    bundles_dir = joinpath(HERE, "bundles")

    ensure_bundles(bundles_dir)

    image_path = if image_arg === nothing
        dest = joinpath(HERE, "test_image.jpg")
        isfile(dest) || (@info "Downloading test image" url = DEFAULT_IMAGE_URL;
                         Downloads.download(DEFAULT_IMAGE_URL, dest))
        dest
    else
        args[image_arg]
    end
    @info "Loading image" image_path
    input, display_img = load_image(image_path)

    backend = use_cpu ? ReactantServer.CPU_BACKEND : ReactantServer.CUDA_BACKEND
    sock = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(sock)[2]); close(sock)
    cfg = ReactantServer.ServerConfig([abspath(bundles_dir)], "",
        ReactantServer.RuntimeConfig(backend, 0, 0.9, true, true),
        ReactantServer.SchedulerConfig(30.0, 64, 30.0),
        ReactantServer.EndpointsConfig("127.0.0.1", port))

    @info "Starting server (compiles bundles before accepting traffic)" backend port
    srv = ReactantServer.serve(cfg; backend = ReactantServer.ReactantBackend(), blocking = false)
    kserve_init()
    try
        model = KServeModel("grpc://127.0.0.1:$port", MODEL; max_batch_size = 1)
        @info "Running inference"
        resp = infer_sync(model, [InferInput("INPUT__0", input)])
        out = InferOutput("OUTPUT__0", resp, Float32)
        @info "Raw output" size = size(out)
        dets = parse_detections(out)
        outfile = joinpath(HERE, "detections.jpg")
        shown = draw(display_img, dets, outfile)
        @info "Done" total_detections = length(dets) drawn = shown output = outfile
        for d in sort(dets; by = x -> -x.score)
            d.score >= DISPLAY_THRESH || continue
            println("  $(label_for(d.class))  score=$(round(d.score; digits=3))  " *
                    "box=($(round(d.x1)), $(round(d.y1)), $(round(d.x2)), $(round(d.y2)))")
        end
    finally
        kserve_shutdown()
        ReactantServer.stop!(srv)
    end
end

main()

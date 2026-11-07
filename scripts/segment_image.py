"""
Segment an image with SAM3 using one or more open-vocabulary text prompts.

SAM3 is promptable (text or geometric prompts), not a class-agnostic
"segment everything" model, so this script runs the model once per
text prompt and overlays all detections on a single output image.

Usage:
  python scripts/segment_image.py --image path/to/image.png
  python scripts/segment_image.py --image path/to/image.png \
      --prompts "face mask" "cup" "cotton swab" "card" "robot gripper" \
      --threshold 0.5 --output path/to/output.png

  # Discover all instances with a generic prompt, then name each one
  # automatically with zero-shot CLIP classification against a broad,
  # built-in vocabulary (no need to know the object names ahead of time):
  python scripts/segment_image.py --image path/to/image.png \
      --discover-prompt "object on the mat" --discover-threshold 0.35
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

NAMER_MODEL_NAME = "openai/clip-vit-large-patch14"

# Broad, generic vocabulary used for automatic zero-shot naming of
# discovered instances. Not scene-specific on purpose, so it generalizes
# across different images. Override/extend with --vocab.
DEFAULT_VOCAB = [
    "face mask", "surgical mask", "cup", "plastic cup", "bottle", "water bottle",
    "marker pen", "pen", "pen cap", "cotton swabs", "cotton swab package",
    "card", "id card", "paper", "document", "digital scale display", "scale",
    "plastic bag", "medical supplies", "bandage", "syringe", "glove",
    "chair", "table", "mat", "floor tile", "metal railing", "handle",
    "robot gripper", "robot arm", "drain cover", "box", "container",
    "phone", "keyboard", "mouse", "monitor", "cable", "wire", "tool",
    "screwdriver", "wrench", "bolt", "screw", "bag", "wallet", "key",
    "bottle cap", "lid", "sticker", "label", "tape", "scissors",
    "book", "notebook", "pencil", "remote control", "light switch",
    "shoe", "cloth", "towel", "tube", "jar", "can", "wheel",
]

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__), "..")

PROMPT_COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 120, 255),
    (255, 165, 0),
    (200, 0, 200),
    (0, 200, 200),
    (255, 0, 150),
    (150, 255, 0),
]

DEFAULT_PROMPTS = ["face mask", "cup", "cotton swab", "card", "robot gripper"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="One or more text prompts, each segmented independently",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <image>_segmented.png next to the input)",
    )
    parser.add_argument(
        "--discover-prompt",
        default=None,
        help=(
            "If set, first find every instance with this generic prompt "
            "(e.g. 'object on the mat'), then automatically name each "
            "instance via zero-shot CLIP classification against --vocab."
        ),
    )
    parser.add_argument(
        "--discover-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold used for --discover-prompt",
    )
    parser.add_argument(
        "--vocab",
        nargs="+",
        default=None,
        help="Override the built-in candidate vocabulary used for auto-naming",
    )
    return parser.parse_args()


def load_namer(device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(NAMER_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(NAMER_MODEL_NAME)
    return model, processor


def encode_vocab(clip_model, clip_processor, vocab, device):
    inputs = clip_processor(text=vocab, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs).pooler_output
    return feats / feats.norm(dim=-1, keepdim=True)


def extract_instance_crop(image: Image.Image, box, pad=20):
    """Crop around the box with generous padding for zero-shot context."""
    img_arr = np.array(image.convert("RGB"))
    h, w = img_arr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return Image.fromarray(img_arr[y0:y1, x0:x1])


def name_instance(clip_model, clip_processor, vocab, vocab_feats, device, crop: Image.Image):
    inputs = clip_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**inputs).pooler_output
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    sims = (img_feats @ vocab_feats.T).squeeze(0)
    best_idx = int(sims.argmax())
    return vocab[best_idx], float(sims[best_idx])


def overlay_detections(image: Image.Image, detections, alpha=0.45):
    """detections: list of (mask: np.ndarray[H,W] bool, box: xyxy, score, prompt, color)"""
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    for mask, _box, _score, _prompt, color in detections:
        color_arr = np.array(color, dtype=np.float32)
        mask = mask.astype(np.float32)[..., None]
        img_arr = img_arr * (1 - alpha * mask) + color_arr * (alpha * mask)
    out = Image.fromarray(img_arr.astype(np.uint8))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for _mask, box, score, prompt, color in detections:
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{prompt} {score:.2f}"
        text_pos = (x0 + 2, max(0, y0 - 12))
        draw.rectangle(
            [text_pos[0], text_pos[1], text_pos[0] + 7 * len(label), text_pos[1] + 11],
            fill=color,
        )
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)
    return out


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading SAM3 image model...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=device)

    image = Image.open(args.image).convert("RGB")

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    def run_prompt(state, prompt, threshold):
        processor.confidence_threshold = threshold
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        results = []
        for mask, box, score in zip(state["masks"], state["boxes"], state["scores"]):
            mask_np = mask.squeeze(0).float().cpu().numpy() > 0.5
            box_np = box.float().cpu().numpy().tolist()
            results.append((mask_np, box_np, float(score)))
        return results

    all_detections = []
    with autocast_ctx:
        state = processor.set_image(image)

        if args.discover_prompt:
            # Pass 1: find every instance generically.
            generic = run_prompt(state, args.discover_prompt, args.discover_threshold)
            print(f"Discovery prompt '{args.discover_prompt}': found {len(generic)} instance(s)")

            # Pass 2: zero-shot classify each instance crop against a generic
            # vocabulary to name it automatically.
            vocab = args.vocab or DEFAULT_VOCAB
            print(f"Loading naming model ({NAMER_MODEL_NAME})...")
            clip_model, clip_processor = load_namer(device)
            vocab_feats = encode_vocab(clip_model, clip_processor, vocab, device)

            for idx, (mask_np, box, score) in enumerate(generic):
                crop = extract_instance_crop(image, box)
                name, name_score = name_instance(
                    clip_model, clip_processor, vocab, vocab_feats, device, crop
                )
                color = PROMPT_COLORS[idx % len(PROMPT_COLORS)]
                print(
                    f"  instance {idx + 1}: score={score:.3f} -> '{name}' "
                    f"(name_sim={name_score:.3f}) box={box}"
                )
                all_detections.append((mask_np, box, score, name, color))
        else:
            for i, prompt in enumerate(args.prompts):
                color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
                detections = run_prompt(state, prompt, args.threshold)
                print(f"Prompt '{prompt}': found {len(detections)} detection(s)")
                for mask_np, box, score in detections:
                    print(f"  score={score:.3f} box={box}")
                    all_detections.append((mask_np, box, score, prompt, color))

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.image)
        output_path = f"{base}_segmented.png"

    result_image = overlay_detections(image, all_detections)
    result_image.save(output_path)
    print(f"Saved segmentation result to {output_path}")
    print(f"Total detections across all prompts: {len(all_detections)}")


if __name__ == "__main__":
    main()

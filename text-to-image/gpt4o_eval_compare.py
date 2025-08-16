#!/usr/bin/env python3
import os
import sys
import argparse
import base64
import csv
import json
from openai import OpenAI
import pandas as pd
from collections import Counter
import random

def encode_image(path: str) -> (str, str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {ext}")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return mime, data

def paired_image_paths(dir1: str, dir2: str):
    imgs1 = sorted([f for f in os.listdir(dir1)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    imgs2 = sorted([f for f in os.listdir(dir2)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if len(imgs1) != len(imgs2):
        sys.exit("Error: directories must contain the same number of images.")
    for a, b in zip(imgs1, imgs2):
        if a != b:
            sys.exit(f"Error: filename mismatch: {a} vs {b}")
    return [os.path.join(dir1, f) for f in imgs1], \
           [os.path.join(dir2, f) for f in imgs2]

def main():
    parser = argparse.ArgumentParser(
        description="Compare two sets of images via GPT-4o structured preferences per prompt."
    )
    parser.add_argument("folderA", help="Path to folder A of images")
    parser.add_argument("folderB", help="Path to folder B of images")
    parser.add_argument("--prompts_file", default=None,
                        help="Path to newline-separated prompts")
    parser.add_argument("--output", "-o", default="results.csv",
                        help="CSV output path (default: results.csv)")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Number of image pairs to evaluate")
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        sys.exit("Error: set OPENAI_API_KEY environment variable")

    imgsA, imgsB = paired_image_paths(args.folderA, args.folderB)
    if args.prompts_file:
        prompts_df = pd.read_csv(args.prompts_file)
        # import pdb; pdb.set_trace()
        prompts = [prompts_df.text[prompts_df.image_id.apply(str) == os.path.basename(path).replace('.png', '')].item() for path in imgsA]
        # import pdb; pdb.set_trace()
    else:
        prompts = [""] * len(imgsA)
    if len(prompts) != len(imgsA):
        sys.exit("Error: number of prompts must equal number of image pairs")
    print('First imageA', imgsA[0])
    print('First imageB', imgsB[0])
    print('First prompt', prompts[0])
    
    if args.num_images is not None:
        imgsA, imgsB = imgsA[: args.num_images], imgsB[: args.num_images]
        prompts = prompts[: args.num_images]

    # ─── Function definition (NO "strict" key!) ───────────────────────────
    functions = [
        {
            "name": "get_image_preference",
            "description": "Returns an explanation and a choice among A, B, or none",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Model’s explanatory text"
                    },
                    "choice": {
                        "type": "string",
                        "enum": ["A", "B", "none"],
                        "description": "Which image is preferred"
                    }
                },
                "required": ["explanation", "choice"],
                "additionalProperties": False
            }
        }
    ]
    # ────────────────────────────────────────────────────────────────────────

    with open(args.output, "w", newline="") as csvfile:
        fieldnames = ["prompt", "imageA", "imageB", "explanation", "choice"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for imgA_path, imgB_path, prompt in zip(imgsA, imgsB, prompts):
            # Random swap
            swapped = random.random() < 0.5
            origA, origB = imgA_path, imgB_path
            if swapped:
                imgA_path, imgB_path = imgB_path, imgA_path  # Swap

            # Encode
            mimeA, dataA = encode_image(imgA_path)
            mimeB, dataB = encode_image(imgB_path)

            text = (
                f"Given the prompt '{prompt}', which image do you prefer, "
                "Image A or Image B, considering factors like image details, "
                "quality, realism, and aesthetics?"
            ) if prompt else (
                "Which image do you prefer, Image A or Image B, "
                "considering factors like image details, quality, realism, and aesthetics?"
            )

            uriA = f"data:{mimeA};base64,{dataA}"
            uriB = f"data:{mimeB};base64,{dataB}"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the get_image_preference "
                        "function to structure your output."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": uriA}},
                        {"type": "image_url", "image_url": {"url": uriB}},
                    ],
                },
            ]

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=functions,
                function_call={"name": "get_image_preference"},
                n=1,
            )

            func_call = resp.choices[0].message.function_call
            args_obj = json.loads(func_call.arguments)
            explanation = args_obj["explanation"]
            choice = args_obj["choice"]

            if swapped:
                if choice == "A":
                    choice = "B"
                elif choice == "B":
                    choice = "A"

                explanation = explanation.replace(" A", " __TMP__").replace(" B", " A").replace(" __TMP__", " B")

            writer.writerow({
                "prompt": prompt,
                "imageA": os.path.basename(origA),
                "imageB": os.path.basename(origB),
                "explanation": explanation,
                "choice": choice
            })
            print(f"[{os.path.basename(imgA_path)}] → choice={choice}")

    print(f"\nDone! Results written to {args.output}")

    output_df = pd.read_csv(args.output)
    counter = Counter(output_df['choice'])
    total = sum(counter.values())
    for choice, count in counter.most_common():
        pct = count / total * 100
        print(f"{choice}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()

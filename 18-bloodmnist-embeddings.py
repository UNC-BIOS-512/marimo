# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "huggingface_hub",
#     "pillow",
#     "pandas",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    device
    return device, mo, model, processor, torch


@app.cell
def _(mo):
    import io
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    # BloodMNIST test split at 224 px (same size CLIP uses)
    _path = hf_hub_download(
        "danjacobellis/bloodmnist_224",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    _df = pd.read_parquet(_path)
    images = [
        Image.open(io.BytesIO(d["bytes"])).convert("RGB") for d in _df["image"]
    ]
    labels = [int(l[0]) for l in _df["label"]]
    mo.vstack(
        [
            mo.md(f"**{len(images)} images**, image size {images[0].size}"),
            mo.hstack([mo.image(im, width=96) for im in images[:8]]),
        ]
    )
    return images, labels, pd


@app.cell
def _(device, images, model, processor, torch):
    _chunks = []
    with torch.no_grad():
        for _i in range(0, len(images), 128):
            _inputs = processor(
                images=images[_i : _i + 128], return_tensors="pt"
            ).to(device)
            _chunks.append(
                model.get_image_features(**_inputs).pooler_output.cpu()
            )
    embeddings = torch.cat(_chunks).numpy()
    embeddings.shape
    return (embeddings,)


@app.cell
def _(embeddings, labels, mo, pd):
    import gzip

    _out = pd.DataFrame(
        embeddings,
        columns=[f"dim_{i + 1}" for i in range(embeddings.shape[1])],
    )
    _out.insert(0, "label", labels)
    _csv_gz = gzip.compress(
        _out.to_csv(index=False, float_format="%.5g").encode()
    )
    mo.download(
        data=_csv_gz,
        filename="bloodmnist_clip_embeddings.csv.gz",
        mimetype="application/gzip",
        label=f"Download embeddings (.csv.gz, {len(_csv_gz) / 1e6:.1f} MB)",
    )
    return


if __name__ == "__main__":
    app.run()

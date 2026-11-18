# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "pillow",
#     "requests",
#     "pandas",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load an encoder from Hugging Face

    We use **CLIP**, the same encoder as in lecture.
    Model card: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
    """)
    return


@app.cell
def _():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    sum(p.numel() for p in model.parameters())
    return model, processor, torch


@app.cell(hide_code=True)
def _(mo, model, torch):
    def _count(m):
        return sum(p.numel() for p in m.parameters())

    mo.ui.table(
        [
            {
                "part": "Image encoder (ViT, 12 layers)",
                "parameters": f"{_count(model.vision_model):,}",
            },
            {
                "part": "Text encoder (Transformer, 12 layers)",
                "parameters": f"{_count(model.text_model):,}",
            },
            {
                "part": "Image projection (768 → 512)",
                "parameters": f"{_count(model.visual_projection):,}",
            },
            {
                "part": "Text projection (512 → 512)",
                "parameters": f"{_count(model.text_projection):,}",
            },
            {"part": "Total", "parameters": f"{_count(model):,}"},
            {
                "part": "Device",
                "parameters": str(next(model.parameters()).device),
            },
            {
                "part": "GPU available?",
                "parameters": str(torch.cuda.is_available()),
            },
        ],
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(model, processor, torch):
    sentences = ["a photo of a cat", "a photo of a dog"]
    with torch.no_grad():
        text_inputs = processor(
            text=sentences, return_tensors="pt", padding=True
        )
        text_emb = model.get_text_features(**text_inputs).pooler_output
    text_emb.shape
    return sentences, text_emb


@app.cell
def _(model, processor, torch):
    import requests
    from PIL import Image

    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(img_url, stream=True).raw)
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt")
        image_emb = model.get_image_features(**image_inputs).pooler_output

    image_emb[0, :20]
    return Image, image, image_emb, requests


@app.cell(hide_code=True)
def _(image, image_emb, mo):
    mo.vstack(
        [
            mo.image(image, alt="two cats on a pink couch", width=350),
            mo.md(f"Image embedding shape: `{tuple(image_emb.shape)}`"),
        ],
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(F, image_emb, mo, sentences, text_emb):
    cat_photo_vec = image_emb[0]
    cat_text_vec = text_emb[0]  # "a photo of a cat"

    cosine = F.cosine_similarity(cat_photo_vec, cat_text_vec, dim=0).item()

    mo.vstack(
        [
            mo.md("### One photo, one sentence"),
            mo.md(
                f"**Photo embedding** (first 8 of {len(cat_photo_vec)}): `{cat_photo_vec[:8].detach().numpy().round(2)}`"
            ),
            mo.md(
                f'**Text embedding** for "{sentences[0]}" (first 8 of {len(cat_text_vec)}): `{cat_text_vec[:8].detach().numpy().round(2)}`'
            ),
            mo.md(f"**Cosine similarity: {cosine:.3f}**"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How was CLIP trained?

    CLIP saw hundreds of millions of (photo, caption) pairs from the web.
    Paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

    In one **batch** of N pairs, the net makes an N x N table of similarity scores.
    Photo 1 should match caption 1, photo 2 caption 2, and so on.
    The **diagonal** is right. Everything else is wrong.

    ### The loss, one photo at a time

    Take one photo. Look at its row of scores. Turn the scores into a share of 1:

    $$\text{share of right caption} = \frac{e^{\text{right score}}}{e^{\text{score 1}} + e^{\text{score 2}} + \dots + e^{\text{score N}}}$$

    If the right caption has a big score, its share is close to 1. If not, the share is small.

    The loss for this photo is:

    $$\text{loss} = -\log(\text{share of right caption})$$

    | share of right caption | loss |
    |---|---|
    | 1.0 (perfect) | 0 |
    | 0.5 | 0.69 |
    | 0.1 | 2.3 |
    | 0.01 | 4.6 |

    Low share, high loss. Training makes the loss smaller.
    We average over all photos in the batch. We do the same for captions.
    """)
    return


@app.cell
def _(Image, mo, requests):
    pairs = [
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "two cats sleeping on a pink couch",
        ),
        (
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "a big brown bear",
        ),
        (
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "a living room with a television",
        ),
        (
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "a bedroom with a bed and books",
        ),
    ]
    batch_images = [
        Image.open(requests.get(u, stream=True).raw).convert("RGB")
        for u, _ in pairs
    ]
    batch_captions = [c for _, c in pairs]
    mo.hstack(
        [
            mo.image(
                im.resize((160, 120)),
                alt=f"{i + 1}: {cap}",
                caption=f"{i + 1}: {cap}",
                width=160,
            )
            for i, (im, cap) in enumerate(zip(batch_images, batch_captions))
        ],
        align="start",
    )
    return batch_captions, batch_images, pairs


@app.cell
def _(batch_captions, batch_images, model, pairs, processor, torch):
    import torch.nn.functional as F

    with torch.no_grad():
        _inp = processor(
            text=batch_captions,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        _img = F.normalize(
            model.get_image_features(
                pixel_values=_inp["pixel_values"]
            ).pooler_output,
            dim=-1,
        )
        _txt = F.normalize(
            model.get_text_features(
                input_ids=_inp["input_ids"],
                attention_mask=_inp["attention_mask"],
            ).pooler_output,
            dim=-1,
        )
        logits = (
            model.logit_scale.exp() * _img @ _txt.T
        )  # rows = photos, columns = captions

    target = torch.arange(len(pairs))  # photo i goes with caption i
    loss = (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
    ) / 2
    loss.item()
    return F, logits, loss


@app.cell
def _(batch_captions, logits, loss, mo, pairs):
    import pandas as pd
    import altair as alt

    _df = pd.DataFrame(
        [
            (
                f"photo {i + 1}",
                f"{j + 1}: {batch_captions[j]}",
                float(logits[i, j]),
            )
            for i in range(len(pairs))
            for j in range(len(pairs))
        ],
        columns=["photo", "caption", "score"],
    )
    _chart = (
        alt.Chart(_df, width=250, height=250)
        .mark_rect()
        .encode(
            x="caption:N",
            y="photo:N",
            color="score:Q",
            tooltip=[
                "photo:N",
                "caption:N",
                alt.Tooltip("score:Q", format=".1f"),
            ],
        )
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                f"Contrastive loss: **{loss.item():.3f}**. The diagonal should be the brightest in each row."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    def _box(x, y, w, h, label, sub="", hot=False):
        fill = "rgba(234,88,12,0.25)" if hot else "rgba(100,116,139,0.18)"
        stroke = "#ea580c" if hot else "currentColor"
        sw = 2.5 if hot else 1
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        s += f'<text x="{x + w / 2}" y="{y + h / 2 - (4 if sub else -5)}" text-anchor="middle" font-size="14" font-weight="600" fill="currentColor">{label}</text>'
        if sub:
            s += f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">{sub}</text>'
        return s

    def _arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>'

    _svg = '<svg viewBox="0 0 900 290" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width:900px">'
    _svg += '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>'
    # image row
    _svg += _box(10, 30, 90, 60, "Photo")
    _svg += _box(150, 30, 160, 60, "Image encoder", "ViT, 12 layers")
    _svg += _box(360, 30, 130, 60, "768 numbers", "from the [CLS] token")
    _svg += _box(
        540, 30, 150, 60, "Linear 768 → 512", "projection layer", hot=True
    )
    _svg += (
        _arrow(100, 60, 150, 60)
        + _arrow(310, 60, 360, 60)
        + _arrow(490, 60, 540, 60)
    )
    # text row
    _svg += _box(10, 190, 90, 60, "Caption")
    _svg += _box(150, 190, 160, 60, "Text encoder", "Transformer, 12 layers")
    _svg += _box(360, 190, 130, 60, "512 numbers", "from the last token")
    _svg += _box(
        540, 190, 150, 60, "Linear 512 → 512", "projection layer", hot=True
    )
    _svg += (
        _arrow(100, 220, 150, 220)
        + _arrow(310, 220, 360, 220)
        + _arrow(490, 220, 540, 220)
    )
    # embeddings + loss
    _svg += _box(740, 30, 150, 60, "Image embedding", "512 numbers", hot=True)
    _svg += _box(740, 190, 150, 60, "Text embedding", "512 numbers", hot=True)
    _svg += _arrow(690, 60, 740, 60) + _arrow(690, 220, 740, 220)
    _svg += _arrow(815, 90, 815, 120) + _arrow(815, 190, 815, 160)
    _svg += '<text x="815" y="146" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">cosine similarity → loss</text>'
    _svg += "</svg>"

    mo.vstack(
        [
            mo.md("## Where do the embeddings come from?"),
            mo.Html(_svg),
            mo.md(
                """
            The embedding is the output of the last **linear projection layer** (orange). It is not the output of the encoder itself.

            - Both towers end in a projection layer. It squeezes each side to **512 numbers**.
            - The 512 numbers from the image and the 512 from the text live in the **same space**. That is why we can compare them.
            - The loss is computed on these 512 numbers. Training pushes matching pairs together.
            - When we "use CLIP as an encoder", we keep this output and throw away the loss.
            """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

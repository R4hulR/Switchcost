"""Export the pilot embedding model's transformer backbone to ONNX.

Pooling and normalization are deliberately NOT part of the exported graph --
they're applied in our own numpy code at serving time (scripts/common/pooling.py)
so the pilot's worker/thread experiments run against a fixed, portable graph
input/output shape, and so those pooling/normalization steps are a distinct,
independently-verified piece of the pipeline (docs/pilot_config_matrix.md §10).

Usage: python scripts/export_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx
import onnxruntime
import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
OPSET = 17
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
MAX_SEQ_LEN_FOR_TRACE = 32  # only affects the export-time example input; ONNX export uses dynamic axes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(REPO_ID)
    revision = info.sha
    print(f"Resolved {REPO_ID} -> commit {revision}")

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=revision)
    backbone = AutoModel.from_pretrained(REPO_ID, revision=revision, add_pooling_layer=False)
    backbone.eval()

    class BackboneOnly(torch.nn.Module):
        """Wraps the transformer so ONNX export traces only last_hidden_state,
        not an unused pooler head -- keeps the exported graph (and every
        inference call against it) limited to what the pilot's pooling code
        (scripts/common/pooling.py) actually consumes."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, token_type_ids, attention_mask):
            return self.m(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

    model = BackboneOnly(backbone)
    model.eval()

    example = tokenizer(
        ["example sentence for tracing the export"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN_FOR_TRACE,
        return_tensors="pt",
    )
    input_names = list(example.keys())  # typically input_ids, attention_mask, token_type_ids
    print("Tokenizer produced inputs:", input_names)

    onnx_path = OUT_DIR / "model.onnx"
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["last_hidden_state"] = {0: "batch", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(example[name] for name in input_names),
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"Wrote {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model passed")

    tokenizer_dir = OUT_DIR / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"Wrote tokenizer to {tokenizer_dir}")

    # Sanity: run the exported graph once via ORT to confirm output shape/names.
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {name: example[name].numpy() for name in input_names}
    ort_outputs = sess.run(None, ort_inputs)
    output_names = [o.name for o in sess.get_outputs()]
    print("ORT output names:", output_names, "shape:", ort_outputs[0].shape)

    model_card = {
        "repo_id": REPO_ID,
        "revision": revision,
        "revision_resolved_at": datetime.now(timezone.utc).isoformat(),
        "opset": OPSET,
        "input_names": input_names,
        "output_names": output_names,
        "hidden_size": backbone.config.hidden_size,
        "max_position_embeddings": getattr(backbone.config, "max_position_embeddings", None),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    card_path = OUT_DIR / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    print(f"Wrote {card_path}")


if __name__ == "__main__":
    main()

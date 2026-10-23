# src/export_onnx.py
import os
# ensure TorchDynamo disabled in this process
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def export(model_name: str, max_length: int, out_path: str, opset_version: int = 14):
    print(f"[export] Loading model/tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    mdl.eval()
    mdl.cpu()

    # Build deterministic dummy input (padded tokenized "hello world")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or tok.unk_token_id)
    if pad_id is None:
        raise RuntimeError("Tokenizer lacks pad/eos/unk token id.")
    input_ids = torch.full((1, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((1, max_length), dtype=torch.long)

    seed = tok("hello world", truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    lid = seed["input_ids"]
    am = seed["attention_mask"]
    input_ids[:, :lid.shape[1]] = lid
    attention_mask[:, :am.shape[1]] = am

    input_ids = input_ids.to(torch.device("cpu"))
    attention_mask = attention_mask.to(torch.device("cpu"))

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    print(f"[export] Exporting ONNX to {out_path} (opset {opset_version}) ...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.onnx.export(
        mdl,
        (input_ids, attention_mask),
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    print("[export] Running ONNX checker ...")
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("[export] ONNX export succeeded:", out_path)


def quantize(in_path: str, out_path: str):
    print(f"[quantize] Quantizing {in_path} -> {out_path} (dynamic INT8)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    print("[quantize] Quantization completed:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--out", default="models/distilbert_mlm.onnx")
    ap.add_argument("--quant_out", default="models/distilbert_mlm_int8.onnx")
    ap.add_argument("--opset", type=int, default=14)   # <-- changed to 14
    args = ap.parse_args()

    export(args.model, args.max_length, args.out, opset_version=args.opset)
    quantize(args.out, args.quant_out)
    print("Exported:", args.out)
    print("Quantized:", args.quant_out)


if __name__ == "__main__":
    main()

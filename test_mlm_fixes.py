"""Smoke test for the MLM pipeline fixes. Run on PACE login node (no GPU needed)."""
import sys

MODEL = 'ddore14/RooseBERT-cont-cased'

print("=== Test 1: AutoTokenizer loading ===")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"OK: loaded {type(tok).__name__}")

print("\n=== Test 2: convert_ids_to_tokens ===")
ids = tok.encode("The immigrant arrived", add_special_tokens=False)
tokens = tok.convert_ids_to_tokens(ids)
print(f"OK: {tokens}")

print("\n=== Test 3: mask_token ===")
print(f"OK: mask_token = '{tok.mask_token}'")

print("\n=== Test 4: vocab dict (needed by convert_embeddings_to_word_probs) ===")
vocab = tok.vocab
assert isinstance(vocab, dict) and len(vocab) > 1000
print(f"OK: vocab size = {len(vocab)}")

print("\n=== Test 5: BertOnlyMLMHead import ===")
try:
    from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    print("OK: imported from transformers.models.bert.modeling_bert")
except ImportError:
    from transformers.modeling_bert import BertOnlyMLMHead
    print("OK: imported from transformers.modeling_bert (older path)")

print("\nAll tests passed.")

"""
COEUS Tokenizer — Wrapper nativo para el tokenizer BPE custom de COEUS.

Carga el tokenizer.json de HuggingFace tokenizers y expone una interfaz
compatible con COEUSTokenizer del training loop y con transformers PreTrainedTokenizerFast.

Vocab: 131,072 tokens (BPE byte-level)
Special tokens cognitivos: think, reason, cot, step, verify, critique, hypothesis, conclude, code...
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# ─── Intentar importar tokenizers (HuggingFace) ──────────────────────────
try:
    from tokenizers import Tokenizer as HFTokenizer
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False

try:
    from transformers import PreTrainedTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ─── IDs de tokens especiales (hardcoded del tokenizer.json) ─────────────
SPECIAL_TOKENS = {
    "pad": {"token": "<|pad|>", "id": 0},
    "unk": {"token": "<|unk|>", "id": 1},
    "bos": {"token": "<|bos|>", "id": 2},
    "eos": {"token": "<|eos|>", "id": 3},
    "sep": {"token": "<|sep|>", "id": 4},
    # ── Tokens cognitivos ──
    "think_start": {"token": "<|think_start|>", "id": 5},
    "think_end":   {"token": "<|think_end|>",   "id": 6},
    "reason_start":{"token": "<|reason_start|>","id": 7},
    "reason_end":  {"token": "<|reason_end|>",  "id": 8},
    "step":        {"token": "<|step|>",        "id": 9},
    "cot_start":   {"token": "<|cot_start|>",   "id": 10},
    "cot_end":     {"token": "<|cot_end|>",     "id": 11},
    "verify":      {"token": "<|verify|>",      "id": 12},
    "critique":    {"token": "<|critique|>",    "id": 13},
    "hypothesis":  {"token": "<|hypothesis|>",  "id": 14},
    "conclude":    {"token": "<|conclude|>",    "id": 15},
    "code_start":  {"token": "<|code_start|>",  "id": 16},
    "code_end":    {"token": "<|code_end|>",    "id": 17},
    "python":      {"token": "<|python|>",      "id": 18},
    "javascript":  {"token": "<|javascript|>",  "id": 19},
}

# Ruta por defecto al tokenizer
_DEFAULT_TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "COEUS_tokenizer_final")


class COEUSTokenizer:
    """
    Tokenizer nativo de COEUS.
    
    Soporta dos backends:
      1. `tokenizers` (HuggingFace tokenizers) — rápido, Rust-based
      2. `transformers` PreTrainedTokenizerFast — compatibilidad total
    
    Uso:
        tok = COEUSTokenizer()                          # auto-detecta backend
        tok = COEUSTokenizer("/path/to/tokenizer_dir")  # directorio con tokenizer.json
        
        encoded = tok.encode("Hola mundo")              # {"input_ids": tensor}
        text = tok.decode(encoded["input_ids"])          # "Hola mundo"
    """
    
    VOCAB_SIZE = 131072
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        max_length: int = 2048,
        padding_side: str = "left",
        truncation_side: str = "left",
    ):
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        
        # Resolver ruta
        if tokenizer_path is None:
            tokenizer_path = _DEFAULT_TOKENIZER_DIR
        
        tokenizer_json = tokenizer_path
        if os.path.isdir(tokenizer_path):
            tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
        
        if not os.path.exists(tokenizer_json):
            raise FileNotFoundError(
                f"Tokenizer no encontrado en: {tokenizer_json}\n"
                f"Asegúrate de que COEUS_tokenizer_final/tokenizer.json existe."
            )
        
        # ── Backend 1: transformers (preferido — más features) ──
        if HAS_TRANSFORMERS:
            self._backend = "transformers"
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_json,
                bos_token="<|bos|>",
                eos_token="<|eos|>",
                unk_token="<|unk|>",
                pad_token="<|pad|>",
                sep_token="<|sep|>",
                padding_side=padding_side,
                truncation_side=truncation_side,
            )
            # Añadir tokens cognitivos como special tokens adicionales
            cognitive_tokens = [v["token"] for k, v in SPECIAL_TOKENS.items() 
                              if k not in ("pad", "unk", "bos", "eos", "sep")]
            self._tokenizer.add_special_tokens({"additional_special_tokens": cognitive_tokens})
            
        # ── Backend 2: tokenizers (sin transformers) ──
        elif HAS_HF_TOKENIZERS:
            self._backend = "tokenizers"
            self._tokenizer = HFTokenizer.from_file(tokenizer_json)
            self._tokenizer.enable_padding(
                pad_id=SPECIAL_TOKENS["pad"]["id"],
                pad_token=SPECIAL_TOKENS["pad"]["token"],
            )
            self._tokenizer.enable_truncation(max_length=max_length)
        else:
            raise ImportError(
                "Se requiere 'tokenizers' o 'transformers' para cargar el COEUS tokenizer.\n"
                "Instala con: pip install tokenizers  (o pip install transformers)"
            )
        
        # ── IDs accesibles directamente ──
        self.vocab_size = self.VOCAB_SIZE
        self.pad_token_id = SPECIAL_TOKENS["pad"]["id"]      # 0
        self.eos_token_id = SPECIAL_TOKENS["eos"]["id"]      # 3
        self.bos_token_id = SPECIAL_TOKENS["bos"]["id"]      # 2
        self.unk_token_id = SPECIAL_TOKENS["unk"]["id"]      # 1
        self.sep_token_id = SPECIAL_TOKENS["sep"]["id"]      # 4
        
        # IDs cognitivos
        self.think_start_id = SPECIAL_TOKENS["think_start"]["id"]   # 5
        self.think_end_id   = SPECIAL_TOKENS["think_end"]["id"]     # 6
        self.reason_start_id= SPECIAL_TOKENS["reason_start"]["id"]  # 7
        self.reason_end_id  = SPECIAL_TOKENS["reason_end"]["id"]    # 8
        self.step_id        = SPECIAL_TOKENS["step"]["id"]          # 9
        self.cot_start_id   = SPECIAL_TOKENS["cot_start"]["id"]     # 10
        self.cot_end_id     = SPECIAL_TOKENS["cot_end"]["id"]       # 11
        self.verify_id      = SPECIAL_TOKENS["verify"]["id"]        # 12
        self.critique_id    = SPECIAL_TOKENS["critique"]["id"]      # 13
        self.hypothesis_id  = SPECIAL_TOKENS["hypothesis"]["id"]    # 14
        self.conclude_id    = SPECIAL_TOKENS["conclude"]["id"]      # 15
        
        logger.info(
            f"COEUS Tokenizer cargado: backend={self._backend}, "
            f"vocab_size={self.vocab_size}, path={tokenizer_json}"
        )
    
    # ─── Encode ────────────────────────────────────────────────────────────
    def encode(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Codifica texto a tokens.
        
        Returns:
            Dict con "input_ids" y opcionalmente "attention_mask"
        """
        max_len = max_length or self.max_length
        
        if self._backend == "transformers":
            kwargs = dict(
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
            )
            # FIX BUG #16/#26: Only pass max_length when truncation is enabled
            # to avoid HF warnings and unintended truncation behavior
            if truncation and max_len is not None:
                kwargs['max_length'] = max_len
            return self._tokenizer(text, **kwargs)
        else:
            # Backend tokenizers
            if isinstance(text, str):
                text = [text]
            
            # FIX: Only enable truncation when requested
            if truncation and max_len:
                self._tokenizer.enable_truncation(max_length=max_len)
            else:
                self._tokenizer.no_truncation()
            
            if padding:
                self._tokenizer.enable_padding(
                    pad_id=self.pad_token_id,
                    pad_token=SPECIAL_TOKENS["pad"]["token"],
                    length=max_len if padding == "max_length" else None,
                )
            
            encoded = self._tokenizer.encode_batch(text)
            
            input_ids = [e.ids for e in encoded]
            attention_mask = [e.attention_mask for e in encoded]
            
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # ─── Decode ────────────────────────────────────────────────────────────
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """Decodifica tokens a texto."""
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 0:
                token_ids = token_ids.unsqueeze(0)
            
            if token_ids.dim() == 1:
                ids = token_ids.tolist()
                return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            else:
                # Batch decode
                return [
                    self._tokenizer.decode(row.tolist(), skip_special_tokens=skip_special_tokens)
                    for row in token_ids
                ]
        else:
            # List[int]
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    # ─── Utilidades cognitivas ──────────────────────────────────────────────
    def wrap_thinking(self, text: str) -> str:
        """Envuelve texto en tokens de pensamiento: <|think_start|>...<|think_end|>"""
        return f"<|think_start|>{text}<|think_end|>"
    
    def wrap_reasoning(self, text: str) -> str:
        """Envuelve texto en tokens de razonamiento."""
        return f"<|reason_start|>{text}<|reason_end|>"
    
    def wrap_chain_of_thought(self, steps: List[str]) -> str:
        """Construye cadena de pensamiento con tokens de paso."""
        inner = "<|step|>".join(steps)
        return f"<|cot_start|>{inner}<|cot_end|>"
    
    def wrap_code(self, code: str, language: str = "python") -> str:
        """Envuelve código con tokens de lenguaje."""
        lang_token = f"<|{language}|>" if f"<|{language}|>" in [v["token"] for v in SPECIAL_TOKENS.values()] else ""
        return f"<|code_start|>{lang_token}{code}<|code_end|>"
    
    def format_hypothesis_verification(self, hypothesis: str, verification: str, conclusion: str) -> str:
        """Formatea un ciclo completo de razonamiento científico."""
        return (
            f"<|hypothesis|>{hypothesis}"
            f"<|verify|>{verification}"
            f"<|critique|>{verification}"
            f"<|conclude|>{conclusion}"
        )
    
    # ─── Propiedades ────────────────────────────────────────────────────────
    @property
    def special_token_ids(self) -> Dict[str, int]:
        """Retorna todos los IDs de tokens especiales."""
        return {k: v["id"] for k, v in SPECIAL_TOKENS.items()}
    
    @property 
    def cognitive_token_ids(self) -> Dict[str, int]:
        """Retorna solo los IDs de tokens cognitivos (sin pad/unk/bos/eos/sep)."""
        return {k: v["id"] for k, v in SPECIAL_TOKENS.items() 
                if k not in ("pad", "unk", "bos", "eos", "sep")}
    
    def get_token_id(self, token_name: str) -> int:
        """Obtiene el ID de un token especial por nombre."""
        if token_name in SPECIAL_TOKENS:
            return SPECIAL_TOKENS[token_name]["id"]
        raise KeyError(f"Token especial '{token_name}' no encontrado. "
                      f"Disponibles: {list(SPECIAL_TOKENS.keys())}")
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return (
            f"COEUSTokenizer(vocab_size={self.vocab_size}, "
            f"backend='{self._backend}', "
            f"max_length={self.max_length}, "
            f"special_tokens={len(SPECIAL_TOKENS)})"
        )


# ─── Factory function ──────────────────────────────────────────────────────
def create_coeus_tokenizer(
    tokenizer_path: Optional[str] = None,
    max_length: int = 2048,
    **kwargs,
) -> COEUSTokenizer:
    """
    Factory para crear el tokenizer de COEUS.
    
    Args:
        tokenizer_path: Ruta al directorio o archivo tokenizer.json.
                       Si None, busca en COEUS_tokenizer_final/
        max_length: Longitud máxima de secuencia
    
    Returns:
        COEUSTokenizer configurado
    """
    return COEUSTokenizer(
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        **kwargs,
    )


# ─── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("COEUS Tokenizer — Self-Test")
    print("=" * 60)
    
    try:
        tok = COEUSTokenizer()
        print(f"\n✓ Tokenizer cargado: {tok}")
        
        # Test encode/decode
        test_text = "COEUS es un modelo de lenguaje avanzado."
        encoded = tok.encode(test_text, padding=False)
        input_ids = encoded["input_ids"]
        print(f"\n✓ Encode: '{test_text}'")
        print(f"  → IDs ({input_ids.shape}): {input_ids[0][:20].tolist()}...")
        
        decoded = tok.decode(input_ids[0])
        print(f"  → Decode: '{decoded}'")
        
        # Test tokens cognitivos
        thinking = tok.wrap_thinking("Analicemos el problema paso a paso")
        print(f"\n✓ Wrap thinking: {thinking}")
        
        cot = tok.wrap_chain_of_thought(["Premisa A", "Razonamiento B", "Conclusión C"])
        print(f"✓ Chain of thought: {cot}")
        
        # Test special token IDs
        print(f"\n✓ Special tokens: {tok.special_token_ids}")
        print(f"✓ Cognitive tokens: {tok.cognitive_token_ids}")
        
        # Test batch
        batch = tok.encode(["Texto uno", "Texto dos más largo que el primero"], padding=True)
        print(f"\n✓ Batch encode: {batch['input_ids'].shape}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
=============================================================================
🐋 1-BIT DISTILLED R1 - BitNet b1.58 + R1 Whitepaper Simulation
=============================================================================
Architecture: 1-bit (Ternary {-1, 0, 1}) simulated weights.
Reasoning: Emergent RL behaviors (Self-Correction, "Aha!" moments).
UI: Faithful replication of chat.deepseek.com.

WEIGHT SCALE: ~7B parameter equivalent (BitNet b1.58 2B/7B paper)
  - Each ternary weight = 1.58 bits ≈ 2 bits packed storage
  - 7B params @ 1.58 bits ≈ ~1.38 GB (vs 14 GB for fp16)
=============================================================================
"""
import os
import sys
import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import re
import queue
import math
from dataclasses import dataclass
from typing import List

# =============================================================================
# 1-BIT WEIGHT SCALE CONSTANTS (Realistic 7B-param BitNet b1.58)
# =============================================================================

# BitNet b1.58 7B architecture (approximate, based on LLaMA-like design):
#   - Layers: 32 transformer blocks
#   - Hidden dim: 4096
#   - FFN intermediate: 11008
#   - Attention heads: 32, head_dim: 128
#   - Total params: ~7,000,000,000

NUM_LAYERS        = 32          # Transformer depth
HIDDEN_DIM        = 4096        # d_model
FFN_DIM           = 11008       # FFN intermediate size (2.7x hidden)
NUM_HEADS         = 32          # Multi-head attention heads
HEAD_DIM          = 128         # HIDDEN_DIM // NUM_HEADS
VOCAB_SIZE        = 32000       # Tokenizer vocab (LLaMA-style)

# Ternary storage: pack 4 ternary values into 1 byte (log2(3)≈1.58 bits each)
# Total bits = 7B * 1.58 ≈ 11.06 Gb = ~1.38 GB
TOTAL_PARAMS_B    = 7_000_000_000
BITS_PER_WEIGHT   = 1.58
MODEL_SIZE_GB     = (TOTAL_PARAMS_B * BITS_PER_WEIGHT) / (8 * 1024**3)

# Layer-wise weight matrix sizes (rows x cols = param count per weight tensor)
ATTN_QKV_PARAMS   = HIDDEN_DIM * (3 * HIDDEN_DIM)        # Q+K+V projection: 4096*12288 = 50,331,648
ATTN_OUT_PARAMS   = HIDDEN_DIM * HIDDEN_DIM               # Output projection: 4096*4096  = 16,777,216
FFN_GATE_PARAMS   = HIDDEN_DIM * FFN_DIM                  # Gate proj:         4096*11008 = 45,088,768
FFN_UP_PARAMS     = HIDDEN_DIM * FFN_DIM                  # Up proj:           4096*11008 = 45,088,768
FFN_DOWN_PARAMS   = FFN_DIM    * HIDDEN_DIM               # Down proj:         11008*4096 = 45,088,768
EMBED_PARAMS      = VOCAB_SIZE * HIDDEN_DIM               # Token embedding:   32000*4096 = 131,072,000

PARAMS_PER_LAYER  = ATTN_QKV_PARAMS + ATTN_OUT_PARAMS + FFN_GATE_PARAMS + FFN_UP_PARAMS + FFN_DOWN_PARAMS
TOTAL_LAYER_PARAMS = PARAMS_PER_LAYER * NUM_LAYERS         # ~6.55B
TOTAL_EMBED_PARAMS = EMBED_PARAMS * 2                      # embed + lm_head ≈ 262M
SIMULATED_TOTAL   = TOTAL_LAYER_PARAMS + TOTAL_EMBED_PARAMS

# =============================================================================
# 1-BIT (TERNARY) QUANTIZATION LOGIC
# =============================================================================

class BitNetLayer:
    """
    Simulates a BitNet b1.58 layer with {-1, 0, 1} ternary weights.
    
    Real weight matrix sizes (7B scale):
      - QKV projection:  [4096, 12288]  → 50,331,648 ternary params
      - Output proj:     [4096,  4096]  → 16,777,216 ternary params
      - FFN gate/up:     [4096, 11008]  → 45,088,768 ternary params each
      - FFN down:        [11008, 4096]  → 45,088,768 ternary params
    
    For simulation we use a compressed representative slice (1/1024 scale)
    to avoid allocating 7B integers in RAM, while reporting true sizes.
    """

    # Simulation compression ratio (full allocation would be ~7GB of Python ints)
    SIM_RATIO = 1024

    def __init__(self, rows: int, cols: int, name: str = "weight"):
        self.name        = name
        self.rows        = rows          # True matrix rows
        self.cols        = cols          # True matrix cols
        self.true_params = rows * cols   # Actual param count at 7B scale
        self.true_bits   = self.true_params * BITS_PER_WEIGHT
        self.true_bytes  = self.true_bits / 8

        # Simulate a compressed slice (rows//SIM_RATIO × cols//SIM_RATIO)
        sim_r = max(1, rows  // self.SIM_RATIO)
        sim_c = max(1, cols  // self.SIM_RATIO)
        self.sim_weights: List[int] = [
            random.choice([-1, 0, 1]) for _ in range(sim_r * sim_c)
        ]
        self.sim_rows = sim_r
        self.sim_cols = sim_c

        # Weight statistics (mirrors real BitNet quantization)
        total = len(self.sim_weights)
        self.sparsity   = self.sim_weights.count(0)  / total   # ~1/3 zeros
        self.pos_ratio  = self.sim_weights.count(1)  / total
        self.neg_ratio  = self.sim_weights.count(-1) / total

    def forward(self, inputs: List[float]) -> float:
        """
        1-bit matrix-vector multiply on the compressed slice.
        Real operation: Y = W_ternary @ X  (no multiplications, only add/sub)
        """
        dot = 0.0
        n = min(len(inputs), self.sim_cols)
        for row in range(self.sim_rows):
            row_sum = 0.0
            for col in range(n):
                w = self.sim_weights[row * self.sim_cols + col]
                if   w ==  1: row_sum += inputs[col]
                elif w == -1: row_sum -= inputs[col]
            dot += row_sum
        return math.tanh(dot / max(self.sim_rows, 1))

    @property
    def size_str(self) -> str:
        if self.true_bytes >= 1024**3:
            return f"{self.true_bytes/1024**3:.2f} GB"
        elif self.true_bytes >= 1024**2:
            return f"{self.true_bytes/1024**2:.1f} MB"
        else:
            return f"{self.true_bytes/1024:.1f} KB"

    def summary(self) -> str:
        return (
            f"{self.name:20s}  shape=({self.rows:6d},{self.cols:6d})  "
            f"params={self.true_params:,}  size={self.size_str}  "
            f"sparsity={self.sparsity:.1%}  +={self.pos_ratio:.1%}  -={self.neg_ratio:.1%}"
        )


class TransformerBlock:
    """One full transformer block at 7B scale (attention + FFN), all 1-bit."""
    def __init__(self, layer_idx: int):
        self.idx = layer_idx
        # Attention weight matrices
        self.attn_qkv = BitNetLayer(HIDDEN_DIM, 3 * HIDDEN_DIM, f"attn_qkv_{layer_idx}")
        self.attn_out = BitNetLayer(HIDDEN_DIM, HIDDEN_DIM,     f"attn_out_{layer_idx}")
        # FFN weight matrices (SwiGLU style)
        self.ffn_gate = BitNetLayer(HIDDEN_DIM, FFN_DIM,        f"ffn_gate_{layer_idx}")
        self.ffn_up   = BitNetLayer(HIDDEN_DIM, FFN_DIM,        f"ffn_up_{layer_idx}")
        self.ffn_down = BitNetLayer(FFN_DIM,    HIDDEN_DIM,     f"ffn_down_{layer_idx}")

    @property
    def total_params(self) -> int:
        return (self.attn_qkv.true_params + self.attn_out.true_params +
                self.ffn_gate.true_params + self.ffn_up.true_params  +
                self.ffn_down.true_params)

    def forward(self, hidden: List[float]) -> List[float]:
        # Simplified: run through one representative weight per sub-layer
        h = hidden[:]
        attn_out  = self.attn_qkv.forward(h[:HIDDEN_DIM // self.attn_qkv.SIM_RATIO])
        ffn_out   = self.ffn_gate.forward(h[:HIDDEN_DIM // self.ffn_gate.SIM_RATIO])
        result    = attn_out + ffn_out
        return [result] * len(hidden)


# =============================================================================
# R1 REASONING ENGINE (1-BIT CORE, 7B SCALE)
# =============================================================================

class OneBitR1Engine:
    def __init__(self):
        self.is_ready   = False
        self.model_name = "DeepSeek-R1-1Bit-7B-Distill"
        self.deep_mode  = False

        # Build realistic-scale 7B transformer blocks (sim-compressed)
        self.blocks: List[TransformerBlock] = []
        # Token embedding (32000 × 4096)
        self.token_embed = BitNetLayer(VOCAB_SIZE, HIDDEN_DIM, "token_embed")
        # LM head (4096 × 32000) — tied weights in real models
        self.lm_head     = BitNetLayer(HIDDEN_DIM, VOCAB_SIZE, "lm_head")

    def boot_sequence(self, callback):
        callback("Initializing 1-bit Quantized Weights (7B scale)...")
        callback(f"[SYS] Architecture: {NUM_LAYERS}L × {HIDDEN_DIM}d | FFN={FFN_DIM} | Heads={NUM_HEADS}×{HEAD_DIM}")
        callback(f"[SYS] Estimated param count: {SIMULATED_TOTAL/1e9:.2f}B")
        callback(f"[SYS] Storage @ 1.58 bits/weight: {MODEL_SIZE_GB:.2f} GB (vs {TOTAL_PARAMS_B*2/8/1024**3:.1f} GB fp16)")
        time.sleep(0.15)

        callback(f"[SYS] Loading token embedding: {self.token_embed.summary()}")
        time.sleep(0.1)

        for i in range(NUM_LAYERS):
            block = TransformerBlock(i)
            self.blocks.append(block)
            if i % 8 == 0 or i == NUM_LAYERS - 1:
                layer_gb = block.total_params * BITS_PER_WEIGHT / (8 * 1024**3)
                callback(f"[SYS] Layer {i:02d}/{NUM_LAYERS-1} loaded | "
                         f"params={block.total_params/1e6:.0f}M | {layer_gb:.3f} GB")
                time.sleep(0.08)

        callback(f"[SYS] LM head: {self.lm_head.summary()}")
        callback("[SYS] Scaling weights to {-1, 0, 1} (BitNet b1.58)...")
        time.sleep(0.1)
        callback("[SYS] Loading R1 Reinforcement Learning Policy...")
        time.sleep(0.1)
        callback("[SYS] Calibrating Self-Correction Thresholds...")
        time.sleep(0.1)
        callback(f"✓ 1-Bit R1 7B Engine Ready. ({MODEL_SIZE_GB:.2f} GB on disk)")
        self.is_ready = True

    def _generate_vector(self, query: str) -> List[float]:
        vec = [0.0] * HIDDEN_DIM
        for i, ch in enumerate(query):
            vec[i % HIDDEN_DIM] += ord(ch) / 255.0
        return vec

    def generate_thought_chain(self, query: str, message_queue: queue.Queue):
        hidden = self._generate_vector(query)

        # Run through first and last block for representativeness
        block_0 = self.blocks[0]  if self.blocks else None
        block_n = self.blocks[-1] if self.blocks else None
        act_0   = block_0.ffn_gate.forward(hidden[:HIDDEN_DIM // BitNetLayer.SIM_RATIO]) if block_0 else 0.0
        act_n   = block_n.attn_qkv.forward(hidden[:HIDDEN_DIM // BitNetLayer.SIM_RATIO]) if block_n else 0.0

        thoughts_text = ""

        def stream_thought(text, delay_range=(0.010, 0.025)):
            nonlocal thoughts_text
            for ch in text:
                thoughts_text += ch
                message_queue.put(("thought", thoughts_text))
                time.sleep(random.uniform(*delay_range))
            thoughts_text += "\n\n"
            message_queue.put(("thought", thoughts_text))
            time.sleep(0.18)

        stream_thought(
            f"[Layer 0  ffn_gate] activation={act_0:.6f}  "
            f"| shape=({HIDDEN_DIM},{FFN_DIM}) | params={FFN_GATE_PARAMS:,} | sparsity={block_0.ffn_gate.sparsity:.1%}"
        )
        stream_thought(
            f"[Layer {NUM_LAYERS-1:2d} attn_qkv] activation={act_n:.6f}  "
            f"| shape=({HIDDEN_DIM},{3*HIDDEN_DIM}) | params={ATTN_QKV_PARAMS:,}"
        )
        stream_thought(
            f"Total model weight storage: {MODEL_SIZE_GB:.3f} GB  "
            f"({SIMULATED_TOTAL/1e9:.3f}B params @ {BITS_PER_WEIGHT} bits/weight)"
        )
        stream_thought("Phase 1: Problem Decomposition. Identifying core constraints via 1-bit attention.")

        stream_thought("Initial hypothesis: Standard linear solution is applicable.")
        time.sleep(0.5)

        if act_0 > 0.05 or self.deep_mode:
            stream_thought("Wait... re-evaluating. Ternary weight distribution suggests non-linear edge case in layers 8-16.")
            stream_thought("Correction: Backtracking to previous hidden state. Re-calculating with ternary logic.")
            stream_thought("Aha! Pattern confirmed — FFN sparsity creates shortcut path through zero weights.")
        else:
            stream_thought("Logic path verified by internal consistency checks across all 32 layers.")

        stream_thought("Synthesis: Formulating response from 1-bit distilled 7B knowledge.")
        self._generate_response(query, message_queue)

    def _generate_response(self, query: str, message_queue: queue.Queue):
        sparsity_avg = sum(b.ffn_gate.sparsity for b in self.blocks) / max(len(self.blocks), 1)
        final_text = (
            f"After processing through {NUM_LAYERS} ternary transformer blocks "
            f"({SIMULATED_TOTAL/1e9:.1f}B params, {MODEL_SIZE_GB:.2f} GB), "
            f"with average FFN sparsity {sparsity_avg:.1%} (zero weights free of compute), "
            f"and applying R1-style self-correction across the reasoning chain, "
            f"the model has determined the most efficient solution. "
            f"The 1-bit quantization reduces memory by {(1 - MODEL_SIZE_GB / (TOTAL_PARAMS_B*2/8/1024**3))*100:.0f}% "
            f"vs fp16 while preserving reasoning capability through emergent ternary representations."
        )
        stream = ""
        for ch in final_text:
            stream += ch
            message_queue.put(("answer", stream))
            time.sleep(0.009)
        message_queue.put(("done", None))


# =============================================================================
# UI COMPONENTS (DeepSeek Accurate Styling)
# =============================================================================

class CollapsibleThought(tk.Frame):
    def __init__(self, parent, colors):
        super().__init__(parent, bg=colors["chat_bg"])
        self.colors = colors
        self.is_expanded = True
        self._build()

    def _build(self):
        c = self.colors
        header = tk.Frame(self, bg=c["chat_bg"], cursor="hand2")
        header.pack(fill="x")
        self.toggle_btn = tk.Label(
            header, text="▾  Thinking...",
            font=("Helvetica", 10, "bold"),
            bg=c["chat_bg"], fg=c["think_accent"],
            cursor="hand2", padx=0, pady=4
        )
        self.toggle_btn.pack(side="left")
        self.toggle_btn.bind("<Button-1>", self.toggle)

        self.content_outer = tk.Frame(self, bg=c["chat_bg"])
        self.content_outer.pack(fill="x", pady=(4, 0))
        tk.Frame(self.content_outer, bg=c["think_accent"], width=3).pack(side="left", fill="y")

        mono = "Menlo" if sys.platform == "darwin" else "Consolas"
        self.text_label = tk.Label(
            self.content_outer, text="",
            font=(mono, 9), bg=c["chat_bg"], fg=c["think_text"],
            justify="left", anchor="nw", wraplength=640,
            padx=12, pady=8
        )
        self.text_label.pack(side="left", fill="x", expand=True)

    def toggle(self, event=None):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.content_outer.pack(fill="x", pady=(4, 0))
            self.toggle_btn.config(text="▾  Thinking...")
        else:
            self.content_outer.pack_forget()
            self.toggle_btn.config(text="▸  Thought Process")

    def update_text(self, text):    self.text_label.config(text=text)
    def mark_done(self, elapsed):   self.toggle_btn.config(text=f"▾  Thought for {elapsed:.0f}s")


class DeepSeekApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSeek-R1 1-Bit 7B Simulation")
        self.root.geometry("1120x760")
        self.root.configure(bg='#141414')

        self.colors = {
            "sidebar_bg":    "#0A0A0A",
            "sidebar_hover": "#1A1A1A",
            "sidebar_text":  "#4D6BFE",
            "sidebar_sub":   "#3A55CC",
            "new_chat_bg":   "#111111",
            "new_chat_fg":   "#4D6BFE",
            "chat_bg":       "#141414",
            "user_text":     "#4D6BFE",
            "bot_text":      "#4D6BFE",
            "bot_name":      "#4D6BFE",
            "think_accent":  "#4D6BFE",
            "think_text":    "#7A96FF",
            "input_bg":      "#1A1A1A",
            "input_border":  "#2A2A2A",
            "input_text":    "#4D6BFE",
            "send_bg":       "#000000",
            "send_fg":       "#4D6BFE",
            "placeholder":   "#1E3080",
            "divider":       "#1F1F1F",
            "user_bubble":   "#1A1A1A",
        }

        self.engine    = OneBitR1Engine()
        self.msg_queue = queue.Queue()
        self.deep_mode = False
        self._think_start = 0.0

        self.setup_ui()
        self.root.after(100, self.process_queue)
        threading.Thread(target=self.engine.boot_sequence, args=(self.update_status,), daemon=True).start()

    def setup_ui(self):
        c = self.colors

        self.sidebar = tk.Frame(self.root, bg=c["sidebar_bg"], width=280)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="🐋 Distilled R1 7B", font=("Helvetica", 14, "bold"),
                 bg=c["sidebar_bg"], fg=c["sidebar_text"]).pack(padx=16, pady=16, anchor="w")

        # Model stats panel
        stats_frame = tk.Frame(self.sidebar, bg=c["sidebar_hover"])
        stats_frame.pack(fill="x", padx=12, pady=4)
        stats = [
            ("Params",   f"{SIMULATED_TOTAL/1e9:.2f}B"),
            ("Storage",  f"{MODEL_SIZE_GB:.2f} GB"),
            ("Layers",   str(NUM_LAYERS)),
            ("Hidden",   str(HIDDEN_DIM)),
            ("FFN dim",  str(FFN_DIM)),
            ("Heads",    f"{NUM_HEADS}×{HEAD_DIM}"),
            ("Vocab",    f"{VOCAB_SIZE:,}"),
            ("Bits/w",   f"{BITS_PER_WEIGHT}"),
        ]
        for label, val in stats:
            row = tk.Frame(stats_frame, bg=c["sidebar_hover"])
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=label, font=("Helvetica", 8),  bg=c["sidebar_hover"], fg=c["sidebar_sub"],  anchor="w").pack(side="left")
            tk.Label(row, text=val,   font=("Helvetica", 8, "bold"), bg=c["sidebar_hover"], fg=c["sidebar_text"], anchor="e").pack(side="right")

        tk.Button(self.sidebar, text="＋  New Chat", font=("Helvetica", 10, "bold"),
                  bg=c["new_chat_bg"], fg=c["new_chat_fg"], relief="flat", bd=0,
                  command=self.clear_chat).pack(fill="x", padx=16, pady=(12, 4), ipady=8)

        self.deep_btn = tk.Button(self.sidebar, text="⚡  DeepThink  OFF",
                                  font=("Helvetica", 9, "bold"),
                                  bg=c["sidebar_hover"], fg=c["sidebar_sub"],
                                  relief="flat", bd=0, command=self.toggle_deep)
        self.deep_btn.pack(fill="x", padx=16, pady=2, ipady=6)

        self.status_label = tk.Label(self.sidebar, text="Initializing...",
                                     font=("Helvetica", 7), bg=c["sidebar_bg"],
                                     fg=c["sidebar_sub"], wraplength=240, justify="left")
        self.status_label.pack(side="bottom", padx=12, pady=12)

        main = tk.Frame(self.root, bg=c["chat_bg"])
        main.pack(side="right", fill="both", expand=True)

        topbar = tk.Frame(main, bg=c["chat_bg"],
                          highlightthickness=1, highlightbackground=c["divider"], height=52)
        topbar.pack(fill="x")
        tk.Label(topbar, text=f"DeepSeek-R1 1-Bit 7B  ({MODEL_SIZE_GB:.2f} GB | {SIMULATED_TOTAL/1e9:.1f}B params)",
                 font=("Helvetica", 10, "bold"), bg=c["chat_bg"], fg=c["bot_name"]).pack(side="left", padx=20, pady=14)

        chat_frame = tk.Frame(main, bg=c["chat_bg"])
        chat_frame.pack(fill="both", expand=True)
        self.canvas     = tk.Canvas(chat_frame, bg=c["chat_bg"], highlightthickness=0)
        self.scroll_frame = tk.Frame(self.canvas, bg=c["chat_bg"])
        vsb = ttk.Scrollbar(chat_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self._win_id = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",       lambda e: self.canvas.itemconfig(self._win_id, width=e.width))

        input_inner = tk.Frame(main, bg=c["chat_bg"])
        input_inner.pack(fill="x", padx=24, pady=20)
        box_frame = tk.Frame(input_inner, bg=c["input_bg"],
                             highlightthickness=1, highlightbackground=c["input_border"])
        box_frame.pack(fill="x")
        self.entry = tk.Text(box_frame, bg=c["input_bg"], fg=c["input_text"],
                             font=("Helvetica", 12), relief="flat", bd=0, height=1, padx=14, pady=12)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send() or "break")
        tk.Button(box_frame, text="↑", font=("Helvetica", 12, "bold"),
                  bg=c["send_bg"], fg=c["send_fg"], relief="flat",
                  command=self.send).pack(side="right", padx=6)

    def toggle_deep(self):
        self.deep_mode = not self.deep_mode
        self.engine.deep_mode = self.deep_mode
        self.deep_btn.config(text=f"⚡  DeepThink  {'ON' if self.deep_mode else 'OFF'}")

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def clear_chat(self):
        for w in self.scroll_frame.winfo_children(): w.destroy()

    def send(self):
        text = self.entry.get("1.0", tk.END).strip()
        if not text or not self.engine.is_ready: return
        self.entry.delete("1.0", tk.END)
        self._add_user_message(text)
        self._build_bot_container()
        self._think_start = time.time()
        threading.Thread(target=self.engine.generate_thought_chain,
                         args=(text, self.msg_queue), daemon=True).start()

    def _add_user_message(self, text):
        f = tk.Frame(self.scroll_frame, bg=self.colors["chat_bg"])
        f.pack(fill="x", padx=32, pady=10)
        tk.Label(f, text=text, bg=self.colors["user_bubble"], fg=self.colors["user_text"],
                 font=("Helvetica", 11), wraplength=520, justify="left",
                 padx=14, pady=10).pack(side="right")
        self._scroll_bottom()

    def _build_bot_container(self):
        self.current_bot_frame    = tk.Frame(self.scroll_frame, bg=self.colors["chat_bg"])
        self.current_bot_frame.pack(fill="x", padx=32, pady=10, anchor="w")
        tk.Label(self.current_bot_frame, text="🐋 Cat R1 (1-Bit 7B)",
                 font=("Helvetica", 11, "bold"),
                 bg=self.colors["chat_bg"], fg=self.colors["bot_name"]).pack(anchor="w")
        self.current_thought_block = None
        self.current_answer_label  = None

    def _scroll_bottom(self):
        self.root.after(50, lambda: self.canvas.yview_moveto(1.0))

    def process_queue(self):
        try:
            while True:
                mode, content = self.msg_queue.get_nowait()
                if mode == "thought":
                    if not self.current_thought_block:
                        self.current_thought_block = CollapsibleThought(self.current_bot_frame, self.colors)
                        self.current_thought_block.pack(fill="x", pady=5)
                    self.current_thought_block.update_text(content)
                elif mode == "answer":
                    if not self.current_answer_label:
                        self.current_answer_label = tk.Label(
                            self.current_bot_frame, text="",
                            bg=self.colors["chat_bg"], fg=self.colors["bot_text"],
                            font=("Helvetica", 11), justify="left", wraplength=600)
                        self.current_answer_label.pack(anchor="w", pady=5)
                    self.current_answer_label.config(text=content)
                elif mode == "done":
                    if self.current_thought_block:
                        self.current_thought_block.mark_done(time.time() - self._think_start)
                self._scroll_bottom()
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app  = DeepSeekApp(root)
    root.mainloop()

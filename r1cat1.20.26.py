"""
=============================================================================
🐋 4-BIT R1CAT - R1 Distilled (Exact Behavior Simulation)
=============================================================================
Architecture: 4-bit uniformly quantized weights (simulated).
Base: DeepSeek R1 Distilled (fluent English capabilities)
UI: Faithful replication of chat.deepseek.com.

WEIGHT SCALE: ~14B parameter equivalent (4-bit)
  - 14B params @ 4 bits = 7 GB (vs 28 GB for fp16)
=============================================================================
"""
import os
import sys
import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import queue
import math
from typing import List

# =============================================================================
# 4-BIT WEIGHT SCALE CONSTANTS (Realistic 14B-param 4-bit Model)
# =============================================================================

NUM_LAYERS        = 44          # Transformer depth
HIDDEN_DIM        = 5120        # d_model
FFN_DIM           = 13824       # FFN intermediate size (2.7x hidden)
NUM_HEADS         = 40          # Multi-head attention heads
HEAD_DIM          = 128         # HIDDEN_DIM // NUM_HEADS
VOCAB_SIZE        = 32000       # Tokenizer vocab

# 4-bit quantization
TOTAL_PARAMS_B    = 14_284_226_560   # from layer formulas below
BITS_PER_WEIGHT   = 4.0
MODEL_SIZE_GB     = (TOTAL_PARAMS_B * BITS_PER_WEIGHT) / (8 * 1024**3)

# Layer-wise weight matrix sizes (rows x cols = param count per weight tensor)
ATTN_QKV_PARAMS   = HIDDEN_DIM * (3 * HIDDEN_DIM)        # Q+K+V projection: 5120*15360 = 78,643,200
ATTN_OUT_PARAMS   = HIDDEN_DIM * HIDDEN_DIM               # Output projection: 5120*5120  = 26,214,400
FFN_GATE_PARAMS   = HIDDEN_DIM * FFN_DIM                  # Gate proj:         5120*13824 = 70,778,880
FFN_UP_PARAMS     = HIDDEN_DIM * FFN_DIM                  # Up proj:           5120*13824 = 70,778,880
FFN_DOWN_PARAMS   = FFN_DIM    * HIDDEN_DIM               # Down proj:         13824*5120 = 70,778,880
EMBED_PARAMS      = VOCAB_SIZE * HIDDEN_DIM               # Token embedding:   32000*5120 = 163,840,000

PARAMS_PER_LAYER  = ATTN_QKV_PARAMS + ATTN_OUT_PARAMS + FFN_GATE_PARAMS + FFN_UP_PARAMS + FFN_DOWN_PARAMS
TOTAL_LAYER_PARAMS = PARAMS_PER_LAYER * NUM_LAYERS         # ≈13.96B
TOTAL_EMBED_PARAMS = EMBED_PARAMS * 2                      # embed + lm_head ≈ 327.68M
SIMULATED_TOTAL   = TOTAL_LAYER_PARAMS + TOTAL_EMBED_PARAMS  # 14,284,226,560

# =============================================================================
# 4-BIT QUANTIZATION LOGIC (simulated as uniform random int4)
# =============================================================================

class BitNetLayer:
    """
    Simulates a 4-bit quantized layer with weights in range [-8,7] (symmetric int4).
    """
    SIM_RATIO = 1024   # compression factor

    def __init__(self, rows: int, cols: int, name: str = "weight"):
        self.name        = name
        self.rows        = rows
        self.cols        = cols
        self.true_params = rows * cols
        self.true_bits   = self.true_params * BITS_PER_WEIGHT
        self.true_bytes  = self.true_bits / 8

        # Simulate a compressed slice (rows//SIM_RATIO × cols//SIM_RATIO)
        sim_r = max(1, rows  // self.SIM_RATIO)
        sim_c = max(1, cols  // self.SIM_RATIO)
        self.sim_weights: List[int] = [
            random.randint(-8, 7) for _ in range(sim_r * sim_c)   # symmetric int4
        ]
        self.sim_rows = sim_r
        self.sim_cols = sim_c

        # Weight statistics
        total = len(self.sim_weights)
        self.pos_ratio  = sum(1 for w in self.sim_weights if w > 0) / total
        self.neg_ratio  = sum(1 for w in self.sim_weights if w < 0) / total
        self.zero_ratio = sum(1 for w in self.sim_weights if w == 0) / total

    def forward(self, inputs: List[float]) -> float:
        """
        4-bit matrix-vector multiply on the compressed slice.
        """
        dot = 0.0
        n = min(len(inputs), self.sim_cols)
        for row in range(self.sim_rows):
            row_sum = 0.0
            for col in range(n):
                w = self.sim_weights[row * self.sim_cols + col]
                row_sum += w * inputs[col]
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
            f"pos={self.pos_ratio:.1%}  neg={self.neg_ratio:.1%}  zero={self.zero_ratio:.1%}"
        )


class TransformerBlock:
    """One full transformer block at 14B scale (attention + FFN), all 4-bit."""
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
        h = hidden[:]
        attn_out  = self.attn_qkv.forward(h[:HIDDEN_DIM // self.attn_qkv.SIM_RATIO])
        ffn_out   = self.ffn_gate.forward(h[:HIDDEN_DIM // self.ffn_gate.SIM_RATIO])
        result    = attn_out + ffn_out
        return [result] * len(hidden)


# =============================================================================
# R1 DISTILLED REASONING ENGINE (EXACT R1 BEHAVIOR SIMULATION)
# =============================================================================

class DistilledR1Engine:
    def __init__(self):
        self.is_ready   = False
        self.model_name = "R1Cat-Distill-4Bit-14B"
        self.deep_mode  = False
        self.distilled  = True 

        # Build realistic-scale 14B transformer blocks (sim-compressed)
        self.blocks: List[TransformerBlock] = []
        # Token embedding (32000 × 5120)
        self.token_embed = BitNetLayer(VOCAB_SIZE, HIDDEN_DIM, "token_embed")
        # LM head (5120 × 32000) — tied weights in real models
        self.lm_head     = BitNetLayer(HIDDEN_DIM, VOCAB_SIZE, "lm_head")

    def boot_sequence(self, callback):
        callback("Initializing 4-bit Quantized Weights (14B scale)...")
        callback(f"[SYS] Architecture: {NUM_LAYERS}L × {HIDDEN_DIM}d | FFN={FFN_DIM} | Heads={NUM_HEADS}×{HEAD_DIM}")
        callback(f"[SYS] Total param count: {SIMULATED_TOTAL/1e9:.2f}B")
        callback(f"[SYS] Storage @ 4 bits/weight: {MODEL_SIZE_GB:.2f} GB (vs {SIMULATED_TOTAL*2/1024**3:.1f} GB fp16)")
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
        callback("[SYS] Applying 4-bit quantization (symmetric int4 range -8..7)...")
        time.sleep(0.1)
        callback("[SYS] R1 Distillation Policy Active: CoT Reasoning + Self-Reflection enabled.")
        time.sleep(0.1)
        callback(f"✓ R1Cat Distilled 4-Bit 14B Ready. ({MODEL_SIZE_GB:.2f} GB on disk)")
        self.is_ready = True

    def _generate_vector(self, query: str) -> List[float]:
        vec = [0.0] * HIDDEN_DIM
        for i, ch in enumerate(query):
            vec[i % HIDDEN_DIM] += ord(ch) / 255.0
        return vec

    # -------------------------------------------------------------------------
    # EXACT R1 STYLE REASONING CHAIN
    # -------------------------------------------------------------------------
    def _build_reasoning_steps(self, query: str) -> List[str]:
        """
        Generates a reasoning chain that mimics DeepSeek R1 exactly:
        1. Analysis
        2. Initial Plan
        3. Execution
        4. Self-Correction / Verification (The "Wait..." moment)
        5. Final Synthesis
        """
        steps = []
        q_lower = query.lower().strip()

        # --- PHASE 1: Initial Analysis (Mimics R1's "Okay, the user wants...") ---
        steps.append(f"Alright, the user is asking: \"{query}\"")
        steps.append("I need to process this request and generate a helpful response.")

        # --- PHASE 2: Categorization & Knowledge Retrieval ---
        if any(greet in q_lower for greet in ["hello", "hi", "hey", "greetings"]):
            steps.append("Analyzing intent: This appears to be a standard greeting.")
            steps.append("I should reply politely and ask how I can assist.")
            # R1 often does a "Wait, let me check if there's more context"
            steps.append("Checking context... No prior conversation history found.")
            steps.append("Proceeding with standard greeting protocol.")
        
        elif "your name" in q_lower:
            steps.append("Identifying query type: Question about self-identity.")
            steps.append("Retrieving model configuration: R1Cat, 4-bit, Distilled.")
            steps.append("I will formulate an answer that highlights the 4-bit efficiency and R1 lineage.")

        elif "?" in q_lower:
            steps.append("Query type detected: General Question.")
            steps.append("Breaking down the components of the question to understand the core need.")
            
            # Deep Think Logic (Expansion)
            if self.deep_mode:
                steps.append("DeepThink mode is active. I should perform a more thorough analysis.")
                steps.append("Let me simulate potential answers and check for logical fallacies.")
                steps.append("Searching internal knowledge base for relevant facts...")
                steps.append("Retrieved potential context. Now synthesizing...")
                
            # --- PHASE 3: The "Wait..." Moment (R1's signature self-correction) ---
            # R1 often hallucinates a check or corrects itself mid-stream.
            if random.random() < 0.6:
                steps.append("Wait, I should make sure I'm addressing the specific nuance here.")
                steps.append("Is there any ambiguity? Let me re-read the question.")
                steps.append(f"Re-reading: \"{query}\"")
                steps.append("Okay, I think I have the correct angle. I'll proceed with the answer.")
            else:
                steps.append("The intent is clear. I will provide a direct answer.")

        else:
            steps.append("Intent classification: Statement or conversational filler.")
            steps.append("I will acknowledge the input and guide the conversation forward.")
            if self.deep_mode:
                steps.append("Analyzing sentiment... Tone seems neutral/positive.")

        # --- PHASE 4: Final Logic Check ---
        steps.append("I have completed the reasoning process.")
        steps.append("Constructing the final response now.")

        return steps

    # -------------------------------------------------------------------------
    # Generate and stream thought chain
    # -------------------------------------------------------------------------
    def generate_thought_chain(self, query: str, message_queue: queue.Queue):
        hidden = self._generate_vector(query) 

        # Run through first and last block for representativeness
        block_0 = self.blocks[0] if self.blocks else None
        block_n = self.blocks[-1] if self.blocks else None
        act_0 = block_0.ffn_gate.forward(hidden[:HIDDEN_DIM // BitNetLayer.SIM_RATIO]) if block_0 else 0.0
        act_n = block_n.attn_qkv.forward(hidden[:HIDDEN_DIM // BitNetLayer.SIM_RATIO]) if block_n else 0.0

        # Build reasoning steps (Exact R1 Logic)
        reasoning_steps = self._build_reasoning_steps(query)

        thoughts_text = ""

        def stream_thought(text, delay_range=(0.015, 0.030)):
            nonlocal thoughts_text
            for ch in text:
                thoughts_text += ch
                message_queue.put(("thought", thoughts_text))
                time.sleep(random.uniform(*delay_range))
            thoughts_text += "\n\n"
            message_queue.put(("thought", thoughts_text))
            time.sleep(0.2)

        # Stream each reasoning step
        for step in reasoning_steps:
            stream_thought(step)

        # Generate final answer
        self._generate_response(query, message_queue)

    # -------------------------------------------------------------------------
    # Generate final response (R1 style)
    # -------------------------------------------------------------------------
    def _generate_response(self, query: str, message_queue: queue.Queue):
        """Generate a fluent English response based on the query."""
        q_lower = query.lower()

        # Response templates – R1 style
        if any(greet in q_lower for greet in ["hello", "hi", "hey", "greetings"]):
            base = "Hello! I am R1Cat, a distilled reasoning model running on 4-bit quantized weights. How can I assist you today?"
        elif "how are you" in q_lower:
            base = "I am functioning optimally. My quantization logic is stable, and my reasoning chains are active. How are you?"
        elif "your name" in q_lower:
            base = "I am **R1Cat-Distill-4Bit-14B**. I am a simulation of a 14B parameter model compressed to 4-bits, designed to mimic the reasoning capabilities of DeepSeek R1."
        elif "?" in q_lower:
            base = (f"Based on my analysis of your question, here is the answer: "
                    f"{self._generate_generic_answer(q_lower)} "
                    f"Let me know if you need more details.")
        else:
            base = (f"I've noted your statement: \"{query}\". "
                    f"If you have any specific questions or need assistance with a task, feel free to ask!")

        # Stream the response
        stream = ""
        for ch in base:
            stream += ch
            message_queue.put(("answer", stream))
            time.sleep(0.009)
        message_queue.put(("done", None))

    def _generate_generic_answer(self, query: str) -> str:
        """Helper to produce a generic answer for questions."""
        if "what" in query:
            return "The definition varies by context, but generally, it refers to a well-known concept."
        elif "why" in query:
            return "The primary reasons are often attributed to underlying structural and historical factors."
        elif "how" in query:
            return "The process involves a sequence of logical steps that I can outline for you."
        else:
            return "This is a nuanced topic that requires considering multiple perspectives."


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

    def update_text(self, text):
        self.text_label.config(text=text)

    def mark_done(self, elapsed):
        self.toggle_btn.config(text=f"▾  Thought for {elapsed:.0f}s")


class R1CatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("R1Cat - Distilled 4-Bit 14B (DeepSeek R1 Distilled)")
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

        self.engine    = DistilledR1Engine()
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

        tk.Label(self.sidebar, text="🐋 R1Cat Distilled 14B", font=("Helvetica", 14, "bold"),
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
        tk.Label(topbar, text=f"R1Cat Distilled 4-Bit 14B  (R1 distillation, {MODEL_SIZE_GB:.2f} GB | {SIMULATED_TOTAL/1e9:.1f}B params)",
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
        tk.Label(self.current_bot_frame, text="🐋 R1Cat (Distilled 4-Bit 14B)",
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
    app  = R1CatApp(root)
    root.mainloop()

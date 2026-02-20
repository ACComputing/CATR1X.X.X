"""
=============================================================================
🐱 CAT R1 - DEEPSEEK-NANO DISTILLATION ENGINE
=============================================================================
Based on:
  • DeepSeek-R1 Technical Report (GRPO Training, "Aha Moment" Discovery)
  • OpenAI O1 Reasoning Paper (Test-Time Compute Scaling, Process Supervision)
  • DeepSeek-V3 MoE Architecture (MLA, DeepSeekMoE)

Key Innovations Preserved:
  1. GRPO (Group Relative Policy Optimization) - No value function needed
  2. Emergent "Aha Moments" - Self-verification without explicit training
  3. Test-Time Compute Scaling - Dynamic reasoning depth
  4. Process Supervision - Reward per reasoning step, not just outcome
=============================================================================
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import sys
import queue
import webbrowser
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque

# =============================================================================
# WHITEPAPER REFERENCE: O1-STYLE REASONING & DEEPSEEK GRPO
# =============================================================================

WHITEPAPER_O1 = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    OPENAI O1 REASONING FRAMEWORK                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  TEST-TIME COMPUTE SCALING                                                ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • O1 allocates additional compute at inference time for harder problems  ║
║  • Generates hidden "thought tokens" before producing final answer        ║
║  • Performance scales with compute budget (O(n log n) for difficulty)     ║
║                                                                           ║
║  PROCESS SUPERVISION                                                      ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Reward given for CORRECT reasoning steps, not just final answer        ║
║  • Prevents reward hacking where model finds wrong path to right answer   ║
║  • Uses per-step verifier (PRM - Process Reward Model)                    ║
║                                                                           ║
║  SELF-CORRECTION MECHANISMS                                               ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Backtracking: Model can abandon failed reasoning paths                 ║
║  • Branching: Explore multiple solution approaches simultaneously         ║
║  • Verification: Self-check intermediate results before proceeding        ║
║                                                                           ║
║  KEY INSIGHT: Reasoning is a PROCESS, not just input→output mapping       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

WHITEPAPER_DEEPSEEK_GRPO = """
╔═══════════════════════════════════════════════════════════════════════════╗
║              DEEPSEEK-R1: GRPO & THE "AHA MOMENT" DISCOVERY               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  GRPO (Group Relative Policy Optimization)                                ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Eliminates need for critic/value function (saves 50% memory)           ║
║  • Uses GROUP of samples per prompt, computes baseline from group mean    ║
║  • Advantage = (reward_i - mean(rewards_group)) / std(rewards_group)      ║
║                                                                           ║
║  THE "AHA MOMENT"                                                         ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  DeepSeek researchers discovered:                                         ║
║                                                                           ║
║  "During training, the model spontaneously learned to:                    ║
║   1. Allocate more thinking time to difficult problems                    ║
║   2. Self-verify its intermediate reasoning steps                         ║
║   3. Reflect on mistakes and revise its approach                          ║
║   4. Generate aha-moment-style insights without explicit supervision"     ║
║                                                                           ║
║  This was NOT trained explicitly - it EMERGED from GRPO optimization!     ║
║                                                                           ║
║  GRPO LOSS FUNCTION                                                       ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  L(θ) = -E[ Σ π_θ(a_t|s_t) / π_old(a_t|s_t) × A_t × clip(...) ]          ║
║         + β × KL(π_θ || π_ref)                                            ║
║                                                                           ║
║  Where A_t = (r_i - μ_group) / σ_group (group-relative advantage)         ║
║                                                                           ║
║  KEY INSIGHT: Simple objective → Complex emergent reasoning behavior      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

WHITEPAPER_MLA_MOE = """
╔═══════════════════════════════════════════════════════════════════════════╗
║            DEEPSEEK-V3 ARCHITECTURE: MLA + DEEPSEEKMOE                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  MLA (Multi-Head Latent Attention)                                        ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Compresses KV cache by 93% (key-value compression)                     ║
║  • Projects queries/keys/values to latent space before attention          ║
║  • Enables 128K+ context with minimal memory overhead                      ║
║                                                                           ║
║  DeepSeekMoE (Mixture of Experts)                                         ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • 256 total experts, top-8 active per token (sparse activation)          ║
║  • 1 shared expert (always active) + routed experts                       ║
║  • Each expert is FFN with SwiGLU activation                              ║
║  • Load balancing via auxiliary loss (avoids expert collapse)             ║
║                                                                           ║
║  KNOWLEDGE DISTILLATION                                                    ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Teacher: DeepSeek-V3 (671B params, 37B active)                         ║
║  • Student: Cat-R1-Nano (1.5B params, distilled)                          ║
║  • Distillation loss: KL(student_logits || teacher_logits) + MSE(hidden)  ║
║                                                                           ║
║  INFERENCE EFFICIENCY                                                     ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  • Only 6% of parameters active per forward pass                          ║
║  • Enables consumer GPU inference at 50+ tokens/sec                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# ARCHITECTURE DATA STRUCTURES
# =============================================================================

@dataclass
class ExpertConfig:
    """MoE Expert Configuration (DeepSeekMoE style)"""
    total_experts: int = 16
    active_experts: int = 2          # top-k routing
    shared_experts: int = 1          # always active
    expert_dim: int = 512            # FFN hidden dim
    activation: str = "swiglu"


@dataclass
class MLAConfig:
    """Multi-Head Latent Attention Configuration"""
    num_heads: int = 32
    head_dim: int = 64
    kv_compression_ratio: float = 0.07  # 93% compression
    latent_dim: int = 256
    rope_theta: float = 10000.0


@dataclass
class GRPOConfig:
    """GRPO Training Configuration"""
    group_size: int = 8              # samples per prompt
    temperature: float = 0.7
    kl_penalty: float = 0.01         # β in GRPO loss
    max_steps: int = 32              # max reasoning steps
    reward_scale: float = 1.0


@dataclass
class ReasoningStep:
    """Single step in chain-of-thought reasoning"""
    content: str
    step_type: str                   # "think", "verify", "branch", "conclude"
    confidence: float
    expert_ids: List[int]
    self_corrected: bool = False
    aha_moment: bool = False


# =============================================================================
# CAT R1 INFERENCE ENGINE (SIMULATING DEEPSEEK-R1 BEHAVIOR)
# =============================================================================

class CatInferenceEngine:
    """
    Simulates DeepSeek-R1 distilled reasoning with:
    - GRPO-style group sampling
    - Emergent "aha moments"
    - O1-style test-time compute scaling
    - MLA + MoE architecture simulation
    """
    
    def __init__(self):
        self.is_ready = False
        self.model_mode = "Cat-R1-Nano"
        self.deep_mode = False
        
        # Architecture configs
        self.expert_cfg = ExpertConfig()
        self.mla_cfg = MLAConfig()
        self.grpo_cfg = GRPOConfig()
        
        # Model specs
        self.total_params = 1.5e9      # 1.5B parameters
        self.active_params = 0.8e9     # ~800M active (sparse MoE)
        self.num_layers = 24
        self.vocab_size = 102400
        self.context_length = 4096
        
        # Reasoning state
        self.reasoning_history: deque = deque(maxlen=100)
        self.total_reasoning_steps = 0
        self.aha_moments_count = 0
        
    def boot_sequence(self, callback):
        """Initialize model with architecture-aware boot sequence."""
        boot_steps = [
            f"╔═══ CAT R1 NANO ═══╗",
            f"║ Params: {self.total_params/1e9:.1f}B total, {self.active_params/1e9:.1f}B active ║",
            f"╚═══════════════════╝",
            "",
            "Loading architecture components...",
            f"  ► {self.num_layers} Transformer layers",
            f"  ► MLA: {self.mla_cfg.num_heads} heads, {self.mla_cfg.kv_compression_ratio:.0%} KV compression",
            f"  ► MoE: {self.expert_cfg.total_experts} experts, top-{self.expert_cfg.active_experts} routing",
            f"  ► GRPO: {self.grpo_cfg.group_size} samples/prompt group",
            "",
            "Applying knowledge distillation from DeepSeek-V3...",
            "  ► Teacher: 671B params, 37B active",
            "  ► Student: 1.5B params (this model)",
            "  ► Distillation: KL + MSE loss",
            "",
            "Initializing reasoning capabilities...",
            "  ► Chain-of-thought (O1-style)",
            "  ► Self-verification (GRPO emergent)",
            "  ► Test-time compute scaling",
            "",
            "✓ Cat R1 Nano Engine Online :3",
        ]
        
        for step in boot_steps:
            callback(step)
            time.sleep(random.uniform(0.05, 0.15))
        self.is_ready = True
        
    def select_experts(self) -> Tuple[List[int], List[float]]:
        """Simulate MoE expert routing with load balancing."""
        # Select top-k experts (simulated routing weights)
        all_experts = list(range(1, self.expert_cfg.total_experts + 1))
        
        # Simulate router producing weights
        weights = [random.random() for _ in all_experts]
        
        # Add shared expert (always active)
        shared = [0]  # expert 0 is shared
        
        # Select top-k routed experts
        sorted_idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        routed = [all_experts[i] for i in sorted_idx[:self.expert_cfg.active_experts]]
        routed_weights = [weights[i] for i in sorted_idx[:self.expert_cfg.active_experts]]
        
        # Normalize weights
        total = sum(routed_weights) + 1  # +1 for shared expert
        routed_weights = [w/total for w in routed_weights]
        shared_weight = 1/total
        
        return shared + routed, routed_weights + [shared_weight]
    
    def compute_grpo_advantage(self, rewards: List[float]) -> List[float]:
        """
        Compute GRPO-style group-relative advantages.
        
        A_i = (r_i - μ_group) / σ_group
        
        This eliminates the need for a value function!
        """
        if len(rewards) < 2:
            return [0.0]
        
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5 + 1e-8  # avoid div by zero
        
        advantages = [(r - mean) / std for r in rewards]
        return advantages
    
    def generate_reasoning_step(
        self, 
        query: str, 
        step_num: int,
        prev_steps: List[ReasoningStep],
        message_queue
    ) -> ReasoningStep:
        """Generate a single reasoning step with GRPO-aware simulation."""
        
        # Route through experts
        expert_ids, weights = self.select_experts()
        
        # Simulate different step types based on progress
        step_types = ["think", "verify", "branch", "conclude"]
        
        # Early steps: thinking
        # Middle steps: verification and branching
        # Late steps: conclusion
        if step_num < 3:
            step_type = "think"
        elif step_num < 6:
            step_type = random.choice(["think", "verify", "verify"])
        elif step_num < 8:
            step_type = random.choice(["verify", "branch", "conclude"])
        else:
            step_type = "conclude"
            
        # Generate content based on step type
        contents = {
            "think": [
                f"Parsing query: '{query[:30]}...'",
                f"Decomposing into sub-problems...",
                f"Activating expert {random.choice(expert_ids)} for domain knowledge...",
                f"Retrieving relevant context from {self.context_length}-token window...",
                f"Computing attention over {self.mla_cfg.num_heads} heads...",
            ],
            "verify": [
                "Self-verification: checking logical consistency...",
                "Backtracking check: is current path optimal?",
                "Reward signal: computing GRPO advantage...",
                "Cross-referencing with known facts...",
                "Confidence estimation: computing certainty...",
            ],
            "branch": [
                "Exploring alternative reasoning path...",
                "Branching: considering counterfactual...",
                "Multi-path evaluation: comparing 3 approaches...",
                "Divergent thinking: what if assumption is wrong?",
            ],
            "conclude": [
                "Synthesizing final answer...",
                "Formatting response in cat-friendly tone...",
                "Final confidence: computing ensemble certainty...",
                "Preparing output token stream...",
            ]
        }
        
        content = random.choice(contents.get(step_type, contents["think"]))
        
        # Simulate "aha moment" emergence (rare, happens during verification)
        aha_moment = False
        if step_type == "verify" and random.random() < 0.15:
            aha_moment = True
            self.aha_moments_count += 1
            content = "💡 AHA! Self-correction detected - revising reasoning path..."
            
        # Compute simulated confidence (GRPO advantage)
        confidence = random.uniform(0.7, 0.99)
        
        step = ReasoningStep(
            content=content,
            step_type=step_type,
            confidence=confidence,
            expert_ids=expert_ids,
            aha_moment=aha_moment
        )
        
        self.total_reasoning_steps += 1
        return step
    
    def generate(self, query: str, message_queue: queue.Queue):
        """Main generation loop with O1-style reasoning and GRPO."""
        
        # Log expert routing
        expert_ids, weights = self.select_experts()
        message_queue.put(("debug", f"Experts: {expert_ids} | Weights: {[f'{w:.2f}' for w in weights]}"))
        
        # Determine reasoning depth based on query complexity
        # (O1-style test-time compute scaling)
        base_steps = 5
        if self.deep_mode:
            base_steps = 12  # More compute for harder problems
        if len(query) > 100:
            base_steps += 3
            
        # Simulate GRPO group sampling (internal, not shown to user)
        group_rewards = [random.uniform(0.3, 1.0) for _ in range(self.grpo_cfg.group_size)]
        advantages = self.compute_grpo_advantage(group_rewards)
        best_idx = advantages.index(max(advantages))
        message_queue.put(("debug", f"GRPO group best: sample {best_idx} (advantage: {advantages[best_idx]:.2f})"))
        
        # Generate reasoning steps
        thoughts_text = ""
        prev_steps = []
        
        for step_num in range(base_steps):
            step = self.generate_reasoning_step(query, step_num, prev_steps, message_queue)
            prev_steps.append(step)
            
            # Format thought with step type indicator
            indicator = {
                "think": "🧠",
                "verify": "✓",
                "branch": "↗",
                "conclude": "🎯"
            }.get(step.step_type, "●")
            
            aha_marker = " 💡 AHA!" if step.aha_moment else ""
            thoughts_text += f"{indicator} [{step.step_type.upper()}] {step.content}{aha_marker}\n"
            thoughts_text += f"   Experts: {step.expert_ids} | Confidence: {step.confidence:.0%}\n\n"
            
            message_queue.put(("thought", thoughts_text))
            time.sleep(random.uniform(0.3, 0.6))
            
            if step.step_type == "conclude":
                break
        
        # Generate final answer with cat persona
        responses = [
            "Meow‑hematical analysis complete! 🐱\n\nAfter careful reasoning through the problem, here's what I found: *purrs contentedly*\n\nThe answer involves multiple considerations that I've verified through self-reflection. Hope this helps, nya~",
            "Reasoning finished! :3\n\nMy GRPO-trained weights converged on this solution after exploring several reasoning paths. *curls tail thoughtfully*\n\nAnything else I can help with, meow?",
            "Aha! I've got it! 💡🐱\n\nThrough emergent self-verification (a behavior that arose naturally from GRPO training), I arrived at this conclusion. Pretty cool how that works, right?\n\n*mrrp* Let me know if you need clarification!",
            "DeepSeek‑style reasoning complete! owo\n\nI allocated extra compute time for this one and verified my intermediate steps. The result should be reliable!\n\nHappy to explain any part of my thought process, nya~"
        ]
        
        answer = random.choice(responses)
        
        # Stream answer character by character
        current_text = ""
        for char in answer:
            current_text += char
            message_queue.put(("answer", current_text))
            time.sleep(0.015)
            
        # Final stats
        stats = f"\n\n📊 Stats: {len(prev_steps)} steps | {self.aha_moments_count} aha moments | {self.total_reasoning_steps} total steps this session"
        current_text += stats
        message_queue.put(("answer", current_text))
        message_queue.put(("done", None))


# =============================================================================
# UI COMPONENTS
# =============================================================================

class CollapsibleThought(tk.Frame):
    """DeepSeek-style collapsible reasoning block with O1-style formatting."""
    
    def __init__(self, parent, colors):
        super().__init__(parent, bg=colors["bg"], pady=5)
        self.colors = colors
        self.is_expanded = True

        # Header with toggle
        self.header = tk.Frame(self, bg=colors["think_bg"], cursor="hand2")
        self.header.pack(fill="x")
        self.header.bind("<Button-1>", self.toggle)

        self.toggle_label = tk.Label(
            self.header, 
            text="▼ Thought Process (O1-style Reasoning)",
            font=("Arial", 9, "bold"),
            bg=colors["think_bg"], 
            fg=colors["primary"],
            padx=10, pady=6
        )
        self.toggle_label.pack(side="left")
        self.toggle_label.bind("<Button-1>", self.toggle)

        # GRPO badge
        self.badge = tk.Label(
            self.header,
            text="GRPO",
            font=("Arial", 7, "bold"),
            bg=colors["primary"],
            fg="white",
            padx=6, pady=2
        )
        self.badge.pack(side="right", padx=10)

        # Content frame
        self.content_frame = tk.Frame(self, bg=colors["think_bg"])
        self.content_frame.pack(fill="x")

        mono_font = "Menlo" if sys.platform == "darwin" else "Consolas"
        self.text_label = tk.Label(
            self.content_frame, 
            text="", 
            font=(mono_font, 9),
            bg=colors["think_bg"], 
            fg=colors["think_text"],
            justify="left", 
            anchor="w", 
            wraplength=550,
            padx=15, pady=10
        )
        self.text_label.pack(fill="x")

    def toggle(self, event=None):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.content_frame.pack(fill="x")
            self.toggle_label.config(text="▼ Thought Process (O1-style Reasoning)")
        else:
            self.content_frame.pack_forget()
            self.toggle_label.config(text="▶ Show Thoughts")


class WhitepaperWindow(tk.Toplevel):
    """Window displaying O1 and DeepSeek whitepaper summaries."""
    
    def __init__(self, parent, colors):
        super().__init__(parent)
        self.title("📚 Whitepaper Reference: O1 & DeepSeek-R1")
        self.geometry("750x600")
        self.configure(bg=colors["bg"])
        
        self.colors = colors
        self.current_page = 0
        self.pages = [
            ("O1 Reasoning Framework", WHITEPAPER_O1),
            ("DeepSeek GRPO & Aha Moment", WHITEPAPER_DEEPSEEK_GRPO),
            ("MLA + DeepSeekMoE Architecture", WHITEPAPER_MLA_MOE),
        ]
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self, bg=self.colors["sidebar"])
        title_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            title_frame,
            text="📚 WHITEPAPER REFERENCE",
            font=("Arial", 14, "bold"),
            bg=self.colors["sidebar"],
            fg=self.colors["primary"],
            pady=15
        ).pack()
        
        # Navigation tabs
        tab_frame = tk.Frame(self, bg=self.colors["bg"])
        tab_frame.pack(fill="x", padx=20)
        
        self.tab_buttons = []
        for i, (name, _) in enumerate(self.pages):
            btn = tk.Button(
                tab_frame,
                text=name,
                font=("Arial", 9),
                bg=self.colors["primary"] if i == 0 else self.colors["sidebar"],
                fg="white",
                relief="flat",
                padx=15, pady=8,
                command=lambda idx=i: self.show_page(idx)
            )
            btn.pack(side="left", padx=2)
            self.tab_buttons.append(btn)
        
        # Content
        self.content_text = tk.Text(
            self,
            bg=self.colors["bg"],
            fg=self.colors["text_p"],
            font=("Menlo" if sys.platform == "darwin" else "Consolas", 9),
            wrap="word",
            padx=20, pady=20,
            relief="flat"
        )
        self.content_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.show_page(0)
        
    def show_page(self, idx):
        self.current_page = idx
        
        # Update tab buttons
        for i, btn in enumerate(self.tab_buttons):
            if i == idx:
                btn.config(bg=self.colors["primary"])
            else:
                btn.config(bg=self.colors["sidebar"])
        
        # Update content
        name, content = self.pages[idx]
        self.content_text.config(state="normal")
        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", content)
        self.content_text.config(state="disabled")


class CatSeekApp:
    """Main application with O1/DeepSeek-style reasoning UI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cat R1 - Local Intelligence (DeepSeek‑Nano Distill)")
        self.root.geometry("1150x800")

        self.colors = {
            "bg": "#050505",
            "sidebar": "#0f172a",
            "primary": "#3b82f6",
            "border": "#1e293b",
            "user_bubble": "#1d4ed8",
            "bot_bubble": "#1e293b",
            "think_bg": "#0f172a",
            "think_text": "#94a3b8",
            "text_p": "#f8fafc",
            "text_s": "#64748b",
            "aha": "#fbbf24",  # Gold for aha moments
        }

        self.root.configure(bg=self.colors["bg"])
        self.engine = CatInferenceEngine()
        self.msg_queue = queue.Queue()
        self.deep_mode = False

        self.setup_styles()
        self.setup_ui()

        self.root.after(100, self.process_queue)
        threading.Thread(
            target=self.engine.boot_sequence,
            args=(self.update_status,),
            daemon=True
        ).start()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=self.colors["border"],
            troughcolor=self.colors["bg"],
            bordercolor=self.colors["bg"],
            arrowcolor="white"
        )

    def setup_ui(self):
        # =========== SIDEBAR ===========
        self.sidebar = tk.Frame(self.root, width=280, bg=self.colors["sidebar"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo/Title
        tk.Label(
            self.sidebar, 
            text="🐱 Cat R1",
            font=("Arial", 24, "bold"),
            bg=self.colors["sidebar"], 
            fg=self.colors["primary"]
        ).pack(pady=(30, 5))

        tk.Label(
            self.sidebar, 
            text="DeepSeek‑Nano Distill",
            font=("Arial", 10),
            bg=self.colors["sidebar"], 
            fg=self.colors["text_s"]
        ).pack()
        
        tk.Label(
            self.sidebar,
            text="1.5B params • GRPO trained",
            font=("Arial", 8),
            bg=self.colors["sidebar"],
            fg=self.colors["text_s"]
        ).pack()

        # Model selection
        tk.Label(
            self.sidebar, 
            text="MODEL",
            font=("Arial", 8, "bold"),
            bg=self.colors["sidebar"], 
            fg=self.colors["text_s"]
        ).pack(anchor="w", padx=20, pady=(30, 5))

        self.model_var = tk.StringVar(value="Cat-R1-Nano")
        for m in ["Cat-R1-Nano", "Cat-R1-Micro"]:
            rb = tk.Radiobutton(
                self.sidebar, 
                text=m,
                variable=self.model_var, 
                value=m,
                command=self.change_model,
                bg=self.colors["sidebar"],
                fg="white",
                selectcolor=self.colors["primary"],
                activebackground=self.colors["sidebar"],
                font=("Arial", 10)
            )
            rb.pack(anchor="w", padx=30, pady=2)

        # DeepThink toggle
        self.deepthink_btn = tk.Button(
            self.sidebar, 
            text="🔍 DeepThink OFF",
            bg="#000000",
            fg=self.colors["primary"],
            font=("Arial", 10, "bold"),
            relief="flat",
            activebackground="#000000",
            activeforeground=self.colors["primary"],
            command=self.toggle_deepthink
        )
        self.deepthink_btn.pack(anchor="w", padx=20, pady=(20, 5), fill="x")

        # Whitepaper button
        self.whitepaper_btn = tk.Button(
            self.sidebar,
            text="📚 Whitepaper Reference",
            bg="#000000",
            fg=self.colors["primary"],
            font=("Arial", 10, "bold"),
            relief="flat",
            activebackground="#000000",
            activeforeground=self.colors["primary"],
            command=self.show_whitepaper
        )
        self.whitepaper_btn.pack(anchor="w", padx=20, pady=5, fill="x")

        # Online chat button
        self.chat_btn = tk.Button(
            self.sidebar, 
            text="💬 Chat.deepseek.com",
            bg="#000000",
            fg=self.colors["primary"],
            font=("Arial", 10, "bold"),
            relief="flat",
            activebackground="#000000",
            activeforeground=self.colors["primary"],
            command=self.open_chat
        )
        self.chat_btn.pack(anchor="w", padx=20, pady=5, fill="x")

        # Stats display
        self.stats_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar"])
        self.stats_frame.pack(side="bottom", fill="x", pady=20, padx=20)
        
        self.stats_label = tk.Label(
            self.stats_frame,
            text="Steps: 0 | Aha: 0",
            font=("Arial", 8),
            bg=self.colors["sidebar"],
            fg=self.colors["text_s"]
        )
        self.stats_label.pack()

        self.status_label = tk.Label(
            self.stats_frame, 
            text="Initializing...",
            font=("Arial", 8),
            bg=self.colors["sidebar"],
            fg=self.colors["primary"],
            wraplength=240
        )
        self.status_label.pack(pady=(5, 0))

        # =========== MAIN CHAT AREA ===========
        self.main_container = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_container.pack(side="right", fill="both", expand=True)

        # Scrollable chat area
        self.canvas = tk.Canvas(
            self.main_container, 
            bg=self.colors["bg"], 
            highlightthickness=0
        )
        self.scroll_frame = tk.Frame(self.canvas, bg=self.colors["bg"])
        self.scrollbar = ttk.Scrollbar(
            self.main_container, 
            orient="vertical", 
            command=self.canvas.yview
        )

        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="top", fill="both", expand=True, padx=40, pady=20)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Input area
        input_frame = tk.Frame(self.main_container, bg=self.colors["bg"])
        input_frame.pack(side="bottom", fill="x", padx=40, pady=20)

        self.entry = tk.Entry(
            input_frame,
            bg=self.colors["sidebar"], 
            fg="white",
            insertbackground="white",
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["primary"]
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=12, padx=(0, 15))
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.send_btn = tk.Button(
            input_frame, 
            text="Send ➤",
            bg="#000000", 
            fg=self.colors["primary"],
            font=("Arial", 10, "bold"),
            relief="flat", 
            padx=25,
            command=self.send_message,
            activebackground="#000000",
            activeforeground=self.colors["primary"]
        )
        self.send_btn.pack(side="right")

    def toggle_deepthink(self):
        self.deep_mode = not self.deep_mode
        self.engine.deep_mode = self.deep_mode
        state_text = "ON" if self.deep_mode else "OFF"
        self.deepthink_btn.config(text=f"🔍 DeepThink {state_text}")
        self.update_status(f"DeepThink {'enabled' if self.deep_mode else 'disabled'} (more compute)")

    def show_whitepaper(self):
        WhitepaperWindow(self.root, self.colors)

    def open_chat(self):
        webbrowser.open("https://chat.deepseek.com")

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def change_model(self):
        self.engine.model_mode = self.model_var.get()
        self.update_status(f"Switched to {self.engine.model_mode}")

    def send_message(self):
        query = self.entry.get().strip()
        if not query or not self.engine.is_ready:
            return
        self.entry.delete(0, tk.END)
        self.add_bubble("YOU", query, is_bot=False)
        self.active_thought_block = None
        self.active_answer_label = None
        self.current_wrapper = self.create_bot_wrapper()
        threading.Thread(
            target=self.engine.generate,
            args=(query, self.msg_queue),
            daemon=True
        ).start()

    def add_bubble(self, sender, text, is_bot):
        wrapper = tk.Frame(self.scroll_frame, bg=self.colors["bg"])
        wrapper.pack(fill="x", anchor="w", pady=10)

        tk.Label(
            wrapper, 
            text=sender,
            font=("Arial", 8, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["primary"] if is_bot else self.colors["text_s"]
        ).pack(anchor="w")

        bubble = tk.Label(
            wrapper, 
            text=text,
            bg=self.colors["bot_bubble"] if is_bot else self.colors["user_bubble"],
            fg="white", 
            font=("Arial", 11),
            justify="left", 
            wraplength=550,
            padx=15, pady=10
        )
        bubble.pack(anchor="w", pady=5)
        self.root.after(10, lambda: self.canvas.yview_moveto(1.0))
        return bubble

    def create_bot_wrapper(self):
        wrapper = tk.Frame(self.scroll_frame, bg=self.colors["bg"])
        wrapper.pack(fill="x", anchor="w", pady=10)

        tk.Label(
            wrapper, 
            text="CAT R1",
            font=("Arial", 8, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["primary"]
        ).pack(anchor="w")

        self.debug_label = tk.Label(
            wrapper, 
            text="",
            font=("Courier", 8),
            bg=self.colors["bg"],
            fg="#10b981"
        )
        self.debug_label.pack(anchor="w")
        return wrapper

    def process_queue(self):
        try:
            while True:
                mode, content = self.msg_queue.get_nowait()
                if mode == "debug":
                    self.debug_label.config(text=content)
                elif mode == "thought":
                    if not self.active_thought_block:
                        self.active_thought_block = CollapsibleThought(
                            self.current_wrapper, 
                            self.colors
                        )
                        self.active_thought_block.pack(fill="x", pady=5)
                    self.active_thought_block.update_text(content)
                elif mode == "answer":
                    if not self.active_answer_label:
                        self.active_answer_label = tk.Label(
                            self.current_wrapper, 
                            text="",
                            bg=self.colors["bot_bubble"],
                            fg="white", 
                            font=("Arial", 11),
                            justify="left", 
                            wraplength=550,
                            padx=15, pady=10
                        )
                        self.active_answer_label.pack(anchor="w", pady=5)
                    self.active_answer_label.config(text=content)
                    
                    # Update stats
                    self.stats_label.config(
                        text=f"Steps: {self.engine.total_reasoning_steps} | Aha: {self.engine.aha_moments_count}"
                    )
                elif mode == "done":
                    pass
                self.canvas.yview_moveto(1.0)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = CatSeekApp(root)
    root.mainloop()

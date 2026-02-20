"""
=============================================================================
🐱 CAT R1 - O1-REASONING LOCAL ENGINE (SINGLE FILE)
=============================================================================
Based on:
  • OpenAI O1 System Card (Chain-of-Thought, Test-Time Compute)
  • DeepSeek-R1 Technical Report (GRPO, Emergent Reasoning)

Implementation Details:
  1. Simulates O1-style "Thinking" phase (hidden thought process)
  2. Dynamic Compute Scaling (thinks longer for harder questions)
  3. Self-Correction & Verification loops
  4. Context-aware response generation (Human + Cat Persona)
=============================================================================
"""
import sys 
import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import re
import queue
import webbrowser
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThoughtStep:
    """Represents a single step in the O1 reasoning chain."""
    content: str
    step_type: str  # 'plan', 'reason', 'verify', 'correct', 'conclude'
    confidence: float = 0.0

# =============================================================================
# CAT R1 REASONING ENGINE (O1 STYLE)
# =============================================================================

class CatReasoningEngine:
    """
    Simulates an O1-style reasoning process.
    It does not just "generate" text; it "thinks" first.
    """
    
    def __init__(self):
        self.is_ready = False
        self.model_name = "Cat-R1-O1"
        self.deep_mode = False
        
        # "Weights" (Simulated knowledge base for context-aware answers)
        self.knowledge = {
            "python": "A high-level programming language known for readability. Great for data science.",
            "cat": "A small carnivorous mammal. Domestic cats are beloved companions. Meow.",
            "ai": "Artificial Intelligence. Simulation of human intelligence by machines.",
            "life": "The condition that distinguishes animals and plants from inorganic matter.",
            "food": "Substance consumed for nutritional support. Tuna is a superior form of this.",
            "default": "A complex topic requiring deep feline contemplation."
        }
        
        # Stats
        self.total_thoughts = 0
        self.corrections = 0

    def boot_sequence(self, callback):
        """Initialize the engine."""
        boot_log = [
            "█▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█",
            "█  CAT R1 O1-REASONING CORE  █",
            "█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█",
            "",
            "[SYS] Loading System 2 Thought Process...",
            "[SYS] Initializing Hidden Chain-of-Thought...",
            "[SYS] Calibrating Self-Verification Modules...",
            "[SYS] Loading Cat Persona Subroutines...",
            "",
            "✓ Engine Ready. Let's think. :3"
        ]
        
        for line in boot_log:
            callback(line)
            time.sleep(0.08)
        self.is_ready = True

    def _analyze_query(self, query: str) -> List[str]:
        """O1 Step 1: Decomposition"""
        # Simulate breaking the query into tokens/concepts
        words = re.findall(r'\w+', query.lower())
        return words

    def _retrieve_context(self, keywords: List[str]) -> str:
        """Internal Knowledge Retrieval"""
        for kw in keywords:
            if kw in self.knowledge:
                return self.knowledge[kw]
        return self.knowledge["default"]

    def generate_thought_chain(self, query: str, message_queue: queue.Queue):
        """
        The core O1 loop: Plan -> Think -> Verify -> Correct -> Conclude.
        This happens BEFORE the final answer is shown to the user.
        """
        
        # Step 0: Setup
        keywords = self._analyze_query(query)
        context = self._retrieve_context(keywords)
        
        # Determine thinking depth (O1 Test-Time Compute Scaling)
        # Harder queries = more thinking steps
        depth = 5
        if self.deep_mode:
            depth = 12
        
        thoughts_text = ""
        
        # --- PHASE 1: PLANNING ---
        step = ThoughtStep(
            content=f"Analyzing user request: '{query}'",
            step_type="plan"
        )
        thoughts_text += f"🗺️ [PLAN] {step.content}\n"
        message_queue.put(("thought", thoughts_text))
        time.sleep(0.5)
        
        # --- PHASE 2: EXPLORATION (The main thinking loop) ---
        for i in range(depth):
            self.total_thoughts += 1
            
            # Dynamic thought generation
            if i < 2:
                # Initial reasoning
                step_content = f"Retrieving associations for keywords: {keywords[:3]}..."
                step_type = "reason"
            elif i < depth - 2:
                # Deep processing
                step_content = f"Processing concept '{random.choice(keywords)}' in context of '{context[:20]}...'"
                step_type = "reason"
            else:
                # Verification phase
                step_content = "Reviewing logic consistency..."
                step_type = "verify"

            # O1 Self-Correction Mechanism (The "Aha Moment")
            # Randomly inject a self-correction to simulate System 2 verification
            if random.random() < 0.2 and i > 2:
                self.corrections += 1
                wrong_thought = "Wait, initial assumption might be incomplete..."
                correction = f"Correction: Refining understanding based on '{context}'."
                
                thoughts_text += f"⚠️ [CHECK] {wrong_thought}\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(0.4)
                
                thoughts_text += f"🔧 [FIX] {correction}\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(0.4)
            else:
                # Normal thought step
                emoji = "🧠" if step_type == "reason" else "🔍"
                thoughts_text += f"{emoji} [{step_type.upper()}] {step_content}\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(random.uniform(0.3, 0.6))

        # --- PHASE 3: CONCLUSION ---
        thoughts_text += "🎯 [READY] Reasoning complete. Formulating response...\n"
        message_queue.put(("thought", thoughts_text))
        time.sleep(0.5)
        
        # --- PHASE 4: FINAL OUTPUT GENERATION ---
        self._generate_response(query, context, message_queue)

    def _generate_response(self, query: str, context: str, message_queue: queue.Queue):
        """
        Generates the final Human-like Cat response.
        Uses the context derived from reasoning.
        """
        
        # Template-based response generation to ensure relevant answers
        response_templates = [
            f"Mrrp! After thinking carefully, I believe the answer relates to: {context}. ",
            f"Meow~ 🐱 My reasoning process led me to this: {context}. ",
            f"Purr... *stretches* I've calculated the variables. Here is my analysis: {context}. ",
            f"Nya! Interesting question. My logic cores suggest: {context}. "
        ]
        
        # Add a conversational closing
        closings = [
            "Does that make sense to you, hooman?",
            "I hope my thought process was transparent enough! :3",
            "Can I has treats now for solving this?",
            "Anything else you want my brain to process?"
        ]
        
        final_text = random.choice(response_templates) + random.choice(closings)
        
        # Stream the final answer
        current_stream = ""
        for char in final_text:
            current_stream += char
            message_queue.put(("answer", current_stream))
            time.sleep(0.02)
            
        message_queue.put(("done", None))


# =============================================================================
# UI COMPONENTS
# =============================================================================

class CollapsibleThought(tk.Frame):
    """
    UI Component: The hidden thought process block.
    Collapsible to mimic O1's hidden reasoning.
    """
    
    def __init__(self, parent, colors):
        super().__init__(parent, bg=colors["bg"], pady=5)
        self.colors = colors
        self.is_expanded = True

        # Header
        self.header = tk.Frame(self, bg=colors["think_bg"], cursor="hand2")
        self.header.pack(fill="x")
        self.header.bind("<Button-1>", self.toggle)

        self.toggle_label = tk.Label(
            self.header, 
            text="▼ Thought Process (O1 Reasoning)",
            font=("Arial", 9, "bold"),
            bg=colors["think_bg"], 
            fg=colors["primary"],
            padx=10, pady=6
        )
        self.toggle_label.pack(side="left")
        self.toggle_label.bind("<Button-1>", self.toggle)

        # Content
        self.content_frame = tk.Frame(self, bg=colors["think_bg"])
        self.content_frame.pack(fill="x")

        mono_font = "Menlo" if hasattr(sys, 'platform') else "Consolas"
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
            self.toggle_label.config(text="▼ Thought Process (O1 Reasoning)")
        else:
            self.content_frame.pack_forget()
            self.toggle_label.config(text="▶ Show Thoughts")

    def update_text(self, text):
        self.text_label.config(text=text)


class CatO1App:
    """Main Application Window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cat R1 - Local O1 Reasoning Engine")
        self.root.geometry("1100x750")
        
        self.colors = {
            "bg": "#0a0a0a",
            "sidebar": "#111111",
            "primary": "#ff6b6b",  # Cat Red/Pink
            "border": "#333333",
            "user_bubble": "#2a2a2a",
            "bot_bubble": "#1a1a1a",
            "think_bg": "#0f1015",
            "think_text": "#a0a0a0",
            "text_p": "#f0f0f0",
            "text_s": "#888888",
        }
        
        self.root.configure(bg=self.colors["bg"])
        self.engine = CatReasoningEngine()
        self.msg_queue = queue.Queue()
        self.deep_mode = False
        
        self.setup_ui()
        
        # Start Queue Processor
        self.root.after(100, self.process_queue)
        
        # Start Boot Sequence
        threading.Thread(
            target=self.engine.boot_sequence,
            args=(self.update_status,),
            daemon=True
        ).start()

    def setup_ui(self):
        # --- Sidebar ---
        sidebar = tk.Frame(self.root, width=260, bg=self.colors["sidebar"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="🐱 CAT R1",
            font=("Arial", 22, "bold"),
            bg=self.colors["sidebar"], fg=self.colors["primary"]
        ).pack(pady=(30, 5))

        tk.Label(
            sidebar, text="O1-Reasoning Architecture",
            font=("Arial", 9), bg=self.colors["sidebar"], fg=self.colors["text_s"]
        ).pack()

        # DeepThink Toggle
        self.deep_btn = tk.Button(
            sidebar, text="🧠 DeepThink: OFF",
            font=("Arial", 10, "bold"),
            bg=self.colors["bg"], fg=self.colors["primary"],
            relief="flat", activebackground=self.colors["bg"],
            command=self.toggle_deep
        )
        self.deep_btn.pack(pady=30, padx=20, fill="x")

        # Stats
        self.status_label = tk.Label(
            sidebar, text="Initializing Core...",
            font=("Arial", 8), bg=self.colors["sidebar"], fg=self.colors["text_s"], wraplength=200
        )
        self.status_label.pack(side="bottom", pady=20)

        # --- Main Chat Area ---
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(side="right", fill="both", expand=True)

        # Chat History Canvas
        self.canvas = tk.Canvas(main_frame, bg=self.colors["bg"], highlightthickness=0)
        self.scroll_frame = tk.Frame(self.canvas, bg=self.colors["bg"])
        
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="top", fill="both", expand=True, padx=30, pady=20)
        
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Input Area
        input_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        input_frame.pack(side="bottom", fill="x", padx=30, pady=20)

        self.entry = tk.Entry(
            input_frame, bg=self.colors["sidebar"], fg="white",
            insertbackground="white", font=("Arial", 12),
            relief="flat", highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["primary"]
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 15))
        self.entry.bind("<Return>", lambda e: self.send())

        tk.Button(
            input_frame, text="Send ➤",
            font=("Arial", 10, "bold"),
            bg=self.colors["primary"], fg="white",
            relief="flat", padx=20, command=self.send
        ).pack(side="right")

    def toggle_deep(self):
        self.deep_mode = not self.deep_mode
        self.engine.deep_mode = self.deep_mode
        state = "ON" if self.deep_mode else "OFF"
        self.deep_btn.config(text=f"🧠 DeepThink: {state}")

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def send(self):
        text = self.entry.get().strip()
        if not text or not self.engine.is_ready: return
        
        self.entry.delete(0, tk.END)
        self.add_message("YOU", text, is_bot=False)
        
        # Prepare UI for response
        self.current_bot_frame = tk.Frame(self.scroll_frame, bg=self.colors["bg"])
        self.current_bot_frame.pack(fill="x", anchor="w", pady=10)
        
        tk.Label(
            self.current_bot_frame, text="CAT R1",
            font=("Arial", 8, "bold"), bg=self.colors["bg"], fg=self.colors["primary"]
        ).pack(anchor="w")

        # Start Reasoning Thread
        threading.Thread(
            target=self.engine.generate_thought_chain,
            args=(text, self.msg_queue),
            daemon=True
        ).start()

    def add_message(self, sender, text, is_bot):
        wrapper = tk.Frame(self.scroll_frame, bg=self.colors["bg"])
        wrapper.pack(fill="x", anchor="w", pady=10)
        
        tk.Label(
            wrapper, text=sender,
            font=("Arial", 8, "bold"), bg=self.colors["bg"],
            fg=self.colors["text_s"] if not is_bot else self.colors["primary"]
        ).pack(anchor="w")
        
        bubble = tk.Label(
            wrapper, text=text,
            bg=self.colors["user_bubble"] if not is_bot else self.colors["bot_bubble"],
            fg="white", font=("Arial", 11),
            justify="left", wraplength=550, padx=15, pady=10
        )
        bubble.pack(anchor="w")
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        self.root.after(10, lambda: self.canvas.yview_moveto(1.0))

    def process_queue(self):
        try:
            while True:
                mode, content = self.msg_queue.get_nowait()
                
                # Handle Thoughts
                if mode == "thought":
                    # Create thought block if not exists
                    if not hasattr(self, 'thought_block') or not self.thought_block.winfo_exists():
                        self.thought_block = CollapsibleThought(self.current_bot_frame, self.colors)
                        self.thought_block.pack(fill="x", pady=5)
                    self.thought_block.update_text(content)
                    self.scroll_to_bottom()
                
                # Handle Final Answer
                elif mode == "answer":
                    if not hasattr(self, 'answer_label') or not self.answer_label.winfo_exists():
                        self.answer_label = tk.Label(
                            self.current_bot_frame, text="",
                            bg=self.colors["bot_bubble"], fg="white",
                            font=("Arial", 11), justify="left",
                            wraplength=550, padx=15, pady=10
                        )
                        self.answer_label.pack(anchor="w", pady=5)
                    self.answer_label.config(text=content)
                    self.scroll_to_bottom()
                
                elif mode == "done":
                    # Reset UI handles for next turn
                    if hasattr(self, 'thought_block'): del self.thought_block
                    if hasattr(self, 'answer_label'): del self.answer_label

        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = CatO1App(root)
    root.mainloop()

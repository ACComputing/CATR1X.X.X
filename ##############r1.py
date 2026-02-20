"""
=============================================================================
🐱 CAT R1 - DeepSeek chat.deepseek.com ACCURATE UI
=============================================================================
UI faithfully replicates chat.deepseek.com:
  • Dark sidebar (#1C1C1E) with nav icons
  • White main chat area
  • Blue "New Chat" button
  • User messages right-aligned, plain text
  • Bot messages left-aligned with avatar
  • Collapsible "Thought Process" block with left blue border (like DeepSeek R1)
  • Streaming token output
  • Bottom input bar with rounded textarea, send button
  • DeepThink toggle button in sidebar
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
from dataclasses import dataclass
from typing import List

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThoughtStep:
    content: str
    step_type: str
    confidence: float = 0.0

# =============================================================================
# CAT R1 REASONING ENGINE
# =============================================================================

class CatReasoningEngine:
    def __init__(self):
        self.is_ready = False
        self.model_name = "Cat-R1"
        self.deep_mode = False

        self.knowledge = {
            "python": "A high-level programming language known for readability. Great for data science.",
            "cat": "A small carnivorous mammal. Domestic cats are beloved companions. Meow.",
            "ai": "Artificial Intelligence — the simulation of human intelligence by machines.",
            "life": "The condition that distinguishes living organisms from inorganic matter.",
            "food": "Substance consumed for nutritional support. Tuna is a superior form of this.",
            "default": "A complex topic requiring deep feline contemplation."
        }

        self.total_thoughts = 0
        self.corrections = 0

    def boot_sequence(self, callback):
        boot_log = [
            "Initializing Cat R1 Engine...",
            "[SYS] Loading Chain-of-Thought Modules...",
            "[SYS] Calibrating Self-Verification...",
            "[SYS] Loading Cat Persona...",
            "✓ Ready. Let's think. :3"
        ]
        for line in boot_log:
            callback(line)
            time.sleep(0.1)
        self.is_ready = True

    def _analyze_query(self, query: str) -> List[str]:
        return re.findall(r'\w+', query.lower())

    def _retrieve_context(self, keywords: List[str]) -> str:
        for kw in keywords:
            if kw in self.knowledge:
                return self.knowledge[kw]
        return self.knowledge["default"]

    def generate_thought_chain(self, query: str, message_queue: queue.Queue):
        keywords = self._analyze_query(query)
        context = self._retrieve_context(keywords)
        depth = 10 if self.deep_mode else 5
        thoughts_text = ""

        # PLAN
        thoughts_text += f"Analyzing user request: '{query}'\n\n"
        message_queue.put(("thought", thoughts_text))
        time.sleep(0.4)

        # EXPLORE
        for i in range(depth):
            self.total_thoughts += 1

            if i < 2:
                step_content = f"Retrieving associations for keywords: {keywords[:3]}..."
            elif i < depth - 2:
                step_content = f"Processing concept '{random.choice(keywords) if keywords else 'data'}' against context: '{context[:30]}...'"
            else:
                step_content = "Verifying logical consistency across reasoning steps..."

            if random.random() < 0.2 and i > 2:
                self.corrections += 1
                thoughts_text += f"Wait, initial assumption might be incomplete. Let me reconsider...\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(0.35)
                thoughts_text += f"Correction: Refining understanding based on broader context of '{context[:25]}'.\n\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(0.35)
            else:
                thoughts_text += f"{step_content}\n"
                message_queue.put(("thought", thoughts_text))
                time.sleep(random.uniform(0.25, 0.5))

        thoughts_text += "\nReasoning complete. Formulating response...\n"
        message_queue.put(("thought", thoughts_text))
        time.sleep(0.4)

        self._generate_response(query, context, message_queue)

    def _generate_response(self, query: str, context: str, message_queue: queue.Queue):
        templates = [
            f"Mrrp! After careful deliberation, I believe the answer relates to: {context} Does that make sense to you, hooman?",
            f"Meow~ 🐱 My reasoning process led me here: {context} I hope my thought process was transparent enough! :3",
            f"Purr... *stretches* I've run the calculations. My analysis: {context} Can I has treats now for solving this?",
            f"Nya! Interesting question. My logic cores conclude: {context} Anything else you want my brain to process?"
        ]
        final_text = random.choice(templates)
        current_stream = ""
        for char in final_text:
            current_stream += char
            message_queue.put(("answer", current_stream))
            time.sleep(0.018)
        message_queue.put(("done", None))


# =============================================================================
# COLLAPSIBLE THOUGHT BLOCK — DeepSeek style (blue left border)
# =============================================================================

class CollapsibleThought(tk.Frame):
    def __init__(self, parent, colors):
        super().__init__(parent, bg=colors["chat_bg"])
        self.colors = colors
        self.is_expanded = True
        self._build()

    def _build(self):
        c = self.colors

        # Header row
        header = tk.Frame(self, bg=c["chat_bg"], cursor="hand2")
        header.pack(fill="x", pady=(0, 0))

        # Thinking icon + label
        self.toggle_btn = tk.Label(
            header,
            text="▾  Thinking...",
            font=("Helvetica", 10, "bold"),
            bg=c["chat_bg"],
            fg=c["think_accent"],
            cursor="hand2",
            padx=0, pady=4
        )
        self.toggle_btn.pack(side="left")
        self.toggle_btn.bind("<Button-1>", self.toggle)
        header.bind("<Button-1>", self.toggle)

        # Content block with blue left border
        self.content_outer = tk.Frame(self, bg=c["chat_bg"])
        self.content_outer.pack(fill="x", pady=(4, 0))

        # Blue left border
        self.left_bar = tk.Frame(self.content_outer, bg=c["think_accent"], width=3)
        self.left_bar.pack(side="left", fill="y")

        # Text area
        mono = "Menlo" if sys.platform == "darwin" else "Consolas"
        self.text_label = tk.Label(
            self.content_outer,
            text="",
            font=(mono, 9),
            bg=c["chat_bg"],
            fg=c["think_text"],
            justify="left",
            anchor="nw",
            wraplength=640,
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

    def mark_done(self, elapsed: float):
        self.toggle_btn.config(text=f"▾  Thought for {elapsed:.0f}s")


# =============================================================================
# MAIN APP — DeepSeek chat.deepseek.com accurate layout
# =============================================================================

class CatR1App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat R1")
        self.root.geometry("1120x760")
        self.root.minsize(800, 600)
        self.root.configure(bg='#141414')

        # ── Full dark theme: black/gray bg, blue text, black buttons ──────
        self.colors = {
            # Sidebar
            "sidebar_bg":    "#0A0A0A",
            "sidebar_hover": "#1A1A1A",
            "sidebar_text":  "#4D6BFE",
            "sidebar_sub":   "#3A55CC",
            "new_chat_bg":   "#111111",
            "new_chat_fg":   "#4D6BFE",

            # Main chat
            "chat_bg":       "#141414",
            "user_text":     "#4D6BFE",
            "bot_text":      "#4D6BFE",
            "bot_name":      "#4D6BFE",

            # Thinking block
            "think_accent":  "#4D6BFE",
            "think_text":    "#3A55CC",

            # Input
            "input_bg":      "#1A1A1A",
            "input_border":  "#2A2A2A",
            "input_text":    "#4D6BFE",
            "send_bg":       "#000000",
            "send_fg":       "#4D6BFE",
            "placeholder":   "#1E3080",

            # Misc
            "divider":       "#1F1F1F",
            "user_bubble":   "#1A1A1A",
        }

        self.engine = CatReasoningEngine()
        self.msg_queue = queue.Queue()
        self.deep_mode = False
        self._think_start = 0.0

        self.setup_ui()
        self.root.after(100, self.process_queue)

        threading.Thread(
            target=self.engine.boot_sequence,
            args=(self.update_status,),
            daemon=True
        ).start()

    # ──────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────

    def setup_ui(self):
        c = self.colors

        # ── Sidebar ───────────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, bg=c["sidebar_bg"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo_frame = tk.Frame(self.sidebar, bg=c["sidebar_bg"])
        logo_frame.pack(fill="x", padx=16, pady=(20, 8))

        tk.Label(
            logo_frame, text="🐱  Cat R1",
            font=("Helvetica", 15, "bold"),
            bg=c["sidebar_bg"], fg=c["sidebar_text"]
        ).pack(side="left")

        # New Chat button
        new_btn = tk.Button(
            self.sidebar,
            text="＋  New Chat",
            font=("Helvetica", 10, "bold"),
            bg=c["new_chat_bg"], fg=c["new_chat_fg"],
            activebackground="#3A55D4", activeforeground=c["new_chat_fg"],
            relief="flat", bd=0, cursor="hand2",
            command=self.clear_chat
        )
        new_btn.pack(fill="x", padx=16, pady=(4, 16), ipady=9)

        # DeepThink toggle
        self.deep_btn = tk.Button(
            self.sidebar,
            text="⚡  DeepThink  OFF",
            font=("Helvetica", 9, "bold"),
            bg=c["sidebar_hover"], fg=c["sidebar_sub"],
            activebackground="#3A3A3C", activeforeground=c["sidebar_text"],
            relief="flat", bd=0, cursor="hand2",
            command=self.toggle_deep
        )
        self.deep_btn.pack(fill="x", padx=16, pady=2, ipady=7)

        # Model label
        tk.Label(
            self.sidebar, text="Model: Cat-R1-Reasoning",
            font=("Helvetica", 8), bg=c["sidebar_bg"], fg=c["sidebar_sub"]
        ).pack(padx=16, pady=(16, 0), anchor="w")

        # Status label (bottom)
        self.status_label = tk.Label(
            self.sidebar, text="Initializing...",
            font=("Helvetica", 8), bg=c["sidebar_bg"],
            fg=c["sidebar_sub"], wraplength=220, justify="left"
        )
        self.status_label.pack(side="bottom", padx=16, pady=16, anchor="w")

        # ── Main area ─────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=c["chat_bg"])
        main.pack(side="right", fill="both", expand=True)

        # Top bar
        topbar = tk.Frame(main, bg=c["chat_bg"],
                          highlightthickness=1, highlightbackground=c["divider"],
                          height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="Cat R1",
            font=("Helvetica", 13, "bold"),
            bg=c["chat_bg"], fg=c["bot_name"]
        ).pack(side="left", padx=20, pady=14)

        # Scrollable chat canvas
        chat_frame = tk.Frame(main, bg=c["chat_bg"])
        chat_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(chat_frame, bg=c["chat_bg"],
                                highlightthickness=0, bd=0)
        self.scroll_frame = tk.Frame(self.canvas, bg=c["chat_bg"])

        vsb = ttk.Scrollbar(chat_frame, orient="vertical",
                            command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self._window_id = self.canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )
        self.scroll_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # ── Input bar ─────────────────────────────────────────────────────
        input_outer = tk.Frame(main, bg=c["chat_bg"])
        input_outer.pack(fill="x", padx=0, pady=0)

        # Thin top divider
        tk.Frame(input_outer, bg=c["divider"], height=1).pack(fill="x")

        input_inner = tk.Frame(input_outer, bg=c["chat_bg"])
        input_inner.pack(fill="x", padx=24, pady=12)

        # Rounded-looking input box
        box_frame = tk.Frame(
            input_inner,
            bg=c["input_bg"],
            highlightthickness=1,
            highlightbackground=c["input_border"]
        )
        box_frame.pack(fill="x")

        self.entry = tk.Text(
            box_frame,
            bg=c["input_bg"], fg=c["input_text"],
            insertbackground=c["input_text"],
            font=("Helvetica", 12),
            relief="flat", bd=0,
            height=1,
            wrap="word",
            padx=14, pady=12
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: None)  # allow shift-enter newline
        self._set_placeholder()

        # Send button
        self.send_btn = tk.Button(
            box_frame,
            text="↑",
            font=("Helvetica", 13, "bold"),
            bg=c["send_bg"], fg=c["send_fg"],
            activebackground="#3A55D4", activeforeground=c["send_fg"],
            relief="flat", bd=0, cursor="hand2",
            width=3,
            command=self.send
        )
        self.send_btn.pack(side="right", padx=6, pady=6)

        # Hint text
        tk.Label(
            input_inner,
            text="Cat R1 can make mistakes. Verify important information.",
            font=("Helvetica", 8),
            bg=c["chat_bg"], fg=c["think_text"]
        ).pack(pady=(6, 0))

    # ──────────────────────────────────────────────────────────────────────
    # INPUT HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _set_placeholder(self):
        self.entry.config(fg=self.colors["placeholder"])
        self.entry.insert("1.0", "Message Cat R1...")
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._restore_placeholder)
        self._has_placeholder = True

    def _clear_placeholder(self, event=None):
        if self._has_placeholder:
            self.entry.delete("1.0", tk.END)
            self.entry.config(fg=self.colors["input_text"])
            self._has_placeholder = False

    def _restore_placeholder(self, event=None):
        if not self.entry.get("1.0", tk.END).strip():
            self.entry.config(fg=self.colors["placeholder"])
            self.entry.insert("1.0", "Message Cat R1...")
            self._has_placeholder = True

    def _on_enter(self, event):
        if not (event.state & 0x1):  # Shift not held
            self.send()
            return "break"

    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        self.canvas.itemconfig(self._window_id, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ──────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ──────────────────────────────────────────────────────────────────────

    def toggle_deep(self):
        self.deep_mode = not self.deep_mode
        self.engine.deep_mode = self.deep_mode
        if self.deep_mode:
            self.deep_btn.config(
                text="⚡  DeepThink  ON",
                bg="#2D2F5E",
                fg=self.colors["new_chat_bg"]
            )
        else:
            self.deep_btn.config(
                text="⚡  DeepThink  OFF",
                bg=self.colors["sidebar_hover"],
                fg=self.colors["sidebar_sub"]
            )

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def clear_chat(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()

    def send(self):
        if self._has_placeholder:
            return
        text = self.entry.get("1.0", tk.END).strip()
        if not text or not self.engine.is_ready:
            return

        # Clear input
        self.entry.delete("1.0", tk.END)
        self._has_placeholder = False

        # Add user message (right-aligned)
        self._add_user_message(text)

        # Prepare bot response container
        self._build_bot_container()

        self._think_start = time.time()

        threading.Thread(
            target=self.engine.generate_thought_chain,
            args=(text, self.msg_queue),
            daemon=True
        ).start()

    def _add_user_message(self, text: str):
        c = self.colors
        # Outer wrapper right-aligned
        wrapper = tk.Frame(self.scroll_frame, bg=c["chat_bg"])
        wrapper.pack(fill="x", padx=32, pady=(20, 4))

        # Right-side bubble
        bubble_frame = tk.Frame(wrapper, bg=c["chat_bg"])
        bubble_frame.pack(side="right")

        bubble = tk.Label(
            bubble_frame,
            text=text,
            bg=c["user_bubble"],
            fg=c["user_text"],
            font=("Helvetica", 11),
            justify="left",
            wraplength=520,
            padx=14, pady=10
        )
        bubble.pack()
        self._scroll_bottom()

    def _build_bot_container(self):
        c = self.colors
        # Wrapper
        self.current_bot_frame = tk.Frame(self.scroll_frame, bg=c["chat_bg"])
        self.current_bot_frame.pack(fill="x", padx=32, pady=(16, 20), anchor="w")

        # Avatar + name row
        hdr = tk.Frame(self.current_bot_frame, bg=c["chat_bg"])
        hdr.pack(fill="x", anchor="w", pady=(0, 8))

        tk.Label(
            hdr, text="🐱",
            font=("Helvetica", 18),
            bg=c["chat_bg"]
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            hdr, text="Cat R1",
            font=("Helvetica", 11, "bold"),
            bg=c["chat_bg"], fg=c["bot_name"]
        ).pack(side="left")

        # Reset streaming refs
        self.current_thought_block = None
        self.current_answer_label = None

    def _scroll_bottom(self):
        self.root.after(20, lambda: self.canvas.yview_moveto(1.0))

    # ──────────────────────────────────────────────────────────────────────
    # QUEUE PROCESSING
    # ──────────────────────────────────────────────────────────────────────

    def process_queue(self):
        c = self.colors
        try:
            while True:
                mode, content = self.msg_queue.get_nowait()

                if mode == "thought":
                    if (self.current_thought_block is None or
                            not self.current_thought_block.winfo_exists()):
                        self.current_thought_block = CollapsibleThought(
                            self.current_bot_frame, c
                        )
                        self.current_thought_block.pack(
                            fill="x", pady=(0, 12), anchor="w"
                        )
                    self.current_thought_block.update_text(content)
                    self._scroll_bottom()

                elif mode == "answer":
                    if (self.current_answer_label is None or
                            not self.current_answer_label.winfo_exists()):
                        self.current_answer_label = tk.Label(
                            self.current_bot_frame,
                            text="",
                            bg=c["chat_bg"], fg=c["bot_text"],
                            font=("Helvetica", 11),
                            justify="left",
                            wraplength=680,
                            padx=0, pady=6
                        )
                        self.current_answer_label.pack(anchor="w")
                    self.current_answer_label.config(text=content)
                    self._scroll_bottom()

                elif mode == "done":
                    elapsed = time.time() - self._think_start
                    if (self.current_thought_block is not None and
                            self.current_thought_block.winfo_exists()):
                        self.current_thought_block.mark_done(elapsed)
                    self.current_thought_block = None
                    self.current_answer_label = None

        except queue.Empty:
            pass
        finally:
            self.root.after(40, self.process_queue)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()

    # Modern look on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = CatR1App(root)
    root.mainloop()

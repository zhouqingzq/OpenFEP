import type { ChatMessage } from "./model.js";

export interface ChatViewOptions {
  root: HTMLElement;
  onSend?: (text: string) => void;
}

export class ChatView {
  private readonly listEl: HTMLElement;
  private readonly composer: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;

  constructor(options: ChatViewOptions) {
    const root = options.root;
    root.innerHTML = `
      <section class="chat-panel">
        <div class="chat-list" role="log" aria-live="polite"></div>
        <form class="composer" autocomplete="off">
          <textarea rows="2" placeholder="输入消息，Enter 发送" aria-label="消息输入"></textarea>
          <button type="submit">发送</button>
        </form>
      </section>
    `;
    this.listEl = root.querySelector(".chat-list") as HTMLElement;
    this.composer = root.querySelector("textarea") as HTMLTextAreaElement;
    this.sendButton = root.querySelector("button") as HTMLButtonElement;

    const form = root.querySelector(".composer") as HTMLFormElement;
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.submitComposer(options.onSend);
    });
    this.composer.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submitComposer(options.onSend);
      }
    });
  }

  setComposerEnabled(enabled: boolean): void {
    this.composer.disabled = !enabled;
    this.sendButton.disabled = !enabled;
  }

  appendMessage(message: ChatMessage): void {
    const row = document.createElement("article");
    row.className = `message message-${message.role}`;
    row.dataset.messageId = message.id;
    const label = message.role === "user" ? "你" : message.role === "proactive" ? "主动" : "助手";
    row.innerHTML = `
      <header>${label}${message.turnIndex !== undefined ? ` · turn ${message.turnIndex}` : ""}</header>
      <p></p>
    `;
    const paragraph = row.querySelector("p");
    if (paragraph) {
      paragraph.textContent = message.text;
    }
    this.listEl.appendChild(row);
    this.listEl.scrollTop = this.listEl.scrollHeight;
  }

  showSuppressionToast(reasonCode: string): void {
    const toast = document.createElement("div");
    toast.className = "toast toast-suppression";
    toast.textContent = `主动消息被抑制：${reasonCode || "unknown"}`;
    this.listEl.appendChild(toast);
    window.setTimeout(() => toast.remove(), 6000);
    this.listEl.scrollTop = this.listEl.scrollHeight;
  }

  private submitComposer(onSend?: (text: string) => void): void {
    const text = this.composer.value.trim();
    if (!text || !onSend) {
      return;
    }
    this.composer.value = "";
    onSend(text);
  }
}

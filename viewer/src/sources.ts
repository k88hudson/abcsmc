// Ways to get run.jsonl lines into the page. Each source calls `onLines`
// with complete lines as they arrive and returns a function that stops it.

import { LineSplitter } from "./run";

export type OnLines = (lines: string[]) => void;

export const supportsFilePicker = (): boolean =>
  typeof window !== "undefined" && "showOpenFilePicker" in window;

// One-shot read of a File (from an input or a drop).
export async function readFile(file: File, onLines: OnLines): Promise<void> {
  const splitter = new LineSplitter();
  const reader = file.stream().getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    onLines(splitter.push(value));
  }
  onLines(splitter.flush());
}

// Bytes of the file's start compared on every poll to notice a restart.
const HEADER_BYTES = 512;

// Re-reads a file handle on an interval, delivering only the bytes past what
// was already read. Works while another process appends to the file. A run
// restarted into the same path is detected by its first line changing (the
// run_started event carries the run id), not only by the file shrinking, so
// a fast restart that outgrows the old offset within one poll still resets.
export function tailHandle(
  handle: FileSystemFileHandle,
  onLines: OnLines,
  onError: (error: unknown) => void,
  intervalMs = 1000,
): () => void {
  let offset = 0;
  let header = "";
  let splitter = new LineSplitter();
  let stopped = false;
  let busy = false;

  const poll = async () => {
    if (stopped || busy) return;
    busy = true;
    try {
      const file = await handle.getFile();
      const head = await file.slice(0, HEADER_BYTES).text();
      const restarted =
        file.size < offset ||
        (offset > 0 && head.slice(0, header.length) !== header);
      if (restarted) {
        offset = 0;
        splitter = new LineSplitter();
      }
      if (offset === 0) header = head;
      if (file.size > offset) {
        const bytes = new Uint8Array(await file.slice(offset).arrayBuffer());
        offset = file.size;
        onLines(splitter.push(bytes));
      }
    } catch (error) {
      onError(error);
    } finally {
      busy = false;
    }
  };

  void poll();
  const timer = setInterval(() => void poll(), intervalMs);
  return () => {
    stopped = true;
    clearInterval(timer);
  };
}

declare global {
  interface Window {
    showOpenFilePicker?: (options?: {
      multiple?: boolean;
      types?: { description?: string; accept: Record<string, string[]> }[];
    }) => Promise<FileSystemFileHandle[]>;
  }
}

export async function pickRunFile(): Promise<FileSystemFileHandle | null> {
  if (!window.showOpenFilePicker) return null;
  try {
    const [handle] = await window.showOpenFilePicker({
      multiple: false,
      types: [
        {
          description: "abcsmc run log",
          accept: { "application/jsonl": [".jsonl"] },
        },
      ],
    });
    return handle ?? null;
  } catch (error) {
    if ((error as DOMException).name === "AbortError") return null;
    throw error;
  }
}

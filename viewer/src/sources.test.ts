import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tailHandle } from "./sources";

// A handle whose file either serves `content` or throws on read.
function fakeHandle(state: { content: string; fail?: boolean }) {
  const getFile = async () => {
    const file = new File([state.content], "run.jsonl");
    if (!state.fail) return file;
    const unreadable = () =>
      Promise.reject(
        new DOMException(
          "The requested file could not be read",
          "NotReadableError",
        ),
      );
    return {
      size: file.size,
      slice: () => ({ text: unreadable, arrayBuffer: unreadable }),
    } as unknown as File;
  };
  return { name: "run.jsonl", getFile } as unknown as FileSystemFileHandle;
}

describe("tailHandle", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("delivers appended lines and retries a read the writer interrupted", async () => {
    const state = { content: "a\n", fail: false };
    const lines: string[] = [];
    const errors: unknown[] = [];
    const stop = tailHandle(
      fakeHandle(state),
      (l) => lines.push(...l),
      (e) => errors.push(e),
      10,
    );
    await vi.advanceTimersByTimeAsync(5);
    expect(lines).toEqual(["a"]);

    state.content = "a\nb\n";
    state.fail = true;
    await vi.advanceTimersByTimeAsync(40);
    expect(lines).toEqual(["a"]);
    expect(errors).toEqual([]);

    state.fail = false;
    await vi.advanceTimersByTimeAsync(10);
    expect(lines).toEqual(["a", "b"]);
    expect(errors).toEqual([]);
    stop();
  });

  it("reports a read that keeps failing, then clears it on recovery", async () => {
    const state = { content: "a\n", fail: true };
    const errors: unknown[] = [];
    const stop = tailHandle(
      fakeHandle(state),
      () => {},
      (e) => errors.push(e),
      10,
    );
    await vi.advanceTimersByTimeAsync(35);
    expect(errors).toEqual([]);
    await vi.advanceTimersByTimeAsync(10);
    expect(errors).toHaveLength(1);
    expect((errors[0] as DOMException).name).toBe("NotReadableError");
    await vi.advanceTimersByTimeAsync(30);
    expect(errors).toHaveLength(1);

    state.fail = false;
    await vi.advanceTimersByTimeAsync(10);
    expect(errors).toEqual([expect.anything(), null]);
    stop();
  });

  it("reports other errors at once", async () => {
    const handle = {
      name: "run.jsonl",
      getFile: () => Promise.reject(new Error("gone")),
    } as unknown as FileSystemFileHandle;
    const errors: unknown[] = [];
    const stop = tailHandle(
      handle,
      () => {},
      (e) => errors.push(e),
      10,
    );
    await vi.advanceTimersByTimeAsync(5);
    expect(errors).toHaveLength(1);
    stop();
  });
});

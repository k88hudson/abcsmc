// Remembers the last run that was open, in IndexedDB, so the page can reopen
// it on the next visit. An opened run is kept as its file handle, which still
// points at the file on disk; browsers without handles keep a copy of the
// file as it was instead.

export type Recent =
  | { kind: "handle"; name: string; handle: FileSystemFileHandle }
  | { kind: "file"; name: string; file: File };

const DB = "abcsmc-viewer";
const STORE = "recent";
const KEY = "last";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const request = run(db.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}

// Storage is a convenience: every failure (private mode, quota) is ignored.
export async function saveRecent(recent: Recent): Promise<void> {
  try {
    await withStore("readwrite", (store) => store.put(recent, KEY));
  } catch {
    // Nothing to reopen next time.
  }
}

export async function loadRecent(): Promise<Recent | null> {
  try {
    return (
      (await withStore<Recent | undefined>("readonly", (store) =>
        store.get(KEY),
      )) ?? null
    );
  } catch {
    return null;
  }
}

export async function clearRecent(): Promise<void> {
  try {
    await withStore("readwrite", (store) => store.delete(KEY));
  } catch {
    // Already gone, as far as the page can tell.
  }
}

type PermissionHandle = FileSystemFileHandle & {
  queryPermission?: (options: { mode: "read" }) => Promise<PermissionState>;
  requestPermission?: (options: { mode: "read" }) => Promise<PermissionState>;
};

// Whether a stored handle can be read without asking.
export async function canRead(handle: FileSystemFileHandle): Promise<boolean> {
  const h = handle as PermissionHandle;
  return (await h.queryPermission?.({ mode: "read" })) === "granted";
}

// Asks for read access; must be called from a click.
export async function requestRead(
  handle: FileSystemFileHandle,
): Promise<boolean> {
  const h = handle as PermissionHandle;
  return (await h.requestPermission?.({ mode: "read" })) === "granted";
}

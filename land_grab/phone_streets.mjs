// The phone's half of "download the streets on the visitor's own device".
//
// Streamlit runs the default export in the visitor's browser (see land_grab/browser_streets.py for the
// Python half). It asks the public Overpass servers for the streets (the question text is written by
// Python, so the area and the street filter are the same as on our server), keeps only what the planner
// uses, squeezes it, and hands it back with setStateValue("reply", ...).
//
// The other exports are plain helpers so they can be tested with `node --test tests/js`.

export const REPLY_FORMAT = 1; // must match REPLY_FORMAT in browser_streets.py
const TOTAL_MS = 10 * 60 * 1000; // give up after 10 minutes
const STALL_MS = 120 * 1000; // ... or when a server sends nothing for 2 minutes
const FORGET_MS = 3000; // a component that is gone this long is really gone (not just redrawn)

export const MESSAGES = {
  downloading: "Downloading streets to your phone…",
  offline: "Your phone seems to be offline.",
  busy: "The map servers are busy.",
  tooBig: "The streets for this area are too big to send from a phone.",
  empty: "The map server didn't send back any streets for that place.",
  cancelled: "The download was stopped.",
};

export class StreetsError extends Error {}

const sleepMs = (ms, signal) =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new StreetsError(MESSAGES.cancelled));
      },
      { once: true },
    );
  });

export function megabytes(bytes) {
  return (bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0) + " MB";
}

// --- one server answer -------------------------------------------------------------------------

// Read a response body piece by piece, so the progress can count the bytes, then parse it as JSON.
export async function readJson(response, onBytes, stallMs, abortAttempt) {
  const chunks = [];
  let total = 0;
  let timer = null;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(abortAttempt, stallMs);
  };
  try {
    arm();
    if (response.body && response.body.getReader) {
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        total += value.length;
        onBytes(value.length);
        arm();
      }
    } else {
      const buffer = new Uint8Array(await response.arrayBuffer());
      chunks.push(buffer);
      total = buffer.length;
      onBytes(total);
    }
  } finally {
    clearTimeout(timer);
  }
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.length;
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

// Ask one query. Tries the servers in order like the Python download: the first one is retried after each
// wait in `waits` when it is busy; each of the others gets one try. Returns Overpass's JSON answer.
export async function fetchQuery(query, request, io) {
  const { servers, waits, busyStatuses } = request;
  const busy = new Set(busyStatuses);
  let lastProblem = MESSAGES.busy;
  for (let index = 0; index < servers.length; index++) {
    const tries = index === 0 ? waits.length + 1 : 1;
    for (let attempt = 0; attempt < tries; attempt++) {
      if (io.signal.aborted) throw new StreetsError(MESSAGES.cancelled);
      const attemptAbort = new AbortController();
      const stop = () => attemptAbort.abort();
      io.signal.addEventListener("abort", stop, { once: true });
      let retry = false;
      try {
        io.onNote(index === 0 && attempt === 0 ? "" : "Trying again…");
        const response = await io.fetch(servers[index], {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: "data=" + encodeURIComponent(query),
          signal: attemptAbort.signal,
        });
        if (busy.has(response.status)) {
          lastProblem = MESSAGES.busy;
          retry = true;
        } else if (!response.ok) {
          lastProblem = `The map server answered ${response.status}.`;
          break; // this server does not like our question: try the next one
        } else {
          const json = await readJson(response, io.onBytes, io.stallMs ?? STALL_MS, stop);
          if (json && Array.isArray(json.elements) && !/runtime error/i.test(json.remark || "")) return json;
          lastProblem = /runtime error/i.test(json?.remark || "") ? MESSAGES.busy : MESSAGES.empty;
          retry = lastProblem === MESSAGES.busy;
        }
      } catch (error) {
        if (io.signal.aborted) throw new StreetsError(MESSAGES.cancelled);
        if (io.isOffline()) throw new StreetsError(MESSAGES.offline);
        lastProblem = MESSAGES.busy; // a dropped connection, a stall or a server we cannot reach: wait and try again
        retry = true;
      } finally {
        io.signal.removeEventListener("abort", stop);
      }
      if (retry && attempt < tries - 1) {
        io.onNote("The map server is busy, waiting a little…");
        await sleepMs(waits[attempt] * 1000 * (io.waitScale ?? 1), io.signal);
      } else if (!retry) {
        break;
      }
    }
    if (index < servers.length - 1) io.onNote("Trying another map server…");
  }
  throw new StreetsError(lastProblem);
}

// --- shrinking and squeezing ---------------------------------------------------------------------

// Keep only what the planner uses: point positions, each street's type/junction/oneway, and its points.
export function shrink(responses) {
  const nodes = new Map();
  const ways = new Map();
  for (const response of responses) {
    for (const element of response.elements) {
      if (element.type === "node") {
        nodes.set(element.id, [element.id, element.lat, element.lon]);
      } else if (element.type === "way") {
        const tags = element.tags || {};
        ways.set(element.id, [element.id, tags.highway || "", tags.junction || "", tags.oneway || "", element.nodes]);
      }
    }
  }
  return { v: REPLY_FORMAT, nodes: [...nodes.values()], ways: [...ways.values()] };
}

function toBase64(bytes) {
  let text = "";
  for (let at = 0; at < bytes.length; at += 0x8000) {
    text += String.fromCharCode.apply(null, bytes.subarray(at, at + 0x8000));
  }
  return btoa(text);
}

// Text -> { encoding, payload }. Gzipped (about 5 to 10 times smaller) where the browser can, else plain.
export async function pack(text) {
  if (typeof CompressionStream === "function") {
    const stream = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    return { encoding: "gzip-base64", payload: toBase64(bytes) };
  }
  return { encoding: "json", payload: text };
}

// The whole job: fetch every query, shrink, squeeze. Always resolves to a reply for Python:
// { id, ok: true, encoding, payload } or { id, ok: false, error }.
export async function download(request, io) {
  try {
    const responses = [];
    for (const query of request.queries) responses.push(await fetchQuery(query, request, io));
    io.onNote("Packing the streets…");
    const small = shrink(responses);
    if (!small.ways.length) throw new StreetsError(MESSAGES.empty);
    const reply = await pack(JSON.stringify(small));
    if (reply.payload.length > request.maxPayloadChars) throw new StreetsError(MESSAGES.tooBig);
    return { id: request.id, ok: true, ...reply };
  } catch (error) {
    const text = error instanceof StreetsError ? error.message : "The phone could not download the streets.";
    return { id: request.id, ok: false, error: text };
  }
}

// --- the Streamlit component -----------------------------------------------------------------------

const jobs = new Map(); // one per request id, so a redraw never starts the download twice

function startJob(request, setStateValue) {
  const abort = new AbortController();
  const job = { id: request.id, bytes: 0, note: "", abort, listeners: new Set(), setStateValue, goneAt: null, done: false };
  const changed = () => job.listeners.forEach((listener) => listener(job));
  const timeout = setTimeout(() => abort.abort(), TOTAL_MS);
  const io = {
    fetch: (...args) => fetch(...args),
    signal: abort.signal,
    isOffline: () => navigator.onLine === false,
    onBytes: (n) => {
      job.bytes += n;
      changed();
    },
    onNote: (note) => {
      job.note = note;
      changed();
    },
  };
  download(request, io).then((reply) => {
    clearTimeout(timeout);
    job.done = true;
    job.setStateValue("reply", reply);
    changed();
  });
  jobs.set(request.id, job);
  return job;
}

export default function (component) {
  const { data, setStateValue, parentElement } = component;
  if (!data || !data.id) return undefined;
  const job = jobs.get(data.id) ?? startJob(data, setStateValue);
  job.setStateValue = setStateValue; // the newest one is the one that works after a redraw
  job.goneAt = null;

  const line = document.createElement("div");
  line.setAttribute("role", "status");
  line.setAttribute("aria-live", "polite");
  line.style.cssText = "font: inherit; padding: .25rem 0; color: #4a5f66;";
  parentElement.replaceChildren(line);
  const show = () => {
    if (job.done) {
      line.textContent = ""; // Python takes over from here and shows its own progress
      return;
    }
    const parts = [`${MESSAGES.downloading.replace("…", "")} ${megabytes(job.bytes)} so far`];
    if (job.note) parts.push(job.note);
    line.textContent = parts.join(" · ");
  };
  job.listeners.add(show);
  show();

  return () => {
    job.listeners.delete(show);
    job.goneAt = Date.now();
    setTimeout(() => {
      // Really removed (a newer search replaced it, or the page was left): stop downloading.
      if (job.goneAt !== null && !job.done && Date.now() - job.goneAt >= FORGET_MS) {
        job.abort.abort();
        jobs.delete(job.id);
      }
    }, FORGET_MS + 50);
  };
}

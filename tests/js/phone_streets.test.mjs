// Tests for the phone's half. Run with: node --test tests/js
import assert from "node:assert/strict";
import { gunzipSync } from "node:zlib";
import test from "node:test";

import { MESSAGES, REPLY_FORMAT, download, fetchQuery, megabytes, pack, shrink } from "../../land_grab/phone_streets.mjs";

const REQUEST = {
  id: "abc",
  queries: ["q1"],
  servers: ["https://one/interpreter", "https://two/interpreter", "https://three/interpreter"],
  waits: [5, 15, 30],
  busyStatuses: [429, 502, 503, 504],
  maxPayloadChars: 1_000_000,
};

const GOOD = {
  elements: [
    { type: "node", id: 1, lat: 51.5, lon: -0.1, tags: { highway: "crossing" } },
    { type: "node", id: 2, lat: 51.501, lon: -0.1 },
    { type: "way", id: 10, nodes: [1, 2], tags: { highway: "residential", oneway: "yes", name: "x" } },
  ],
};

function answer(status, body) {
  const bytes = new TextEncoder().encode(typeof body === "string" ? body : JSON.stringify(body));
  return {
    status,
    ok: status >= 200 && status < 300,
    body: {
      getReader() {
        let sent = false;
        return { read: async () => (sent ? { done: true } : ((sent = true), { done: false, value: bytes })) };
      },
    },
  };
}

function makeIo(responses, extra = {}) {
  const calls = [];
  const notes = [];
  const controller = new AbortController();
  return {
    calls,
    notes,
    controller,
    bytes: 0,
    fetch: async (url, options) => {
      calls.push({ url, options });
      const next = responses.shift();
      if (next instanceof Error) throw next;
      return next;
    },
    signal: controller.signal,
    isOffline: () => false,
    onBytes() {},
    onNote: (note) => notes.push(note),
    waitScale: 0,
    stallMs: 1000,
    ...extra,
  };
}

test("the query text is sent as a form post to the first server", async () => {
  const io = makeIo([answer(200, GOOD)]);
  const json = await fetchQuery("a;b&c=d", REQUEST, io);
  assert.equal(json.elements.length, 3);
  assert.equal(io.calls[0].url, REQUEST.servers[0]);
  assert.equal(io.calls[0].options.method, "POST");
  assert.equal(io.calls[0].options.body, "data=" + encodeURIComponent("a;b&c=d"));
});

test("a busy first server is retried after waiting, then succeeds", async () => {
  const io = makeIo([answer(429, ""), answer(504, ""), answer(200, GOOD)]);
  await fetchQuery("q", REQUEST, io);
  assert.deepEqual(io.calls.map((c) => c.url), [REQUEST.servers[0], REQUEST.servers[0], REQUEST.servers[0]]);
  assert.ok(io.notes.some((n) => /busy/i.test(n)));
});

test("a server that stays busy hands over to the next server", async () => {
  const io = makeIo([answer(503, ""), answer(503, ""), answer(503, ""), answer(503, ""), answer(200, GOOD)]);
  await fetchQuery("q", REQUEST, io);
  assert.equal(io.calls.length, 5); // first server: 1 try + 3 retries, then the second server
  assert.equal(io.calls[4].url, REQUEST.servers[1]);
});

test("a server that refuses the question (400) is skipped without retrying", async () => {
  const io = makeIo([answer(400, "bad"), answer(200, GOOD)]);
  await fetchQuery("q", REQUEST, io);
  assert.deepEqual(io.calls.map((c) => c.url), [REQUEST.servers[0], REQUEST.servers[1]]);
});

test("a runtime-error remark counts as busy", async () => {
  const io = makeIo([answer(200, { elements: [], remark: "runtime error: Query timed out" }), answer(200, GOOD)]);
  const json = await fetchQuery("q", REQUEST, io);
  assert.equal(json.elements.length, 3);
});

test("all servers busy ends with the busy message", async () => {
  const io = makeIo(Array.from({ length: 10 }, () => answer(503, "")));
  await assert.rejects(fetchQuery("q", REQUEST, io), { message: MESSAGES.busy });
});

test("being offline gives the offline message straight away", async () => {
  const io = makeIo([new TypeError("Failed to fetch")], { isOffline: () => true });
  await assert.rejects(fetchQuery("q", REQUEST, io), { message: MESSAGES.offline });
  assert.equal(io.calls.length, 1);
});

test("a blocked request (e.g. CORS) while online tries every server then gives up", async () => {
  const io = makeIo(Array.from({ length: 10 }, () => new TypeError("Failed to fetch")));
  await assert.rejects(fetchQuery("q", REQUEST, io), { message: MESSAGES.busy });
  assert.deepEqual([...new Set(io.calls.map((c) => c.url))], REQUEST.servers);
});

test("stopping the download stops the retries", async () => {
  const io = makeIo([answer(503, "")]);
  io.controller.abort();
  await assert.rejects(fetchQuery("q", REQUEST, io), { message: MESSAGES.cancelled });
  assert.equal(io.calls.length, 0);
});

test("progress counts the bytes that arrive", async () => {
  let seen = 0;
  const io = makeIo([answer(200, GOOD)], { onBytes: (n) => (seen += n) });
  await fetchQuery("q", REQUEST, io);
  assert.equal(seen, JSON.stringify(GOOD).length);
});

test("shrink keeps only what the planner uses and merges repeats", () => {
  const small = shrink([GOOD, GOOD]);
  assert.deepEqual(small, {
    v: REPLY_FORMAT,
    nodes: [[1, 51.5, -0.1], [2, 51.501, -0.1]],
    ways: [[10, "residential", "", "yes", [1, 2]]],
  });
});

test("pack gzips and the result unpacks to the same text", async () => {
  const text = JSON.stringify(shrink([GOOD]));
  const reply = await pack(text);
  assert.equal(reply.encoding, "gzip-base64");
  assert.equal(gunzipSync(Buffer.from(reply.payload, "base64")).toString(), text);
});

test("download gives a reply for Python with the request id", async () => {
  const io = makeIo([answer(200, GOOD)]);
  const reply = await download(REQUEST, io);
  assert.equal(reply.ok, true);
  assert.equal(reply.id, "abc");
  const data = JSON.parse(gunzipSync(Buffer.from(reply.payload, "base64")).toString());
  assert.equal(data.v, REPLY_FORMAT);
  assert.equal(data.ways.length, 1);
});

test("download never throws: failures become ok:false with a message", async () => {
  const io = makeIo([new TypeError("x")], { isOffline: () => true });
  const reply = await download(REQUEST, io);
  assert.deepEqual(reply, { id: "abc", ok: false, error: MESSAGES.offline });
});

test("an answer with no streets is a failure, not an empty success", async () => {
  const io = makeIo([answer(200, { elements: [] })]);
  const reply = await download(REQUEST, io);
  assert.equal(reply.ok, false);
  assert.equal(reply.error, MESSAGES.empty);
});

test("an answer too big to send is refused with a clear message", async () => {
  const io = makeIo([answer(200, GOOD)]);
  const reply = await download({ ...REQUEST, maxPayloadChars: 10 }, io);
  assert.equal(reply.ok, false);
  assert.equal(reply.error, MESSAGES.tooBig);
});

test("several queries are all fetched", async () => {
  const io = makeIo([answer(200, GOOD), answer(200, GOOD)]);
  const reply = await download({ ...REQUEST, queries: ["a", "b"] }, io);
  assert.equal(reply.ok, true);
  assert.equal(io.calls.length, 2);
});

test("megabytes reads well", () => {
  assert.equal(megabytes(2_500_000), "2.5 MB");
  assert.equal(megabytes(42_000_000), "42 MB");
});

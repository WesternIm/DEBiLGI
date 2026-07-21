import { randomBytes, randomUUID } from "node:crypto";
import http from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const sidecarDirectory = dirname(fileURLToPath(import.meta.url));
const host = process.env.PUTER_SIDECAR_HOST || "127.0.0.1";
const port = Number.parseInt(process.env.PUTER_SIDECAR_PORT || "43127", 10);
const sharedSecret = process.env.PUTER_SIDECAR_SECRET || "";
const semanticModel = process.env.PUTER_SEMANTIC_MODEL || "google/gemini-3.5-flash";
const bridgeToken = randomBytes(24).toString("base64url");
const bridgeTtlMs = 15_000;
const jobTimeoutMs = 38_000;
const maximumQueueSize = 16;

const [bridgeTemplate, systemPrompt] = await Promise.all([
  readFile(resolve(sidecarDirectory, "bridge.html"), "utf8"),
  readFile(resolve(sidecarDirectory, "semantic-instruction.txt"), "utf8"),
]);
const bridgePage = bridgeTemplate
  .replaceAll("__BRIDGE_TOKEN__", bridgeToken)
  .replaceAll("__SEMANTIC_MODEL__", semanticModel);

const queuedJobs = [];
const pendingJobs = new Map();
let bridgeState = {
  clientId: null,
  lastSeenAt: 0,
  signedIn: false,
  status: "offline",
  lastError: null,
};

function sendJson(response, status, payload) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(JSON.stringify(payload));
}

function sendEmpty(response, status = 204) {
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end();
}

function readJson(request) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > 256 * 1024) {
        rejectBody(new Error("request_too_large"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        resolveBody(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        rejectBody(new Error("invalid_json"));
      }
    });
    request.on("error", rejectBody);
  });
}

function isBridgeRequest(request) {
  return request.headers["x-debilgi-bridge"] === bridgeToken;
}

function bridgeIsActive(now = Date.now()) {
  return Boolean(
    bridgeState.clientId
      && bridgeState.signedIn
      && now - bridgeState.lastSeenAt <= bridgeTtlMs,
  );
}

function bridgeStatus() {
  return {
    active: bridgeIsActive(),
    signed_in: bridgeState.signedIn,
    bridge_status: bridgeState.status,
    last_seen_at: bridgeState.lastSeenAt || null,
    last_error: bridgeState.lastError,
    queued_jobs: queuedJobs.length,
    pending_jobs: pendingJobs.size,
    model: semanticModel,
  };
}

function updateBridgeState(body) {
  const reportedError = Object.hasOwn(body, "last_error")
    ? (body.last_error ? String(body.last_error).slice(0, 300) : null)
    : bridgeState.lastError;
  bridgeState = {
    clientId: String(body.client_id || bridgeState.clientId || "browser"),
    lastSeenAt: Date.now(),
    signedIn: body.signed_in === true,
    status: String(body.status || "ready").slice(0, 80),
    lastError: reportedError,
  };
}

function removeQueuedJob(jobId) {
  const index = queuedJobs.findIndex((item) => item.id === jobId);
  if (index >= 0) queuedJobs.splice(index, 1);
}

function settleJob(jobId, result) {
  const pending = pendingJobs.get(jobId);
  if (!pending) return false;
  clearTimeout(pending.timer);
  pendingJobs.delete(jobId);
  removeQueuedJob(jobId);
  pending.resolve(result);
  return true;
}

function enqueueJob(requestPayload) {
  return new Promise((resolveJob) => {
    const id = randomUUID();
    const job = {
      id,
      model: semanticModel,
      system_prompt: systemPrompt.trim(),
      request: requestPayload,
      created_at: new Date().toISOString(),
    };
    const timer = setTimeout(() => {
      settleJob(id, { ok: false, status: 504, error: "bridge_job_timeout" });
    }, jobTimeoutMs);
    pendingJobs.set(id, { resolve: resolveJob, timer });
    queuedJobs.push(job);
  });
}

function parseJsonObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
  try {
    const parsed = JSON.parse(cleaned);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : null;
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    try {
      const parsed = JSON.parse(cleaned.slice(start, end + 1));
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed
        : null;
    } catch {
      return null;
    }
  }
}

function sanitizedContent(value) {
  const parsed = parseJsonObject(value);
  if (!parsed) return null;
  for (const forbidden of ["answer", "reply", "response", "response_text", "personality"]) {
    delete parsed[forbidden];
  }
  const hasInterpretation = typeof parsed.interpretation === "string"
    && parsed.interpretation.trim().length > 0;
  const hasFacts = Array.isArray(parsed.facts) && parsed.facts.length > 0;
  if (!hasInterpretation && !hasFacts) return null;
  const encoded = JSON.stringify(parsed);
  return encoded.length <= 128 * 1024 ? encoded : null;
}

async function handleConsult(request, response) {
  if (sharedSecret && request.headers.authorization !== `Bearer ${sharedSecret}`) {
    sendJson(response, 401, { error: "unauthorized" });
    return;
  }
  if (!bridgeIsActive()) {
    sendJson(response, 503, {
      error: "bridge_unavailable",
      message: "Open the local Puter bridge and press Start.",
    });
    return;
  }
  if (pendingJobs.size >= maximumQueueSize) {
    sendJson(response, 429, { error: "bridge_queue_full" });
    return;
  }
  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    sendJson(response, 400, { error: error.message || "invalid_request" });
    return;
  }
  if (!body.request || typeof body.request !== "object" || Array.isArray(body.request)) {
    sendJson(response, 400, { error: "missing_consultation_request" });
    return;
  }
  const result = await enqueueJob(body.request);
  if (!result.ok) {
    sendJson(response, result.status || 502, { error: result.error || "consultation_failed" });
    return;
  }
  sendJson(response, 200, {
    provider: "puter-browser",
    model: semanticModel,
    content: result.content,
  });
}

async function handleBridgeNext(request, response) {
  if (!isBridgeRequest(request)) {
    sendJson(response, 401, { error: "unauthorized_bridge" });
    return;
  }
  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    sendJson(response, 400, { error: error.message || "invalid_request" });
    return;
  }
  updateBridgeState(body);
  const job = queuedJobs.shift();
  if (!job) {
    sendEmpty(response);
    return;
  }
  sendJson(response, 200, job);
}

async function handleBridgeResult(request, response) {
  if (!isBridgeRequest(request)) {
    sendJson(response, 401, { error: "unauthorized_bridge" });
    return;
  }
  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    sendJson(response, 400, { error: error.message || "invalid_request" });
    return;
  }
  updateBridgeState(body);
  const jobId = String(body.id || "");
  if (!pendingJobs.has(jobId)) {
    sendJson(response, 410, { error: "job_expired" });
    return;
  }
  if (body.error) {
    settleJob(jobId, {
      ok: false,
      status: 502,
      error: String(body.error).slice(0, 300) || "puter_browser_error",
    });
    sendJson(response, 202, { accepted: true });
    return;
  }
  const content = sanitizedContent(body.content);
  if (content === null) {
    sendJson(response, 422, { error: "invalid_semantic_payload" });
    return;
  }
  settleJob(jobId, { ok: true, content });
  sendJson(response, 202, { accepted: true });
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${host}:${port}`);
  if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/bridge")) {
    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
    });
    response.end(bridgePage);
    return;
  }
  if (request.method === "GET" && url.pathname === "/health") {
    sendJson(response, 200, {
      status: "ok",
      provider: "puter-browser",
      mode: "browser-bridge",
      node: process.version,
      ...bridgeStatus(),
    });
    return;
  }
  if (request.method === "GET" && url.pathname === "/v1/bridge/status") {
    sendJson(response, 200, bridgeStatus());
    return;
  }
  if (request.method === "POST" && url.pathname === "/v1/bridge/next") {
    await handleBridgeNext(request, response);
    return;
  }
  if (request.method === "POST" && url.pathname === "/v1/bridge/result") {
    await handleBridgeResult(request, response);
    return;
  }
  if (request.method === "POST" && url.pathname === "/v1/consult") {
    await handleConsult(request, response);
    return;
  }
  sendJson(response, 404, { error: "not_found" });
});

server.listen(port, host, () => {
  console.log(
    `DEBiLGI Puter browser bridge listening on http://${host}:${port}/bridge `
      + `(model=${semanticModel})`,
  );
});

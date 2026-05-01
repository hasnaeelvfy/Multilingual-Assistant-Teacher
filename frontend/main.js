import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODEL_URL = "./models/robot.glb";

const canvas = document.getElementById("c");
const scenePane = document.getElementById("scenePane");
const listenBtn = document.getElementById("listenBtn");
const languageSelect = document.getElementById("languageSelect");
const statusEl = document.getElementById("statusText");
const userTextEl = document.getElementById("userTextEl");
const correctedEl = document.getElementById("correctedText");
const whyEl = document.getElementById("whyText");
const answerEl = document.getElementById("answerText");
const userTextRow = document.getElementById("userTextRow");
const correctedRow = document.getElementById("correctedRow");
const whyRow = document.getElementById("whyRow");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f19);

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
camera.position.set(-0.6, 1.5, 3.1);

const renderer = new THREE.WebGLRenderer({ antialias: true, canvas, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.35, 1.0, 0);
controls.enablePan = false;
controls.enableDamping = true;
controls.minDistance = 1.2;
controls.maxDistance = 6.5;
controls.maxPolarAngle = Math.PI * 0.6;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.25);
key.position.set(0.8, 2.8, 2.6);
scene.add(key);
const fill = new THREE.DirectionalLight(0x98b9ff, 0.35);
fill.position.set(-2.2, 1.4, 1.6);
scene.add(fill);
const talkLight = new THREE.PointLight(0x8cc8ff, 0.0, 2.5, 2.0);
talkLight.position.set(-0.35, 1.15, 0.9);
scene.add(talkLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(1.2, 64),
  new THREE.MeshBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.65 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = 0;
floor.position.x = -1.4;
floor.scale.set(1.7, 1.0, 1.2);
scene.add(floor);

let robotGroup = new THREE.Group();
scene.add(robotGroup);

let baseScale = 1;
let groundY = 0; // <-- Store the grounded Y position so animation never overrides it
let lastTurn = null;
let modelPivot = null;
let mouthNodes = [];
let handNodes = [];
let handPose = [];
let welcomeDone = false;
let isListening = false;
let isBusy = false;
let selectedLanguage = "en";

function resize() {
  const w = Math.max(1, scenePane?.clientWidth || window.innerWidth);
  const h = Math.max(1, scenePane?.clientHeight || window.innerHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

function setStatus(text, muted = false) {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.classList.toggle("muted", muted);
}

function setResponseCard({ user_text, corrected, explanation, answer }) {
  if (userTextEl) {
    const t = String(user_text ?? "").trim();
    userTextEl.textContent = t || "—";
    userTextEl.classList.toggle("muted", !t);
  }
  if (correctedEl) {
    const t = String(corrected ?? "").trim();
    correctedEl.textContent = t || "—";
    correctedEl.classList.toggle("muted", !t);
  }
  if (whyEl) {
    const t = String(explanation ?? "").trim();
    whyEl.textContent = t || "—";
    whyEl.classList.toggle("muted", !t);
  }
  if (answerEl) {
    const t = String(answer ?? "").trim();
    answerEl.textContent = t || "—";
    answerEl.classList.toggle("muted", !t);
  }
}

function setWelcomeMode(on) {
  if (userTextRow) userTextRow.classList.toggle("hidden", on);
  if (correctedRow) correctedRow.classList.toggle("hidden", on);
  if (whyRow) whyRow.classList.toggle("hidden", on);
}

function buildSpeakText(state) {
  const userText = String(state?.user_text ?? "").trim();
  const corrected = String(state?.corrected ?? "").trim();
  const explanation = String(state?.explanation ?? "").trim();
  const answer = String(state?.answer ?? "").trim();

  const parts = [];
  const needsCorrection =
    Boolean(corrected) && Boolean(userText) && corrected.toLowerCase() !== userText.toLowerCase();

  if (needsCorrection) {
    parts.push(`Your sentence should be: ${corrected}.`);
    if (explanation) parts.push(explanation);
  }
  if (answer) parts.push(`Now, to answer you: ${answer}`);
  return parts.join(" ").trim() || answer || "";
}

function buildSubtitle(state) {
  const corrected = state?.corrected ? `Corrected: ${state.corrected}` : "";
  const why = state?.explanation ? `Why: ${state.explanation}` : "";
  const answer = state?.answer ? `Answer: ${state.answer}` : "";
  return [corrected, why, answer].filter(Boolean).join("\n");
}

// --- Procedural animation (no rig required) ---
let speakingTarget = 0;
let speaking = 0;

function damp(current, target, lambda, dt) {
  return THREE.MathUtils.lerp(target, current, Math.exp(-lambda * dt));
}

function applyRobotMotion(t, dt) {
  if (!robotGroup) return;

  speaking = damp(speaking, speakingTarget, 10.0, dt);

  const idleBreath = 0.012 * Math.sin(t * 1.1);
  const idleSway = 0.02 * Math.sin(t * 0.55);

  const talkYaw = 0.16 * Math.sin(t * 6.8);
  const talkPulse = 0.016 * Math.sin(t * 10.0);

  const yaw = idleSway + talkYaw * speaking;
  if (modelPivot) modelPivot.rotation.y = yaw;

  const s = baseScale * (1 + idleBreath + talkPulse * speaking);
  robotGroup.scale.setScalar(s);

  // FIX: use groundY as the base so robot always stays on the floor
  robotGroup.position.y = groundY + (0.015 * Math.sin(t * 6.0)) * speaking;

  const mouthPulse = 1 + 0.1 * speaking * (0.5 + 0.5 * Math.sin(t * 18.0));
  for (const n of mouthNodes) {
    n.scale.y = n.userData.baseScaleY * mouthPulse;
    n.position.z = n.userData.basePosZ + 0.006 * speaking * Math.sin(t * 17.0);
  }

  for (let i = 0; i < handNodes.length; i += 1) {
    const node = handNodes[i];
    const base = handPose[i];
    const dir = i % 2 === 0 ? 1 : -1;
    node.rotation.x = base.x + dir * 0.22 * speaking * Math.sin(t * 5.2 + i * 0.4);
    node.rotation.z = base.z + dir * 0.16 * speaking * Math.sin(t * 3.7 + i * 0.3);
  }

  talkLight.intensity = 0.1 + 0.9 * speaking;
  talkLight.distance = 2.2 + 1.4 * speaking;
}

function setListeningUi(on) {
  isListening = on;
  if (!listenBtn) return;
  listenBtn.classList.toggle("listening", on);
  listenBtn.textContent = on ? "⏹ Stop Listening" : "🎤 Start Listening";
  if (languageSelect) languageSelect.disabled = on || isBusy;
}

function languageLabel(code) {
  if (code === "fr") return "French";
  if (code === "ar") return "Arabic";
  if (code === "es") return "Spanish";
  return "English";
}

async function loadRobot() {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(MODEL_URL);

  robotGroup.clear();
  modelPivot = new THREE.Group();
  robotGroup.add(modelPivot);
  const model = gltf.scene;
  modelPivot.add(model);
  mouthNodes = [];
  handNodes = [];
  handPose = [];

  model.traverse((obj) => {
    if (!obj || !obj.isObject3D) return;
    const name = String(obj.name || "").toLowerCase();

    if (obj.isMesh) {
      if (/(mouth|jaw|lip)/.test(name)) {
        obj.userData.baseScaleY = obj.scale.y;
        obj.userData.basePosZ = obj.position.z;
        mouthNodes.push(obj);
      }
    }

    if (/(arm|hand|shoulder|wrist)/.test(name)) {
      handNodes.push(obj);
      handPose.push({ x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z });
    }
  });

  if (mouthNodes.length === 0) {
    let fallback = null;
    model.traverse((obj) => {
      if (fallback || !obj?.isMesh) return;
      const name = String(obj.name || "").toLowerCase();
      if (/head|face/.test(name)) fallback = obj;
    });
    if (fallback) {
      fallback.userData.baseScaleY = fallback.scale.y;
      fallback.userData.basePosZ = fallback.position.z;
      mouthNodes.push(fallback);
    }
  }

  // Center model at origin
  const box = new THREE.Box3().setFromObject(model);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  model.position.sub(center);

  // Scale
  const sizeLen = size.length() || 1;
  baseScale = 2.0 / sizeLen;
  robotGroup.scale.setScalar(baseScale);

  // FIX: Place robot so feet are exactly at y=0, then save that as groundY
  const box2 = new THREE.Box3().setFromObject(robotGroup);
  robotGroup.position.y = -box2.min.y;
  groundY = robotGroup.position.y; // <-- save grounded Y for animation

  // Face camera
  modelPivot.rotation.y = Math.PI;

  // Fit camera
  const box3 = new THREE.Box3().setFromObject(robotGroup);
  const sphere = new THREE.Sphere();
  box3.getBoundingSphere(sphere);
  const radius = Math.max(0.001, sphere.radius);

  const fov = THREE.MathUtils.degToRad(camera.fov);
  const distance = (radius / Math.sin(fov / 2)) * 1.05;
  const target = sphere.center.clone();

  // Move robot to left side (x only, never touch y)
  const presentationOffset = new THREE.Vector3(-1.4, 0, 0);
  target.add(presentationOffset);
  robotGroup.position.x += presentationOffset.x;

  // Floor stays at y=0, aligned with robot x
  floor.position.y = 0;
  floor.position.x = robotGroup.position.x;

  controls.target.copy(target);
  controls.update();
  camera.position.set(target.x + radius * 0.5, target.y + radius * 0.16, target.z + distance * 1.26);
  camera.near = Math.max(0.01, distance / 100);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  talkLight.position.set(target.x + 0.1, target.y + 0.14, 0.95);
  key.position.set(target.x + 1.35, target.y + 1.9, 2.2);
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function connectWS() {
  const ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    setStatus("Connected", true);
    ws._pingTimer = window.setInterval(() => {
      try { ws.send("ping"); } catch { }
    }, 20000);
  };

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg?.type === "speech_state") {
      speakingTarget = msg.speaking ? 1 : 0;
      document.body.classList.toggle("speaking", Boolean(msg.speaking));
      return;
    }
    if (msg?.type !== "assistant_state") return;

    const state = msg.state || {};
    if (typeof state.turn !== "undefined" && state.turn !== null) {
      if (state.turn === lastTurn) return;
      lastTurn = state.turn;
    }

    const response = {
      corrected: state.corrected || "",
      why: state.explanation || "",
      answer: state.answer || "",
    };
    if (state.selected_language) {
      selectedLanguage = String(state.selected_language);
      if (languageSelect) languageSelect.value = selectedLanguage;
    }

    setWelcomeMode(false);
    setResponseCard({
      user_text: state.user_text,
      corrected: response.corrected,
      explanation: response.why,
      answer: response.answer,
    });
  };

  ws.onclose = () => {
    if (ws._pingTimer) window.clearInterval(ws._pingTimer);
    setStatus("Disconnected. Reconnecting…", true);
    speakingTarget = 0;
    setTimeout(connectWS, 800);
  };

  ws.onerror = () => { };
}

async function loadSelectedLanguage() {
  try {
    const r = await fetch("/api/language");
    if (!r.ok) return;
    const data = await r.json();
    selectedLanguage = String(data?.selected_language || "en");
    if (languageSelect) languageSelect.value = selectedLanguage;
  } catch { }
}

function setupLanguageSelector() {
  if (!languageSelect) return;
  languageSelect.addEventListener("change", async () => {
    if (isListening || isBusy) return;
    const next = String(languageSelect.value || "en");
    try {
      const r = await fetch("/api/language", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_language: next }),
      });
      if (!r.ok) throw new Error(`Failed to set language (${r.status})`);
      const data = await r.json();
      selectedLanguage = String(data?.selected_language || next);
      languageSelect.value = selectedLanguage;
      setStatus(`Language: ${languageLabel(selectedLanguage)}`, true);
    } catch (e) {
      console.warn(e);
      languageSelect.value = selectedLanguage;
      setStatus("Could not change language.", true);
    }
  });
}

async function askAssistant(userText) {
  const t = String(userText ?? "").trim();
  if (!t) return;
  setStatus("Thinking…", true);
  setResponseCard({ user_text: "", corrected: "", explanation: "", answer: "…" });
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: t }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    setStatus("Error", true);
    setResponseCard({ user_text: "", corrected: "", explanation: "", answer: msg || `Request failed (${r.status})` });
    return;
  }
  setStatus("Done", true);
}

async function listenOnce() {
  setStatus("Listening…", true);
  setResponseCard({ user_text: "", corrected: "", explanation: "", answer: "…" });
  const r = await fetch("/api/listen", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!r.ok) {
    let msg = "";
    try {
      const j = await r.json();
      msg = j?.detail || JSON.stringify(j);
    } catch {
      msg = await r.text().catch(() => "");
    }
    setStatus("Error", true);
    setResponseCard({ user_text: "", corrected: "", explanation: "", answer: msg || `Request failed (${r.status})` });
    return;
  }
  setStatus("Thinking…", true);
}

function setupListeningButton() {
  if (!listenBtn) return;

  listenBtn.addEventListener("click", () => {
    if (isBusy) return;

    if (!isListening) {
      isBusy = true;
      listenBtn.disabled = true;
      setListeningUi(true);
      setStatus("Listening…", true);

      fetch("/api/listen/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      })
        .then(async (r) => {
          if (!r.ok) {
            let msg = "";
            try {
              const j = await r.json();
              msg = j?.detail || JSON.stringify(j);
            } catch {
              msg = await r.text().catch(() => "");
            }
            throw new Error(msg || `Request failed (${r.status})`);
          }
          setStatus("Listening…", true);
        })
        .catch((err) => {
          console.warn(err);
          setListeningUi(false);
          setStatus("Error", true);
          setResponseCard({ user_text: "", corrected: "", explanation: "", answer: String(err?.message || err) });
        })
        .finally(() => {
          listenBtn.disabled = false;
          isBusy = false;
          if (languageSelect) languageSelect.disabled = false;
        });
      return;
    }

    isBusy = true;
    listenBtn.disabled = true;
    setStatus("Stopping…", true);
    setListeningUi(false);

    fetch("/api/listen/stop", { method: "POST" })
      .then(async (r) => {
        if (!r.ok) {
          let msg = "";
          try {
            const j = await r.json();
            msg = j?.detail || JSON.stringify(j);
          } catch {
            msg = await r.text().catch(() => "");
          }
          throw new Error(msg || `Request failed (${r.status})`);
        }
        setStatus("Thinking…", true);
      })
      .catch((err) => {
        console.warn(err);
        setStatus("Error", true);
        setResponseCard({ user_text: "", corrected: "", explanation: "", answer: String(err?.message || err) });
      })
      .finally(() => {
        listenBtn.disabled = false;
        isBusy = false;
        if (languageSelect) languageSelect.disabled = false;
      });
  });
}

async function start() {
  try {
    await loadRobot();
  } catch (e) {
    console.warn(e);
    setStatus("Could not load robot.glb. Check frontend/models/robot.glb", true);
  }

  connectWS();
  await loadSelectedLanguage();
  setupLanguageSelector();
  setupListeningButton();
  setStatus(`Idle · Language: ${languageLabel(selectedLanguage)}`, true);
  setListeningUi(false);

  setWelcomeMode(true);
  setResponseCard({ user_text: "", corrected: "", explanation: "", answer: "..." });
  if (!welcomeDone) {
    welcomeDone = true;
    try {
      await fetch("/api/welcome", { method: "POST" });
    } catch (e) {
      console.warn(e);
      setResponseCard({
        user_text: "",
        corrected: "",
        explanation: "",
        answer: "Welcome message failed. Please click Start Listening.",
      });
    }
  }

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    const t = clock.elapsedTime;
    controls.update();
    applyRobotMotion(t, dt);
    renderer.render(scene, camera);
  }
  animate();
}

start();
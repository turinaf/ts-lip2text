const datasetSelect = document.getElementById("datasetSelect");
const promptInput = document.getElementById("promptInput");
const generateBtn = document.getElementById("generateBtn");
const startCameraBtn = document.getElementById("startCameraBtn");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const submitBtn = document.getElementById("submitBtn");
const statusText = document.getElementById("statusText");

const videoEl = document.getElementById("videoEl");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");

const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const resultPrompt = document.getElementById("resultPrompt");
const resultProb = document.getElementById("resultProb");
const resultThreshold = document.getElementById("resultThreshold");
const resultFrames = document.getElementById("resultFrames");
const resultBadge = document.getElementById("resultBadge");

let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let faceMesh = null;
let cameraPipeline = null;

function setStatus(text, cls) {
  statusText.textContent = text;
  statusText.className = `status ${cls}`;
}

function resizeOverlay() {
  overlayCanvas.width = videoEl.videoWidth || videoEl.clientWidth;
  overlayCanvas.height = videoEl.videoHeight || videoEl.clientHeight;
}

function drawResults(results) {
  overlayCtx.save();
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  overlayCtx.drawImage(results.image, 0, 0, overlayCanvas.width, overlayCanvas.height);

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    for (const landmarks of results.multiFaceLandmarks) {
      drawConnectors(overlayCtx, landmarks, FACEMESH_LIPS, {
        color: "#f18f01",
        lineWidth: 2,
      });
      drawConnectors(overlayCtx, landmarks, FACEMESH_FACE_OVAL, {
        color: "rgba(15, 139, 141, 0.45)",
        lineWidth: 1,
      });
    }
  }

  overlayCtx.restore();
}

async function setupFaceMesh() {
  if (faceMesh) {
    return;
  }

  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
  });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  faceMesh.onResults(drawResults);
}

async function fetchPrompt() {
  const dataset = datasetSelect.value;
  setStatus("Loading prompt...", "busy");

  const res = await fetch(`/api/prompt?dataset=${encodeURIComponent(dataset)}`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to fetch prompt.");
  }

  const data = await res.json();
  promptInput.value = data.prompt;
  setStatus("Prompt ready. Start camera when you are ready.", "ok");
}

async function startCamera() {
  await setupFaceMesh();

  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 960 },
      height: { ideal: 540 },
      facingMode: "user",
    },
    audio: true,
  });

  videoEl.srcObject = stream;
  await videoEl.play();
  resizeOverlay();

  cameraPipeline = new Camera(videoEl, {
    onFrame: async () => {
      await faceMesh.send({ image: videoEl });
    },
    width: videoEl.videoWidth || 960,
    height: videoEl.videoHeight || 540,
  });
  cameraPipeline.start();

  recordBtn.disabled = false;
  setStatus("Camera started. Begin recording when ready.", "ok");
}

function startRecording() {
  if (!stream) {
    throw new Error("Start camera first.");
  }

  recordedChunks = [];
  recordedBlob = null;

  mediaRecorder = new MediaRecorder(stream, {
    mimeType: "video/webm;codecs=vp8,opus",
  });
  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(recordedChunks, { type: "video/webm" });
    submitBtn.disabled = !recordedBlob;
    setStatus("Recording stopped. Submit to verify.", "ok");
  };

  mediaRecorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  submitBtn.disabled = true;
  setStatus("Recording in progress...", "busy");
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  stopBtn.disabled = true;
  recordBtn.disabled = false;
}

async function submitRecording() {
  if (!recordedBlob) {
    throw new Error("No recording found. Record a clip first.");
  }

  const dataset = datasetSelect.value;
  const prompt = promptInput.value.trim();
  if (!prompt) {
    throw new Error("Prompt cannot be empty.");
  }

  setStatus("Uploading and verifying...", "busy");

  const formData = new FormData();
  formData.append("dataset", dataset);
  formData.append("prompt_text", prompt);
  formData.append("video", recordedBlob, "capture.webm");

  const res = await fetch("/api/verify", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Verification failed.");
  }

  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");

  resultPrompt.textContent = data.tokens.join(" ");
  resultProb.textContent = data.probability.toFixed(4);
  resultThreshold.textContent = data.threshold.toFixed(4);
  resultFrames.textContent = `${data.frames} / ${data.fps.toFixed(1)}`;

  resultBadge.textContent = data.accepted ? "MATCHED" : "NOT MATCHED";
  resultBadge.className = `badge ${data.accepted ? "pass" : "fail"}`;

  setStatus("Verification complete.", "ok");
}

generateBtn.addEventListener("click", async () => {
  try {
    await fetchPrompt();
  } catch (err) {
    setStatus(err.message, "err");
  }
});

startCameraBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (err) {
    setStatus(err.message, "err");
  }
});

recordBtn.addEventListener("click", () => {
  try {
    startRecording();
  } catch (err) {
    setStatus(err.message, "err");
  }
});

stopBtn.addEventListener("click", () => {
  try {
    stopRecording();
  } catch (err) {
    setStatus(err.message, "err");
  }
});

submitBtn.addEventListener("click", async () => {
  try {
    await submitRecording();
  } catch (err) {
    setStatus(err.message, "err");
  }
});

datasetSelect.addEventListener("change", () => {
  promptInput.value = "";
  resultContent.classList.add("hidden");
  resultEmpty.classList.remove("hidden");
  setStatus("Dataset changed. Generate a new prompt.", "idle");
});

window.addEventListener("resize", resizeOverlay);

fetchPrompt().catch((err) => setStatus(err.message, "err"));

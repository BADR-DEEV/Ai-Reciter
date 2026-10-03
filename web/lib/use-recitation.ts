"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Surah, Update } from "./types";

type State = "idle" | "connecting" | "listening" | "stopping" | "complete";
type Resources = { socket?: WebSocket; context?: AudioContext; stream?: MediaStream; node?: AudioWorkletNode; source?: AudioNode; playback?: AudioBufferSourceNode; timer?: ReturnType<typeof setInterval>; heartbeat?: ReturnType<typeof setInterval>; stopTimeout?: ReturnType<typeof setTimeout>; flushTimeout?: ReturnType<typeof setTimeout>; finishing?: boolean };
const endpoint = process.env.NEXT_PUBLIC_RECITER_WS || "ws://127.0.0.1:8000/ws/recite";

export function useRecitation(surah: Surah | null, startAyah = 1) {
  const [state, setState] = useState<State>("idle");
  const [update, setUpdate] = useState<Update | null>(null);
  const [error, setError] = useState("");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [device, setDevice] = useState("");
  const [demo, setDemo] = useState(false);
  const [fileName, setFileName] = useState("");
  const [connection, setConnection] = useState("Ready");
  const resources = useRef<Resources>({});
  const generation = useRef(0);

  const release = useCallback(() => {
    const r = resources.current;
    resources.current = {};
    r.node?.disconnect();
    r.source?.disconnect();
    if (r.playback) { r.playback.onended = null; try { r.playback.stop(); } catch { /* Already stopped. */ } }
    r.stream?.getTracks().forEach(track => track.stop());
    if (r.context && r.context.state !== "closed") void r.context.close();
    if (r.socket) { r.socket.onclose = null; r.socket.onmessage = null; r.socket.onerror = null; r.socket.onopen = null; r.socket.close(); }
    clearInterval(r.timer);
    clearInterval(r.heartbeat);
    clearTimeout(r.stopTimeout);
    clearTimeout(r.flushTimeout);
  }, []);

  const reset = useCallback(() => {
    generation.current++;
    release(); setState("idle"); setUpdate(null); setError(""); setLevel(0); setSeconds(0); setDemo(false); setDevice(""); setFileName(""); setConnection("Ready");
  }, [release]);

  useEffect(() => { reset(); return () => { generation.current++; release(); }; }, [surah?.id, startAyah, reset, release]);

  const finish = useCallback(() => {
    const r = resources.current;
    if (r.finishing) return;
    r.finishing = true;
    if (r.playback) { r.playback.onended = null; try { r.playback.stop(); } catch { /* Already ended. */ } }
    r.source?.disconnect(); r.stream?.getTracks().forEach(track => track.stop());
    clearInterval(r.timer); setLevel(0);
    if (r.socket?.readyState !== WebSocket.OPEN) { generation.current++; release(); setState("idle"); return; }
    setState("stopping");
    // Flush the final partial PCM frame before requesting the last decode.
    r.node?.port.postMessage({ type: "flush" });
    r.flushTimeout = setTimeout(() => {
      if (resources.current === r && r.socket?.readyState === WebSocket.OPEN) r.socket.send(JSON.stringify({ type: "stop" }));
    }, 250);
    r.stopTimeout = setTimeout(() => { release(); setState("idle"); setError("Final decoding timed out. Your existing results are preserved. Choose a starting ayah to resume."); }, 45000);
  }, [release]);

  const start = useCallback(async (file?: File) => {
    if (!surah) return;
    reset(); setState("connecting");
    const token = generation.current;
    try {
      let stream: MediaStream | undefined;
      if (file) {
        if (file.size > 50 * 1024 * 1024) throw new Error("Please choose an audio file smaller than 50 MB.");
        setFileName(file.name);
      } else {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access needs localhost or HTTPS.");
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        if (generation.current !== token) { stream.getTracks().forEach(track => track.stop()); return; }
        resources.current.stream = stream;
      }
      const context = new AudioContext();
      resources.current.context = context;
      let recording: AudioBuffer | undefined;
      if (file) {
        const bytes = await file.arrayBuffer();
        if (generation.current !== token) return;
        try { recording = await context.decodeAudioData(bytes); }
        catch { throw new Error("Cannot decode this audio file. Try a WAV, MP3, or M4A recording."); }
        if (generation.current !== token) return;
        if (recording.duration > 600) throw new Error("Please choose a recording no longer than 10 minutes.");
        if (!recording.length) throw new Error("This recording contains no audio.");
      }
      await context.audioWorklet.addModule("/audio/pcm-worklet.js");
      if (generation.current !== token) return;
      const node = new AudioWorkletNode(context, "pcm-recorder");
      resources.current.node = node;
      const source = recording ? context.createBufferSource() : context.createMediaStreamSource(stream!);
      if (source instanceof AudioBufferSourceNode) {
        source.buffer = recording!;
        source.onended = () => { if (generation.current === token) finish(); };
        resources.current.playback = source;
      }
      resources.current.source = source;
      const socket = new WebSocket(endpoint);
      resources.current.socket = socket;
      let ready = false;
      const timeout = setTimeout(() => { if (!ready && generation.current === token) { setError("The model service did not respond. Start the Python backend and try again."); release(); setState("idle"); } }, 15000);
      resources.current.stopTimeout = timeout;
      socket.onopen = () => socket.send(JSON.stringify({ surah: surah.id, start_ayah: startAyah }));
      socket.onmessage = event => {
        if (generation.current !== token) return;
        let message;
        try { message = JSON.parse(event.data); } catch { setError("Invalid response from the model service."); release(); setState("idle"); return; }
        if (message.type === "ready") {
          ready = true; clearTimeout(timeout);
          if (resources.current.finishing) return;
          setDevice(message.device); setState("listening");
          source.connect(node); node.connect(context.destination);
          void context.resume();
          if (source instanceof AudioBufferSourceNode) source.start();
          resources.current.timer = setInterval(() => setSeconds(value => value + 1), 1000);
          let lastResponse = Date.now();
          const original = socket.onmessage;
          socket.onmessage = event => { lastResponse = Date.now(); original?.call(socket, event); };
          resources.current.heartbeat = setInterval(() => {
            if (socket.readyState !== WebSocket.OPEN) return;
            if (Date.now() - lastResponse > 20000) {
              setError("The connection stopped responding. Results are preserved; resume from your starting ayah.");
              release(); setState("idle"); return;
            }
            socket.send(JSON.stringify({ type: "ping" }));
          }, 5000);
        } else if (message.type === "pong") {
          const lag = message.audio_seconds - message.decoded_seconds;
          setConnection(lag > 8 && message.busy ? `Model catching up · ${Math.round(lag)}s pending` : message.busy ? "Decoding on device" : "Connected · waiting for speech");
        } else if (message.type === "update" || message.type === "finished") {
          setConnection(`Last decode ${message.latency_ms || 0}ms`);
          setUpdate(message);
          if (message.complete || message.type === "finished") { release(); setLevel(0); setState(message.complete ? "complete" : "idle"); }
        } else if (message.type === "error") { setError(message.message); release(); setState("idle"); }
      };
      node.port.onmessage = event => {
        if (generation.current !== token) return;
        if (event.data?.type === "flushed") {
          clearTimeout(resources.current.flushTimeout);
          if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "stop" }));
          return;
        }
        const samples = new Float32Array(event.data);
        const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
        setLevel(Math.min(1, rms * 12));
        if (ready && socket.readyState === WebSocket.OPEN) {
          if (socket.bufferedAmount > 256000) { setError("The inference connection is too slow. Please restart your session."); release(); setState("idle"); }
          else socket.send(event.data);
        }
      };
      socket.onerror = () => { if (generation.current === token) { setError("Cannot connect to the local model. Start the backend on port 8000."); release(); setState("idle"); } };
      socket.onclose = () => { if (generation.current === token) { release(); setLevel(0); setState("idle"); setError("The inference connection closed. You can start a new session."); } };
    } catch (cause) {
      if (generation.current !== token) return;
      release(); setState("idle");
      setError(cause instanceof Error ? (cause.name === "NotAllowedError" ? "Microphone permission was denied. Allow access in your browser and try again." : cause.message) : "Could not start recording.");
    }
  }, [surah, startAyah, reset, release, finish]);

  const stop = useCallback(() => {
    if (demo) { release(); setState("idle"); setLevel(0); return; }
    finish();
  }, [demo, release, finish]);

  const startDemo = useCallback(() => {
    if (!surah) return;
    reset(); setDemo(true); setState("listening"); setDevice("demo");
    let index = 0, wordIndex = 0, ticks = 0;
    const results: Update["results"] = {};
    setUpdate({ type: "update", current: surah.ayahs[0].ayah, results: {}, transcript: "", complete: false, advanced: false });
    resources.current.timer = setInterval(() => {
      const ayah = surah.ayahs[index];
      const expected = ayah.normalized.split(" ");
      const missed = index === 2;
      wordIndex = missed ? expected.length : wordIndex + 1;
      const wordResults: Update["results"][number]["words"] = expected.map((text, i) => ({ index: i, text,
        status: i >= wordIndex ? "pending" : missed || (index === 0 && expected.length > 2 && i === 1) ? "missed" : "correct" }));
      const score = wordResults.filter(word => word.status === "correct").length / expected.length;
      const finalized = wordIndex === expected.length;
      results[ayah.ayah] = { status: finalized && missed ? "missed" : score >= 0.65 ? "correct" : "listening",
        score, missing: wordResults.filter(word => word.status === "missed").map(word => word.text), final: finalized, words: wordResults };
      const transcript = wordResults.filter(word => word.status === "correct").map(word => word.text).join(" ");
      if (finalized) { index++; wordIndex = 0; }
      setSeconds(Math.floor(++ticks * 0.65)); setLevel(0.4 + Math.random() * 0.3);
      const complete = index === surah.ayahs.length;
      setUpdate({ type: "update", current: complete ? null : surah.ayahs[index].ayah, results: { ...results }, transcript, complete, advanced: finalized });
      if (complete) { release(); setState("complete"); setLevel(0); }
    }, 650);
  }, [surah, reset, release]);

  return { state, update, error, level, seconds, device, demo, fileName, connection, start, stop, reset, startDemo };
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const RATE = 16000;
type Handles = { context?: AudioContext; stream?: MediaStream; node?: AudioWorkletNode; source?: AudioNode };

/** Record one utterance as 16 kHz mono PCM, stopping after trailing silence or `maxSeconds`. */
export function useRecorder({ maxSeconds = 12, silenceSeconds = 0.9 } = {}) {
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState("");
  const handles = useRef<Handles>({});
  const chunks = useRef<Float32Array[]>([]);
  const resolver = useRef<((pcm: Float32Array | null) => void) | null>(null);

  const release = useCallback(() => {
    const h = handles.current;
    handles.current = {};
    h.node?.disconnect(); h.source?.disconnect();
    h.stream?.getTracks().forEach(t => t.stop());
    if (h.context && h.context.state !== "closed") void h.context.close();
    setRecording(false); setLevel(0);
  }, []);

  const stop = useCallback(() => {
    const done = resolver.current;
    resolver.current = null;
    release();
    if (!done) return;
    const total = chunks.current.reduce((n, c) => n + c.length, 0);
    const pcm = new Float32Array(total);
    let offset = 0;
    for (const c of chunks.current) { pcm.set(c, offset); offset += c.length; }
    chunks.current = [];
    done(total ? pcm : null);
  }, [release]);

  useEffect(() => () => { resolver.current?.(null); resolver.current = null; release(); }, [release]);

  const record = useCallback(async (): Promise<Float32Array | null> => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) { setError("Microphone access needs localhost or HTTPS."); return null; }
    let stream: MediaStream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true } }); }
    catch { setError("Microphone permission was denied. Allow access in your browser to practise speaking."); return null; }
    const context = new AudioContext();
    handles.current = { context, stream };
    let node: AudioWorkletNode, source: MediaStreamAudioSourceNode;
    try {
      await context.audioWorklet.addModule("/audio/pcm-worklet.js");
      node = new AudioWorkletNode(context, "pcm-recorder");
      source = context.createMediaStreamSource(stream);
    } catch (cause) {
      release();
      setError(`Could not start audio capture: ${cause instanceof Error ? cause.message : cause}`);
      return null;
    }
    handles.current = { context, stream, node, source };
    chunks.current = [];
    let samples = 0, voiced = 0, lastVoice = 0;
    node.port.onmessage = event => {
      if (!(event.data instanceof ArrayBuffer)) return;
      const frame = new Float32Array(event.data);
      chunks.current.push(frame);
      samples += frame.length;
      const rms = Math.sqrt(frame.reduce((s, v) => s + v * v, 0) / frame.length);
      setLevel(Math.min(1, rms * 10));
      if (rms > 0.015) { voiced += frame.length; lastVoice = samples; }
      // Stop after `silenceSeconds` of quiet once at least 0.25 s of voice was heard.
      if ((voiced > RATE * 0.25 && samples - lastVoice > RATE * silenceSeconds) || samples > RATE * maxSeconds) stop();
    };
    source.connect(node); node.connect(context.destination);
    await context.resume();
    setRecording(true);
    return new Promise(resolve => { resolver.current = resolve; });
  }, [stop, maxSeconds, silenceSeconds]);

  return { record, stop, recording, level, error };
}

/** Wrap PCM as a WAV blob so learners can hear themselves back. */
export function wavUrl(pcm: Float32Array) {
  const buffer = new ArrayBuffer(44 + pcm.length * 2), view = new DataView(buffer);
  const text = (o: number, s: string) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}

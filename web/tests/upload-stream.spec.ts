import { test, expect } from "@playwright/test";

test("upload sends all 36 seconds in credited batches, without a microphone or realtime replay", async ({ page }) => {
  let samples = 0, stopped = false, inputMode = "";
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new Error("Upload must not request microphone access"); };
  });
  await page.routeWebSocket("**/ws/recite", socket => {
    socket.onMessage(message => {
      if (typeof message === "string") {
        const config = JSON.parse(message);
        if (config.surah) {
          inputMode = config.input_mode;
          socket.send(JSON.stringify({ type: "ready", device: "test", upload_credit_seconds: 12 }));
        } else if (config.type === "stop") {
          stopped = true;
          socket.send(JSON.stringify({ type: "finished", current: 1, results: {}, transcript: "", complete: false, advanced: false, decoded_seconds: samples / 16000 }));
        } else if (config.type === "ping") {
          socket.send(JSON.stringify({ type: "pong", audio_seconds: samples / 16000, decoded_seconds: samples / 16000, busy: false }));
        }
      } else {
        expect(message.length).toBeLessThanOrEqual(16384);
        samples += message.length / 4;
        socket.send(JSON.stringify({ type: "update", current: 1, results: {}, transcript: "", complete: false, advanced: false, decoded_seconds: samples / 16000, pending_seconds: 0 }));
      }
    });
  });
  const count = 36 * 16000;
  const wav = Buffer.alloc(44 + count * 2);
  wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write("data", 36); wav.writeUInt32LE(count * 2, 40);
  await page.goto("/studio");
  await expect(page.getByRole("button", { name: "Upload audio to test", exact: true })).toBeEnabled();
  const start = Date.now();
  await page.locator('input[type="file"]').setInputFiles({ name: "36-seconds.wav", mimeType: "audio/wav", buffer: wav });
  await expect.poll(() => stopped, { timeout: 12000 }).toBe(true);
  expect(Date.now() - start).toBeLessThan(12000);
  expect(samples).toBe(count);
  expect(inputMode).toBe("file");
  await expect(page.locator(".upload-file-note")).toContainText("Accelerated local processing");
});

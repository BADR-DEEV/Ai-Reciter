// Microphone audio never leaves the local inference connection.
// Resample with a continuous fractional accumulator, preserving block boundaries.
class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0;
    this.sum = 0;
    this.weight = 0;
    this.frame = new Float32Array(2048);
    this.offset = 0;
    this.port.onmessage = event => {
      if (event.data?.type !== "flush") return;
      if (this.offset) {
        const tail = this.frame.slice(0, this.offset);
        this.port.postMessage(tail.buffer, [tail.buffer]);
        this.offset = 0;
      }
      this.port.postMessage({ type: "flushed" });
    };
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      let remaining = 1;
      while (remaining > 0.000001) {
        const take = Math.min(remaining, this.ratio - this.phase);
        this.sum += value * take;
        this.weight += take;
        this.phase += take;
        remaining -= take;
        if (this.phase >= this.ratio - 0.000001) {
          // Browser resampling can overshoot PCM full scale slightly. Clip
          // legitimate peaks before the server's bounded-sample validation.
          this.frame[this.offset++] = Math.max(-1, Math.min(1, this.sum / this.weight));
          this.phase = this.sum = this.weight = 0;
          if (this.offset === this.frame.length) {
            this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
            this.frame = new Float32Array(2048);
            this.offset = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PCMRecorder);

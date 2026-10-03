import { qaloonG2P } from "@/lib/qaloon-g2p";

export function PhoneticAid({ text }: { text: string }) {
  const result = qaloonG2P(text);
  return <div className="phonetic-aid" lang="en" dir="ltr">
    <p>{result.text}</p>
    <small>Qālūn source · draft reading aid · pause at the end</small>
    <details><summary>Pronunciation key & review notes</summary>
      <p>ā / ī / ū: long vowels; doubled letters: hold the consonant; ʿ: ʿayn; ʾ: hamza; ḥ / ṣ / ḍ / ṭ / ẓ have no exact English equivalents.</p>
      <p>This spelling cannot teach tajweed, nasalization or permitted Qālūn variants. Listen to a verified Qālūn recitation and consult a teacher. Not scholar-approved.</p>
      {result.warnings.map(note => <p key={note}>{note}</p>)}
    </details>
  </div>;
}

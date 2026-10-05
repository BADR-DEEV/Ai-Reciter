import { NextRequest } from "next/server";
import { readerAudioState, startReaderAudioDownload } from "@/lib/reader-audio";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json(await readerAudioState(), { headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: NextRequest) {
  // Only this app's own pages may start a download.
  const site = request.headers.get("sec-fetch-site");
  if (site && site !== "same-origin") return new Response("Cross-site request refused", { status: 403 });
  return Response.json(await startReaderAudioDownload(), { headers: { "Cache-Control": "no-store" } });
}

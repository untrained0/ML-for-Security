import { sampleImages, DatasetError } from '@/server/datasets';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * GET /api/datasets/images?source=mnist|cifar10&classes=7,1&train=100&valid=500&test=all
 * A fresh random sample from the original dataset (binary; see sampleImages for the layout).
 */
export async function GET(request: Request) {
  const q = new URL(request.url).searchParams;
  const int = (k: string, fallback: number) => {
    const v = parseInt(q.get(k) ?? '', 10);
    return Number.isFinite(v) && v >= 0 ? v : fallback;
  };
  try {
    const body = sampleImages({
      source: q.get('source') ?? 'mnist',
      classes: (q.get('classes') ?? '').split(',').map(Number),
      train: int('train', 100),
      valid: int('valid', 500),
      test: q.get('test') === 'all' ? 'all' : int('test', 500),
    });
    return new Response(new Uint8Array(body), {
      headers: { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store' },
    });
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: e instanceof DatasetError ? e.status : 500 });
  }
}

import { sampleRegression, DatasetError } from '@/server/datasets';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * GET /api/datasets/regression?key=warfarin|loan|house&n=1400
 * n random records of the complete preprocessed file, as CSV (response in column 0).
 */
export async function GET(request: Request) {
  const q = new URL(request.url).searchParams;
  const n = Math.min(20000, Math.max(1, parseInt(q.get('n') ?? '1400', 10) || 1400));
  try {
    return new Response(sampleRegression(q.get('key') ?? '', n), {
      headers: { 'Content-Type': 'text/csv; charset=utf-8', 'Cache-Control': 'no-store' },
    });
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: e instanceof DatasetError ? e.status : 500 });
  }
}

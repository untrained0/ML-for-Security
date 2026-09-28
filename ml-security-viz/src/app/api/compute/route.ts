/** GET /api/compute — which backend the attack server uses (CUDA GPU or CPU), and why. */

import { serverCompute } from '@/server/compute';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export function GET() {
  return Response.json(serverCompute(), { headers: { 'Cache-Control': 'no-store' } });
}

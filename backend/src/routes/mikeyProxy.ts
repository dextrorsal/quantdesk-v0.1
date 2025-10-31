import { Router, Request, Response } from 'express';

// Simple proxy to MIKEY-AI service (port 3000)
// Avoids CORS by routing through the backend API gateway (port 3002)

const router = Router();
const MIKEY_BASE = process.env.MIKEY_BASE_URL || 'http://localhost:3000';

router.get('/health', async (_req: Request, res: Response) => {
  try {
    const r = await fetch(`${MIKEY_BASE}/health`);
    const data = await r.json();
    res.status(r.status).json(data);
  } catch (err: any) {
    res.status(502).json({ success: false, error: 'MIKEY-AI unavailable', details: err?.message });
  }
});

router.get('/llm/status', async (_req: Request, res: Response) => {
  try {
    const r = await fetch(`${MIKEY_BASE}/api/v1/llm/status`);
    const data = await r.json();
    res.status(r.status).json(data);
  } catch (err: any) {
    res.status(502).json({ success: false, error: 'LLM status unavailable', details: err?.message });
  }
});

router.post('/query', async (req: Request, res: Response) => {
  try {
    const r = await fetch(`${MIKEY_BASE}/api/v1/ai/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: req.body?.query || '' })
    });
    const data = await r.json();
    res.status(r.status).json(data);
  } catch (err: any) {
    res.status(502).json({ success: false, error: 'MIKEY query failed', details: err?.message });
  }
});

export default router;



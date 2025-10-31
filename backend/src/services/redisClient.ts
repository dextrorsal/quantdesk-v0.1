// Dynamic Redis import to avoid loading Redis module when not needed
let createClient: any = null;
let RedisClientType: any = null;

// Environment-aware key building
const ENV_NAME = process.env.ENV_NAME || process.env.NODE_ENV || 'dev';
const makeKey = (...parts: string[]) => `qd:${ENV_NAME}:${parts.join(':')}`;

// Simple jittered backoff between reconnect attempts
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const getBackoffMs = (attempt: number) => Math.min(1000 * Math.pow(2, attempt) + Math.floor(Math.random() * 200), 15000);

let client: any = null;
let subscriber: any = null;

// Lazy initialization - only create clients when actually needed
const createRedisClient = (): any => {
  // Skip Redis in development if no Redis URL is configured
  if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
    console.log('⚠️  Redis disabled in development mode');
    return null;
  }
  
  // Dynamically import Redis only when needed
  if (!createClient) {
    try {
      const redis = require('redis');
      createClient = redis.createClient;
      RedisClientType = redis.RedisClientType;
    } catch (error) {
      console.error('Failed to load Redis module:', error);
      return null;
    }
  }
  
  const url = process.env.REDIS_URL || 'redis://localhost:6379';
  const newClient = createClient({ url, socket: { reconnectStrategy: () => 0 } });
  newClient.on('error', (err) => console.error('Redis client error', err));
  return newClient;
};

const createRedisSubscriber = (): any => {
  // Skip Redis in development if no Redis URL is configured
  if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
    console.log('⚠️  Redis disabled in development mode');
    return null;
  }
  
  // Dynamically import Redis only when needed
  if (!createClient) {
    try {
      const redis = require('redis');
      createClient = redis.createClient;
      RedisClientType = redis.RedisClientType;
    } catch (error) {
      console.error('Failed to load Redis module:', error);
      return null;
    }
  }
  
  const url = process.env.REDIS_URL || 'redis://localhost:6379';
  const newSubscriber = createClient({ url, socket: { reconnectStrategy: () => 0 } });
  newSubscriber.on('error', (err) => console.error('Redis subscriber error', err));
  return newSubscriber;
};

export const getRedis = (): any => {
  if (client) return client;
  client = createRedisClient();
  return client;
};

export const getRedisSubscriber = (): any => {
  if (subscriber) return subscriber;
  subscriber = createRedisSubscriber();
  return subscriber;
};

export const connectRedis = async (): Promise<void> => {
  // Skip Redis connection in development if no Redis URL is configured
  if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
    console.log('⚠️  Skipping Redis connection in development mode');
    return;
  }

  const c = createRedisClient();
  const s = createRedisSubscriber();
  
  if (!c || !s) {
    console.log('⚠️  Redis clients not available');
    return;
  }
  
  let attempt = 0;
  while (!(c as any).isOpen) {
    try {
      await c.connect();
    } catch (e) {
      attempt += 1;
      const wait = getBackoffMs(attempt);
      console.warn(`Redis connect failed (client). Retrying in ${wait}ms...`, e);
      await sleep(wait);
    }
  }
  attempt = 0;
  while (!(s as any).isOpen) {
    try {
      await s.connect();
    } catch (e) {
      attempt += 1;
      const wait = getBackoffMs(attempt);
      console.warn(`Redis connect failed (subscriber). Retrying in ${wait}ms...`, e);
      await sleep(wait);
    }
  }
};

export const pingRedis = async (): Promise<{ ok: boolean; info?: unknown; error?: string }> => {
  try {
    const c = getRedis();
    if (!c) {
      return { ok: true, info: 'Redis disabled in development mode' };
    }
    if (!(c as any).isOpen) await c.connect();
    const pong = await c.ping();
    return { ok: pong === 'PONG' };
  } catch (e: unknown) {
    const error = e instanceof Error ? e.message : String(e);
    return { ok: false, error };
  }
};

export const setSession = async (walletPubkey: string, data: unknown, ttlSec = 3600) => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, skipping session storage');
    return;
  }
  await c.set(makeKey('session', walletPubkey), JSON.stringify(data), { EX: ttlSec });
};

export const getSession = async (walletPubkey: string): Promise<Record<string, unknown> | null> => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, returning null session');
    return null;
  }
  const val = await c.get(makeKey('session', walletPubkey));
  return val ? JSON.parse(val) : null;
};

export const setPresence = async (channel: string, walletPubkey: string, ttlSec = 65) => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, skipping presence storage');
    return;
  }
  await c.set(makeKey('presence', channel, walletPubkey), '1', { EX: ttlSec });
};

export const getPresenceCount = async (channel: string): Promise<number> => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, returning 0 presence count');
    return 0;
  }
  const keys = await c.keys(makeKey('presence', channel, '*'));
  return keys.length;
};

export const rateLimitSimple = async (key: string, limit: number, windowSec: number): Promise<boolean> => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, allowing request (no rate limiting)');
    return true;
  }
  const nowBucket = Math.floor(Date.now() / 1000 / windowSec);
  const redisKey = makeKey('rl', key, String(nowBucket));
  const count = await c.incr(redisKey);
  if (count === 1) await c.expire(redisKey, windowSec);
  return count <= limit;
};

export const publishRedisMessage = async (channel: string, message: string): Promise<void> => {
  const c = getRedis();
  if (!c) {
    console.log('Redis not available, skipping message publish');
    return;
  }
  await c.publish(makeKey('pubsub', channel), message);
};



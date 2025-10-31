import { Connection, Keypair, PublicKey, Transaction, sendAndConfirmTransaction } from '@solana/web3.js'
import { HermesClient } from '@pythnetwork/hermes-client'
import { PythSolanaReceiver } from '@pythnetwork/pyth-solana-receiver'
import { Logger } from '../utils/logger'

type WorkerConfig = {
  rpcUrl: string
  receiverProgramId: string
  feedIds: string[]
  walletSecretBase64?: string
  batchSize?: number
  intervalMs?: number
}

export class PythReceiverWorker {
  private readonly logger: Logger
  private readonly connection: Connection
  private readonly hermes: HermesClient
  private readonly receiver: PythSolanaReceiver
  private readonly feedIds: string[]
  private readonly batchSize: number
  private readonly intervalMs: number
  private readonly wallet?: Keypair
  private timer: NodeJS.Timeout | null = null

  constructor(cfg: WorkerConfig) {
    this.logger = new Logger()
    this.connection = new Connection(cfg.rpcUrl, 'confirmed')
    this.hermes = new HermesClient(process.env['PYTH_BASE_URL'] || 'https://hermes.pyth.network')
    this.receiver = new PythSolanaReceiver(new PublicKey(cfg.receiverProgramId))
    this.feedIds = cfg.feedIds
    this.batchSize = cfg.batchSize ?? 10
    this.intervalMs = cfg.intervalMs ?? 5000
    if (cfg.walletSecretBase64) {
      try {
        const secret = Buffer.from(cfg.walletSecretBase64, 'base64')
        this.wallet = Keypair.fromSecretKey(secret)
      } catch {
        this.logger.warn('Invalid wallet secret; worker will run in dry mode (no on-chain submit).')
      }
    } else {
      this.logger.warn('No wallet provided; worker will run in dry mode (no on-chain submit).')
    }
  }

  start() {
    if (this.timer) return
    this.logger.info(`🛰️ PythReceiverWorker starting with ${this.feedIds.length} feeds`) 
    this.timer = setInterval(() => { this.tick().catch((e) => this.logger.warn('Worker tick failed:', e?.message || e)) }, this.intervalMs)
    // run immediately
    this.tick().catch((e) => this.logger.warn('Initial tick failed:', e?.message || e))
  }

  stop() {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
  }

  private async tick(): Promise<void> {
    if (this.feedIds.length === 0) return
    const batches: string[][] = []
    for (let i = 0; i < this.feedIds.length; i += this.batchSize) {
      batches.push(this.feedIds.slice(i, i + this.batchSize))
    }

    for (const ids of batches) {
      // 1) Fetch binary updates
      const resp = await this.hermes.getLatestPriceUpdates(ids, { parsed: false })
      const vaas: Uint8Array[] = Array.isArray(resp.binary) ? resp.binary.map((b: any) => Buffer.from(b, 'base64')) : []
      if (vaas.length === 0) continue

      // 2) Build Receiver instruction(s)
      const ix = await this.receiver.updatePriceFeedsInstruction(vaas)
      if (!this.wallet) {
        this.logger.info(`Dry run: fetched ${vaas.length} updates for ${ids.length} ids`) 
        continue
      }

      // 3) Submit transaction
      const tx = new Transaction().add(ix)
      tx.feePayer = this.wallet.publicKey
      tx.recentBlockhash = (await this.connection.getLatestBlockhash('finalized')).blockhash
      tx.sign(this.wallet)
      try {
        const sig = await sendAndConfirmTransaction(this.connection, tx, [this.wallet], { commitment: 'confirmed' })
        this.logger.info(`✅ Submitted Pyth Receiver update tx: ${sig}`)
      } catch (e: any) {
        this.logger.warn(`Receiver submit failed: ${e?.message || e}`)
      }
    }
  }
}

export function getEnvFeedIds(): string[] {
  const env = process.env || {}
  const prefix = 'PYTH_PRICE_FEED_'
  const hex64 = /^0x[a-fA-F0-9]{64}$/
  const ids: string[] = []
  for (const [k, v] of Object.entries(env)) {
    if (!k.startsWith(prefix)) continue
    const id = String(v || '').trim()
    if (hex64.test(id)) ids.push(id.toLowerCase())
  }
  return ids
}


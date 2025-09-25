-- exchanges table
CREATE TABLE IF NOT EXISTS exchanges (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- markets table
CREATE TABLE IF NOT EXISTS markets (
    id SERIAL PRIMARY KEY,
    exchange_id INTEGER REFERENCES exchanges(id),
    type TEXT,
    symbol TEXT,
    base_asset TEXT,
    quote_asset TEXT,
    UNIQUE (exchange_id, symbol)
);

-- candles table
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    market_id INTEGER REFERENCES markets(id),
    resolution TEXT,
    ts TIMESTAMPTZ,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE (market_id, resolution, ts)
);

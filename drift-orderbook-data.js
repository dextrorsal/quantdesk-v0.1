// Drift Order Book Data Extracted
const driftOrderBookData = {
  structure: {
    container: "h-full border-container-border border relative overflow-hidden border-none rounded-md-old new-brand:rounded-sm bg-container-bg",
    header: "flex flex-col gap-sm bg-container-bg p-sm",
    tabs: "Orderbook | Recent Trades",
    controls: "$1 USD dropdown",
    columns: "Price | Size (USD) | Total (USD)",
    rows: "flex-col-reverse layout with buy/sell sections"
  },
  styling: {
    backgroundColor: "rgb(8, 15, 24)", // bg-container-bg
    borderColor: "rgb(16, 26, 39)", // border-container-border
    borderRadius: "6px",
    fontFamily: "Marfa, Lato, Arial",
    fontSize: "14px",
    lineHeight: "21px",
    height: "704px",
    width: "260px"
  },
  colors: {
    buyOrders: "var(--color-positive-green)", // Green for buy orders
    sellOrders: "var(--color-negative-red)", // Red for sell orders
    textPrimary: "rgb(0, 0, 0)",
    textSecondary: "rgb(0, 0, 0)",
    background: "rgb(8, 15, 24)",
    border: "rgb(16, 26, 39)"
  },
  layout: {
    rowHeight: "22px",
    padding: "8px", // p-sm
    gap: "8px", // gap-sm
    flexDirection: "column",
    justifyContent: "space-between"
  }
};

export default driftOrderBookData;

# Terminal Shortcuts & Command Palette

QuantDesk’s terminal interface uses a command palette and keyboard shortcuts so power users can move fast. Here’s the condensed cheat sheet for navigating the workspace.

## Launching the Command Palette

- Press the **` (backtick)** key anywhere in the terminal to open the palette.
- Type to filter commands; use **↑ / ↓** to highlight and **Enter** to run.
- Hit **Esc** to close or **Tab** to auto-complete a highlighted command.

## High-Value Commands

| Command | What It Opens |
|---------|----------------|
| `QM` | Quote Monitor with live perp pricing |
| `CHART` | Advanced chart with indicators and overlays |
| `ORDER` | Trading ticket for market/limit/bracket orders |
| `POSITIONS` | Active positions with P&L and liquidation buffers |
| `PF` | Portfolio overview and performance metrics |
| `ALERT` | Price alert configuration |
| `FLOW` | Smart money & whale activity feed |
| `DEFI` | DeFi protocol stats and TVL snapshots |
| `NEWS` | Streaming headlines relevant to current markets |
| `HELP` | In-terminal documentation and command list |

## Keyboard Shortcuts

### Trading Actions
- **Ctrl + B** – Quick buy ticket for the selected market.
- **Ctrl + S** – Quick sell ticket.
- **Ctrl + C** – Close the highlighted position.
- **Ctrl + A** – Create a new alert at the current price.

### Window Management
- **Ctrl + W** – Close the focused window.
- **Ctrl + M** – Minimize it to the dock.
- **Ctrl + Shift + C** – Clear all open windows.
- **Ctrl + L** – Save the current layout; run `LAYOUT` to load later.

### Miscellaneous
- **S** – Show the full keyboard shortcut list.
- **ERR** – Report an issue directly to the team.
- **CHAT** – Jump into the community chat panel.

## Workflow Tips

1. Launch **`QM`** for quick market scanning, then open **`CHART`** on the asset you want to trade.
2. Keep **`POSITIONS`** docked so liquidation buffers and P&L stay visible while you work.
3. Save layouts for different moods (scalping, research, monitoring) and swap using the `LAYOUT` command.
4. Use alerts (`ALERT`) instead of babysitting charts—MIKEY pings you when conditions change.

The command palette will keep growing as new features roll out; hit **`HELP`** or check the in-terminal release notes to see the latest additions.

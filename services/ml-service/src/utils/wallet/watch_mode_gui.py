"""
GUI-based watch mode for Solana wallet monitoring with multi-wallet support
"""

import sys
import asyncio
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QFrame, QScrollArea, QListWidget, QSplitter, QComboBox, QMessageBox, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QPalette, QColor, QFont
import click
from datetime import datetime
from .price_service import get_price_service
from .wallet_manager import WalletManager
from solana.rpc.async_api import AsyncClient
import json
import os
import subprocess
import requests

class SolanaWatchWorker(QThread):
    """Background worker to fetch wallet and price updates"""
    balance_updated = pyqtSignal(str, float, float, float)  # wallet_name, sol_balance, usd_price, cad_price
    transactions_updated = pyqtSignal(str, list)  # wallet_name, transactions
    
    def __init__(self, wallet_configs):
        super().__init__()
        self.wallet_configs = wallet_configs
        self.running = True
        self._last_usd_price = 0
        self._last_cad_price = 0

    def run(self):
        while self.running:
            try:
                # Get SOL price
                response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd,cad")
                prices = response.json()["solana"]
                self._last_usd_price = prices["usd"]
                self._last_cad_price = prices["cad"]

                # Get balances and transactions for each wallet
                for name, config in self.wallet_configs.items():
                    # Get balance
                    result = subprocess.run(
                        ["solana", "balance", "--url", "devnet", config["pubkey"]],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        try:
                            balance = float(result.stdout.strip().split()[0])
                            self.balance_updated.emit(name, balance, self._last_usd_price, self._last_cad_price)
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing balance for {name}: {e}")
                            continue

                    # Get recent transactions using signatures command
                    result = subprocess.run(
                        ["solana", "signatures", "--url", "devnet", config["pubkey"], "--limit", "10"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        transactions = []
                        lines = result.stdout.strip().split("\n")
                        for line in lines[2:]:  # Skip header lines
                            if not line:  # Skip empty lines
                                continue
                            try:
                                # Format: Signature | Status | Confirmations | Slot | Timestamp
                                parts = line.split()
                                if len(parts) >= 5:  # Ensure we have all required fields
                                    transactions.append({
                                        "signature": parts[0],
                                        "status": parts[1],
                                        "timestamp": parts[4]  # Already in readable format
                                    })
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing transaction: {e}")
                                continue
                                
                        self.transactions_updated.emit(name, transactions)

            except Exception as e:
                print(f"Error in watch worker: {e}")

            # Sleep for 5 seconds before next update
            self.thread().msleep(5000)

    def stop(self):
        """Stop the worker thread"""
        self.running = False

class WalletSelectorWidget(QWidget):
    wallet_changed = pyqtSignal(str)  # Emits selected wallet name

    def __init__(self, wallet_configs):
        super().__init__()
        self.wallet_configs = wallet_configs
        
        layout = QHBoxLayout()
        
        # Create wallet selector combo box
        self.selector = QComboBox()
        self.selector.addItems(wallet_configs.keys())
        self.selector.currentTextChanged.connect(self.wallet_changed.emit)
        
        # Add label and combo box to layout
        layout.addWidget(QLabel("Select Wallet:"))
        layout.addWidget(self.selector)
        layout.addStretch()
        
        self.setLayout(layout)

class BalanceWidget(QWidget):
    """Widget displaying wallet balance with fiat values"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create labels with larger font
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        
        # SOL Balance
        self.sol_balance = QLabel("Loading...")
        self.sol_balance.setFont(font)
        self.sol_balance.setStyleSheet("color: #03E1FF;")  # Solana blue
        
        # USD Balance
        self.usd_balance = QLabel("")
        self.usd_balance.setFont(font)
        self.usd_balance.setStyleSheet("color: white;")
        
        # CAD Balance
        self.cad_balance = QLabel("")
        self.cad_balance.setFont(font)
        self.cad_balance.setStyleSheet("color: white;")
        
        # Add labels to layout with some spacing
        layout.addWidget(QLabel("Current Balance:"))
        layout.addWidget(self.sol_balance)
        layout.addSpacing(10)
        layout.addWidget(self.usd_balance)
        layout.addWidget(self.cad_balance)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_balance(self, sol_balance, usd_price, cad_price):
        """Update the balance display with new values"""
        # Format SOL balance
        self.sol_balance.setText(f"◎ {sol_balance:.4f} Solana")
        
        # Calculate and format fiat values
        usd_value = sol_balance * usd_price
        cad_value = sol_balance * cad_price
        
        # Update USD display (bold, no emoji)
        self.usd_balance.setText(f"${usd_value:,.2f} USD")
        
        # Update CAD display (bold, italic, with maple leaf)
        self.cad_balance.setText(f"${cad_value:,.2f} 🍁")

class TransactionWidget(QWidget):
    """Widget displaying recent transactions"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create transaction table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Signature", "Status", "Timestamp"])
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                gridline-color: #3d3d3d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                padding: 5px;
                border: 1px solid #2b2b2b;
                font-weight: bold;
            }
        """)
        
        # Set column widths
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Signature
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)    # Status
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)    # Timestamp
        self.table.setColumnWidth(1, 100)  # Status column width
        self.table.setColumnWidth(2, 150)  # Timestamp column width
        
        # Add table to layout
        layout.addWidget(QLabel("Recent Transactions:"))
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def update_transactions(self, transactions):
        """Update the transaction table with new data"""
        self.table.setRowCount(len(transactions))
        
        for i, tx in enumerate(transactions):
            # Create signature item with monospace font
            sig_item = QTableWidgetItem(tx["signature"])
            sig_item.setFont(QFont("Monospace"))
            self.table.setItem(i, 0, sig_item)
            
            # Create status item with colored background
            status_item = QTableWidgetItem(tx["status"])
            if tx["status"].lower() == "success":
                status_item.setBackground(QColor("#2e7d32"))  # Green
            elif tx["status"].lower() == "failed":
                status_item.setBackground(QColor("#c62828"))  # Red
            else:
                status_item.setBackground(QColor("#f9a825"))  # Yellow for pending
            self.table.setItem(i, 1, status_item)
            
            # Create timestamp item
            time_item = QTableWidgetItem(tx["timestamp"])
            self.table.setItem(i, 2, time_item)

class WalletDashboard(QFrame):
    """Dashboard widget showing balance and transactions for a wallet"""
    def __init__(self, wallet_name, wallet_pubkey):
        super().__init__()
        self.wallet_name = wallet_name
        self.wallet_pubkey = wallet_pubkey
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Wallet header
        header = QLabel(f"Dashboard - {self.wallet_name}")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #5555FF;")
        layout.addWidget(header)
        
        # Address display
        address_layout = QHBoxLayout()
        address_label = QLabel("Address:")
        address_label.setStyleSheet("font-weight: bold;")
        address_layout.addWidget(address_label)
        
        address_value = QLabel(f"{self.wallet_pubkey[:20]}...{self.wallet_pubkey[-8:]}")
        address_value.setStyleSheet("font-family: monospace;")
        address_layout.addWidget(address_value)
        address_layout.addStretch()
        
        address_widget = QWidget()
        address_widget.setLayout(address_layout)
        layout.addWidget(address_widget)
        
        # Balance display
        self.balance_widget = BalanceWidget()
        layout.addWidget(self.balance_widget)
        
        # Transaction history
        self.transaction_widget = TransactionWidget()
        layout.addWidget(self.transaction_widget)
        
        self.setLayout(layout)
    
    def update_balance(self, sol_balance, usd_price, cad_price):
        self.balance_widget.update_balance(sol_balance, usd_price, cad_price)
    
    def update_transactions(self, transactions):
        self.transaction_widget.update_transactions(transactions)

class WatchModeWindow(QMainWindow):
    """Main window for the multi-wallet watch mode GUI"""
    def __init__(self, wallet_configs):
        super().__init__()
        self.wallet_configs = wallet_configs
        self.current_wallet = next(iter(wallet_configs.keys()))  # Default to first wallet
        self.dashboards = {}  # Store wallet dashboards
        self.setup_ui()
        self.setup_worker()
        
        # Create dashboard for MAIN wallet if it exists
        if "MAIN" in self.wallet_configs:
            self.create_dashboard("MAIN")
        elif self.wallet_configs:  # If no MAIN wallet, create dashboard for first available wallet
            first_wallet = next(iter(self.wallet_configs.keys()))
            self.create_dashboard(first_wallet)
        
    def setup_ui(self):
        self.setWindowTitle("Solana Multi-Wallet Watch Mode")
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create wallet selector header
        header_layout = QHBoxLayout()
        self.wallet_selector = WalletSelectorWidget(self.wallet_configs)
        self.wallet_selector.wallet_changed.connect(self.change_wallet)
        header_layout.addWidget(self.wallet_selector)
        
        # Add refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 15px;
                color: white;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_current_wallet)
        header_layout.addWidget(refresh_btn)
        
        # Add header to main layout
        layout.addLayout(header_layout)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(line)
        
        # Create tab widget for wallet dashboards
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_wallet_tab)
        layout.addWidget(self.tabs)
    
    def setup_worker(self):
        """Setup the background worker for updates"""
        self.worker = SolanaWatchWorker(self.wallet_configs)
        self.worker.balance_updated.connect(self.update_wallet_balance)
        self.worker.transactions_updated.connect(self.update_wallet_transactions)
        self.worker.start()
    
    def create_dashboard(self, wallet_name):
        """Create a new dashboard for the specified wallet"""
        if wallet_name not in self.wallet_configs:
            return
            
        if wallet_name not in self.dashboards:
            config = self.wallet_configs[wallet_name]
            dashboard = WalletDashboard(wallet_name, config["pubkey"])
            self.dashboards[wallet_name] = dashboard
            
            # Add tab with wallet icon
            tab_index = self.tabs.addTab(dashboard, f"🔑 {wallet_name}")
            self.tabs.setCurrentIndex(tab_index)
            
            # Update the wallet selector
            self.wallet_selector.selector.setCurrentText(wallet_name)
    
    def change_wallet(self, wallet_name):
        """Handle wallet selection"""
        self.current_wallet = wallet_name
        self.create_dashboard(wallet_name)
        
        # Switch to the dashboard tab
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).endswith(wallet_name):
                self.tabs.setCurrentIndex(i)
                break
    
    def refresh_current_wallet(self):
        """Force refresh of the current wallet's data"""
        if self.current_wallet in self.dashboards:
            # Trigger an immediate update for the current wallet
            config = self.wallet_configs[self.current_wallet]
            self.worker.balance_updated.emit(
                self.current_wallet, 
                float(subprocess.check_output(["solana", "balance", "--url", "devnet", config["pubkey"]], text=True).strip()),
                self.worker._last_usd_price,
                self.worker._last_cad_price
            )
    
    def close_wallet_tab(self, index):
        """Close a wallet dashboard tab"""
        widget = self.tabs.widget(index)
        wallet_name = next((name for name, dash in self.dashboards.items() if dash == widget), None)
        if wallet_name:
            del self.dashboards[wallet_name]
        self.tabs.removeTab(index)
        
        # If we closed the current wallet's tab, switch to another one
        if self.tabs.count() > 0:
            self.tabs.setCurrentIndex(0)
            tab_text = self.tabs.tabText(0)
            new_wallet = tab_text.split("🔑 ")[-1]
            self.current_wallet = new_wallet
            self.wallet_selector.selector.setCurrentText(new_wallet)
    
    def update_wallet_balance(self, wallet_name, sol_balance, usd_price, cad_price):
        """Update balance for a specific wallet"""
        if wallet_name in self.dashboards:
            self.dashboards[wallet_name].update_balance(sol_balance, usd_price, cad_price)
    
    def update_wallet_transactions(self, wallet_name, transactions):
        """Update transactions for a specific wallet"""
        if wallet_name in self.dashboards:
            self.dashboards[wallet_name].update_transactions(transactions)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.worker.stop()
        self.worker.wait()
        event.accept()

def launch_watch_mode(wallet_configs: dict):
    """Launch the watch mode GUI with multiple wallet support"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    app.setPalette(palette)
    
    # Create and show the main window
    window = WatchModeWindow(wallet_configs)
    window.show()
    
    # Start the application
    sys.exit(app.exec())

if __name__ == "__main__":
    # Test with sample wallet configs
    sample_configs = {
        "MAIN": {"pubkey": "your_pubkey_here", "name": "MAIN"},
        "TRADE": {"pubkey": "another_pubkey_here", "name": "TRADE"}
    }
    launch_watch_mode(sample_configs) 
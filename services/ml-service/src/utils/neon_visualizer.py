"""
Simple tool to visualize trading data from Neon database.
Perfect for beginners to understand their data!
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime, timedelta

class NeonVisualizer:
    def __init__(self, connection_string):
        """
        Initialize the visualizer with your Neon database connection
        
        Example:
            connection_string = "postgresql://user:pass@ep-cool-night-123456.us-east-2.aws.neon.tech/neondb"
            visualizer = NeonVisualizer(connection_string)
        """
        self.engine = create_engine(connection_string)
        
    def view_recent_data(self, symbol, days=7):
        """
        View the most recent price data for a symbol
        
        Args:
            symbol: The trading pair (like 'BTC/USD')
            days: How many days of data to show
        """
        # Get price data
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM price_data
        WHERE symbol = %s
        AND timestamp >= NOW() - INTERVAL '%s days'
        ORDER BY timestamp
        """
        df = pd.read_sql(query, self.engine, params=[symbol, days])
        
        # Create the candlestick chart
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=(f'{symbol} Price', 'Volume'),
                           row_heights=[0.7, 0.3])

        # Add candlestick
        fig.add_trace(
            go.Candlestick(x=df['timestamp'],
                          open=df['open'],
                          high=df['high'],
                          low=df['low'],
                          close=df['close'],
                          name='Price'),
            row=1, col=1
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(x=df['timestamp'],
                  y=df['volume'],
                  name='Volume'),
            row=2, col=1
        )

        fig.update_layout(
            title=f'{symbol} - Last {days} Days',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False
        )

        fig.show()
        
    def view_indicators(self, symbol, days=7):
        """
        View all your trading indicators for a symbol
        
        Args:
            symbol: The trading pair (like 'BTC/USD')
            days: How many days of data to show
        """
        # Get indicator data
        query = """
        SELECT timestamp, rsi_14, wt_value, cci_20, adx_20, rsi_9
        FROM trading_signals
        WHERE symbol = %s
        AND timestamp >= NOW() - INTERVAL '%s days'
        ORDER BY timestamp
        """
        df = pd.read_sql(query, self.engine, params=[symbol, days])
        
        # Create subplots for each indicator
        fig = make_subplots(rows=5, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('RSI (14)', 'Wave Trend', 'CCI (20)', 
                                         'ADX (20)', 'RSI (9)'))

        # Add each indicator
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_14'], 
                                name='RSI (14)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['wt_value'], 
                                name='Wave Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cci_20'], 
                                name='CCI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx_20'], 
                                name='ADX'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_9'], 
                                name='RSI (9)'), row=5, col=1)

        fig.update_layout(
            title=f'{symbol} Indicators - Last {days} Days',
            height=1200  # Make it tall enough to see everything
        )

        fig.show()
        
    def view_model_performance(self, symbol, days=30):
        """
        View how well your model is doing
        
        Args:
            symbol: The trading pair (like 'BTC/USD')
            days: How many days of data to show
        """
        # Get prediction data
        query = """
        SELECT 
            DATE_TRUNC('hour', timestamp) as hour,
            AVG(prediction) as avg_prediction,
            AVG(confidence) as avg_confidence,
            AVG(CASE WHEN actual_return > 0 THEN 1 ELSE 0 END) as actual_up
        FROM model_predictions
        WHERE symbol = %s
        AND timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY DATE_TRUNC('hour', timestamp)
        ORDER BY hour
        """
        df = pd.read_sql(query, self.engine, params=[symbol, days])
        
        # Create visualization
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Model Predictions vs Reality',
                                         'Prediction Confidence'))

        # Add prediction vs reality
        fig.add_trace(go.Scatter(x=df['hour'], y=df['avg_prediction'],
                                name='Model Prediction',
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['hour'], y=df['actual_up'],
                                name='Actual Movement',
                                line=dict(color='green')), row=1, col=1)

        # Add confidence
        fig.add_trace(go.Scatter(x=df['hour'], y=df['avg_confidence'],
                                name='Confidence',
                                fill='tozeroy'), row=2, col=1)

        fig.update_layout(
            title=f'{symbol} Model Performance - Last {days} Days',
            yaxis_title='Probability Up',
            yaxis2_title='Confidence'
        )

        fig.show()
        
    def get_quick_stats(self, symbol):
        """
        Get some quick statistics about your trading data
        
        Args:
            symbol: The trading pair (like 'BTC/USD')
        """
        # Get basic stats
        queries = {
            'data_points': """
                SELECT COUNT(*) as count 
                FROM price_data 
                WHERE symbol = %s
            """,
            'date_range': """
                SELECT 
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM price_data 
                WHERE symbol = %s
            """,
            'predictions': """
                SELECT 
                    COUNT(*) as total,
                    AVG(CASE WHEN actual_return > 0 AND prediction > 0.5 THEN 1 
                        WHEN actual_return < 0 AND prediction < 0.5 THEN 1 
                        ELSE 0 END) as accuracy
                FROM model_predictions
                WHERE symbol = %s
            """
        }
        
        stats = {}
        for name, query in queries.items():
            result = pd.read_sql(query, self.engine, params=[symbol])
            stats[name] = result
            
        print(f"\n=== Quick Stats for {symbol} ===")
        print(f"Total price data points: {stats['data_points']['count'].iloc[0]:,}")
        print(f"Date range: {stats['date_range']['earliest'].iloc[0]} to {stats['date_range']['latest'].iloc[0]}")
        if stats['predictions']['total'].iloc[0] > 0:
            print(f"Model predictions: {stats['predictions']['total'].iloc[0]:,}")
            print(f"Model accuracy: {stats['predictions']['accuracy'].iloc[0]*100:.1f}%")
        else:
            print("No model predictions yet!")
            
    def plot_training_batch(self, df):
        """
        Visualize a batch of training data with all indicators
        
        Args:
            df: DataFrame with price and indicator data
        """
        fig = make_subplots(rows=3, cols=2,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price', 'Volume',
                                         'RSI (14)', 'Wave Trend',
                                         'CCI (20)', 'ADX (20)'))

        # Price
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='Price'), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                            name='Volume'), row=1, col=2)

        # Indicators
        if 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'],
                                   name='RSI (14)'), row=2, col=1)
        if 'wt_value' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['wt_value'],
                                   name='Wave Trend'), row=2, col=2)
        if 'cci_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['cci_20'],
                                   name='CCI (20)'), row=3, col=1)
        if 'adx_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['adx_20'],
                                   name='ADX (20)'), row=3, col=2)

        fig.update_layout(height=1000,
                         title='Training Data Visualization',
                         showlegend=True)
        fig.show() 
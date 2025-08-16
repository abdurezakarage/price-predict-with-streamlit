

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import talib
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
from typing import Dict, List, Optional, Tuple
# from quantitative_analysis import QuantitativeAnalyzer

# Configure page
st.set_page_config(
    page_title="📊 Quantitative Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

class StreamlitQuantDashboard:
    """
    Streamlit-based interactive dashboard for quantitative analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit dashboard."""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Setup page configuration and styling."""
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .metric-positive {
            color: #28a745;
        }
        .metric-negative {
            color: #dc3545;
        }
        .metric-neutral {
            color: #6c757d;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = None
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_stock_data(symbol: str, period: str) -> Optional[pd.DataFrame]:
       
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            st.error(f"Error loading data for {symbol}: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300)
    def calculate_technical_indicators(stock_data: pd.DataFrame) -> Dict:
       
        try:
            # Extract OHLCV data and ensure they are float64 for TA-Lib
            open_prices = stock_data['Open'].astype(np.float64).values
            high_prices = stock_data['High'].astype(np.float64).values
            low_prices = stock_data['Low'].astype(np.float64).values
            close_prices = stock_data['Close'].astype(np.float64).values
            volume = stock_data['Volume'].astype(np.float64).values
            
            indicators = {}
            
            # Moving Averages
            indicators['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
            indicators['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
            indicators['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
            indicators['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
            indicators['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # Momentum Indicators
            indicators['RSI'] = talib.RSI(close_prices, timeperiod=14)
            indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high_prices, low_prices, close_prices)
            indicators['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Trend Indicators
            indicators['MACD'], indicators['MACD_SIGNAL'], indicators['MACD_HIST'] = talib.MACD(close_prices)
            indicators['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Volatility Indicators
            indicators['BBANDS_UPPER'], indicators['BBANDS_MIDDLE'], indicators['BBANDS_LOWER'] = talib.BBANDS(close_prices)
            indicators['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Volume Indicators
            indicators['OBV'] = talib.OBV(close_prices, volume)
            indicators['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {e}")
            # Return empty indicators with default RSI to prevent downstream errors
            return {
                'RSI': np.array([50.0] * len(stock_data)),
                'SMA_20': np.array([np.nan] * len(stock_data)),
                'SMA_50': np.array([np.nan] * len(stock_data))
            }
    
    @staticmethod
    @st.cache_data(ttl=300)
    def calculate_financial_metrics(stock_data: pd.DataFrame) -> Dict:
     
        try:
            metrics = {}
            
            close_prices = stock_data['Close']
            returns = close_prices.pct_change().dropna()
            
            # Basic metrics
            metrics['current_price'] = float(close_prices.iloc[-1])
            metrics['price_change'] = float(close_prices.iloc[-1] - close_prices.iloc[-2]) if len(close_prices) > 1 else 0.0
            metrics['price_change_pct'] = float((metrics['price_change'] / close_prices.iloc[-2]) * 100) if len(close_prices) > 1 else 0.0
            
            # Volatility
            metrics['volatility'] = float(returns.std() * np.sqrt(252))
            
            # Risk metrics
            metrics['sharpe_ratio'] = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() != 0 else 0.0
            
            # Calculate max drawdown
            peak = close_prices.expanding(min_periods=1).max()
            drawdown = (close_prices - peak) / peak
            metrics['max_drawdown'] = float(drawdown.min())
            
            # Volume metrics
            metrics['avg_volume'] = float(stock_data['Volume'].mean())
            metrics['volume_trend'] = float(stock_data['Volume'].iloc[-5:].mean() / stock_data['Volume'].iloc[-20:].mean()) if len(stock_data) >= 20 else 1.0
            
            # Additional metrics
            metrics['daily_return_mean'] = float(returns.mean())
            metrics['daily_return_std'] = float(returns.std())
            metrics['var_95'] = float(np.percentile(returns, 5))
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating financial metrics: {e}")
            return {}
    
    def create_sidebar_controls(self) -> Tuple[str, str, bool]:
     
        st.sidebar.markdown('<div class="sidebar-header">📊 Analysis Controls</div>', unsafe_allow_html=True)
        
        # Stock symbol input
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            symbol = st.selectbox(
                "📈 Stock Symbol",
                options=default_symbols,
                index=0,
                help="Select a stock symbol for analysis"
            )
        
        with col2:
            custom_symbol = st.text_input(
                "Custom",
                placeholder="AAPL",
                help="Enter custom symbol"
            )
        
        # Use custom symbol if provided
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        # Time period selection
        period_options = {
            '1 Month': '1mo',
            '3 Months': '3mo',
            '6 Months': '6mo',
            '1 Year': '1y',
            '2 Years': '2y',
            '5 Years': '5y'
        }
        
        period_label = st.sidebar.selectbox(
            "⏱️ Time Period",
            options=list(period_options.keys()),
            index=3,  # Default to 1 Year
            help="Select analysis time period"
        )
        period = period_options[period_label]
        
        # Analysis controls
        st.sidebar.markdown("---")
        run_analysis = st.sidebar.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            help="Click to run comprehensive analysis"
        )
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox(
            "🔄 Auto Refresh (5 min)",
            value=False,
            help="Automatically refresh analysis every 5 minutes"
        )
        
        if auto_refresh:
            # Auto-refresh logic
            if st.session_state.last_update is None or \
               (datetime.now() - st.session_state.last_update).seconds > 300:
                run_analysis = True
        
        # Additional controls
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Display Options")
        
        show_volume = st.sidebar.checkbox("📊 Show Volume", value=True)
        show_indicators = st.sidebar.checkbox("📈 Show Indicators", value=True)
        chart_height = st.sidebar.slider("📏 Chart Height", 400, 800, 600)
        
        # Store display options in session state
        st.session_state.show_volume = show_volume
        st.session_state.show_indicators = show_indicators
        st.session_state.chart_height = chart_height
        
        return symbol, period, run_analysis
    
    def display_kpi_cards(self, metrics: Dict, indicators: Dict):
       
        if not metrics:
            st.warning("No metrics available. Please run analysis first.")
            return
        
        # Create KPI columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Current Price
        with col1:
            price = metrics.get('current_price', 0)
            price_change = metrics.get('price_change', 0)
            price_change_pct = metrics.get('price_change_pct', 0)
            
            delta_color = "normal"
            if price_change > 0:
                delta_color = "normal"
            elif price_change < 0:
                delta_color = "inverse"
            
            st.metric(
                label="💰 Current Price",
                value=f"${price:.2f}",
                delta=f"{price_change_pct:+.2f}%",
                delta_color=delta_color
            )
        
        # RSI
        with col2:
            rsi_values = indicators.get('RSI', np.array([50.0]))
            if isinstance(rsi_values, np.ndarray) and len(rsi_values) > 0:
                valid_rsi = rsi_values[~np.isnan(rsi_values)]
                current_rsi = valid_rsi[-1] if len(valid_rsi) > 0 else 50.0
            else:
                current_rsi = 50.0
            
            rsi_status = "🟢 Neutral"
            if current_rsi > 70:
                rsi_status = "🔴 Overbought"
            elif current_rsi < 30:
                rsi_status = "🟢 Oversold"
            
            st.metric(
                label="📊 RSI (14)",
                value=f"{current_rsi:.1f}",
                help=f"Relative Strength Index: {rsi_status}"
            )
        
        # Volatility
        with col3:
            volatility = metrics.get('volatility', 0)
            vol_status = "🟢 Low" if volatility < 0.2 else "🟡 Medium" if volatility < 0.4 else "🔴 High"
            
            st.metric(
                label="📈 Volatility",
                value=f"{volatility:.1%}",
                help=f"Annualized volatility: {vol_status}"
            )
        
        # Sharpe Ratio
        with col4:
            sharpe = metrics.get('sharpe_ratio', 0)
            sharpe_status = "🔴 Poor" if sharpe < 0 else "🟡 Fair" if sharpe < 1 else "🟢 Good" if sharpe < 2 else "🟢 Excellent"
            
            st.metric(
                label="⚖️ Sharpe Ratio",
                value=f"{sharpe:.2f}",
                help=f"Risk-adjusted return: {sharpe_status}"
            )
    
    def create_price_chart(self, stock_data: pd.DataFrame, indicators: Dict, symbol: str) -> go.Figure:
      
        # Determine number of rows based on display options
        row_count = 1
        if st.session_state.get('show_volume', True):
            row_count += 1
        
        # Create subplots
        fig = make_subplots(
            rows=row_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f'{symbol} Price Chart'] + (['Volume'] if st.session_state.get('show_volume', True) else []),
            row_heights=[0.7, 0.3] if row_count == 2 else [1.0]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Close Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add technical indicators if enabled
        if st.session_state.get('show_indicators', True):
            # Moving averages
            if 'SMA_20' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=indicators['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', dash='dash', width=1),
                        hovertemplate='<b>SMA 20</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=indicators['SMA_50'],
                        name='SMA 50',
                        line=dict(color='red', dash='dash', width=1),
                        hovertemplate='<b>SMA 50</b><br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'BBANDS_UPPER' in indicators and 'BBANDS_LOWER' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=indicators['BBANDS_UPPER'],
                        name='BB Upper',
                        line=dict(color='gray', dash='dot', width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=indicators['BBANDS_LOWER'],
                        name='BB Lower',
                        line=dict(color='gray', dash='dot', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        hovertemplate='<b>Bollinger Bands</b><br>Date: %{x}<br>Upper: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        if st.session_state.get('show_volume', True):
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(stock_data['Close'], stock_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6,
                    hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=2 if row_count == 2 else 1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=st.session_state.get('chart_height', 600),
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row_count, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if st.session_state.get('show_volume', True) and row_count == 2:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_indicators_chart(self, stock_data: pd.DataFrame, indicators: Dict, symbol: str) -> go.Figure:
        """
        Create technical indicators chart.
        
        Args:
            stock_data (pd.DataFrame): Stock data
            indicators (Dict): Technical indicators
            symbol (str): Stock symbol
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('RSI (14)', 'MACD', 'Stochastic Oscillator'),
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # RSI
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=indicators['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>RSI</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, 
                         annotation_text="Overbought (70)", annotation_position="right")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1,
                         annotation_text="Oversold (30)", annotation_position="right")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
        
        # MACD
        if 'MACD' in indicators and 'MACD_SIGNAL' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=indicators['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=indicators['MACD_SIGNAL'],
                    name='Signal',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            if 'MACD_HIST' in indicators:
                colors = ['green' if x >= 0 else 'red' for x in indicators['MACD_HIST']]
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=indicators['MACD_HIST'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=2, col=1
                )
        
        # Stochastic
        if 'STOCH_K' in indicators and 'STOCH_D' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=indicators['STOCH_K'],
                    name='%K',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=indicators['STOCH_D'],
                    name='%D',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
            
            # Add stochastic levels
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            height=600,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update y-axis ranges
        fig.update_yaxes(range=[0, 100], row=1, col=1)  # RSI
        fig.update_yaxes(range=[0, 100], row=3, col=1)  # Stochastic
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def display_financial_metrics(self, metrics: Dict):
        """
        Display detailed financial metrics.
        
        Args:
            metrics (Dict): Financial metrics
        """
        if not metrics:
            st.warning("No financial metrics available.")
            return
        
        st.subheader("💰 Financial Metrics")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Price Statistics")
            st.metric("Current Price", f"${metrics.get('current_price', 0):.2f}")
            st.metric("Price Change", f"${metrics.get('price_change', 0):.2f}")
            st.metric("Price Change %", f"{metrics.get('price_change_pct', 0):.2f}%")
        
        with col2:
            st.markdown("#### 📈 Risk Metrics")
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        
        with col3:
            st.markdown("#### 📊 Return Metrics")
            st.metric("Daily Return Mean", f"{metrics.get('daily_return_mean', 0):.4f}")
            st.metric("Daily Return Std", f"{metrics.get('daily_return_std', 0):.4f}")
            st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2%}")
        
        # Volume metrics
        st.markdown("#### 📊 Volume Analysis")
        vol_col1, vol_col2 = st.columns(2)
        
        with vol_col1:
            st.metric("Average Volume", f"{metrics.get('avg_volume', 0):,.0f}")
        
        with vol_col2:
            volume_trend = metrics.get('volume_trend', 1)
            trend_text = "📈 Increasing" if volume_trend > 1 else "📉 Decreasing" if volume_trend < 1 else "➡️ Stable"
            st.metric("Volume Trend", f"{volume_trend:.2f}", help=trend_text)
    
    def run_analysis(self, symbol: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict, Dict]:
        """
        Run comprehensive analysis for the given symbol and period.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period
            
        Returns:
            Tuple: Stock data, technical indicators, financial metrics
        """
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load stock data
            status_text.text("📥 Loading stock data...")
            progress_bar.progress(20)
            
            stock_data = StreamlitQuantDashboard.load_stock_data(symbol, period)
            if stock_data is None:
                st.error(f"Failed to load data for {symbol}")
                return None, {}, {}
            
            # Calculate technical indicators
            status_text.text("📊 Calculating technical indicators...")
            progress_bar.progress(60)
            
            indicators = StreamlitQuantDashboard.calculate_technical_indicators(stock_data)
            
            # Calculate financial metrics
            status_text.text("💰 Calculating financial metrics...")
            progress_bar.progress(80)
            
            metrics = StreamlitQuantDashboard.calculate_financial_metrics(stock_data)
            
            # Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Update session state
            st.session_state.analysis_data = {
                'stock_data': stock_data,
                'indicators': indicators,
                'metrics': metrics,
                'symbol': symbol,
                'period': period
            }
            st.session_state.last_update = datetime.now()
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return stock_data, indicators, metrics
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            progress_bar.empty()
            status_text.empty()
            return None, {}, {}
    
    def run_dashboard(self):
        """Run the main dashboard application."""
        # Header
        st.markdown('<h1 class="main-header">📊 Quantitative Analysis Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Real-time Technical Analysis using TA-Lib and PyNance")
        
        # Sidebar controls
        symbol, period, run_analysis = self.create_sidebar_controls()
        
        # Main content area
        if run_analysis or st.session_state.analysis_data:
            if run_analysis:
                stock_data, indicators, metrics = self.run_analysis(symbol, period)
            else:
                # Use cached data
                analysis_data = st.session_state.analysis_data
                stock_data = analysis_data.get('stock_data')
                indicators = analysis_data.get('indicators', {})
                metrics = analysis_data.get('metrics', {})
                symbol = analysis_data.get('symbol', symbol)
            
            if stock_data is not None:
                # Display last update time
                if st.session_state.last_update:
                    st.sidebar.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
                
                # KPI Cards
                st.markdown("### 📊 Key Performance Indicators")
                self.display_kpi_cards(metrics, indicators)
                
                # Analysis tabs
                tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "🔍 Technical Indicators", "💰 Financial Metrics"])
                
                with tab1:
                    st.plotly_chart(
                        self.create_price_chart(stock_data, indicators, symbol),
                        use_container_width=True
                    )
                    
                    # Price statistics
                    if len(stock_data) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Data Points", f"{len(stock_data):,}")
                        with col2:
                            st.metric("Date Range", f"{(stock_data.index.max() - stock_data.index.min()).days} days")
                        with col3:
                            st.metric("Avg Daily Volume", f"{stock_data['Volume'].mean():,.0f}")
                        with col4:
                            price_range = ((stock_data['High'].max() - stock_data['Low'].min()) / stock_data['Close'].mean()) * 100
                            st.metric("Price Range %", f"{price_range:.1f}%")
                
                with tab2:
                    st.plotly_chart(
                        self.create_indicators_chart(stock_data, indicators, symbol),
                        use_container_width=True
                    )
                    
                    # Indicator summary
                    if indicators:
                        st.markdown("#### 📊 Current Indicator Values")
                        ind_col1, ind_col2, ind_col3 = st.columns(3)
                        
                        with ind_col1:
                            if 'RSI' in indicators and isinstance(indicators['RSI'], np.ndarray):
                                valid_rsi = indicators['RSI'][~np.isnan(indicators['RSI'])]
                                rsi_val = valid_rsi[-1] if len(valid_rsi) > 0 else 0
                                st.metric("RSI", f"{rsi_val:.1f}")
                            
                            if 'ADX' in indicators and isinstance(indicators['ADX'], np.ndarray):
                                valid_adx = indicators['ADX'][~np.isnan(indicators['ADX'])]
                                adx_val = valid_adx[-1] if len(valid_adx) > 0 else 0
                                st.metric("ADX", f"{adx_val:.1f}")
                        
                        with ind_col2:
                            if 'MACD' in indicators and isinstance(indicators['MACD'], np.ndarray):
                                valid_macd = indicators['MACD'][~np.isnan(indicators['MACD'])]
                                macd_val = valid_macd[-1] if len(valid_macd) > 0 else 0
                                st.metric("MACD", f"{macd_val:.4f}")
                            
                            if 'CCI' in indicators and isinstance(indicators['CCI'], np.ndarray):
                                valid_cci = indicators['CCI'][~np.isnan(indicators['CCI'])]
                                cci_val = valid_cci[-1] if len(valid_cci) > 0 else 0
                                st.metric("CCI", f"{cci_val:.1f}")
                        
                        with ind_col3:
                            if 'ATR' in indicators and isinstance(indicators['ATR'], np.ndarray):
                                valid_atr = indicators['ATR'][~np.isnan(indicators['ATR'])]
                                atr_val = valid_atr[-1] if len(valid_atr) > 0 else 0
                                st.metric("ATR", f"{atr_val:.2f}")
                            
                            if 'WILLR' in indicators and isinstance(indicators['WILLR'], np.ndarray):
                                valid_willr = indicators['WILLR'][~np.isnan(indicators['WILLR'])]
                                willr_val = valid_willr[-1] if len(valid_willr) > 0 else 0
                                st.metric("Williams %R", f"{willr_val:.1f}")
                
                with tab3:
                    self.display_financial_metrics(metrics)
                    
                    # Risk analysis chart
                    if metrics:
                        st.markdown("#### 📊 Risk Analysis Visualization")
                        
                        # Create risk metrics chart
                        risk_metrics = ['volatility', 'sharpe_ratio', 'max_drawdown', 'var_95']
                        risk_values = [abs(metrics.get(m, 0)) for m in risk_metrics]  # Use absolute values for visualization
                        risk_labels = ['Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)']
                        
                        fig_risk = go.Figure(data=[
                            go.Bar(x=risk_labels, y=risk_values, marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                        ])
                        
                        fig_risk.update_layout(
                            title="Risk Metrics Overview",
                            xaxis_title="Metrics",
                            yaxis_title="Values",
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_risk, use_container_width=True)
        
        else:
            # Welcome screen
            st.markdown("""
            ### 🚀 Welcome to the Quantitative Analysis Dashboard
            
            This dashboard provides real-time technical analysis using industry-standard indicators and financial metrics.
            
            **Features:**
            - 📊 **Real-time Data**: Live stock data from Yahoo Finance
            - 📈 **Technical Indicators**: 20+ indicators using TA-Lib
            - 💰 **Financial Metrics**: Comprehensive risk and return analysis
            - 🔄 **Auto Refresh**: Optional 5-minute auto-refresh
            - 📱 **Interactive Charts**: Zoom, pan, and hover for details
            
            **To get started:**
            1. Select a stock symbol from the sidebar
            2. Choose your preferred time period
            3. Click "🚀 Run Analysis"
            
            """)
            
            # Quick access buttons for popular stocks
            st.markdown("#### 🔥 Quick Analysis")
            quick_cols = st.columns(5)
            popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
            
            for i, stock in enumerate(popular_stocks):
                with quick_cols[i]:
                    if st.button(f"📊 {stock}", use_container_width=True):
                        # Set symbol in session state and trigger analysis
                        st.session_state.quick_symbol = stock
                        st.rerun()


def main():
    """Main function to run the Streamlit dashboard."""
    dashboard = StreamlitQuantDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()

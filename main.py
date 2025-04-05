import time 
import hmac 
import hashlib 
import requests 
import json 
import logging 
import threading 
import numpy as np 
import pandas as pd 
from datetime import datetime 

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s') 

# --- CONFIGURATION --- 
API_BASE_URL = "https://mock-api.roostoo.com" 
API_KEY = "9obwwLuXTFuunAvjMKQ94YWBaWvSwwUK1PPdtRjWVoUYGJHSDiiXNqGx82XbeSY8"
SECRET_KEY = "vEb7XYDIns2h7CUt0k7IHNdnrBvRgyBPl90yZ2UnJZOIVbkWjteNsxUYkADL8B7v"
RISK_FREE_RATE = 0.001  # 0.1% risk-free rate 
 
# For simulation/demo purposes 
TRADE_PAIR = "BTC/USD" 
FETCH_INTERVAL = 5  # seconds between market data fetches during recording 
TRADING_INTERVAL = 10  # seconds between trading decisions during simulation 
 
# --- API CLIENT --- 
class RoostooAPIClient: 
    def _init_(self, api_key, secret_key, base_url=API_BASE_URL): 
        self.api_key = api_key 
        self.secret_key = secret_key.encode()  # must be bytes for hmac 
        self.base_url = base_url 
 
    def _get_timestamp(self): 
        return str(int(time.time() * 1000)) 
 
    def _sign(self, params: dict): 
        sorted_items = sorted(params.items()) 
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items]) 
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest() 
        return signature, query_string 
 
    def _headers(self, params: dict, is_signed=False): 
        headers = { 
            "Content-Type": "application/x-www-form-urlencoded" 
        } 
        if is_signed: 
            signature, _ = self._sign(params) 
            headers["RST-API-KEY"] = self.api_key 
            headers["MSG-SIGNATURE"] = signature 
        return headers 
 
    def _handle_response(self, response): 
        if response.status_code != 200: 
            logging.error(f"HTTP Error: {response.status_code} {response.text}") 
            return None 
        try: 
            data = response.json() 
        except Exception as e: 
            logging.error(f"JSON decode error: {e}") 
            return None 
        return data 
 
    def get_ticker(self, pair=None): 
        url = f"{self.base_url}/v3/ticker" 
        params = { 
            "timestamp": self._get_timestamp() 
        } 
        if pair: 
            params["pair"] = pair 
        headers = self._headers(params, is_signed=False) 
        response = requests.get(url, params=params, headers=headers) 
        return self._handle_response(response) 
 
# --- DATA RECORDER --- 
class DataRecorder: 
    """ 
    Records market data (timestamp and price) for a specified duration. 
    """ 
    def _init_(self, api_client, trade_pair, fetch_interval=10): 
        self.api_client = api_client 
        self.trade_pair = trade_pair 
        self.fetch_interval = fetch_interval 
        self.data = []  # List of dictionaries: {"timestamp": ..., "price": ...} 
 
    def record(self, duration_sec): 
        logging.info(f"Starting data recording for {duration_sec} seconds...") 
        start_time = time.time() 
        while time.time() - start_time < duration_sec: 
            ticker_data = self.api_client.get_ticker(pair=self.trade_pair) 
            if ticker_data and ticker_data.get("Success"): 
                try: 
                    price = float(ticker_data["Data"][self.trade_pair]["LastPrice"]) 
                    record_time = datetime.now() 
                    self.data.append({"timestamp": record_time, "price": price}) 
                    logging.info(f"Recorded price: {price} at {record_time.strftime('%Y-%m-%d %H:%M:%S')}") 
                except Exception as e: 
                    logging.error(f"Error processing ticker data: {e}") 
            else: 
                logging.error("Failed to fetch ticker data during recording.") 
            time.sleep(self.fetch_interval) 
        logging.info("Data recording completed.") 
 
    def get_dataframe(self): 
        return pd.DataFrame(self.data) 

# --- SIMPLE THRESHOLD STRATEGY --- 
class SimpleThresholdStrategy:
    """
    Simple Threshold Strategy:
    - Buy when price is below the threshold
    - Sell when price is above the threshold
    - No profit targets, just immediate price comparison with threshold
    """
    def _init_(self, lookback_period=5):
        # Strategy parameters
        self.lookback_period = lookback_period
        
        # Price history
        self.prices = []
        
        # Threshold tracking
        self.threshold = None
        self.position_status = "CASH"  # CASH, HOLDING
        
        # Logging and performance tracking
        self.trade_count = 0
        self.profitable_trades = 0
        self.buy_price = None
        
    def update_price(self, price):
        self.prices.append(price)
        # Keep reasonable history for calculations
        if len(self.prices) > self.lookback_period * 3:
            self.prices.pop(0)
        
        # Initialize threshold if not set
        if self.threshold is None and len(self.prices) >= self.lookback_period:
            self._calculate_threshold()
            logging.info(f"Initial Threshold set to: {self.threshold:.6f}")
        
        # Update threshold dynamically with each new price
        elif len(self.prices) >= self.lookback_period:
            self._update_threshold()
            
    def _calculate_threshold(self):
        """Calculate initial threshold based on recent price history"""
        recent_prices = self.prices[-self.lookback_period:]
        avg_price = np.mean(recent_prices)
        self.threshold = avg_price
        
    def _update_threshold(self):
        """Dynamically update the threshold based on recent prices"""
        recent_prices = self.prices[-self.lookback_period:]
        self.threshold = np.mean(recent_prices)
        logging.info(f"Updated Threshold: {self.threshold:.6f}")
    
    def generate_signal(self):
        """
        Generate trading signal based solely on price relative to threshold:
        - If price is below threshold and we're not holding, BUY
        - If price is above threshold and we're holding, SELL
        - Otherwise HOLD
        """
        if len(self.prices) < self.lookback_period or self.threshold is None:
            return "HOLD"  # Not enough data
            
        current_price = self.prices[-1]
        signal = "HOLD"
        
        # Log current price vs threshold and status
        logging.info(f"Current Price: {current_price:.6f} | Threshold: {self.threshold:.6f} | Status: {self.position_status}")
        
        # Buying logic - Price is below threshold, opportunity to buy
        if current_price < self.threshold and self.position_status == "CASH":
            signal = "BUY"
            self.position_status = "HOLDING"
            self.buy_price = current_price
            logging.info(f"BUY Signal: Price {current_price:.6f} below Threshold {self.threshold:.6f}")
            
        # Selling logic - Price is above threshold, time to sell
        elif current_price > self.threshold and self.position_status == "HOLDING":
            signal = "SELL"
            self.position_status = "CASH"
            
            # Track trade performance
            self.trade_count += 1
            if self.buy_price is not None and current_price > self.buy_price:
                self.profitable_trades += 1
                profit_pct = ((current_price / self.buy_price) - 1) * 100
                logging.info(f"SELL Signal (PROFIT): {profit_pct:.2f}% gain from {self.buy_price:.6f} to {current_price:.6f}")
            else:
                loss_pct = ((current_price / self.buy_price) - 1) * 100 if self.buy_price else 0
                logging.info(f"SELL Signal (LOSS): {loss_pct:.2f}% loss from {self.buy_price:.6f} to {current_price:.6f}")
            
            self.buy_price = None
            
        return signal

# --- RISK MANAGEMENT --- 
class RiskManager: 
    def _init_(self): 
        self.portfolio_values = []  # Recorded portfolio value over time 
 
    def update_portfolio(self, value): 
        self.portfolio_values.append(value) 
 
    def calculate_sharpe_ratio(self): 
        if len(self.portfolio_values) < 2: 
            return 0 
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1] 
        excess_returns = returns - RISK_FREE_RATE 
        mean_return = np.mean(excess_returns) 
        std_return = np.std(excess_returns) 
        if std_return == 0: 
            return 0 
        sharpe_ratio = mean_return / std_return 
        return sharpe_ratio 
 
# --- SIMULATION BOT --- 
class SimulationBot: 
    """ 
    Simulates trading using recorded historical data. 
    """ 
    def _init_(self, recorded_df, strategy, risk_manager, initial_cash=100000): 
        self.data = recorded_df.sort_values(by="timestamp").reset_index(drop=True) 
        self.strategy = strategy 
        self.risk_manager = risk_manager 
        self.cash = initial_cash 
        self.holdings = 0.0  # In BTC 
        self.trade_log = [] 
        self.entry_prices = []  # Track entry prices for each position
 
    def update_portfolio_value(self, price): 
        portfolio_value = self.cash + self.holdings * price 
        self.risk_manager.update_portfolio(portfolio_value) 
        return portfolio_value 
 
    def simulate_trade(self, signal, price): 
        # For simulation, define a fixed trade amount (0.01 BTC) 
        trade_amount = 0.01 
        if signal == "BUY" and self.cash >= trade_amount * price: 
            # Simulate buying 
            self.holdings += trade_amount 
            purchase_amount = trade_amount * price
            self.cash -= purchase_amount
            self.entry_prices.append(price)  # Track entry price
            self.trade_log.append({
                "timestamp": datetime.now(), 
                "action": "BUY", 
                "price": price, 
                "amount": trade_amount,
                "cash_spent": purchase_amount,
                "remaining_cash": self.cash
            }) 
            logging.info(f"Simulated BUY: {trade_amount} BTC at {price}, Spent: {purchase_amount:.6f}, Cash: {self.cash:.2f}") 
        elif signal == "SELL" and self.holdings >= trade_amount: 
            # Simulate selling 
            self.holdings -= trade_amount 
            sale_amount = trade_amount * price
            self.cash += sale_amount
            
            # Calculate profit/loss if we have entry prices
            if self.entry_prices:
                entry_price = self.entry_prices.pop(0)  # FIFO method
                profit = sale_amount - (trade_amount * entry_price)
                profit_pct = ((price / entry_price) - 1) * 100
                logging.info(f"Trade P&L: {profit:.6f} ({profit_pct:.2f}%)")
            
            self.trade_log.append({
                "timestamp": datetime.now(), 
                "action": "SELL", 
                "price": price, 
                "amount": trade_amount,
                "cash_received": sale_amount,
                "cash_balance": self.cash
            }) 
            logging.info(f"Simulated SELL: {trade_amount} BTC at {price}, Received: {sale_amount:.6f}, Cash: {self.cash:.2f}") 
        else:
            logging.info("No simulated trade executed (either HOLD signal or insufficient funds/holdings).")
            
    def run_simulation(self): 
        logging.info("Starting simulation on recorded data...") 
        for index, row in self.data.iterrows(): 
            price = row["price"] 
            current_time = row["timestamp"] 
            
            # Update strategy with current price 
            self.strategy.update_price(price) 
            signal = self.strategy.generate_signal() 
            logging.info(f"Time: {current_time} | Price: {price} | Signal: {signal}") 
            
            # Simulate trade decision at each data point 
            if signal in ["BUY", "SELL"]: 
                self.simulate_trade(signal, price) 
                
            portfolio_value = self.update_portfolio_value(price) 
            logging.info(f"Portfolio Value at {current_time}: {portfolio_value:.2f}") 
            
            # For simulation pacing (optional), you can add a short delay here 
            time.sleep(0.5) 
            
        # Final performance metrics 
        if not self.data.empty:
            final_price = self.data.iloc[-1]["price"] 
            final_value = self.cash + self.holdings * final_price
            sharpe_ratio = self.risk_manager.calculate_sharpe_ratio() 
            logging.info(f"Simulation Completed. Final Portfolio Value: {final_value:.2f}") 
            logging.info(f"Cash: {self.cash:.2f} | Holdings: {self.holdings} BTC at {final_price} each")
            logging.info(f"Calculated Sharpe Ratio: {sharpe_ratio:.4f}") 
            return final_value, sharpe_ratio, self.trade_log
        else:
            logging.error("No data available for simulation")
            return 0, 0, []

# --- MAIN EXECUTION --- 
def main(): 
    # Initialize API client 
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY) 
    
    # --- Phase 1: Data Recording --- 
    record_duration = 120  # Extended to 120 seconds for more data points
    recorder = DataRecorder(api_client, TRADE_PAIR, fetch_interval=15)  # Increased interval to avoid rate limits
    recorder.record(record_duration) 
    recorded_df = recorder.get_dataframe() 
    
    if recorded_df.empty: 
        logging.error("No data recorded. Exiting simulation.") 
        return 
        
    logging.info("Recorded Data:") 
    logging.info(recorded_df) 
    
    # --- Phase 2: Simulation --- 
    # Create strategy and risk manager instances 
    # Using the new SimpleThresholdStrategy
    strategy = SimpleThresholdStrategy(lookback_period=3)
    risk_manager = RiskManager() 
    
    # Create simulation bot with recorded data 
    simulation_bot = SimulationBot(recorded_df, strategy, risk_manager, initial_cash=100000) 
    final_value, sharpe_ratio, trade_log = simulation_bot.run_simulation() 
    
    # Output simulation summary 
    logging.info("Simulation Summary:") 
    logging.info(f"Final Portfolio Value: {final_value:.2f}") 
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}") 
    logging.info("Trade Log:") 
    for trade in trade_log: 
        logging.info(trade) 
    
    if strategy.trade_count > 0:
        win_rate = (strategy.profitable_trades / strategy.trade_count) * 100
        logging.info(f"Win Rate: {win_rate:.2f}% ({strategy.profitable_trades}/{strategy.trade_count})")

if __name__ == "_main_": 
    main()
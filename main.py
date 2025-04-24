import time
import hmac
import hashlib
import requests
import json
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import yfinance as yf
import talib
import config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = config.API_KEY
SECRET_KEY = config.SECRET_KEY
RISK_FREE_RATE = 0.001  # 0.1% risk-free rate
FETCH_INTERVAL = 10  # seconds between market data fetches
TRADING_INTERVAL = 20  # seconds between trading decisions
POSITION_SIZE_PCT = 0.05  # Risk 5% of portfolio per coin
BUYING_COMMISSION = 0.001  # 0.1% commission on buys
SELLING_COMMISSION = 0.001  # 0.1% commission on sells
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.06  # 6% take profit
MAX_COINS = 5  # Maximum number of coins to trade simultaneously
MIN_SCORE_THRESHOLD = 0.1  # Minimum score to consider a coin for trading
YF_HISTORICAL_PERIOD = "1y"  # Fetch 1 year of historical data
YF_INTERVAL = "1d"  # Daily data

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BBANDS_PERIOD = 20
BBANDS_NBDEV = 2
STOCH_K = 14
STOCH_D = 3
STOCH_SLOWD = 3

# Validate TAKE_PROFIT_PCT and STOP_LOSS_PCT
MIN_TAKE_PROFIT = (1 + BUYING_COMMISSION) / (1 - SELLING_COMMISSION) - 1
MIN_STOP_LOSS = 1 - (1 - BUYING_COMMISSION) / (1 + SELLING_COMMISSION)
if TAKE_PROFIT_PCT <= MIN_TAKE_PROFIT:
    logging.warning(f"TAKE_PROFIT_PCT ({TAKE_PROFIT_PCT}) is below minimum ({MIN_TAKE_PROFIT}). Adjusting to {MIN_TAKE_PROFIT + 0.001}.")
    TAKE_PROFIT_PCT = MIN_TAKE_PROFIT + 0.001
if STOP_LOSS_PCT <= MIN_STOP_LOSS:
    logging.warning(f"STOP_LOSS_PCT ({STOP_LOSS_PCT}) is below minimum ({MIN_STOP_LOSS}). Adjusting to {MIN_STOP_LOSS + 0.001}.")
    STOP_LOSS_PCT = MIN_STOP_LOSS + 0.001

# --- API CLIENT ---
class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()
        self.base_url = base_url

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, params: dict):
        sorted_items = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items])
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest()
        return signature, query_string

    def _headers(self, params: dict, is_signed=False):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
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
            logging.error(f"JSON decode error: {e}, Response: {response.text}")
            return None
        return data

    def list_of_coins(self):
        response = requests.get(self.base_url + "/v3/exchangeInfo")
        try:
            return [*self._handle_response(response)["TradePairs"]]
        except Exception as e:
            logging.error(f"Error in list_of_coins: {e}")
            return ["BTC/USD", "ETH/USD"]  # Default fallback

    def get_ticker(self, pair=None):
        try:
            url = f"{self.base_url}/v3/ticker"
            params = {"timestamp": self._get_timestamp()}
            if pair:
                params["pair"] = pair
            headers = self._headers(params, is_signed=False)
            response = requests.get(url, params=params, headers=headers)
            return self._handle_response(response)
        except Exception as e:
            logging.error(f"Error in get_ticker: {e}")
            return None

    def get_balance(self):
        try:
            params = {"timestamp": self._get_timestamp()}
            response = requests.get(
                f"{self.base_url}/v3/balance",
                params=params,
                headers=self._headers(params, is_signed=True))
            data = self._handle_response(response)
            return data if data else {"SpotWallet": {"USD": {"Free": 10000}}}
        except Exception as e:
            logging.error(f"Error in get_balance: {e}")
            return {"SpotWallet": {"USD": {"Free": 10000}}}

    def place_order(self, coin, side, qty, price=None):
        try:
            params = {
                "timestamp": self._get_timestamp(),
                "pair": f"{coin}/USD",
                "side": side,
                "quantity": qty,
                "type": "MARKET" if not price else "LIMIT",
            }
            if price:
                params["price"] = price
            response = requests.post(
                f"{self.base_url}/v3/place_order",
                data=params,
                headers=self._headers(params, is_signed=True))
            return self._handle_response(response)
        except Exception as e:
            logging.error(f"Error in place_order: {e}")
            return None

    def cancel_order(self, pair):
        try:
            params = {"timestamp": self._get_timestamp(), "pair": pair}
            response = requests.post(
                f"{self.base_url}/v3/cancel_order",
                data=params,
                headers=self._headers(params, is_signed=True))
            return self._handle_response(response)
        except Exception as e:
            logging.error(f"Error in cancel_order: {e}")
            return None

# --- COIN SELECTION STRATEGY ---
class CoinSelector:
    def __init__(self, api_client):
        self.api_client = api_client
        self.price_history = {}
        self.trade_history = {}
        self.historical_data = {}
        self.ticker_mapping = {
            "BTC/USD": "BTC-USD",
            "ETH/USD": "ETH-USD",
            "LTC/USD": "LTC-USD",
            # Add more mappings as needed
        }

    def update_trade_history(self, coin, trade):
        if coin not in self.trade_history:
            self.trade_history[coin] = []
        self.trade_history[coin].append(trade)
        if len(self.trade_history[coin]) > 50:
            self.trade_history[coin].pop(0)

    def fetch_historical_data(self, ticker):
        try:
            if ticker in self.historical_data:
                return self.historical_data[ticker]
            asset = yf.Ticker(ticker)
            hist = asset.history(period=YF_HISTORICAL_PERIOD, interval=YF_INTERVAL)
            if hist.empty:
                logging.warning(f"No historical data for {ticker}")
                return None
            self.historical_data[ticker] = hist
            return hist
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def calculate_historical_metrics(self, hist):
        if hist is None or len(hist) < 50:
            return 0, 0, 0
        closes = hist["Close"]
        returns = closes.pct_change().dropna()
        annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100  # 252 trading days
        volatility = returns.std() * np.sqrt(252) * 100
        ma50 = closes.rolling(window=50).mean().iloc[-1]
        ma200 = closes.rolling(window=200).mean().iloc[-1]
        ma_signal = 1 if ma50 > ma200 else 0  # Golden cross signal
        return annualized_return, volatility, ma_signal

    def calculate_coin_score(self, coin, pair):
        score = 0
        # Short-term metrics
        if pair in self.price_history and len(self.price_history[pair]) >= 10:
            prices = np.array(self.price_history[pair])
            short_term_volatility = np.std(prices) / np.mean(prices)
            score += short_term_volatility * 50
        else:
            score += 0.1

        if coin in self.trade_history and len(self.trade_history[coin]) > 0:
            profits = [t["profit_pct"] for t in self.trade_history[coin] if "profit_pct" in t]
            avg_profit = np.mean(profits) if profits else 0
            win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0.5
            score += avg_profit * 10 + win_rate * 20
        else:
            score += 0.2

        # Long-term metrics
        ticker = self.ticker_mapping.get(pair, None)
        if ticker:
            hist = self.fetch_historical_data(ticker)
            annualized_return, long_term_volatility, ma_signal = self.calculate_historical_metrics(hist)
            score += annualized_return * 0.5  # Reward high returns
            score += long_term_volatility * 0.2  # Moderate weight for volatility
            score += ma_signal * 10  # Boost for bullish trend
        else:
            # logging.warning(f"No yfinance ticker mapping for {pair}")
            score += 0.1

        return max(score, MIN_SCORE_THRESHOLD)

    def select_coins(self, max_coins=MAX_COINS):
        available_pairs = self.api_client.list_of_coins()
        if not available_pairs:
            return [("BTC", "BTC/USD")]

        for pair in available_pairs:
            ticker_data = self.api_client.get_ticker(pair=pair)
            if ticker_data and ticker_data.get("Success"):
                price = float(ticker_data["Data"][pair]["LastPrice"])
                if pair not in self.price_history:
                    self.price_history[pair] = []
                self.price_history[pair].append(price)
                if len(self.price_history[pair]) > 20:
                    self.price_history[pair].pop(0)

        coin_scores = []
        for pair in available_pairs:
            coin = pair.split("/")[0]
            score = self.calculate_coin_score(coin, pair)
            coin_scores.append((coin, pair, score))

        coin_scores.sort(key=lambda x: x[2], reverse=True)
        selected = [(c, p) for c, p, s in coin_scores[:max_coins]]
        
        if len(selected) < max_coins and len(coin_scores) > len(selected):
            selected.extend([(c, p) for c, p, s in coin_scores[len(selected):max_coins]])

        if not selected:
            selected = [(coin_scores[0][0], coin_scores[0][1])]

        logging.info(f"Selected coins: {[c for c, p in selected]}, Scores: {[s for _, _, s in coin_scores[:len(selected)]]}")
        return selected

# --- TRADING STRATEGY ---
class AutonomousStrategy:
    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period
        self.strategies = {}
        self.price_data = {}
        self.strategy_performance = {}
        self.available_strategies = [
            "mean_reversion",
            "macd_crossover",
            "rsi_strategy",
            "bollinger_bands",
            "combined"
        ]

    def get_strategy_state(self, coin):
        if coin not in self.strategies:
            self.strategies[coin] = {
                "no": 0,
                "price_mean": 0,
                "position_status": "CASH",
                "buy_price": None,
                "stop_loss_price": None,
                "take_profit_price": None,
                "active_strategy": "mean_reversion"  # Default strategy
            }
        if coin not in self.price_data:
            self.price_data[coin] = []
        if coin not in self.strategy_performance:
            self.strategy_performance[coin] = {strat: 0 for strat in self.available_strategies}
        return self.strategies[coin]

    def update_price_data(self, coin, price):
        self.price_data[coin].append(price)
        if len(self.price_data[coin]) > max(RSI_PERIOD, MACD_SLOW, BBANDS_PERIOD, STOCH_K) + 10:
            self.price_data[coin].pop(0)

    def calculate_indicators(self, coin):
        prices = np.array(self.price_data[coin])
        if len(prices) < max(RSI_PERIOD, MACD_SLOW, BBANDS_PERIOD, STOCH_K) + 1:
            return None

        # RSI
        rsi = talib.RSI(prices, timeperiod=RSI_PERIOD)[-1]

        # MACD
        macd, signal, _ = talib.MACD(prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        macd, signal = macd[-1], signal[-1]

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(prices, timeperiod=BBANDS_PERIOD, nbdevup=BBANDS_NBDEV, nbdevdn=BBANDS_NBDEV)
        upper, middle, lower = upper[-1], middle[-1], lower[-1]

        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(prices, prices, prices, fastk_period=STOCH_K, slowk_period=STOCH_D, slowd_period=STOCH_SLOWD)
        slowk, slowd = slowk[-1], slowd[-1]

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "stoch_k": slowk,
            "stoch_d": slowd
        }

    def update_price_mean(self, coin, price):
        state = self.get_strategy_state(coin)
        self.update_price_data(coin, price)
        if state["price_mean"] == 0:
            state["price_mean"] = price
        else:
            state["price_mean"] = (state["price_mean"] * state["no"] + price) / (state["no"] + 1)
        state["no"] += 1

    def set_risk_levels(self, coin, entry_price):
        state = self.get_strategy_state(coin)
        state["buy_price"] = entry_price
        state["stop_loss_price"] = entry_price * (1 - STOP_LOSS_PCT)
        state["take_profit_price"] = entry_price * (1 + TAKE_PROFIT_PCT)
        logging.info(f"{coin} - Set Stop Loss: {state['stop_loss_price']:.6f}, Take Profit: {state['take_profit_price']:.6f}")

    def select_best_strategy(self, coin):
        # Select strategy with highest performance score
        state = self.get_strategy_state(coin)
        if state["no"] < self.lookback_period:
            return state["active_strategy"]
        best_strategy = max(self.strategy_performance[coin], key=self.strategy_performance[coin].get)
        state["active_strategy"] = best_strategy
        logging.info(f"{coin} - Selected strategy: {best_strategy}")
        return best_strategy

    def mean_reversion_strategy(self, coin, price, indicators):
        state = self.get_strategy_state(coin)
        if state["no"] <= self.lookback_period:
            return "HOLD"

        if state["position_status"] == "HOLDING":
            if price <= state["stop_loss_price"]:
                logging.info(f"{coin} - Stop Loss Triggered at {price:.6f}")
                return "SELL"
            if price >= state["take_profit_price"]:
                logging.info(f"{coin} - Take Profit Triggered at {price:.6f}")
                return "SELL"

        if state["price_mean"] > price and state["position_status"] == "CASH":
            signal = "BUY"
            state["position_status"] = "HOLDING"
            self.set_risk_levels(coin, price * 1.001)
            logging.info(f"{coin} - BUY Signal (Mean Reversion): Price {price:.6f} below Mean {state['price_mean']:.6f}")
        elif price > state["price_mean"] and state["position_status"] == "HOLDING":
            if price > state["buy_price"] * 1.003:
                signal = "SELL"
                state["position_status"] = "CASH"
                profit_pct = ((price / state["buy_price"]) - 1) * 100 if state["buy_price"] else 0
                logging.info(f"{coin} - SELL Signal (Mean Reversion): Profit {profit_pct:.2f}%")
                state["buy_price"] = None
                state["stop_loss_price"] = None
                state["take_profit_price"] = None
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
        return signal

    def macd_crossover_strategy(self, coin, price, indicators):
        state = self.get_strategy_state(coin)
        if not indicators or state["no"] <= self.lookback_period:
            return "HOLD"

        if state["position_status"] == "HOLDING":
            if price <= state["stop_loss_price"]:
                logging.info(f"{coin} - Stop Loss Triggered at {price:.6f}")
                return "SELL"
            if price >= state["take_profit_price"]:
                logging.info(f"{coin} - Take Profit Triggered at {price:.6f}")
                return "SELL"

        macd = indicators["macd"]
        signal_line = indicators["macd_signal"]
        rsi = indicators["rsi"]

        if macd > signal_line and rsi < RSI_OVERBOUGHT and state["position_status"] == "CASH":
            signal = "BUY"
            state["position_status"] = "HOLDING"
            self.set_risk_levels(coin, price * 1.001)
            logging.info(f"{coin} - BUY Signal (MACD): MACD {macd:.6f} > Signal {signal_line:.6f}, RSI {rsi:.2f}")
        elif macd < signal_line and rsi > RSI_OVERSOLD and state["position_status"] == "HOLDING":
            signal = "SELL"
            state["position_status"] = "CASH"
            profit_pct = ((price / state["buy_price"]) - 1) * 100 if state["buy_price"] else 0
            logging.info(f"{coin} - SELL Signal (MACD): MACD {macd:.6f} < Signal {signal_line:.6f}, RSI {rsi:.2f}, Profit {profit_pct:.2f}%")
            state["buy_price"] = None
            state["stop_loss_price"] = None
            state["take_profit_price"] = None
        else:
            signal = "HOLD"
        return signal

    def rsi_strategy(self, coin, price, indicators):
        state = self.get_strategy_state(coin)
        if not indicators or state["no"] <= self.lookback_period:
            return "HOLD"

        if state["position_status"] == "HOLDING":
            if price <= state["stop_loss_price"]:
                logging.info(f"{coin} - Stop Loss Triggered at {price:.6f}")
                return "SELL"
            if price >= state["take_profit_price"]:
                logging.info(f"{coin} - Take Profit Triggered at {price:.6f}")
                return "SELL"

        rsi = indicators["rsi"]
        stoch_k = indicators["stoch_k"]
        stoch_d = indicators["stoch_d"]

        if rsi < RSI_OVERSOLD and stoch_k < 20 and stoch_k > stoch_d and state["position_status"] == "CASH":
            signal = "BUY"
            state["position_status"] = "HOLDING"
            self.set_risk_levels(coin, price * 1.001)
            logging.info(f"{coin} - BUY Signal (RSI): RSI {rsi:.2f}, Stoch K {stoch_k:.2f}, Stoch D {stoch_d:.2f}")
        elif rsi > RSI_OVERBOUGHT and stoch_k > 80 and stoch_k < stoch_d and state["position_status"] == "HOLDING":
            signal = "SELL"
            state["position_status"] = "CASH"
            profit_pct = ((price / state["buy_price"]) - 1) * 100 if state["buy_price"] else 0
            logging.info(f"{coin} - SELL Signal (RSI): RSI {rsi:.2f}, Stoch K {stoch_k:.2f}, Stoch D {stoch_d:.2f}, Profit {profit_pct:.2f}%")
            state["buy_price"] = None
            state["stop_loss_price"] = None
            state["take_profit_price"] = None
        else:
            signal = "HOLD"
        return signal

    def bollinger_bands_strategy(self, coin, price, indicators):
        state = self.get_strategy_state(coin)
        if not indicators or state["no"] <= self.lookback_period:
            return "HOLD"

        if state["position_status"] == "HOLDING":
            if price <= state["stop_loss_price"]:
                logging.info(f"{coin} - Stop Loss Triggered at {price:.6f}")
                return "SELL"
            if price >= state["take_profit_price"]:
                logging.info(f"{coin} - Take Profit Triggered at {price:.6f}")
                return "SELL"

        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        if price < bb_lower and rsi < RSI_OVERSOLD and state["position_status"] == "CASH":
            signal = "BUY"
            state["position_status"] = "HOLDING"
            self.set_risk_levels(coin, price * 1.001)
            logging.info(f"{coin} - BUY Signal (BBands): Price {price:.6f} < Lower {bb_lower:.6f}, RSI {rsi:.2f}")
        elif price > bb_upper and rsi > RSI_OVERBOUGHT and state["position_status"] == "HOLDING":
            signal = "SELL"
            state["position_status"] = "CASH"
            profit_pct = ((price / state["buy_price"]) - 1) * 100 if state["buy_price"] else 0
            logging.info(f"{coin} - SELL Signal (BBands): Price {price:.6f} > Upper {bb_upper:.6f}, RSI {rsi:.2f}, Profit {profit_pct:.2f}%")
            state["buy_price"] = None
            state["stop_loss_price"] = None
            state["take_profit_price"] = None
        else:
            signal = "HOLD"
        return signal

    def combined_strategy(self, coin, price, indicators):
        state = self.get_strategy_state(coin)
        if not indicators or state["no"] <= self.lookback_period:
            return "HOLD"

        if state["position_status"] == "HOLDING":
            if price <= state["stop_loss_price"]:
                logging.info(f"{coin} - Stop Loss Triggered at {price:.6f}")
                return "SELL"
            if price >= state["take_profit_price"]:
                logging.info(f"{coin} - Take Profit Triggered at {price:.6f}")
                return "SELL"

        signals = [
            self.macd_crossover_strategy(coin, price, indicators),
            self.rsi_strategy(coin, price, indicators),
            self.bollinger_bands_strategy(coin, price, indicators)
        ]

        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")

        if buy_count >= 2 and state["position_status"] == "CASH":
            signal = "BUY"
            state["position_status"] = "HOLDING"
            self.set_risk_levels(coin, price * 1.001)
            logging.info(f"{coin} - BUY Signal (Combined): {buy_count}/3 strategies agree")
        elif sell_count >= 2 and state["position_status"] == "HOLDING":
            signal = "SELL"
            state["position_status"] = "CASH"
            profit_pct = ((price / state["buy_price"]) - 1) * 100 if state["buy_price"] else 0
            logging.info(f"{coin} - SELL Signal (Combined): {sell_count}/3 strategies agree, Profit {profit_pct:.2f}%")
            state["buy_price"] = None
            state["stop_loss_price"] = None
            state["take_profit_price"] = None
        else:
            signal = "HOLD"
        return signal

    def update_strategy_performance(self, coin, strategy, signal, profit_pct=0):
        if signal == "SELL" and profit_pct:
            self.strategy_performance[coin][strategy] += profit_pct
        elif signal == "BUY":
            self.strategy_performance[coin][strategy] += 0.1  # Small reward for entering position

    def generate_signal(self, coin, price):
        state = self.get_strategy_state(coin)
        indicators = self.calculate_indicators(coin)
        active_strategy = self.select_best_strategy(coin)

        if active_strategy == "mean_reversion":
            signal = self.mean_reversion_strategy(coin, price, indicators)
        elif active_strategy == "macd_crossover":
            signal = self.macd_crossover_strategy(coin, price, indicators)
        elif active_strategy == "rsi_strategy":
            signal = self.rsi_strategy(coin, price, indicators)
        elif active_strategy == "bollinger_bands":
            signal = self.bollinger_bands_strategy(coin, price, indicators)
        else:  # combined
            signal = self.combined_strategy(coin, price, indicators)

        # Update performance (profit_pct will be set in simulate_trade)
        self.update_strategy_performance(coin, active_strategy, signal)
        logging.info(f"{coin} - Price: {price:.6f} | Strategy: {active_strategy} | Signal: {signal}")
        return signal

# --- RISK MANAGEMENT ---
class RiskManager:
    def __init__(self):
        self.portfolio_values = []

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
    def __init__(self, strategy, risk_manager, api_client, coin_selector, initial_cash=10000):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.api_client = api_client
        self.coin_selector = coin_selector
        self.cash = initial_cash
        self.holdings = {}
        self.trade_log = []
        self.entry_prices = {}
        self.price_history = {}
        self.trade_count = 0
        self.profitable_trades = 0

    def update_portfolio_value(self, prices):
        portfolio_value = self.cash
        for coin, amount in self.holdings.items():
            price = prices.get(coin, 0)
            portfolio_value += amount * price
        self.risk_manager.update_portfolio(portfolio_value)
        return portfolio_value

    def calculate_trade_amount(self, price, portfolio_value, active_positions):
        max_positions = MAX_COINS
        position_adjustment = min(1.0, max_positions / (active_positions + 1))
        risk_amount = portfolio_value * POSITION_SIZE_PCT * position_adjustment
        trade_qty = risk_amount / price
        trade_qty = math.floor(trade_qty * 10000) / 10000
        return max(0.001, trade_qty)

    def simulate_trade(self, coin, pair, signal, price):
        active_positions = sum(1 for amount in self.holdings.values() if amount > 0)
        portfolio_value = self.update_portfolio_value({coin: price})
        trade_amount = self.calculate_trade_amount(price, portfolio_value, active_positions)

        if coin not in self.holdings:
            self.holdings[coin] = 0
        if coin not in self.entry_prices:
            self.entry_prices[coin] = []
        if coin not in self.price_history:
            self.price_history[coin] = []

        self.price_history[coin].append(price)

        if signal == "BUY" and self.cash >= trade_amount * price * (1 + BUYING_COMMISSION) and active_positions < MAX_COINS:
            self.holdings[coin] += trade_amount
            purchase_amount = trade_amount * price
            commission = purchase_amount * BUYING_COMMISSION
            total_cost = purchase_amount + commission
            self.cash -= total_cost
            self.entry_prices[coin].append(price)
            trade = {
                "timestamp": datetime.now(),
                "action": "BUY",
                "coin": coin,
                "pair": pair,
                "price": price,
                "amount": trade_amount,
                "cash_spent": purchase_amount,
                "commission": commission,
                "total_cost": total_cost,
                "cash_balance": self.cash
            }
            self.trade_log.append(trade)
            self.coin_selector.update_trade_history(coin, trade)
            logging.info(f"BUY: {trade_amount} {coin} at {price}, Spent: {purchase_amount:.6f}, Commission: {commission:.6f}, Total: {total_cost:.6f}, Active Positions: {active_positions + 1}")
            self.api_client.place_order(coin, "BUY", trade_amount)
            logging.info(f"Portfolio Value after BUY: {portfolio_value:.2f}")
        elif signal == "SELL" and self.holdings.get(coin, 0) >= trade_amount:
            sale_amount = self.holdings[coin] * price
            commission = sale_amount * SELLING_COMMISSION
            net_proceeds = sale_amount - commission
            trade_amount = self.holdings[coin]
            self.holdings[coin] = 0
            self.cash += net_proceeds
            self.trade_count += 1
            if self.entry_prices[coin]:
                entry_price = self.entry_prices[coin].pop(0)
                buy_cost = trade_amount * entry_price * (1 + BUYING_COMMISSION)
                profit = net_proceeds - buy_cost
                profit_pct = (net_proceeds / buy_cost - 1) * 100 if buy_cost else 0
                trade = {
                    "timestamp": datetime.now(),
                    "action": "SELL",
                    "coin": coin,
                    "pair": pair,
                    "price": price,
                    "amount": trade_amount,
                    "cash_received": sale_amount,
                    "commission": commission,
                    "net_proceeds": net_proceeds,
                    "cash_balance": self.cash,
                    "profit_pct": profit_pct
                }
                if profit > 0:
                    self.profitable_trades += 1
                # Update strategy performance with profit
                state = self.strategy.get_strategy_state(coin)
                self.strategy.update_strategy_performance(coin, state["active_strategy"], "SELL", profit_pct)
                logging.info(f"{coin} - Trade P&L: {profit:.6f} ({profit_pct:.2f}%)")
            else:
                trade = {
                    "timestamp": datetime.now(),
                    "action": "SELL",
                    "coin": coin,
                    "pair": pair,
                    "price": price,
                    "amount": trade_amount,
                    "cash_received": sale_amount,
                    "commission": commission,
                    "net_proceeds": net_proceeds,
                    "cash_balance": self.cash
                }
            self.trade_log.append(trade)
            self.coin_selector.update_trade_history(coin, trade)
            logging.info(f"SELL: {trade_amount} {coin} at {price}, Received: {sale_amount:.6f}, Commission: {commission:.6f}, Net: {net_proceeds:.6f}, Active Positions: {active_positions - 1}")
            self.api_client.place_order(coin, "SELL", trade_amount)
            logging.info(f"Portfolio Value after SELL: {portfolio_value:.2f}")
            save_trade_log_to_file(self.trade_log)
        elif signal == "BUY" and active_positions >= MAX_COINS:
            logging.info(f"{coin} - BUY signal ignored - maximum positions ({MAX_COINS}) reached")

    def run_simulation(self):
        logging.info("Starting multi-coin simulation (runs until manually stopped)...")
        initial_portfolio_value = self.cash
        logging.info(f"Initial Portfolio Value: {initial_portfolio_value:.2f}")

        try:
            while True:
                try:
                    selected_coins = self.coin_selector.select_coins()
                    logging.info(f"Processing {len(selected_coins)} coins: {[c for c, _ in selected_coins]}")

                    prices = {}
                    for coin, pair in selected_coins:
                        try:
                            ticker_data = self.api_client.get_ticker(pair=pair)
                            if ticker_data and ticker_data.get("Success"):
                                price = float(ticker_data["Data"][pair]["LastPrice"])
                                prices[coin] = price
                                current_time = datetime.now()

                                logging.info(f"Time: {current_time} | Coin: {coin} | Price: {price}")
                                self.strategy.update_price_mean(coin, price)
                                signal = self.strategy.generate_signal(coin, price)
                                logging.info(f"{coin} - Signal: {signal}")

                                if signal in ["BUY", "SELL"]:
                                    self.simulate_trade(coin, pair, signal, price)

                        except Exception as e:
                            logging.error(f"Error processing {coin}: {e}")

                    portfolio_value = self.update_portfolio_value(prices)
                    active_positions = sum(1 for amount in self.holdings.values() if amount > 0)
                    logging.info(f"Portfolio Value: {portfolio_value:.2f}, Active Positions: {active_positions}")

                except Exception as e:
                    logging.error(f"Error in simulation loop: {e}")

                time.sleep(FETCH_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Bot interrupted by user. Closing all open positions...")
            final_prices = {}
            for coin, amount in list(self.holdings.items()):
                if amount > 0:
                    try:
                        pair = next((p for c, p in self.coin_selector.select_coins() if c == coin), None)
                        if not pair:
                            logging.error(f"No pair found for {coin} during final sell")
                            continue
                        lookback_samples = min(len(self.price_history.get(coin, [])), int(60 / FETCH_INTERVAL))
                        recent_prices = self.price_history[coin][-lookback_samples:] if coin in self.price_history else []
                        highest_price = max(recent_prices) if recent_prices else 0
                        ticker_data = self.api_client.get_ticker(pair=pair)
                        if ticker_data and ticker_data.get("Success"):
                            current_price = float(ticker_data["Data"][pair]["LastPrice"])
                            highest_price = max(highest_price, current_price) if highest_price else current_price

                        final_prices[coin] = highest_price
                        sale_amount = amount * highest_price
                        commission = sale_amount * SELLING_COMMISSION
                        net_proceeds = sale_amount - commission
                        self.cash += net_proceeds
                        trade = {
                            "timestamp": datetime.now(),
                            "action": "FINAL_SELL",
                            "coin": coin,
                            "pair": pair,
                            "price": highest_price,
                            "amount": amount,
                            "cash_received": sale_amount,
                            "commission": commission,
                            "net_proceeds": net_proceeds,
                            "cash_balance": self.cash
                        }
                        self.trade_log.append(trade)
                        self.coin_selector.update_trade_history(coin, trade)
                        logging.info(f"Final SELL: {amount} {coin} at {highest_price:.2f}, Net Proceeds: {net_proceeds:.2f}")
                        self.api_client.place_order(coin, "SELL", amount)
                        self.holdings[coin] = 0
                    except Exception as e:
                        logging.error(f"Error during final sell for {coin}: {e}")

            final_portfolio_value = self.update_portfolio_value(final_prices)
            sharpe_ratio = self.risk_manager.calculate_sharpe_ratio()
            logging.info(f"Simulation Terminated. Final Portfolio Value: {final_portfolio_value:.2f}")
            # logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            logging.info(f"Win Rate: {self.profitable_trades/self.trade_count*100:.2f}% ({self.profitable_trades}/{self.trade_count})" if self.trade_count > 0 else "No trades executed")
            save_trade_log_to_file(self.trade_log)
            return final_portfolio_value, sharpe_ratio, self.trade_log

# --- UTILITY FUNCTIONS ---
def save_trade_log_to_file(trade_log):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_log_{timestamp}.txt"
        with open(filename, "w") as file:
            file.write(f"Trade Log - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 80 + "\n\n")
            file.write(f"Total Trades: {len(trade_log)}\n\n")
            file.write("DETAILED TRADE LOG:\n")
            file.write("-" * 80 + "\n")
            for i, trade in enumerate(trade_log, 1):
                file.write(f"Trade #{i}:\n")
                file.write(f"  Timestamp: {trade['timestamp']}\n")
                file.write(f"  Action: {trade['action']}\n")
                file.write(f"  Coin: {trade['coin']}\n")
                file.write(f"  Pair: {trade['pair']}\n")
                file.write(f"  Price: {trade['price']:.6f}\n")
                file.write(f"  Amount: {trade['amount']}\n")
                if 'cash_spent' in trade:
                    file.write(f"  Cash Spent: {trade['cash_spent']:.6f}\n")
                    file.write(f"  Buy Commission: {trade['commission']:.6f}\n")
                    file.write(f"  Total Cost: {trade['total_cost']:.6f}\n")
                elif 'cash_received' in trade:
                    file.write(f"  Cash Received: {trade['cash_received']:.6f}\n")
                    file.write(f"  Sell Commission: {trade['commission']:.6f}\n")
                    file.write(f"  Net Proceeds: {trade['net_proceeds']:.6f}\n")
                file.write(f"  Cash Balance: {trade['cash_balance']:.6f}\n")
                if 'profit_pct' in trade:
                    file.write(f"  Profit: {trade['profit_pct']:.2f}%\n")
                file.write("\n")
        logging.info(f"Trade log saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save trade log: {e}")

# --- MAIN EXECUTION ---
def main():
    try:
        api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
        strategy = AutonomousStrategy(lookback_period=20)
        risk_manager = RiskManager()
        coin_selector = CoinSelector(api_client)

        balance_data = api_client.get_balance()
        initial_cash = balance_data["SpotWallet"]["USD"]["Free"] if balance_data and "SpotWallet" in balance_data else 10000
        logging.info(f"Initial cash balance: {initial_cash}")

        simulation_bot = SimulationBot(strategy, risk_manager, api_client, coin_selector, initial_cash=initial_cash)
        final_value, sharpe_ratio, trade_log = simulation_bot.run_simulation()

        logging.info("Simulation Summary:")
        logging.info(f"Final Portfolio Value: {final_value:.2f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        save_trade_log_to_file(trade_log)
    except Exception as e:
        logging.error(f"Critical error in main function: {e}")

if __name__ == "__main__":
    main()

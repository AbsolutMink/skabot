# -*- coding: utf-8 -*-
"""
Bot de Trading Multi-Asset v4.3 - MÓDULO DE ANÁLISIS Y SEÑALES (MAS)
- Sistema completo de indicadores técnicos
- Detección de patrones de velas
- Modo Manual (Asistente) y Automático
- Gestión de riesgo avanzada
"""

import os
import sys
import time
import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, NamedTuple, Set
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ===== CREDENCIALES MT5 =====
class MT5Config:
    LOGIN = 40376616
    PASSWORD = "SeruGir@n36"
    SERVER = "Deriv-Demo"
    TIMEOUT = 60000
    MAGIC_NUMBER = 234567

# ===== MODO DE OPERACIÓN =====
class OperationMode(Enum):
    MANUAL = "manual"      # Solo alertas, usuario decide
    AUTOMATIC = "auto"     # Ejecución automática de señales

# ===== CONFIGURACIÓN OPTIMIZADA V4.3 =====
class TradingConfig:
    # Capital y Riesgo
    CAPITAL = 4000.0
    RISK_PER_TRADE = 0.01          # 1% riesgo por operación
    MAX_DAILY_DRAWDOWN = 0.05      # 5% máxima pérdida diaria
    MAX_DAILY_OPERATIONS = 15      # Máximo 15 operaciones por día
    
    # Gestión de Operaciones
    SL_ATR_MULTIPLIER = 1.5        # Stop Loss = 1.5 x ATR
    TP_ATR_MULTIPLIER = 2.5        # Take Profit = 2.5 x ATR
    MIN_RR_RATIO = 1.5             # Mínimo R:R aceptable
    MAX_TRADE_DURATION = 45        # Minutos máximos por operación
    
    # Lotaje
    MAX_LOT_SIZE = 1.5
    MIN_LOT_SIZE = 0.01
    
    # Análisis
    TIMEFRAME_PRIMARY = mt5.TIMEFRAME_M5    # Timeframe principal
    SIGNAL_COOLDOWN = 300                   # 5 minutos entre señales del mismo símbolo
    
    # Indicadores - Configuraciones
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH = 3
    STOCH_OVERSOLD = 20
    STOCH_OVERBOUGHT = 80
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    EMA_FAST = 9
    EMA_SLOW = 21
    
    BB_PERIOD = 20
    BB_STD = 2
    
    ATR_PERIOD = 14

# ===== CREDENCIALES TELEGRAM =====
TELEGRAM_TOKEN = "8335797861:AAFhkConjYEsTKCBfFfP248rLKEoLCeYE1A"
CHAT_ID = "1565589354"

# ===== CONTROL DEL BOT =====
class BotController:
    def __init__(self):
        self.running = True
        self.paused = False
        self.operation_mode = OperationMode.MANUAL  # Por defecto modo manual
        self.active_operation = False
        self.last_signal_time = {}
        self.daily_operations = 0
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.signals_sent_today = 0
        self.max_daily_loss_reached = False
        
bot_control = BotController()

# ===== CLASES DE DATOS =====
class SignalSource(Enum):
    RSI = "RSI"
    STOCHASTIC = "Oscilador Estocástico"
    MACD = "MACD"
    EMA_CROSS = "Cruce de EMAs"
    BOLLINGER = "Bandas de Bollinger"
    HAMMER = "Patrón Martillo"
    HANGING_MAN = "Patrón Hombre Colgado"
    BULLISH_ENGULFING = "Envolvente Alcista"
    BEARISH_ENGULFING = "Envolvente Bajista"
    MORNING_STAR = "Estrella de la Mañana"
    EVENING_STAR = "Estrella del Atardecer"

class TradingSignal(NamedTuple):
    symbol: str
    direction: str  # BUY o SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    source: SignalSource
    confidence: float
    timestamp: datetime
    suggested_lots: float
    risk_reward: float
    atr_value: float
    description: str

# ===== LOGGING =====
def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"bot_v43_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ===== MÓDULO DE ANÁLISIS Y SEÑALES (MAS) =====
class ModuloAnalisisSeñales:
    def __init__(self, mt5_conn):
        self.mt5 = mt5_conn
        self.last_candle_time = {}
        
    def fetch_rates(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Obtiene datos históricos"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is not None and len(rates) >= count:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
        except Exception as e:
            logging.error(f"Error obteniendo datos de {symbol}: {e}")
        return None
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula TODOS los indicadores técnicos"""
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['tick_volume']
        
        # === RSI ===
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=TradingConfig.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=TradingConfig.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # === ESTOCÁSTICO ===
        lowest_low = l.rolling(window=TradingConfig.STOCH_K).min()
        highest_high = h.rolling(window=TradingConfig.STOCH_K).max()
        df['stoch_k'] = 100 * ((c - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=TradingConfig.STOCH_D).mean()
        
        # === MACD ===
        ema_fast = c.ewm(span=TradingConfig.MACD_FAST, adjust=False).mean()
        ema_slow = c.ewm(span=TradingConfig.MACD_SLOW, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=TradingConfig.MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # === EMAs para cruces ===
        df['ema_fast'] = c.ewm(span=TradingConfig.EMA_FAST, adjust=False).mean()
        df['ema_slow'] = c.ewm(span=TradingConfig.EMA_SLOW, adjust=False).mean()
        
        # === BANDAS DE BOLLINGER ===
        rolling_mean = c.rolling(window=TradingConfig.BB_PERIOD).mean()
        rolling_std = c.rolling(window=TradingConfig.BB_PERIOD).std()
        df['bb_upper'] = rolling_mean + (rolling_std * TradingConfig.BB_STD)
        df['bb_lower'] = rolling_mean - (rolling_std * TradingConfig.BB_STD)
        df['bb_middle'] = rolling_mean
        
        # === ATR (para gestión de riesgo) ===
        high_low = h - l
        high_close = (h - c.shift()).abs()
        low_close = (l - c.shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=TradingConfig.ATR_PERIOD).mean()
        
        return df.dropna()
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[SignalSource]:
        """Detecta patrones de velas japonesas"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Últimas 3 velas para análisis
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Cálculos básicos
        curr_body = abs(curr['close'] - curr['open'])
        curr_upper_wick = curr['high'] - max(curr['close'], curr['open'])
        curr_lower_wick = min(curr['close'], curr['open']) - curr['low']
        curr_is_bullish = curr['close'] > curr['open']
        
        prev_body = abs(prev['close'] - prev['open'])
        prev_is_bullish = prev['close'] > prev['open']
        
        # === PATRONES ALCISTAS ===
        
        # Martillo (Hammer)
        if curr_lower_wick > curr_body * 2 and curr_upper_wick < curr_body * 0.3:
            if not curr_is_bullish or curr_body < prev_body * 0.5:
                patterns.append(SignalSource.HAMMER)
        
        # Envolvente Alcista
        if not prev_is_bullish and curr_is_bullish:
            if curr['open'] < prev['close'] and curr['close'] > prev['open']:
                patterns.append(SignalSource.BULLISH_ENGULFING)
        
        # Estrella de la Mañana (3 velas)
        if len(df) >= 3:
            if not df.iloc[-3]['close'] > df.iloc[-3]['open']:  # Primera vela bajista
                if abs(prev['close'] - prev['open']) < prev_body * 0.3:  # Segunda vela pequeña
                    if curr_is_bullish and curr['close'] > df.iloc[-3]['open']:  # Tercera vela alcista
                        patterns.append(SignalSource.MORNING_STAR)
        
        # === PATRONES BAJISTAS ===
        
        # Hombre Colgado (Hanging Man)
        if curr_lower_wick > curr_body * 2 and curr_upper_wick < curr_body * 0.3:
            if curr_is_bullish and df['close'].iloc[-5:].mean() > df['close'].iloc[-10:-5].mean():
                patterns.append(SignalSource.HANGING_MAN)
        
        # Envolvente Bajista
        if prev_is_bullish and not curr_is_bullish:
            if curr['open'] > prev['close'] and curr['close'] < prev['open']:
                patterns.append(SignalSource.BEARISH_ENGULFING)
        
        # Estrella del Atardecer (3 velas)
        if len(df) >= 3:
            if df.iloc[-3]['close'] > df.iloc[-3]['open']:  # Primera vela alcista
                if abs(prev['close'] - prev['open']) < prev_body * 0.3:  # Segunda vela pequeña
                    if not curr_is_bullish and curr['close'] < df.iloc[-3]['open']:  # Tercera vela bajista
                        patterns.append(SignalSource.EVENING_STAR)
        
        return patterns
    
    def check_all_signals(self, symbol: str) -> Optional[TradingSignal]:
        """
        FUNCIÓN PRINCIPAL DEL MAS
        Verifica TODAS las condiciones y retorna la PRIMERA señal válida encontrada
        """
        try:
            # Obtener datos
            df = self.fetch_rates(symbol, TradingConfig.TIMEFRAME_PRIMARY, 100)
            if df is None or len(df) < 50:
                return None
            
            # Calcular indicadores
            df = self.calculate_all_indicators(df)
            if df is None or len(df) < 2:
                return None
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Información del símbolo para cálculos
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
            
            point = symbol_info.point
            atr = current['atr']
            spread = symbol_info.spread * point
            
            # Variables para la señal
            signal_direction = None
            signal_source = None
            signal_description = ""
            confidence = 0.7  # Confianza base
            
            # === VERIFICAR TODAS LAS CONDICIONES (LA PRIMERA QUE SE CUMPLA GENERA SEÑAL) ===
            
            # 1. RSI - Salida de zonas extremas
            if previous['rsi'] < TradingConfig.RSI_OVERSOLD and current['rsi'] > TradingConfig.RSI_OVERSOLD:
                signal_direction = "BUY"
                signal_source = SignalSource.RSI
                signal_description = f"RSI salió de sobreventa ({current['rsi']:.1f})"
                confidence = 0.75
                
            elif previous['rsi'] > TradingConfig.RSI_OVERBOUGHT and current['rsi'] < TradingConfig.RSI_OVERBOUGHT:
                signal_direction = "SELL"
                signal_source = SignalSource.RSI
                signal_description = f"RSI salió de sobrecompra ({current['rsi']:.1f})"
                confidence = 0.75
            
            # 2. Estocástico - Cruces en zonas extremas
            elif not signal_direction:
                if current['stoch_k'] < TradingConfig.STOCH_OVERSOLD:
                    if previous['stoch_k'] < previous['stoch_d'] and current['stoch_k'] > current['stoch_d']:
                        signal_direction = "BUY"
                        signal_source = SignalSource.STOCHASTIC
                        signal_description = f"Cruce alcista del Estocástico en sobreventa"
                        confidence = 0.8
                        
                elif current['stoch_k'] > TradingConfig.STOCH_OVERBOUGHT:
                    if previous['stoch_k'] > previous['stoch_d'] and current['stoch_k'] < current['stoch_d']:
                        signal_direction = "SELL"
                        signal_source = SignalSource.STOCHASTIC
                        signal_description = f"Cruce bajista del Estocástico en sobrecompra"
                        confidence = 0.8
            
            # 3. MACD - Cruces con línea de señal
            elif not signal_direction:
                if previous['macd'] < previous['macd_signal'] and current['macd'] > current['macd_signal']:
                    signal_direction = "BUY"
                    signal_source = SignalSource.MACD
                    signal_description = f"Cruce alcista del MACD"
                    confidence = 0.75
                    
                elif previous['macd'] > previous['macd_signal'] and current['macd'] < current['macd_signal']:
                    signal_direction = "SELL"
                    signal_source = SignalSource.MACD
                    signal_description = f"Cruce bajista del MACD"
                    confidence = 0.75
            
            # 4. Cruce de EMAs
            elif not signal_direction:
                if previous['ema_fast'] < previous['ema_slow'] and current['ema_fast'] > current['ema_slow']:
                    signal_direction = "BUY"
                    signal_source = SignalSource.EMA_CROSS
                    signal_description = f"EMA{TradingConfig.EMA_FAST} cruzó por encima de EMA{TradingConfig.EMA_SLOW}"
                    confidence = 0.7
                    
                elif previous['ema_fast'] > previous['ema_slow'] and current['ema_fast'] < current['ema_slow']:
                    signal_direction = "SELL"
                    signal_source = SignalSource.EMA_CROSS
                    signal_description = f"EMA{TradingConfig.EMA_FAST} cruzó por debajo de EMA{TradingConfig.EMA_SLOW}"
                    confidence = 0.7
            
            # 5. Bandas de Bollinger - Breakouts
            elif not signal_direction:
                if current['close'] > current['bb_upper'] and previous['close'] <= previous['bb_upper']:
                    signal_direction = "BUY"
                    signal_source = SignalSource.BOLLINGER
                    signal_description = f"Ruptura alcista de Banda de Bollinger superior"
                    confidence = 0.65
                    
                elif current['close'] < current['bb_lower'] and previous['close'] >= previous['bb_lower']:
                    signal_direction = "SELL"
                    signal_source = SignalSource.BOLLINGER
                    signal_description = f"Ruptura bajista de Banda de Bollinger inferior"
                    confidence = 0.65
            
            # 6. Patrones de Velas
            if not signal_direction:
                patterns = self.detect_candlestick_patterns(df)
                
                # Patrones alcistas
                if SignalSource.HAMMER in patterns:
                    signal_direction = "BUY"
                    signal_source = SignalSource.HAMMER
                    signal_description = "Patrón Martillo detectado"
                    confidence = 0.85
                    
                elif SignalSource.BULLISH_ENGULFING in patterns:
                    signal_direction = "BUY"
                    signal_source = SignalSource.BULLISH_ENGULFING
                    signal_description = "Patrón Envolvente Alcista detectado"
                    confidence = 0.9
                    
                elif SignalSource.MORNING_STAR in patterns:
                    signal_direction = "BUY"
                    signal_source = SignalSource.MORNING_STAR
                    signal_description = "Patrón Estrella de la Mañana detectado"
                    confidence = 0.9
                
                # Patrones bajistas
                elif SignalSource.HANGING_MAN in patterns:
                    signal_direction = "SELL"
                    signal_source = SignalSource.HANGING_MAN
                    signal_description = "Patrón Hombre Colgado detectado"
                    confidence = 0.85
                    
                elif SignalSource.BEARISH_ENGULFING in patterns:
                    signal_direction = "SELL"
                    signal_source = SignalSource.BEARISH_ENGULFING
                    signal_description = "Patrón Envolvente Bajista detectado"
                    confidence = 0.9
                    
                elif SignalSource.EVENING_STAR in patterns:
                    signal_direction = "SELL"
                    signal_source = SignalSource.EVENING_STAR
                    signal_description = "Patrón Estrella del Atardecer detectado"
                    confidence = 0.9
            
            # Si encontramos una señal válida, construir el objeto TradingSignal
            if signal_direction and signal_source:
                entry_price = current['close']
                
                # Calcular SL y TP basados en ATR
                sl_distance = atr * TradingConfig.SL_ATR_MULTIPLIER
                tp_distance = atr * TradingConfig.TP_ATR_MULTIPLIER
                
                if signal_direction == "BUY":
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + tp_distance
                else:
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - tp_distance
                
                # Calcular Risk/Reward
                risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
                
                # Solo generar señal si el R:R es aceptable
                if risk_reward >= TradingConfig.MIN_RR_RATIO:
                    # Calcular lotaje sugerido
                    account_balance = mt5.account_info().balance if mt5.account_info() else TradingConfig.CAPITAL
                    risk_amount = account_balance * TradingConfig.RISK_PER_TRADE
                    
                    pip_risk = abs(entry_price - stop_loss) / point
                    pip_value = symbol_info.trade_tick_value
                    
                    suggested_lots = risk_amount / (pip_risk * pip_value) if pip_risk > 0 else TradingConfig.MIN_LOT_SIZE
                    suggested_lots = max(TradingConfig.MIN_LOT_SIZE, 
                                       min(suggested_lots, TradingConfig.MAX_LOT_SIZE))
                    suggested_lots = round(suggested_lots / symbol_info.volume_step) * symbol_info.volume_step
                    
                    return TradingSignal(
                        symbol=symbol,
                        direction=signal_direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        source=signal_source,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        suggested_lots=round(suggested_lots, 2),
                        risk_reward=risk_reward,
                        atr_value=atr,
                        description=signal_description
                    )
            
            return None
            
        except Exception as e:
            logging.error(f"Error en check_all_signals para {symbol}: {e}")
            return None

# ===== GESTOR DE RIESGO =====
class RiskManager:
    def __init__(self):
        self.positions_data = {}
        self.daily_start_balance = 0.0

    def _record_position_data(self, position, sl=None, tp=None):
        """Guarda información relevante de la posición"""
        try:
            open_time = datetime.fromtimestamp(position.time)
        except Exception:
            open_time = datetime.now()

        ticket = position.ticket
        self.positions_data[ticket] = {
            'open_time': open_time,
            'symbol': position.symbol,
            'type': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'sl': position.sl if sl is None else sl,
            'tp': position.tp if tp is None else tp
        }

    def check_daily_drawdown(self) -> bool:
        """Verifica si se alcanzó el drawdown máximo diario"""
        try:
            account = mt5.account_info()
            if not account:
                return False
            
            current_balance = account.balance
            daily_loss_pct = (self.daily_start_balance - current_balance) / self.daily_start_balance
            
            if daily_loss_pct >= TradingConfig.MAX_DAILY_DRAWDOWN:
                bot_control.max_daily_loss_reached = True
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error verificando drawdown: {e}")
            return False
    
    def apply_risk_management(self, position):
        """Aplica gestión de riesgo a una posición abierta"""
        try:
            symbol = position.symbol
            ticket = position.ticket

            # Si ya tiene SL y TP, no hacer nada
            if position.sl != 0 and position.tp != 0:
                if ticket not in self.positions_data:
                    self._record_position_data(position)
                return True

            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
            
            # Obtener el tick actual para precios bid/ask
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            # Obtener datos para calcular ATR
            mas = ModuloAnalisisSeñales(None)
            df = mas.fetch_rates(symbol, TradingConfig.TIMEFRAME_PRIMARY, 50)
            if df is None:
                return False
            
            df = mas.calculate_all_indicators(df)
            atr = df['atr'].iloc[-1]
            
            # Verificar que ATR sea válido
            if atr <= 0 or pd.isna(atr):
                logging.error(f"ATR inválido para {symbol}: {atr}")
                return False
            
            # Calcular distancias de SL y TP
            sl_distance = atr * TradingConfig.SL_ATR_MULTIPLIER
            tp_distance = atr * TradingConfig.TP_ATR_MULTIPLIER
            
            # Obtener el nivel mínimo de stops (StopLevel)
            stop_level = symbol_info.trade_stops_level * symbol_info.point
            spread = symbol_info.spread * symbol_info.point
            
            # Asegurar que las distancias sean mayores al mínimo permitido
            min_distance = max(stop_level, spread * 2)
            sl_distance = max(sl_distance, min_distance * 1.5)
            tp_distance = max(tp_distance, min_distance * 2.5)
            
            # Calcular SL y TP según el tipo de posición
            if position.type == mt5.ORDER_TYPE_BUY:
                # Para BUY: SL debe estar por debajo del bid, TP por encima del bid
                current_price = tick.bid
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                
                # Verificar que SL esté por debajo del precio de apertura
                if sl >= position.price_open:
                    sl = position.price_open - min_distance * 2
                    
            else:  # SELL
                # Para SELL: SL debe estar por encima del ask, TP por debajo del ask
                current_price = tick.ask
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                
                # Verificar que SL esté por encima del precio de apertura
                if sl <= position.price_open:
                    sl = position.price_open + min_distance * 2
            
            # Normalizar precios según los dígitos del símbolo
            sl = round(sl, symbol_info.digits)
            tp = round(tp, symbol_info.digits)
            
            # Verificación adicional de validez
            if position.type == mt5.ORDER_TYPE_BUY:
                if sl >= tick.bid or tp <= tick.bid:
                    logging.error(f"Stops inválidos para BUY: precio={tick.bid}, SL={sl}, TP={tp}")
                    return False
            else:
                if sl <= tick.ask or tp >= tick.ask:
                    logging.error(f"Stops inválidos para SELL: precio={tick.ask}, SL={sl}, TP={tp}")
                    return False
            
            # Preparar request para modificar posición
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp,
                "magic": MT5Config.MAGIC_NUMBER,
                "comment": "Risk Management v4.3"
            }
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"✅ SL/TP aplicados a posición {ticket}: SL={sl:.5f}, TP={tp:.5f}")

                # Guardar datos de la posición
                self._record_position_data(position, sl=sl, tp=tp)
                return True
            else:
                error_msg = result.comment if result else 'Unknown error'
                logging.error(f"❌ Error aplicando SL/TP a {ticket}: {error_msg}")

                # Si el error es por stops inválidos, intentar con valores más conservadores
                if "Invalid stops" in error_msg or "Invalid request" in error_msg:
                    logging.info(f"Reintentando con stops más conservadores para {symbol}")
                    
                    # Usar distancias más grandes
                    sl_distance = max(sl_distance * 2, min_distance * 3)
                    tp_distance = max(tp_distance * 2, min_distance * 5)
                    
                    if position.type == mt5.ORDER_TYPE_BUY:
                        sl = tick.bid - sl_distance
                        tp = tick.bid + tp_distance
                    else:
                        sl = tick.ask + sl_distance
                        tp = tick.ask - tp_distance
                    
                    sl = round(sl, symbol_info.digits)
                    tp = round(tp, symbol_info.digits)
                    
                    request["sl"] = sl
                    request["tp"] = tp
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"✅ SL/TP aplicados (segundo intento): SL={sl:.5f}, TP={tp:.5f}")
                        self._record_position_data(position, sl=sl, tp=tp)
                        return True

                return False
                
        except Exception as e:
            logging.error(f"Error en apply_risk_management: {e}")
            return False
    
    def check_time_limit(self, position) -> bool:
        """Verifica si una posición excedió el tiempo límite"""
        try:
            ticket = position.ticket

            if ticket not in self.positions_data:
                self._record_position_data(position)

            data = self.positions_data.get(ticket)
            if not data or 'open_time' not in data:
                return False

            open_time = data['open_time']

            if isinstance(open_time, datetime):
                time_elapsed = (datetime.now() - open_time).total_seconds() / 60

                if time_elapsed >= TradingConfig.MAX_TRADE_DURATION:
                    return True

            return False

        except Exception as e:
            logging.error(f"Error verificando tiempo límite: {e}")
            return False
    
    def close_position(self, position) -> bool:
        """Cierra una posición específica"""
        try:
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume
            
            # Tipo de orden contraria
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": MT5Config.MAGIC_NUMBER,
                "comment": "Cierre por tiempo"
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"✅ Posición {ticket} cerrada por tiempo límite")
                if ticket in self.positions_data:
                    del self.positions_data[ticket]
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error cerrando posición: {e}")
            return False

# ===== EJECUTOR DE OPERACIONES =====
class TradeExecutor:
    def __init__(self, mt5_conn):
        self.mt5 = mt5_conn
        
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Ejecuta una operación basada en la señal"""
        try:
            symbol = signal.symbol
            symbol_info = mt5.symbol_info(symbol)
            
            if not symbol_info or not symbol_info.visible:
                logging.error(f"❌ Símbolo {symbol} no disponible")
                return False
            
            # Obtener tick actual
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logging.error(f"❌ No se pudo obtener tick para {symbol}")
                return False
            
            # Preparar request con validación de stops
            if signal.direction == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                
                # Verificar que los stops sean válidos para BUY
                stop_level = symbol_info.trade_stops_level * symbol_info.point
                min_distance = max(stop_level, symbol_info.spread * symbol_info.point * 2)
                
                # Ajustar SL si está muy cerca
                if price - signal.stop_loss < min_distance:
                    sl = price - min_distance * 1.5
                else:
                    sl = signal.stop_loss
                    
                # Ajustar TP si está muy cerca
                if signal.take_profit - price < min_distance:
                    tp = price + min_distance * 2.5
                else:
                    tp = signal.take_profit
                    
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                
                # Verificar que los stops sean válidos para SELL
                stop_level = symbol_info.trade_stops_level * symbol_info.point
                min_distance = max(stop_level, symbol_info.spread * symbol_info.point * 2)
                
                # Ajustar SL si está muy cerca
                if signal.stop_loss - price < min_distance:
                    sl = price + min_distance * 1.5
                else:
                    sl = signal.stop_loss
                    
                # Ajustar TP si está muy cerca
                if price - signal.take_profit < min_distance:
                    tp = price - min_distance * 2.5
                else:
                    tp = signal.take_profit
            
            # Normalizar precios
            sl = round(sl, symbol_info.digits)
            tp = round(tp, symbol_info.digits)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": signal.suggested_lots,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": MT5Config.MAGIC_NUMBER,
                "comment": f"MAS v4.3 - {signal.source.value}"
            }
            
            # Log del request para debugging
            logging.info(f"📝 Request: {symbol} {signal.direction} Vol:{signal.suggested_lots} "
                        f"Price:{price:.5f} SL:{sl:.5f} TP:{tp:.5f}")
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"✅ OPERACIÓN EJECUTADA: {symbol} {signal.direction} @ {price:.5f}")
                bot_control.daily_operations += 1
                bot_control.active_operation = True
                return True
            else:
                error_msg = result.comment if result else "Error desconocido"
                error_code = result.retcode if result else "N/A"
                logging.error(f"❌ Error ejecutando operación: {error_msg} (Code: {error_code})")
                
                # Si el error es por stops, intentar sin stops e aplicarlos después
                if "Invalid stops" in error_msg or "Invalid request" in error_msg:
                    logging.info("Intentando sin SL/TP inicial...")
                    request.pop("sl", None)
                    request.pop("tp", None)
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"✅ Operación ejecutada sin stops, se aplicarán después")
                        bot_control.daily_operations += 1
                        bot_control.active_operation = True
                        return True
                        
                return False
                
        except Exception as e:
            logging.error(f"Error en execute_trade: {e}")
            return False

# ===== COMANDOS TELEGRAM =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Bot v4.3 MAS - Iniciado")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_control.running = False
    await update.message.reply_text("🛑 Bot detenido")

async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_control.paused = not bot_control.paused
    status = "pausado ⏸️" if bot_control.paused else "activo ▶️"
    await update.message.reply_text(f"Bot {status}")

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cambia entre modo manual y automático"""
    if bot_control.operation_mode == OperationMode.MANUAL:
        bot_control.operation_mode = OperationMode.AUTOMATIC
        mode = "AUTOMÁTICO 🤖"
    else:
        bot_control.operation_mode = OperationMode.MANUAL
        mode = "MANUAL 👤"
    
    await update.message.reply_text(f"Modo cambiado a: {mode}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra el estado actual del bot"""
    try:
        account = mt5.account_info()
        if not account:
            await update.message.reply_text("❌ No conectado a MT5")
            return
        
        positions = mt5.positions_get()
        num_positions = len(positions) if positions else 0
        
        msg = f"""📊 **ESTADO BOT v4.3 MAS**
        
💰 Balance: ${account.balance:.2f}
💵 Equity: ${account.equity:.2f}
📈 Profit: ${account.profit:.2f}
🎯 Posiciones: {num_positions}

🤖 Modo: {bot_control.operation_mode.value.upper()}
📊 Operaciones hoy: {bot_control.daily_operations}/{TradingConfig.MAX_DAILY_OPERATIONS}
💹 P&L Diario: ${bot_control.daily_pnl:.2f}

Estado: {'🟢 Activo' if bot_control.running else '🔴 Detenido'}
{'⏸️ PAUSADO' if bot_control.paused else ''}
{'🚫 LÍMITE PÉRDIDA DIARIA' if bot_control.max_daily_loss_reached else ''}"""
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Error en cmd_status: {e}")
        await update.message.reply_text("❌ Error obteniendo estado")

async def send_telegram_signal(text: str):
    """Envía señal a Telegram"""
    try:
        bot = Bot(TELEGRAM_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, 
            text=text, 
            parse_mode='HTML',
            disable_web_page_preview=True
        )
        logging.info("✅ Señal enviada a Telegram")
    except Exception as e:
        logging.error(f"❌ Error Telegram: {e}")

# ===== BOT PRINCIPAL V4.3 =====
class TradingBotV43:
    def __init__(self):
        self.mt5_conn = None
        self.mas = None
        self.risk_manager = None
        self.executor = None
        self.telegram_app = None
        self.symbols_to_monitor = []  # Se llenará dinámicamente
        
    def validate_symbols(self):
        """Valida y filtra los símbolos disponibles para trading"""
        valid_symbols = []
        
        # Lista de símbolos preferidos
        preferred_symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD",
            "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"
        ]
        
        for symbol in preferred_symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info and symbol_info.visible:
                # Verificar que el símbolo esté habilitado para trading
                if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    valid_symbols.append(symbol)
                    logging.info(f"✅ Símbolo validado: {symbol}")
                else:
                    logging.warning(f"⚠️ {symbol} no está en modo trading completo")
            else:
                logging.warning(f"⚠️ {symbol} no está disponible")
        
        if not valid_symbols:
            # Si no hay símbolos preferidos, buscar cualquier par de Forex disponible
            all_symbols = mt5.symbols_get()
            for symbol_info in all_symbols[:20]:  # Limitar a 20 para no sobrecargar
                if symbol_info.visible and "USD" in symbol_info.name:
                    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                        valid_symbols.append(symbol_info.name)
                        
        self.symbols_to_monitor = valid_symbols
        logging.info(f"📊 Total símbolos a monitorear: {len(self.symbols_to_monitor)}")
        return len(valid_symbols) > 0
        
    async def setup_telegram(self):
        """Configura Telegram con comandos"""
        try:
            self.telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
            
            self.telegram_app.add_handler(CommandHandler("start", cmd_start))
            self.telegram_app.add_handler(CommandHandler("stop", cmd_stop))
            self.telegram_app.add_handler(CommandHandler("pause", cmd_pause))
            self.telegram_app.add_handler(CommandHandler("status", cmd_status))
            self.telegram_app.add_handler(CommandHandler("mode", cmd_mode))
            
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            
            logging.info("✅ Telegram configurado")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error configurando Telegram: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Inicializa el bot"""
        setup_logging()
        
        # Conectar a MT5
        if not mt5.initialize():
            logging.error("❌ No se pudo inicializar MT5")
            return False
        
        if not mt5.login(MT5Config.LOGIN, MT5Config.PASSWORD, MT5Config.SERVER):
            logging.error("❌ No se pudo hacer login en MT5")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if not account:
            logging.error("❌ No se pudo obtener información de cuenta")
            return False
        
        # Inicializar componentes
        self.mas = ModuloAnalisisSeñales(None)
        self.risk_manager = RiskManager()
        self.executor = TradeExecutor(None)
        
        # Configurar balance inicial del día
        bot_control.daily_start_balance = account.balance
        self.risk_manager.daily_start_balance = account.balance
        
        # Configurar Telegram
        await self.setup_telegram()
        
        # Mensaje inicial
        start_msg = f"""🚀 <b>BOT v4.3 - MAS INICIADO</b>
━━━━━━━━━━━━━━━━━━━━━━━━

💰 Balance: ${account.balance:.2f}
🤖 Modo: {bot_control.operation_mode.value.upper()}

📊 <b>MÓDULO DE ANÁLISIS Y SEÑALES</b>
✅ RSI (14) - Zonas 30/70
✅ Estocástico (14,3,3) - Zonas 20/80
✅ MACD (12,26,9)
✅ Cruce EMAs (9,21)
✅ Bandas de Bollinger (20,2)
✅ 6 Patrones de Velas

⚙️ <b>GESTIÓN DE RIESGO</b>
• Riesgo por operación: {TradingConfig.RISK_PER_TRADE*100}%
• SL: {TradingConfig.SL_ATR_MULTIPLIER} x ATR
• TP: {TradingConfig.TP_ATR_MULTIPLIER} x ATR
• R:R mínimo: {TradingConfig.MIN_RR_RATIO}:1
• Cierre automático: {TradingConfig.MAX_TRADE_DURATION} min

<b>COMANDOS:</b>
/status - Estado actual
/mode - Cambiar modo Manual/Auto
/pause - Pausar/Reanudar
/stop - Detener bot

🔍 Monitoreando mercados...
━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        await send_telegram_signal(start_msg)
        logging.info("✅ Bot v4.3 MAS inicializado correctamente")
        return True
    
    async def check_for_new_candle(self, symbol: str) -> bool:
        """Verifica si hay una nueva vela en M5"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            current_time = datetime.fromtimestamp(tick.time)
            candle_time = current_time.replace(
                minute=(current_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            
            last_time = self.mas.last_candle_time.get(symbol)
            
            if last_time is None or candle_time > last_time:
                self.mas.last_candle_time[symbol] = candle_time
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error verificando nueva vela: {e}")
            return False
    
    async def process_signal(self, signal: TradingSignal):
        """Procesa una señal según el modo de operación"""
        try:
            # Verificar cooldown
            current_time = time.time()
            last_signal = bot_control.last_signal_time.get(signal.symbol, 0)
            
            if current_time - last_signal < TradingConfig.SIGNAL_COOLDOWN:
                return
            
            # Actualizar tiempo de última señal
            bot_control.last_signal_time[signal.symbol] = current_time
            
            # Crear mensaje de señal
            signal_msg = f"""🎯 <b>SEÑAL DE OPORTUNIDAD DETECTADA</b>
━━━━━━━━━━━━━━━━━━━━━━━━

🔔 <b>Fuente:</b> {signal.source.value}
📊 <b>Símbolo:</b> {signal.symbol}
📈 <b>Dirección:</b> {signal.direction}
💵 <b>Precio Entrada:</b> {signal.entry_price:.5f}

📉 <b>Stop Loss:</b> {signal.stop_loss:.5f}
📈 <b>Take Profit:</b> {signal.take_profit:.5f}
⚖️ <b>R:R Ratio:</b> 1:{signal.risk_reward:.1f}

💼 <b>Lotes Sugeridos:</b> {signal.suggested_lots}
🎯 <b>Confianza:</b> {signal.confidence*100:.0f}%

📝 <b>Descripción:</b>
{signal.description}

🤖 <b>Modo:</b> {bot_control.operation_mode.value.upper()}"""
            
            # Procesar según modo
            if bot_control.operation_mode == OperationMode.MANUAL:
                signal_msg += "\n\n⚠️ <b>ACCIÓN REQUERIDA:</b> Abrir operación manualmente"
                await send_telegram_signal(signal_msg)
                
            elif bot_control.operation_mode == OperationMode.AUTOMATIC:
                # Verificar si no hay operación activa
                if not bot_control.active_operation:
                    success = self.executor.execute_trade(signal)
                    
                    if success:
                        signal_msg += "\n\n✅ <b>OPERACIÓN EJECUTADA AUTOMÁTICAMENTE</b>"
                    else:
                        signal_msg += "\n\n❌ <b>ERROR AL EJECUTAR OPERACIÓN</b>"
                    
                    await send_telegram_signal(signal_msg)
                    
        except Exception as e:
            logging.error(f"Error procesando señal: {e}")
    
    async def monitor_positions(self):
        """Monitorea y gestiona posiciones abiertas"""
        try:
            positions = mt5.positions_get()
            if not positions:
                bot_control.active_operation = False
                return
            
            for position in positions:
                # Aplicar gestión de riesgo si no tiene SL/TP
                if position.sl == 0 or position.tp == 0:
                    self.risk_manager.apply_risk_management(position)
                
                # Verificar tiempo límite
                if self.risk_manager.check_time_limit(position):
                    if self.risk_manager.close_position(position):
                        await send_telegram_signal(
                            f"⏰ Posición cerrada por tiempo límite:\n"
                            f"{position.symbol} - Ticket #{position.ticket}"
                        )
            
        except Exception as e:
            logging.error(f"Error monitoreando posiciones: {e}")
    
    async def run_analysis_cycle(self):
        """Ciclo principal de análisis del MAS"""
        while bot_control.running:
            try:
                if bot_control.paused:
                    await asyncio.sleep(5)
                    continue
                
                # Verificar límites diarios
                if bot_control.daily_operations >= TradingConfig.MAX_DAILY_OPERATIONS:
                    logging.info("📊 Límite diario de operaciones alcanzado")
                    await asyncio.sleep(60)
                    continue
                
                if bot_control.max_daily_loss_reached:
                    logging.info("🚫 Límite de pérdida diaria alcanzado")
                    await asyncio.sleep(60)
                    continue
                
                # Verificar drawdown
                if self.risk_manager.check_daily_drawdown():
                    await send_telegram_signal("🚫 LÍMITE DE PÉRDIDA DIARIA ALCANZADO - Bot detenido")
                    await asyncio.sleep(60)
                    continue
                
                # Analizar cada símbolo
                for symbol in self.symbols_to_monitor:
                    # Verificar nueva vela M5
                    if await self.check_for_new_candle(symbol):
                        logging.info(f"🕐 Nueva vela M5 detectada en {symbol}")
                        
                        # Ejecutar MAS
                        signal = self.mas.check_all_signals(symbol)
                        
                        if signal:
                            logging.info(f"📡 Señal detectada: {symbol} - {signal.source.value}")
                            await self.process_signal(signal)
                
                # Monitorear posiciones existentes
                await self.monitor_positions()
                
                # Esperar antes del próximo ciclo
                await asyncio.sleep(10)
                
            except Exception as e:
                logging.error(f"Error en ciclo de análisis: {e}")
                await asyncio.sleep(10)
    
    async def run(self):
        """Ejecuta el bot"""
        if not await self.initialize():
            return
        
        logging.info("🔄 Iniciando análisis de mercados...")
        
        try:
            await self.run_analysis_cycle()
            
        except KeyboardInterrupt:
            logging.info("🛑 Bot detenido por usuario")
        except Exception as e:
            logging.error(f"❌ Error crítico: {e}")
            await send_telegram_signal(f"❌ <b>ERROR CRÍTICO:</b> {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Limpieza al finalizar"""
        try:
            account = mt5.account_info()
            if account:
                final_balance = account.balance
                session_profit = final_balance - bot_control.daily_start_balance
                
                final_msg = f"""🏁 <b>BOT v4.3 MAS FINALIZADO</b>
━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>RESUMEN DE SESIÓN</b>
• Operaciones ejecutadas: {bot_control.daily_operations}
• Balance inicial: ${bot_control.daily_start_balance:.2f}
• Balance final: ${final_balance:.2f}
• P&L Total: ${session_profit:+.2f}

⚠️ Recuerda cerrar posiciones manualmente si es necesario.

Gracias por usar Trading Bot v4.3 MAS!"""
                
                await send_telegram_signal(final_msg)
            
            if self.telegram_app:
                await self.telegram_app.updater.stop()
                await self.telegram_app.stop()
                await self.telegram_app.shutdown()
            
            mt5.shutdown()
            logging.info("✅ Bot finalizado correctamente")
            
        except Exception as e:
            logging.error(f"❌ Error en cleanup: {e}")

# ===== PUNTO DE ENTRADA =====
if __name__ == "__main__":
    print("🚀 Iniciando Trading Bot v4.3 - MAS")
    print("📊 Características:")
    print("  ✅ Módulo de Análisis y Señales completo")
    print("  ✅ 11 fuentes de señales independientes")
    print("  ✅ Modo Manual y Automático")
    print("  ✅ Gestión de riesgo avanzada")
    print("  ✅ Detección de patrones de velas")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    bot = TradingBotV43()
    asyncio.run(bot.run())
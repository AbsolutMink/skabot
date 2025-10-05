from datetime import datetime, timedelta
from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - entorno de pruebas sin dependencias reales
    import MetaTrader5  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - utilizado en pruebas
    mt5_stub_module = types.ModuleType("MetaTrader5")
    mt5_stub_module.TIMEFRAME_M5 = 0
    mt5_stub_module.ORDER_TYPE_BUY = 0
    mt5_stub_module.ORDER_TYPE_SELL = 1
    mt5_stub_module.TRADE_ACTION_SLTP = 2
    mt5_stub_module.TRADE_ACTION_DEAL = 3
    mt5_stub_module.TRADE_RETCODE_DONE = 10009
    mt5_stub_module.SYMBOL_TRADE_MODE_FULL = 0

    def _return_none(*args, **kwargs):
        return None

    def _return_false(*args, **kwargs):
        return False

    mt5_stub_module.copy_rates_from_pos = _return_none
    mt5_stub_module.symbol_info = _return_none
    mt5_stub_module.account_info = _return_none
    mt5_stub_module.symbol_info_tick = _return_none
    mt5_stub_module.order_send = _return_none
    mt5_stub_module.positions_get = _return_none
    mt5_stub_module.positions_total = _return_none
    mt5_stub_module.initialize = lambda: True
    mt5_stub_module.login = lambda *args, **kwargs: True
    mt5_stub_module.shutdown = _return_none
    mt5_stub_module.symbols_get = _return_none

    sys.modules["MetaTrader5"] = mt5_stub_module

try:  # pragma: no cover - entorno de pruebas sin dependencias reales
    import telegram  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - utilizado en pruebas
    telegram_module = types.ModuleType("telegram")

    class _TelegramDummy:  # pragma: no cover - clase auxiliar
        pass

    telegram_module.Bot = _TelegramDummy
    telegram_module.Update = _TelegramDummy
    sys.modules["telegram"] = telegram_module

    telegram_ext_module = types.ModuleType("telegram.ext")
    telegram_ext_module.Application = _TelegramDummy
    telegram_ext_module.CommandHandler = _TelegramDummy

    class _TelegramContextTypes:  # pragma: no cover - clase auxiliar
        DEFAULT_TYPE = object()

    telegram_ext_module.ContextTypes = _TelegramContextTypes
    sys.modules["telegram.ext"] = telegram_ext_module

import pandas as pd

import bottradingv44 as bot


class Dummy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MT5Stub:
    def __init__(self):
        self._symbol_info = Dummy(
            point=0.01,
            spread=2,
            trade_tick_value=1.0,
            volume_step=0.01,
            digits=5,
            trade_stops_level=0,
        )
        self._account_info = Dummy(balance=10_000.0)

    def symbol_info(self, symbol):
        return self._symbol_info

    def account_info(self):
        return self._account_info


def build_indicator_dataframe(prev_updates=None, curr_updates=None, size=60):
    base_row = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "tick_volume": 1000,
        "rsi": 50.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "ema_fast": 100.0,
        "ema_slow": 100.0,
        "bb_upper": 102.0,
        "bb_lower": 98.0,
        "atr": 1.0,
    }

    rows = [base_row.copy() for _ in range(size - 2)]

    previous_row = base_row.copy()
    if prev_updates:
        previous_row.update(prev_updates)
    rows.append(previous_row)

    current_row = base_row.copy()
    if curr_updates:
        current_row.update(curr_updates)
    rows.append(current_row)

    index = pd.date_range(datetime.now() - timedelta(minutes=size - 1), periods=size, freq="min")
    df = pd.DataFrame(rows, index=index)
    return df


class TestModulo(bot.ModuloAnalisisSeñales):
    def __init__(self, df):
        super().__init__(mt5_conn=None)
        self._df = df

    def fetch_rates(self, symbol, timeframe, count):
        return self._df

    def calculate_all_indicators(self, df):
        return self._df


def setup_mt5_stub(monkeypatch):
    stub = MT5Stub()
    monkeypatch.setattr(bot, "mt5", stub)
    return stub


def test_macd_signal_is_reached(monkeypatch):
    setup_mt5_stub(monkeypatch)
    df = build_indicator_dataframe(
        prev_updates={"macd": -0.1, "macd_signal": 0.1},
        curr_updates={"macd": 0.2, "macd_signal": 0.0},
    )

    modulo = TestModulo(df)
    signal = modulo.check_all_signals("EURUSD")

    assert signal is not None
    assert signal.source == bot.SignalSource.MACD
    assert signal.direction == "BUY"


def test_ema_cross_signal_is_reached(monkeypatch):
    setup_mt5_stub(monkeypatch)
    df = build_indicator_dataframe(
        prev_updates={"ema_fast": 99.0, "ema_slow": 100.0, "macd": 0.05, "macd_signal": 0.05},
        curr_updates={"ema_fast": 101.0, "ema_slow": 100.0, "macd": 0.05, "macd_signal": 0.05},
    )

    modulo = TestModulo(df)
    signal = modulo.check_all_signals("EURUSD")

    assert signal is not None
    assert signal.source == bot.SignalSource.EMA_CROSS
    assert signal.direction == "BUY"


def test_bollinger_signal_is_reached(monkeypatch):
    setup_mt5_stub(monkeypatch)
    df = build_indicator_dataframe(
        prev_updates={"close": 100.0, "bb_upper": 102.0},
        curr_updates={"close": 103.0, "bb_upper": 102.0, "bb_lower": 98.0, "ema_fast": 100.0, "ema_slow": 100.0},
    )

    modulo = TestModulo(df)
    signal = modulo.check_all_signals("EURUSD")

    assert signal is not None
    assert signal.source == bot.SignalSource.BOLLINGER
    assert signal.direction == "BUY"

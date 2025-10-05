import sys
import types
from datetime import datetime, timedelta
import unittest

# Stub pandas and numpy modules used only for typing in tests
pandas_stub = types.ModuleType('pandas')
pandas_stub.DataFrame = object
pandas_stub.Series = object
pandas_stub.Timestamp = object
pandas_stub.isna = staticmethod(lambda x: False)
sys.modules.setdefault('pandas', pandas_stub)

numpy_stub = types.ModuleType('numpy')
numpy_stub.ndarray = object
sys.modules.setdefault('numpy', numpy_stub)

# Stub MetaTrader5 module
mt5_stub = types.SimpleNamespace(
    TIMEFRAME_M5=0,
    ORDER_TYPE_BUY=0,
    ORDER_TYPE_SELL=1,
    TRADE_ACTION_SLTP=2,
    TRADE_ACTION_DEAL=3,
    TRADE_RETCODE_DONE=10009,
    account_info=lambda: None,
    positions_get=lambda: [],
    symbol_info=lambda symbol: None,
    symbol_info_tick=lambda symbol: None,
    copy_rates_from_pos=lambda *args, **kwargs: None,
    order_send=lambda request: None
)
sys.modules.setdefault('MetaTrader5', mt5_stub)

# Stub telegram modules required for import
telegram_module = types.ModuleType('telegram')
telegram_module.Bot = object
telegram_module.Update = object
sys.modules.setdefault('telegram', telegram_module)

tg_ext_module = types.ModuleType('telegram.ext')
tg_ext_module.Application = object
tg_ext_module.CommandHandler = object

class DummyContextTypes:
    DEFAULT_TYPE = object

tg_ext_module.ContextTypes = DummyContextTypes
sys.modules.setdefault('telegram.ext', tg_ext_module)

# Import module under test
from bottradingv44 import RiskManager, TradingConfig


class DummyPosition:
    def __init__(self, ticket, open_time, sl=0.0, tp=0.0, order_type=0, symbol='EURUSD'):
        self.ticket = ticket
        self.time = open_time
        self.sl = sl
        self.tp = tp
        self.type = order_type
        self.symbol = symbol


class RiskManagerTimeLimitTests(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()

    def test_existing_stops_are_recorded(self):
        open_timestamp = (datetime.now() - timedelta(minutes=10)).timestamp()
        position = DummyPosition(ticket=1, open_time=open_timestamp, sl=1.0, tp=2.0)

        # Precondition: no data stored yet
        self.assertNotIn(position.ticket, self.risk_manager.positions_data)

        # Should record data even when SL/TP already set
        self.risk_manager.apply_risk_management(position)
        self.assertIn(position.ticket, self.risk_manager.positions_data)

        stored = self.risk_manager.positions_data[position.ticket]
        self.assertAlmostEqual(stored['open_time'].timestamp(), open_timestamp, delta=1)
        self.assertEqual(stored['sl'], position.sl)
        self.assertEqual(stored['tp'], position.tp)

    def test_check_time_limit_records_and_detects_expiry(self):
        open_timestamp = (datetime.now() - timedelta(minutes=TradingConfig.MAX_TRADE_DURATION + 5)).timestamp()
        position = DummyPosition(ticket=2, open_time=open_timestamp, sl=1.0, tp=2.0)

        exceeded = self.risk_manager.check_time_limit(position)

        self.assertTrue(exceeded)
        self.assertIn(position.ticket, self.risk_manager.positions_data)

    def test_check_time_limit_allows_recent_trade(self):
        open_timestamp = datetime.now().timestamp()
        position = DummyPosition(ticket=3, open_time=open_timestamp, sl=1.0, tp=2.0)

        within_limit = self.risk_manager.check_time_limit(position)

        self.assertFalse(within_limit)
        self.assertIn(position.ticket, self.risk_manager.positions_data)


if __name__ == '__main__':
    unittest.main()

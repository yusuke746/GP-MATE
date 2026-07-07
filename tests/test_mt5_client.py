from __future__ import annotations

from types import SimpleNamespace

from data import mt5_client


class _FakeStruct:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def _asdict(self) -> dict[str, object]:
        return dict(self._kwargs)


class _FakeMt5:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    TRADE_ACTION_SLTP = 11
    ORDER_TIME_GTC = 20
    ORDER_FILLING_IOC = 30
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    def __init__(self, positions: list[_FakeStruct] | None = None, order_result: _FakeStruct | None = None) -> None:
        self._positions = positions or []
        self._order_result = order_result or _FakeStruct(retcode=self.TRADE_RETCODE_DONE, order=111, deal=222)
        self.sent_requests: list[dict[str, object]] = []

    def positions_get(self, symbol: str | None = None):
        if symbol is None:
            return list(self._positions)
        return [p for p in self._positions if p._asdict().get("symbol") == symbol]

    def order_send(self, request: dict[str, object]):
        self.sent_requests.append(dict(request))
        return self._order_result

    def symbol_info(self, symbol: str):
        return SimpleNamespace(visible=True)

    def symbol_select(self, symbol: str, visible: bool):
        return True

    def symbol_info_tick(self, symbol: str):
        return SimpleNamespace(ask=2350.5, bid=2349.5)

    def last_error(self):
        return (1, "fake error")


def test_get_positions_keeps_raw_shape(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(
        positions=[
            _FakeStruct(
                ticket=123,
                symbol="GOLD#",
                type=_FakeMt5.POSITION_TYPE_BUY,
                volume=0.2,
                price_open=2300.0,
                price_current=2310.0,
                sl=2280.0,
                tp=2340.0,
                profit=100.0,
            )
        ]
    )
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    positions = mt5_client.get_positions("GOLD#")

    assert len(positions) == 1
    assert positions[0]["ticket"] == 123
    assert positions[0]["type"] == _FakeMt5.POSITION_TYPE_BUY


def test_get_position_details_returns_required_fields(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(
        positions=[
            _FakeStruct(
                ticket=123,
                symbol="GOLD#",
                type=_FakeMt5.POSITION_TYPE_BUY,
                volume=0.2,
                price_open=2300.0,
                price_current=2310.0,
                sl=2280.0,
                tp=2340.0,
                profit=100.0,
            )
        ]
    )
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    positions = mt5_client.get_position_details("GOLD#")

    assert positions == [
        {
            "ticket": 123,
            "symbol": "GOLD#",
            "type": "BUY",
            "volume": 0.2,
            "price_open": 2300.0,
            "price_current": 2310.0,
            "sl": 2280.0,
            "tp": 2340.0,
            "profit": 100.0,
            "raw": {
                "ticket": 123,
                "symbol": "GOLD#",
                "type": 0,
                "volume": 0.2,
                "price_open": 2300.0,
                "price_current": 2310.0,
                "sl": 2280.0,
                "tp": 2340.0,
                "profit": 100.0,
            },
        }
    ]


def test_close_position_builds_sell_request_for_buy_position(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(
        positions=[
            _FakeStruct(
                ticket=123,
                symbol="GOLD#",
                type=_FakeMt5.POSITION_TYPE_BUY,
                volume=0.2,
                price_open=2300.0,
                price_current=2310.0,
                sl=2280.0,
                tp=2340.0,
                profit=100.0,
            )
        ]
    )
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    result = mt5_client.close_position(123)

    assert result["success"] is True
    assert result["action"] == "SELL"
    assert fake_mt5.sent_requests[0]["action"] == _FakeMt5.TRADE_ACTION_DEAL
    assert fake_mt5.sent_requests[0]["type"] == _FakeMt5.ORDER_TYPE_SELL
    assert fake_mt5.sent_requests[0]["position"] == 123
    assert fake_mt5.sent_requests[0]["price"] == 2349.5


def test_close_position_builds_buy_request_for_sell_position(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(
        positions=[
            _FakeStruct(
                ticket=456,
                symbol="GOLD#",
                type=_FakeMt5.POSITION_TYPE_SELL,
                volume=0.3,
                price_open=2305.0,
                price_current=2295.0,
                sl=2320.0,
                tp=2280.0,
                profit=90.0,
            )
        ]
    )
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    result = mt5_client.close_position(456)

    assert result["success"] is True
    assert result["action"] == "BUY"
    assert fake_mt5.sent_requests[0]["type"] == _FakeMt5.ORDER_TYPE_BUY
    assert fake_mt5.sent_requests[0]["position"] == 456
    assert fake_mt5.sent_requests[0]["price"] == 2350.5


def test_close_position_fails_safely_when_ticket_missing(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(positions=[])
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    result = mt5_client.close_position(999)

    assert result["success"] is False
    assert result["reason"] == "Position not found: 999"
    assert fake_mt5.sent_requests == []


def test_modify_sl_builds_sltp_request(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(
        positions=[
            _FakeStruct(
                ticket=123,
                symbol="GOLD#",
                type=_FakeMt5.POSITION_TYPE_BUY,
                volume=0.2,
                price_open=2300.0,
                price_current=2310.0,
                sl=2280.0,
                tp=2340.0,
                profit=100.0,
            )
        ]
    )
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    result = mt5_client.modify_sl(123, 2290.5)

    assert result["success"] is True
    assert result["action"] == "MODIFY_SL"
    assert fake_mt5.sent_requests[0] == {
        "action": _FakeMt5.TRADE_ACTION_SLTP,
        "symbol": "GOLD#",
        "position": 123,
        "sl": 2290.5,
        "tp": 2340.0,
        "magic": mt5_client.ORDER_MAGIC,
        "comment": "GP-MATE modify_sl",
    }


def test_modify_sl_fails_safely_when_ticket_missing(monkeypatch) -> None:
    fake_mt5 = _FakeMt5(positions=[])
    monkeypatch.setattr(mt5_client, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    result = mt5_client.modify_sl(999, 2290.5)

    assert result["success"] is False
    assert result["reason"] == "Position not found: 999"
    assert fake_mt5.sent_requests == []
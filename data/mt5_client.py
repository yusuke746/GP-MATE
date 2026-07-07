from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

import pandas as pd

from config import MT5_LOGIN, MT5_PASSWORD, MT5_PATH, MT5_SERVER

LOGGER = logging.getLogger(__name__)

ORDER_MAGIC = 20260702
ORDER_DEVIATION = 20

try:
    import MetaTrader5 as mt5  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - environment dependent
    mt5 = None


@dataclass(frozen=True)
class AccountInfo:
    login: int
    trade_mode: int
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str


def _position_type_name(position_type: Any) -> str:
    try:
        normalized = int(position_type)
    except Exception:
        return "UNKNOWN"

    buy_value = getattr(mt5, "POSITION_TYPE_BUY", 0) if mt5 is not None else 0
    sell_value = getattr(mt5, "POSITION_TYPE_SELL", 1) if mt5 is not None else 1
    if normalized == buy_value:
        return "BUY"
    if normalized == sell_value:
        return "SELL"
    return f"OTHER({normalized})"


def _normalize_position_dict(position: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket": int(position.get("ticket") or 0),
        "symbol": str(position.get("symbol") or ""),
        "type": _position_type_name(position.get("type")),
        "volume": float(position.get("volume") or 0.0),
        "price_open": float(position.get("price_open") or 0.0),
        "price_current": float(position.get("price_current") or 0.0),
        "sl": float(position.get("sl") or 0.0),
        "tp": float(position.get("tp") or 0.0),
        "profit": float(position.get("profit") or 0.0),
        "raw": dict(position),
    }


def _done_codes() -> set[Any]:
    if mt5 is None:
        return set()
    return {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED}


def _find_raw_position_by_ticket(ticket: int) -> dict[str, Any] | None:
    if mt5 is None:
        return None

    positions = mt5.positions_get()
    if positions is None:
        LOGGER.error("positions_get failed: %s", mt5.last_error())
        return None

    for position in positions:
        position_dict = position._asdict()
        if int(position_dict.get("ticket") or 0) == ticket:
            return position_dict

    return None


def _submit_mt5_request(request: dict[str, Any], action_label: str) -> dict[str, Any]:
    if mt5 is None:
        return {
            "success": False,
            "action": action_label,
            "reason": "MetaTrader5 package unavailable",
            "retcode": None,
        }

    result = mt5.order_send(request)
    if result is None:
        return {
            "success": False,
            "action": action_label,
            "reason": f"order_send returned None: {mt5.last_error()}",
            "retcode": None,
            "request": request,
        }

    result_dict = result._asdict()
    retcode = result_dict.get("retcode")
    return {
        "success": retcode in _done_codes(),
        "action": action_label,
        "retcode": retcode,
        "order": result_dict.get("order"),
        "deal": result_dict.get("deal"),
        "request": request,
        "raw": result_dict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _timeframe_to_mt5_constant(timeframe: str) -> int | None:
    if mt5 is None:
        return None

    mapping: dict[str, int] = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(timeframe.upper())


def _initialize_mt5() -> bool:
    if mt5 is None:
        LOGGER.error("MetaTrader5 package is not available")
        return False

    init_kwargs: dict[str, str] = {}
    if MT5_PATH:
        init_kwargs["path"] = MT5_PATH

    if not mt5.initialize(**init_kwargs):
        LOGGER.error("MT5 initialize failed: %s", mt5.last_error())
        return False

    if MT5_LOGIN is not None and MT5_PASSWORD and MT5_SERVER:
        if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            LOGGER.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

    return True


def _ensure_symbol(symbol: str) -> bool:
    if mt5 is None:
        return False
    info = mt5.symbol_info(symbol)
    if info is not None and info.visible:
        return True
    if info is not None:
        return bool(mt5.symbol_select(symbol, True))
    return False


def connect() -> bool:
    """Initialize and login to MT5 terminal."""
    try:
        return _initialize_mt5()
    except Exception as exc:
        LOGGER.exception("MT5 connect exception: %s", exc)
        return False


def disconnect() -> None:
    """Shutdown MT5 terminal connection."""
    if mt5 is None:
        return
    try:
        mt5.shutdown()
    except Exception as exc:
        LOGGER.exception("MT5 shutdown exception: %s", exc)


def get_rates(symbol: str, tf: str, count: int) -> pd.DataFrame:
    """Fetch OHLCV rates from MT5 and return a pandas DataFrame."""
    if count <= 0:
        return pd.DataFrame()

    if not connect():
        return pd.DataFrame()

    try:
        if mt5 is None:
            return pd.DataFrame()

        mt5_tf = _timeframe_to_mt5_constant(tf)
        if mt5_tf is None:
            LOGGER.error("Unsupported timeframe: %s", tf)
            return pd.DataFrame()

        if not _ensure_symbol(symbol):
            LOGGER.error("Symbol not available on MT5: %s", symbol)
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None:
            LOGGER.error("copy_rates_from_pos failed: %s", mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        if df.empty:
            return df

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
    except Exception as exc:
        LOGGER.exception("get_rates exception: %s", exc)
        return pd.DataFrame()
    finally:
        disconnect()


def get_spread(symbol: str) -> float:
    """Get current spread in points for the given symbol."""
    if not connect():
        return -1.0

    try:
        if mt5 is None:
            return -1.0

        if not _ensure_symbol(symbol):
            LOGGER.error("Symbol not available on MT5: %s", symbol)
            return -1.0

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            LOGGER.error("Failed to get symbol tick/info: %s", symbol)
            return -1.0

        if info.point <= 0:
            return float(tick.ask - tick.bid)

        return float((tick.ask - tick.bid) / info.point)
    except Exception as exc:
        LOGGER.exception("get_spread exception: %s", exc)
        return -1.0
    finally:
        disconnect()


def get_baseline_spread(
    symbol: str,
    samples: int = 20,
    interval_sec: float = 0.5,
) -> float | None:
    """Measure normal spread baseline from recent samples using median.

    Returns None when measurement fails or valid samples are insufficient.
    """
    if samples <= 0:
        return None

    valid_spreads: list[float] = []
    for idx in range(samples):
        try:
            spread = get_spread(symbol)
        except Exception as exc:
            LOGGER.warning("get_baseline_spread sample failed: %s", exc)
            spread = -1.0

        if isinstance(spread, (int, float)) and spread >= 0:
            valid_spreads.append(float(spread))

        if idx < samples - 1 and interval_sec > 0:
            time.sleep(interval_sec)

    if len(valid_spreads) < max(1, samples // 2):
        return None

    return float(median(valid_spreads))


def get_positions(symbol: str) -> list[dict[str, Any]]:
    """Get open positions for a symbol."""
    if not connect():
        return []

    try:
        if mt5 is None:
            return []

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            LOGGER.error("positions_get failed: %s", mt5.last_error())
            return []

        return [p._asdict() for p in positions]
    except Exception as exc:
        LOGGER.exception("get_positions exception: %s", exc)
        return []
    finally:
        disconnect()


def get_position_details(symbol: str) -> list[dict[str, Any]]:
    """Get normalized open position details for a symbol."""
    raw_positions = get_positions(symbol)
    return [_normalize_position_dict(position) for position in raw_positions]


def get_closed_deals(symbol: str, since: datetime) -> list[dict[str, Any]]:
    """Get closed deals (realized PnL) since a given datetime.

    Returns list of dictionaries with realized fields including `profit`.
    """
    if not connect():
        return []

    try:
        if mt5 is None:
            return []

        if since.tzinfo is None:
            since_utc = since.replace(tzinfo=timezone.utc)
        else:
            since_utc = since.astimezone(timezone.utc)

        now_utc = datetime.now(timezone.utc)
        if now_utc <= since_utc:
            return []

        # Pull a slightly wider window to estimate holding time from opening deal.
        history_from = since_utc - timedelta(days=30)
        all_deals = mt5.history_deals_get(history_from, now_utc)
        if all_deals is None:
            LOGGER.error("history_deals_get failed: %s", mt5.last_error())
            return []

        open_by_position: dict[int, tuple[datetime, float]] = {}
        for raw in all_deals:
            deal = raw._asdict()
            if str(deal.get("symbol", "")).upper() != symbol.upper():
                continue

            position_id = int(deal.get("position_id") or 0)
            if position_id <= 0:
                continue

            entry_type = int(deal.get("entry") or -1)
            if entry_type != mt5.DEAL_ENTRY_IN:
                continue

            opened_at = datetime.fromtimestamp(int(deal.get("time") or 0), tz=timezone.utc)
            opened_price = float(deal.get("price") or 0.0)
            prev = open_by_position.get(position_id)
            if prev is None or opened_at < prev[0]:
                open_by_position[position_id] = (opened_at, opened_price)

        closed: list[dict[str, Any]] = []
        for raw in all_deals:
            deal = raw._asdict()
            if str(deal.get("symbol", "")).upper() != symbol.upper():
                continue

            closed_at = datetime.fromtimestamp(int(deal.get("time") or 0), tz=timezone.utc)
            if closed_at < since_utc:
                continue

            entry_type = int(deal.get("entry") or -1)
            if entry_type not in {mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY, mt5.DEAL_ENTRY_INOUT}:
                continue

            position_id = int(deal.get("position_id") or 0)
            opened_info = open_by_position.get(position_id)
            holding_seconds = 0
            entry_price = 0.0
            if opened_info is not None:
                opened_at, entry_price = opened_info
                if closed_at >= opened_at:
                    holding_seconds = int((closed_at - opened_at).total_seconds())

            deal_type = int(deal.get("type") or -1)
            action = "BUY" if deal_type == mt5.DEAL_TYPE_BUY else "SELL"

            closed.append(
                {
                    "deal_id": int(deal.get("ticket") or 0),
                    "position_id": position_id,
                    "symbol": str(deal.get("symbol") or symbol),
                    "action": action,
                    "entry": entry_type,
                    "entry_price": entry_price,
                    "exit_price": float(deal.get("price") or 0.0),
                    "lot": float(deal.get("volume") or 0.0),
                    "profit": float(deal.get("profit") or 0.0),
                    "commission": float(deal.get("commission") or 0.0),
                    "swap": float(deal.get("swap") or 0.0),
                    "fee": float(deal.get("fee") or 0.0),
                    "holding_seconds": holding_seconds,
                    "time_utc": closed_at.isoformat(),
                }
            )

        closed.sort(key=lambda x: str(x.get("time_utc", "")))
        return closed
    except Exception as exc:
        LOGGER.exception("get_closed_deals exception: %s", exc)
        return []
    finally:
        disconnect()


def send_order(symbol: str, action: str, lot: float, sl: float, tp: float) -> dict[str, Any]:
    """Send market order to MT5. Returns a dict-like result for logging."""
    normalized_action = action.upper().strip()
    if normalized_action == "HOLD":
        return {
            "success": False,
            "action": "HOLD",
            "reason": "HOLD action does not send order",
            "retcode": None,
        }

    if normalized_action not in {"BUY", "SELL"}:
        return {
            "success": False,
            "action": normalized_action,
            "reason": "Invalid action",
            "retcode": None,
        }

    if lot <= 0:
        return {
            "success": False,
            "action": normalized_action,
            "reason": "Lot must be positive",
            "retcode": None,
        }

    if not connect():
        return {
            "success": False,
            "action": normalized_action,
            "reason": "MT5 connection failed",
            "retcode": None,
        }

    try:
        if mt5 is None:
            return {
                "success": False,
                "action": normalized_action,
                "reason": "MetaTrader5 package unavailable",
                "retcode": None,
            }

        if not _ensure_symbol(symbol):
            return {
                "success": False,
                "action": normalized_action,
                "reason": f"Symbol unavailable: {symbol}",
                "retcode": None,
            }

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {
                "success": False,
                "action": normalized_action,
                "reason": "Failed to get symbol tick",
                "retcode": None,
            }

        order_type = mt5.ORDER_TYPE_BUY if normalized_action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if normalized_action == "BUY" else tick.bid

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": ORDER_DEVIATION,
            "magic": ORDER_MAGIC,
            "comment": "GP-MATE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        response = _submit_mt5_request(request, normalized_action)
        response.update(
            {
                "volume": lot,
                "price": price,
                "sl": sl,
                "tp": tp,
            }
        )
        return response
    except Exception as exc:
        LOGGER.exception("send_order exception: %s", exc)
        return {
            "success": False,
            "action": normalized_action,
            "reason": str(exc),
            "retcode": None,
        }
    finally:
        disconnect()


def close_position(ticket: int) -> dict[str, Any]:
    """Close an open position by ticket with an opposing market order."""
    if ticket <= 0:
        return {
            "success": False,
            "action": "CLOSE_POSITION",
            "reason": "Ticket must be positive",
            "retcode": None,
        }

    if not connect():
        return {
            "success": False,
            "action": "CLOSE_POSITION",
            "reason": "MT5 connection failed",
            "retcode": None,
        }

    try:
        if mt5 is None:
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": "MetaTrader5 package unavailable",
                "retcode": None,
            }

        position = _find_raw_position_by_ticket(ticket)
        if position is None:
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": f"Position not found: {ticket}",
                "retcode": None,
            }

        symbol = str(position.get("symbol") or "")
        if not symbol or not _ensure_symbol(symbol):
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": f"Symbol unavailable: {symbol}",
                "retcode": None,
            }

        current_type = _position_type_name(position.get("type"))
        if current_type == "BUY":
            close_action = "SELL"
            order_type = mt5.ORDER_TYPE_SELL
        elif current_type == "SELL":
            close_action = "BUY"
            order_type = mt5.ORDER_TYPE_BUY
        else:
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": f"Unsupported position type: {position.get('type')}",
                "retcode": None,
            }

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": "Failed to get symbol tick",
                "retcode": None,
            }

        price = float(tick_info.bid if current_type == "BUY" else tick_info.ask)
        volume = float(position.get("volume") or 0.0)
        if volume <= 0:
            return {
                "success": False,
                "action": "CLOSE_POSITION",
                "reason": "Position volume must be positive",
                "retcode": None,
            }

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": int(ticket),
            "price": price,
            "deviation": ORDER_DEVIATION,
            "magic": ORDER_MAGIC,
            "comment": "GP-MATE close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        response = _submit_mt5_request(request, close_action)
        response.update(
            {
                "position": int(ticket),
                "symbol": symbol,
                "volume": volume,
                "price": price,
            }
        )
        return response
    except Exception as exc:
        LOGGER.exception("close_position exception: %s", exc)
        return {
            "success": False,
            "action": "CLOSE_POSITION",
            "reason": str(exc),
            "retcode": None,
        }
    finally:
        disconnect()


def modify_sl(ticket: int, new_sl: float) -> dict[str, Any]:
    """Modify stop loss for an open position while preserving current TP."""
    if ticket <= 0:
        return {
            "success": False,
            "action": "MODIFY_SL",
            "reason": "Ticket must be positive",
            "retcode": None,
        }

    if new_sl <= 0:
        return {
            "success": False,
            "action": "MODIFY_SL",
            "reason": "new_sl must be positive",
            "retcode": None,
        }

    if not connect():
        return {
            "success": False,
            "action": "MODIFY_SL",
            "reason": "MT5 connection failed",
            "retcode": None,
        }

    try:
        if mt5 is None:
            return {
                "success": False,
                "action": "MODIFY_SL",
                "reason": "MetaTrader5 package unavailable",
                "retcode": None,
            }

        position = _find_raw_position_by_ticket(ticket)
        if position is None:
            return {
                "success": False,
                "action": "MODIFY_SL",
                "reason": f"Position not found: {ticket}",
                "retcode": None,
            }

        symbol = str(position.get("symbol") or "")
        if not symbol or not _ensure_symbol(symbol):
            return {
                "success": False,
                "action": "MODIFY_SL",
                "reason": f"Symbol unavailable: {symbol}",
                "retcode": None,
            }

        current_tp = float(position.get("tp") or 0.0)
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(ticket),
            "sl": float(new_sl),
            "tp": current_tp,
            "magic": ORDER_MAGIC,
            "comment": "GP-MATE modify_sl",
        }

        response = _submit_mt5_request(request, "MODIFY_SL")
        response.update(
            {
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": current_tp,
            }
        )
        return response
    except Exception as exc:
        LOGGER.exception("modify_sl exception: %s", exc)
        return {
            "success": False,
            "action": "MODIFY_SL",
            "reason": str(exc),
            "retcode": None,
        }
    finally:
        disconnect()


def get_account_info() -> dict[str, Any]:
    """Fetch account information from MT5 as a dictionary."""
    if not connect():
        return {
            "success": False,
            "reason": "MT5 connection failed",
        }

    try:
        if mt5 is None:
            return {
                "success": False,
                "reason": "MetaTrader5 package unavailable",
            }

        account_info = mt5.account_info()
        if account_info is None:
            return {
                "success": False,
                "reason": f"account_info failed: {mt5.last_error()}",
            }

        account = AccountInfo(
            login=int(account_info.login),
            trade_mode=int(account_info.trade_mode),
            server=str(account_info.server),
            balance=float(account_info.balance),
            equity=float(account_info.equity),
            margin=float(account_info.margin),
            free_margin=float(account_info.margin_free),
            margin_level=float(account_info.margin_level),
            currency=str(account_info.currency),
        )
        return {
            "success": True,
            "data": account,
        }
    except Exception as exc:
        LOGGER.exception("get_account_info exception: %s", exc)
        return {
            "success": False,
            "reason": str(exc),
        }
    finally:
        disconnect()

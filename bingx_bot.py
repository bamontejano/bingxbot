#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import ccxt.async_support as ccxt

# ============================================================
# CONFIG
# ============================================================

TIMEFRAME = os.getenv("TIMEFRAME", "1m")
POLL_SEC = int(os.getenv("POLL_SEC", "60"))

QUOTE = "USDT"

# Lista fija (bases). Se resolverán solo los swaps disponibles en BingX.
TOP10_BASES = [
    "BTC","ETH","SOL","XRP","DOGE",
    "OP","ARB","APT","SUI","INJ",
    "SEI","WIF","PEPE","RNDR","IMX",
    "FET","TIA","NEAR","AVAX","LINK",
    "ADA","TON","BOME","ORDI","JUP"
]

# Indicadores
RSI_LEN = 14
VWAP_LEN = 120
BB_LEN = 20
BB_K = 2.0
ATR_LEN = 14
VOL_MA_LEN = 20

# Filtros base (anti-ruido)
MIN_ATR_PCT = 0.0012        # 0.12% mínimo para operar en NORMAL (sube si quieres menos ruido)
MIN_BB_WIDTH = 0.0035       # ancho BB mínimo
VOL_MULT = 1.55             # volumen actual >= VOL_MULT * vol_ma
MAX_SPREAD_BPS = 40.0       # bloqueo duro por spread absoluto (bps)

# Gates de productividad
SPREAD_TP_MAX_RATIO = 0.30  # spread_pct / tp_pct <= 0.30

# Rate limiting soft (evitar spam)
MIN_SCORE_TO_ALERT = 62
HIGH_SCORE_OVERRIDE = 85
TARGET_ALERTS_PER_HOUR = 4

ONLY_ALERT_ON_CHANGE = False
SIDE_COOLDOWN_SEC = 300  # 5 min por símbolo y lado

# Runner / expansión (expansion label independiente del modo C)
EXP_ATR_PCT = 0.0020      # 0.20% ATR en 1m -> EXP
EXP_TP2_MULT = 4.0        # TP2 = ATR * 4 (sugerido)
EXP_TRAIL_ATR = 1.0       # trail ≈ ATR * 1 (sugerido)

# Circuit breaker (anti-compresión sostenida por universo de símbolos)
LOW_ATR_WARN = 0.0012         # 0.12% (warn de compresión)
LOW_ATR_FRAC_TRIGGER = 0.75   # >=75% símbolos con ATR% bajo -> compresión
LOW_ATR_CONSEC_POLLS = 3      # durante 3 ciclos seguidos
PAUSE_SEC = 1800              # pausa 30 min
REGIME_WINDOW_SEC = 20 * 60   # ventana 20 min

# ============================================================
# MODO C: NORMAL + AGRESIVO AUTO (para capital pequeño)
# ============================================================

MODE_AUTO = True

# Activación del modo agresivo (expansión real)
AGGR_MIN_ATR_PCT = 0.0016      # 0.16% mínimo para agresivo
AGGR_ATR_RISE_MULT = 1.25      # ATR actual > ATR media (20) * 1.25
AGGR_MIN_BB_WIDTH = 0.0045     # BB width mínimo en agresivo
AGGR_VOL_MULT = 1.80           # volumen más exigente en agresivo

# Parametrización por modo
NORMAL_TP_MULT = 2.0
NORMAL_SL_MULT = 1.2
NORMAL_RSI_MOMENTUM = 50       # LONG >=50 / SHORT <=50

AGGR_TP_MULT = 2.8
AGGR_SL_MULT = 1.0
AGGR_RSI_MOMENTUM = 55         # LONG >=55 / SHORT <=45

# ============================================================
# Telegram
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _need_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Falta variable de entorno: {name}")


async def telegram_send(text: str, session: aiohttp.ClientSession) -> None:
    _need_env("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    _need_env("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    async with session.post(url, json=payload, timeout=20) as r:
        data = await r.json()
        if not data.get("ok", False):
            raise RuntimeError(f"Telegram sendMessage error: {data}")


# ============================================================
# Indicators
# ============================================================

def ema(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n:
        return np.full_like(x, np.nan, dtype=float)
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    if len(close) < n + 1:
        return np.full_like(close, np.nan, dtype=float)
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    ru = ema(up, n)
    rd = ema(down, n)
    rs = ru / np.where(rd == 0, np.nan, rd)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    if len(c) < n + 1:
        return np.full_like(c, np.nan, dtype=float)
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    return ema(tr, n)


def vwap(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    if len(c) < n:
        return np.full_like(c, np.nan, dtype=float)
    tp = (h + l + c) / 3.0
    out = np.full_like(c, np.nan, dtype=float)
    for i in range(n - 1, len(c)):
        vv = v[i - n + 1 : i + 1]
        denom = float(np.sum(vv))
        if denom <= 0:
            out[i] = np.nan
        else:
            out[i] = float(np.sum(tp[i - n + 1 : i + 1] * vv) / denom)
    return out


def bb_width(close: np.ndarray, n: int, k: float) -> np.ndarray:
    if len(close) < n:
        return np.full_like(close, np.nan, dtype=float)
    out = np.full_like(close, np.nan, dtype=float)
    for i in range(n - 1, len(close)):
        w = close[i - n + 1 : i + 1]
        m = float(np.mean(w))
        sd = float(np.std(w))
        upper = m + k * sd
        lower = m - k * sd
        if m == 0:
            out[i] = np.nan
        else:
            out[i] = (upper - lower) / abs(m)
    return out


# ============================================================
# Signal
# ============================================================

@dataclass
class Signal:
    symbol: str
    side: str            # LONG / SHORT / NONE
    quality: str         # A / B / C
    score: int
    price: float
    tp: Optional[float]
    sl: Optional[float]
    spread_bps: float
    atr_value: float
    atr_pct: float
    rsi: float
    vwap: float
    ts_ms: int


# ============================================================
# Bot
# ============================================================

class BingXSignals:
    def __init__(self) -> None:
        self.ex = ccxt.bingx({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.session = aiohttp.ClientSession()
        self.symbols: List[str] = []

        self._last_alert_side: Dict[str, str] = {}
        self._last_alert_ts: Dict[Tuple[str, str], float] = {}
        self._alerts_ts: List[float] = []  # timestamps alerts last hour

        # Circuit breaker state
        self.pause_until = 0.0
        self._regime_hist: List[Tuple[float, float]] = []  # (ts, frac_low_atr)

    async def init(self) -> None:
        await self.ex.load_markets()
        self.symbols = self._resolve_fixed_swaps(TOP10_BASES, quote=QUOTE)
        print(f"Símbolos FUTUROS encontrados en BingX: {self.symbols}")

        try:
            await telegram_send(
                f"🤖 Bot iniciado. Símbolos={len(self.symbols)} | TF={TIMEFRAME} | AUTO={'ON' if MODE_AUTO else 'OFF'}",
                self.session
            )
        except Exception as e:
            print("No pude enviar mensaje de arranque a Telegram:", repr(e))

    def _resolve_fixed_swaps(self, bases: List[str], quote: str) -> List[str]:
        out: List[str] = []
        markets = self.ex.markets or {}
        wanted = set(bases)

        for sym, m in markets.items():
            try:
                if not m.get("swap"):
                    continue
                if m.get("linear") is False:
                    continue
                if m.get("quote") != quote:
                    continue
                base = m.get("base")
                if base in wanted:
                    out.append(sym)
            except Exception:
                continue

        def key_fn(s: str) -> int:
            m = markets.get(s, {})
            b = m.get("base", "")
            try:
                return bases.index(b)
            except ValueError:
                return 9999

        out = sorted(list(dict.fromkeys(out)), key=key_fn)
        return out

    def _soft_rate_limit_ok(self, score: int) -> bool:
        now = time.time()
        self._alerts_ts = [t for t in self._alerts_ts if now - t < 3600]
        if len(self._alerts_ts) < TARGET_ALERTS_PER_HOUR:
            return True
        return score >= HIGH_SCORE_OVERRIDE

    def _cooldown_ok(self, symbol: str, side: str) -> bool:
        now = time.time()
        k = (symbol, side)
        last = self._last_alert_ts.get(k, 0.0)
        return (now - last) >= SIDE_COOLDOWN_SEC

    def _should_alert(self, sig: Signal) -> bool:
        if sig.side == "NONE":
            return False
        if sig.score < MIN_SCORE_TO_ALERT:
            return False
        if sig.spread_bps > MAX_SPREAD_BPS:
            return False
        if ONLY_ALERT_ON_CHANGE and self._last_alert_side.get(sig.symbol) == sig.side:
            return False
        if not self._cooldown_ok(sig.symbol, sig.side):
            return False
        if not self._soft_rate_limit_ok(sig.score):
            return False
        return True

    def _score_quality(self, atr_pct: float, spread_bps: float, vol_ratio: float, bbw: float) -> Tuple[int, str]:
        score = 0

        # volatilidad
        if atr_pct >= 0.0030:
            score += 30
        elif atr_pct >= 0.0020:
            score += 24
        elif atr_pct >= 0.0012:
            score += 18
        else:
            score += 8

        # spread
        if spread_bps <= 2:
            score += 20
        elif spread_bps <= 6:
            score += 16
        elif spread_bps <= 12:
            score += 10
        else:
            score += 4

        # volumen
        if vol_ratio >= 2.0:
            score += 22
        elif vol_ratio >= 1.6:
            score += 18
        elif vol_ratio >= 1.3:
            score += 12
        else:
            score += 6

        # bb width
        if not np.isnan(bbw):
            if bbw >= 0.008:
                score += 18
            elif bbw >= 0.005:
                score += 12
            elif bbw >= 0.003:
                score += 8
            else:
                score += 3

        if score >= 80:
            q = "A"
        elif score >= 65:
            q = "B"
        else:
            q = "C"
        return int(score), q

    def compute_signal(
        self,
        symbol: str,
        spread_bps: float,
        ts_ms: int,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        tp_mult: float,
        sl_mult: float,
        rsi_gate: float,
    ) -> Signal:
        last_price = float(c[-1])

        r = rsi(c, RSI_LEN)
        a = atr(o, h, l, c, ATR_LEN)
        vw = vwap(h, l, c, v, VWAP_LEN)
        bbw = bb_width(c, BB_LEN, BB_K)

        rsi_last = float(r[-1]) if not np.isnan(r[-1]) else np.nan
        atr_last = float(a[-1]) if not np.isnan(a[-1]) else np.nan
        vwap_last = float(vw[-1]) if not np.isnan(vw[-1]) else np.nan
        bbw_last = float(bbw[-1]) if not np.isnan(bbw[-1]) else np.nan

        atr_pct = float(atr_last / last_price) if last_price > 0 and not np.isnan(atr_last) else 0.0

        vol_ma = float(np.mean(v[-VOL_MA_LEN:])) if len(v) >= VOL_MA_LEN else float(np.mean(v))
        vol_ratio = float(v[-1] / vol_ma) if vol_ma > 0 else 0.0

        score, quality = self._score_quality(atr_pct, spread_bps, vol_ratio, bbw_last)

        # Señal base: reclaim VWAP + momentum gate (por modo)
        side = "NONE"
        if not np.isnan(vwap_last) and len(c) >= 3 and not np.isnan(rsi_last):
            reclaim_up = (c[-2] < vw[-2]) and (c[-1] > vwap_last)
            reclaim_down = (c[-2] > vw[-2]) and (c[-1] < vwap_last)

            if reclaim_up and rsi_last >= rsi_gate:
                side = "LONG"
            elif reclaim_down and rsi_last <= (100.0 - rsi_gate):
                side = "SHORT"

        tp = None
        sl = None
        if side != "NONE" and not np.isnan(atr_last):
            dir_sign = 1 if side == "LONG" else -1
            tp = last_price + dir_sign * (tp_mult * atr_last)
            sl = last_price - dir_sign * (sl_mult * atr_last)

        return Signal(
            symbol=symbol,
            side=side,
            quality=quality,
            score=score,
            price=last_price,
            tp=tp,
            sl=sl,
            spread_bps=float(spread_bps),
            atr_value=float(atr_last if not np.isnan(atr_last) else 0.0),
            atr_pct=float(atr_pct),
            rsi=float(rsi_last if not np.isnan(rsi_last) else 50.0),
            vwap=float(vwap_last if not np.isnan(vwap_last) else last_price),
            ts_ms=int(ts_ms),
        )

    async def run(self) -> None:
        last_tick = 0.0

        while True:
            try:
                now = time.time()

                # Circuit breaker: pausa activa
                if now < self.pause_until:
                    if now - last_tick >= 60:
                        mins = int((self.pause_until - now) // 60) + 1
                        print(f"tick {time.strftime('%H:%M:%S')} PAUSED ({mins}m) | symbols= {len(self.symbols)}")
                        last_tick = now
                    await asyncio.sleep(POLL_SEC)
                    continue

                # heartbeat
                if now - last_tick >= 60:
                    alerts_last_hour = len([t for t in self._alerts_ts if now - t < 3600])
                    print(f"tick {time.strftime('%H:%M:%S')} alive | symbols= {len(self.symbols)} | alerts_last_hour= {alerts_last_hour}")
                    last_tick = now

                total = 0
                low = 0

                for sym in self.symbols:
                    limit = max(VWAP_LEN, BB_LEN, ATR_LEN, RSI_LEN, VOL_MA_LEN) + 10
                    ohlcv = await self.ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
                    if not ohlcv or len(ohlcv) < 50:
                        continue

                    arr = np.array(ohlcv, dtype=float)
                    ts = arr[:, 0].astype(np.int64)
                    o = arr[:, 1]
                    h = arr[:, 2]
                    l = arr[:, 3]
                    c = arr[:, 4]
                    v = arr[:, 5]

                    t = await self.ex.fetch_ticker(sym)
                    bid = float(t.get("bid") or 0.0)
                    ask = float(t.get("ask") or 0.0)
                    last_price = float(t.get("last") or c[-1])

                    if bid > 0 and ask > 0 and last_price > 0:
                        spread = ask - bid
                        spread_bps = (spread / last_price) * 10000.0
                    else:
                        spread_bps = 9999.0

                    # indicadores rápidos (pre-filtros)
                    r = rsi(c, RSI_LEN)
                    a = atr(o, h, l, c, ATR_LEN)
                    bbw = bb_width(c, BB_LEN, BB_K)

                    rsi_last = float(r[-1]) if not np.isnan(r[-1]) else np.nan
                    atr_last = float(a[-1]) if not np.isnan(a[-1]) else np.nan
                    bbw_last = float(bbw[-1]) if not np.isnan(bbw[-1]) else np.nan

                    if last_price <= 0 or np.isnan(atr_last):
                        continue

                    atr_pct = float(atr_last / last_price)

                    # régimen (compresión por universo)
                    total += 1
                    if atr_pct < LOW_ATR_WARN:
                        low += 1

                    # volumen ratio
                    vol_ma = float(np.mean(v[-VOL_MA_LEN:])) if len(v) >= VOL_MA_LEN else float(np.mean(v))
                    vol_ratio = float(v[-1] / vol_ma) if vol_ma > 0 else 0.0

                    # ATR rising (para modo agresivo)
                    atr_ma = float(np.nanmean(a[-20:])) if len(a) >= 20 else float(np.nanmean(a))
                    atr_rise = (atr_last / atr_ma) if (atr_ma and not np.isnan(atr_ma) and atr_ma > 0) else 1.0

                    # decidir modo
                    aggressive = False
                    if MODE_AUTO:
                        aggressive = (
                            atr_pct >= AGGR_MIN_ATR_PCT
                            and atr_rise >= AGGR_ATR_RISE_MULT
                            and (not np.isnan(bbw_last) and bbw_last >= AGGR_MIN_BB_WIDTH)
                            and vol_ratio >= AGGR_VOL_MULT
                        )

                    # parámetros por modo
                    tp_mult = AGGR_TP_MULT if aggressive else NORMAL_TP_MULT
                    sl_mult = AGGR_SL_MULT if aggressive else NORMAL_SL_MULT
                    rsi_gate = AGGR_RSI_MOMENTUM if aggressive else NORMAL_RSI_MOMENTUM

                    # filtros base (normal siempre)
                    if atr_pct < MIN_ATR_PCT:
                        continue
                    if not np.isnan(bbw_last) and bbw_last < MIN_BB_WIDTH:
                        continue
                    if vol_ratio < VOL_MULT and not aggressive:
                        # en agresivo el vol_ratio ya tiene su propio umbral más alto
                        continue
                    if aggressive and vol_ratio < AGGR_VOL_MULT:
                        continue

                    sig = self.compute_signal(sym, spread_bps, int(ts[-1]), o, h, l, c, v, tp_mult, sl_mult, rsi_gate)

                    if self._should_alert(sig):
                        # spread/TP gate relativo
                        tp_dist = (tp_mult * sig.atr_value)
                        tp_pct = (tp_dist / sig.price) if sig.price > 0 else 0.0
                        spread_pct = (sig.spread_bps / 10000.0)
                        if tp_pct > 0 and (spread_pct / tp_pct) > SPREAD_TP_MAX_RATIO:
                            continue

                        rr = None
                        if sig.tp is not None and sig.sl is not None:
                            risk = abs(sig.price - sig.sl)
                            reward = abs(sig.tp - sig.price)
                            rr = (reward / risk) if risk > 0 else None
                        rr_txt = f"{rr:.2f}" if rr is not None else "n/a"

                        # etiqueta modo
                        mode_tag = "🟠 AGGR" if aggressive else "🔵 NORM"

                        # TP parcial + runner (solo si EXP o AGGR)
                        tp1 = sig.tp
                        be_price = sig.price
                        runner_note = ""

                        do_runner = (sig.atr_pct >= EXP_ATR_PCT) or aggressive
                        if do_runner and sig.tp is not None:
                            dir_sign = 1 if sig.side == "LONG" else -1
                            tp2 = sig.price + dir_sign * (EXP_TP2_MULT * sig.atr_value)
                            trail = (EXP_TRAIL_ATR * sig.atr_value)
                            runner_note = (
                                f"\n🎯 TP1 (70%) {tp1:.6g}"
                                f"\n🚀 TP2 (30%) {tp2:.6g}"
                                f"\n🛡 Move SL to BE en TP1 ({be_price:.6g})"
                                f"\n🧵 Trail≈{trail:.6g} (sugerido)"
                            )
                        else:
                            runner_note = f"\n🎯 TP único {tp1:.6g}"

                        msg = (
                            f"📣 Señal {sig.side} ({TIMEFRAME}) | {mode_tag} | Calidad: {sig.quality}\n"
                            f"📌 {sig.symbol}\n"
                            f"💰 Entrada: {sig.price:.6g}\n"
                            f"🎯 TP (ATR×{tp_mult:.2g}): {sig.tp:.6g}\n"
                            f"🛑 SL (ATR×{sl_mult:.2g}): {sig.sl:.6g}\n"
                            f"📊 R:R aprox: {rr_txt}\n"
                            f"📏 Spread: {sig.spread_bps:.1f} bps | ATR: {sig.atr_pct*100:.3f}% | atrRise×{atr_rise:.2f}\n"
                            f"📈 RSI({RSI_LEN}): {sig.rsi:.1f} | VWAP({VWAP_LEN}): {sig.vwap:.6g}\n"
                            f"{runner_note}\n"
                            f"⏱️ ts(ms): {sig.ts_ms}"
                        )

                        try:
                            await telegram_send(msg, self.session)
                            self._last_alert_side[sig.symbol] = sig.side
                            self._last_alert_ts[(sig.symbol, sig.side)] = time.time()
                            self._alerts_ts.append(time.time())
                        except Exception as e:
                            print("Error enviando a Telegram:", repr(e))

                # Circuit breaker update
                if total > 0:
                    frac_low = low / total
                    tnow = time.time()
                    self._regime_hist.append((tnow, frac_low))
                    self._regime_hist = [(t, f) for (t, f) in self._regime_hist if tnow - t <= REGIME_WINDOW_SEC]

                    last = [f for (t, f) in self._regime_hist[-LOW_ATR_CONSEC_POLLS:]]
                    if len(last) == LOW_ATR_CONSEC_POLLS and all(f >= LOW_ATR_FRAC_TRIGGER for f in last):
                        self.pause_until = tnow + PAUSE_SEC
                        try:
                            await telegram_send(
                                f"🧊 Mercado en compresión: {frac_low*100:.0f}% low-ATR. Pauso {PAUSE_SEC//60} min.",
                                self.session
                            )
                        except Exception as e:
                            print("No pude avisar pausa por Telegram:", repr(e))

                await asyncio.sleep(POLL_SEC)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print("Error:", repr(e))
                await asyncio.sleep(POLL_SEC)

    async def shutdown(self) -> None:
        try:
            await self.ex.close()
        except Exception:
            pass
        try:
            await self.session.close()
        except Exception:
            pass


async def main() -> None:
    bot = BingXSignals()
    await bot.init()
    try:
        await bot.run()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Bot detenido por el usuario (Ctrl+C).")

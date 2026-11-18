"""poly-maker's market-making strategy, ported to a NautilusTrader Strategy.

Source: sample_codes/poly-maker-main_lh/src/polymaker
  strategy/estimators.py  -> Ewma, VolEstimator, FlowEstimator, MarkoutTracker
  strategy/regime.py      -> RegimeMachine
  strategy/quoting.py     -> construct_yes_quotes
  execution/reconciler.py -> reconcile
  risk/manager.py         -> PolyMakerStrategy._risk
  engine.py               -> PolyMakerStrategy._recompute

Model:
  fair value   fv = microprice + 0.5 * flow_z * tick
  reservation  r  = fv - gamma * vol * inventory_utilisation
  half-spread  d  = delta_min_ticks * tick + c_vol * vol + c_tox * toxicity
  entry        post-only BUY YES at r - d, joining but never improving the touch
  exit         post-only SELL of held YES at fv + d

Differences from the live bot (one YES book, no NO book or wallet):
  * No BUY NO quotes and no YES/NO merge, so inventory is long YES only.
  * Rewards are not simulated. rewards_* default to 0, which disables them.
  * Flow and sweep detection need TradeTick data. Without it flow_z stays 0.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import NamedTuple

from nautilus_trader.model import AggressorSide
from nautilus_trader.model import BookType
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import OrderSide
from nautilus_trader.model import TimeInForce
from nautilus_trader.trading import Strategy
from nautilus_trader.trading import StrategyConfig

_EPS = 1e-9
_REQUOTE_ALERT = "poly-maker-requote"


@dataclass(frozen=True)
class PolyMakerConfig:
    """Defaults are poly-maker's `political-longdated` profile and config.toml risk."""

    instrument_id: InstrumentId
    book_type: BookType = BookType.L2_MBP
    order_id_tag: str = "001"

    # fair value and estimators
    micro_levels: int = 3
    flow_ewma_halflife_s: float = 120.0
    vol_short_halflife_s: float = 10.0
    vol_long_halflife_s: float = 900.0
    markout_horizon_s: float = 300.0
    markout_halflife_s: float = 1800.0

    # spread and skew
    gamma: float = 0.6
    delta_min_ticks: int = 1
    c_vol: float = 1.5
    c_tox: float = 3.0

    # sizing and inventory
    base_size_usdc: float = 22.0
    q_max_usdc: float = 100.0
    q_soft_frac: float = 0.6
    layers: int = 1
    layer_step_ticks: int = 2
    min_order_size: float = 5.0

    # rewards (not simulated)
    rewards_min_size: float = 0.0
    rewards_max_spread: float = 0.0  # in cents, as on Polymarket
    reward_size_mult: float = 1.0

    # placement and churn
    debounce_ms: int = 250
    reprice_ticks: int = 3
    resize_frac: float = 0.6
    min_edge_ticks: int = 1

    # regime
    event_cooloff_s: float = 30.0
    event_jump_ticks: int = 5
    event_sweep_mult: float = 3.0
    event_sweep_frac: float = 0.8
    trend_flow_z: float = 2.6
    trend_vol_ratio: float = 5.0
    reduce_only_hours: float = 24.0
    halt_before_hours: float = 2.0

    # risk
    max_market_notional_usdc: float = 400.0
    daily_loss_kill_usdc: float = 40.0

    # backtest only: flatten with a market order when the run ends
    close_positions_on_stop: bool = True


class Regime(str, Enum):
    QUIET = "quiet"
    TRENDING = "trending"
    REDUCE_ONLY = "reduce_only"
    EVENT = "event"
    HALTED = "halted"


class Quote(NamedTuple):
    side: OrderSide
    price: float
    size: float


class LiveOrder(NamedTuple):
    order_id: object
    side: OrderSide
    price: float
    size: float


# -- estimators ---------------------------------------------------------------


class Ewma:
    """Time-decayed mean. Prior weight decays by 0.5 ** (dt / halflife)."""

    def __init__(self, halflife_s: float) -> None:
        if halflife_s <= 0:
            raise ValueError("halflife must be positive")
        self.halflife = halflife_s
        self.value = 0.0
        self._last_ts = 0.0
        self._ready = False

    def update(self, value: float, ts: float) -> float:
        if not self._ready:
            self.value, self._last_ts, self._ready = value, ts, True
            return self.value
        decay = 0.5 ** (max(0.0, ts - self._last_ts) / self.halflife)
        self.value = decay * self.value + (1.0 - decay) * value
        self._last_ts = ts
        return self.value

    def decay_to(self, ts: float) -> float:
        if self._ready:
            self.value *= 0.5 ** (max(0.0, ts - self._last_ts) / self.halflife)
            self._last_ts = ts
        return self.value


class VolEstimator:
    """Realized volatility of fair-value changes at two horizons."""

    def __init__(self, short_halflife_s: float, long_halflife_s: float) -> None:
        self._short = Ewma(short_halflife_s)
        self._long = Ewma(long_halflife_s)
        self._last_fv: float | None = None

    def update(self, fv: float, ts: float) -> None:
        if self._last_fv is not None:
            squared = (fv - self._last_fv) ** 2
            self._short.update(squared, ts)
            self._long.update(squared, ts)
        self._last_fv = fv

    @property
    def short(self) -> float:
        return math.sqrt(max(0.0, self._short.value))

    @property
    def long(self) -> float:
        return math.sqrt(max(0.0, self._long.value))

    @property
    def ratio(self) -> float:
        return self.short / self.long if self.long > 1e-9 else 1.0


class FlowEstimator:
    """Signed aggressor flow, normalized by average trade size."""

    def __init__(self, halflife_s: float) -> None:
        self._signed = Ewma(halflife_s)
        self._abs = Ewma(halflife_s)

    def update(self, aggressor_is_buy: bool, size: float, ts: float) -> None:
        self._signed.update(size if aggressor_is_buy else -size, ts)
        self._abs.update(abs(size), ts)

    def decay_to(self, ts: float) -> None:
        self._signed.decay_to(ts)
        self._abs.decay_to(ts)

    @property
    def z(self) -> float:
        return self._signed.value / self._abs.value if self._abs.value > 1e-9 else 0.0


class MarkoutTracker:
    """Adverse selection: how fair value moves against our fills after a horizon."""

    def __init__(self, horizon_s: float, halflife_s: float) -> None:
        self._horizon_s = horizon_s
        self._pending: list[tuple[float, bool, float]] = []
        self._markout = Ewma(halflife_s)

    def record_fill(self, is_buy: bool, fv_at_fill: float, ts: float) -> None:
        self._pending.append((fv_at_fill, is_buy, ts + self._horizon_s))

    def evaluate(self, fv_now: float, ts: float) -> None:
        still_pending = []
        for item in self._pending:
            fv_at_fill, is_buy, due_ts = item
            if ts >= due_ts:
                move = fv_now - fv_at_fill
                self._markout.update(move if is_buy else -move, ts)
            else:
                still_pending.append(item)
        self._pending = still_pending

    @property
    def toxicity(self) -> float:
        return max(0.0, -self._markout.value)


# -- regime -------------------------------------------------------------------


class RegimeMachine:
    """Priority: HALTED > EVENT > REDUCE_ONLY > TRENDING > QUIET."""

    def __init__(self) -> None:
        self._event_until = 0.0

    def decide(
        self,
        cfg: PolyMakerConfig,
        *,
        now: float,
        tick: float,
        fv: float,
        prev_fv: float | None,
        vol_ratio: float,
        flow_z: float,
        inventory_util: float,
        hours_to_end: float | None,
        sweep_flagged: bool = False,
        risk_halt: bool = False,
        risk_reduce_only: bool = False,
    ) -> Regime:
        if risk_halt:
            return Regime.HALTED
        if hours_to_end is not None and hours_to_end <= cfg.halt_before_hours:
            return Regime.HALTED

        jump_ticks = abs(fv - prev_fv) / tick if prev_fv is not None else 0.0
        if sweep_flagged or jump_ticks >= cfg.event_jump_ticks:
            self._event_until = now + cfg.event_cooloff_s
            return Regime.EVENT
        if now < self._event_until:
            return Regime.EVENT

        if risk_reduce_only or inventory_util >= 1.0:
            return Regime.REDUCE_ONLY
        if hours_to_end is not None and hours_to_end <= cfg.reduce_only_hours:
            return Regime.REDUCE_ONLY

        if abs(flow_z) >= cfg.trend_flow_z or vol_ratio >= cfg.trend_vol_ratio:
            return Regime.TRENDING
        return Regime.QUIET


# -- pure quote construction --------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def round_to_tick(price: float, tick: float, decimals: int, *, up: bool) -> float:
    n = price / tick
    n = math.ceil(n - _EPS) if up else math.floor(n + _EPS)
    return min(max(round(n * tick, decimals), tick), 1.0 - tick)


def compute_fair_value(microprice: float, flow_z: float, tick: float, weight: float = 0.5) -> float:
    return min(max(microprice + weight * flow_z * tick, tick), 1.0 - tick)


def microprice(bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> float | None:
    """Best prices weighted by opposite-side size. Levels are ordered from the touch."""
    if not bids or not asks:
        return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    bid_size = sum(size for _, size in bids)
    ask_size = sum(size for _, size in asks)
    total = bid_size + ask_size
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_ask * bid_size + best_bid * ask_size) / total


def _place_bid(
    target: float,
    best_bid: float | None,
    best_ask: float | None,
    tick: float,
    decimals: int,
    fv: float,
    min_edge_ticks: int,
) -> float | None:
    price = min(target, fv - min_edge_ticks * tick)
    if best_bid is not None and price >= best_bid:
        price = best_bid
    if best_ask is not None and price >= best_ask:
        price = best_ask - tick
    price = round_to_tick(price, tick, decimals, up=False)
    return price if 0 < price < 1 else None


def _bid_layers(
    top_price: float,
    total_size: float,
    tick: float,
    decimals: int,
    cfg: PolyMakerConfig,
) -> list[Quote]:
    if total_size <= 0:
        return []
    layers = max(1, cfg.layers)
    reward_floor = max(cfg.rewards_min_size * cfg.reward_size_mult, cfg.min_order_size)
    per_layer = round(total_size / layers, 2)
    while layers > 1 and reward_floor > 0 and per_layer < 0.5 * reward_floor:
        layers -= 1
        per_layer = round(total_size / layers, 2)
    if reward_floor > 0 and 0.5 * reward_floor <= per_layer < reward_floor:
        per_layer = reward_floor
    if per_layer < cfg.min_order_size or per_layer <= 0:
        return []

    quotes = []
    for i in range(layers):
        price = round(top_price - i * cfg.layer_step_ticks * tick, decimals)
        if 0 < price < 1:
            quotes.append(Quote(OrderSide.BUY, price, per_layer))
    return quotes


def construct_yes_quotes(
    cfg: PolyMakerConfig,
    *,
    regime: Regime,
    fv: float,
    vol_short: float,
    toxicity: float,
    best_bid: float | None,
    best_ask: float | None,
    position: float,
    tick: float,
    decimals: int,
    risk_size_scale: float = 1.0,
    exit_urgency: float = 0.0,
) -> list[Quote]:
    """The YES-token half of poly-maker's construct_quotes."""
    if regime in (Regime.EVENT, Regime.HALTED):
        return []

    quotes: list[Quote] = []
    q_max_shares = cfg.q_max_usdc / max(fv, tick)
    util = _clamp(position / q_max_shares, -1.0, 1.0) if q_max_shares > 0 else 0.0
    reservation = fv - cfg.gamma * vol_short * util

    base = cfg.delta_min_ticks * tick
    delta = base + cfg.c_vol * vol_short + cfg.c_tox * toxicity
    reward_band = cfg.rewards_max_spread / 100.0
    if regime is Regime.QUIET and reward_band > 0:
        delta = _clamp(delta, base, max(base, reward_band))
    delta = max(delta, tick)

    regime_scale = 0.5 if regime is Regime.TRENDING else 1.0
    toxicity_scale = 1.0 / (1.0 + toxicity * 10.0)
    scale = regime_scale * toxicity_scale * _clamp(risk_size_scale, 0.0, 1.0)

    if regime is not Regime.REDUCE_ONLY and util < cfg.q_soft_frac:
        price = _place_bid(
            reservation - delta, best_bid, best_ask, tick, decimals, fv, cfg.min_edge_ticks,
        )
        if price is not None:
            shares = cfg.base_size_usdc / max(price, tick) * max(scale * (1 - max(util, 0.0)), 0.0)
            quotes += _bid_layers(price, round(shares, 2), tick, decimals, cfg)

    if position >= cfg.min_order_size:
        passive = fv + delta
        floor = best_bid + tick if best_bid is not None else passive
        urgency = max(exit_urgency, 0.5) if regime is Regime.REDUCE_ONLY else exit_urgency
        target = passive * (1.0 - urgency) + floor * urgency
        if best_bid is not None:
            target = max(target, best_bid + tick)
        price = round_to_tick(target, tick, decimals, up=True)
        size = math.floor(position * 100) / 100
        if 0 < price < 1 and size >= cfg.min_order_size:
            quotes.append(Quote(OrderSide.SELL, price, size))
    return quotes


def reconcile(
    targets: list[Quote],
    live: list[LiveOrder],
    *,
    tick: float,
    reprice_ticks: int,
    resize_frac: float,
) -> tuple[list[object], list[Quote]]:
    """Return (order ids to cancel, quotes to place), keeping orders within tolerance."""
    keep: set[object] = set()
    to_place: list[Quote] = []
    price_tolerance = reprice_ticks * tick + _EPS

    for quote in targets:
        for order in live:
            if order.side != quote.side or order.order_id in keep:
                continue
            price_close = abs(order.price - quote.price) <= price_tolerance
            size_close = (
                quote.size <= 0
                or abs(order.size - quote.size) <= resize_frac * quote.size + _EPS
            )
            if price_close and size_close:
                keep.add(order.order_id)
                break
        else:
            to_place.append(quote)

    return [o.order_id for o in live if o.order_id not in keep], to_place


def _headroom(current: float, cap: float) -> float:
    if cap <= 0:
        return 1.0
    fraction = current / cap
    return 1.0 if fraction <= 0.7 else max(0.0, (1.0 - fraction) / 0.3)


# -- Nautilus strategy --------------------------------------------------------


class PolyMakerStrategy(Strategy):
    """Debounced, event-driven maker quoting on one YES instrument."""

    def __init__(self, config: PolyMakerConfig) -> None:
        super().__init__(StrategyConfig(order_id_tag=config.order_id_tag))
        self.cfg = config
        self.instrument = None
        self.tick = 0.0
        self.decimals = 0

        self.vol = VolEstimator(config.vol_short_halflife_s, config.vol_long_halflife_s)
        self.flow = FlowEstimator(config.flow_ewma_halflife_s)
        self.markout = MarkoutTracker(config.markout_horizon_s, config.markout_halflife_s)
        self.regime_machine = RegimeMachine()

        self.last_fv: float | None = None
        self.net_cash = 0.0
        self.sweep_flagged = False
        self.requote_pending = False
        self.regime_counts: Counter[str] = Counter()

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.cfg.instrument_id)
        if self.instrument is None:
            raise RuntimeError(f"Instrument not found: {self.cfg.instrument_id}")
        self.tick = float(self.instrument.price_increment)
        self.decimals = max(0, round(-math.log10(self.tick)))
        self.subscribe_book_deltas(self.cfg.instrument_id, self.cfg.book_type, managed=True)
        self.subscribe_trades(self.cfg.instrument_id)

    def on_book_deltas(self, deltas) -> None:
        self._request_requote()

    def on_trade(self, trade) -> None:
        now = self.clock.timestamp()
        size = trade.size.as_double()
        price = trade.price.as_double()
        is_buy = trade.aggressor_side == AggressorSide.BUY
        self.flow.update(is_buy, size, now)

        # A sweep is a large print that also consumed most of the near-touch depth.
        if size < self.cfg.event_sweep_mult * self.cfg.base_size_usdc / max(price, 0.01):
            return
        book = self.cache.order_book(self.cfg.instrument_id)
        if book is None:
            return
        bid, ask = self._levels(book)
        if not bid or not ask:
            return
        band = 3 * self.tick + _EPS
        if is_buy:
            consumed = sum(s for p, s in ask if p <= ask[0][0] + band)
        else:
            consumed = sum(s for p, s in bid if p >= bid[0][0] - band)
        if consumed > 0 and size >= self.cfg.event_sweep_frac * consumed:
            self.sweep_flagged = True
            self._request_requote()

    def on_order_filled(self, event) -> None:
        price = event.last_px.as_double()
        size = event.last_qty.as_double()
        is_buy = event.order_side == OrderSide.BUY
        self.net_cash += price * size * (-1 if is_buy else 1)
        fv_at_fill = self.last_fv if self.last_fv is not None else price
        self.markout.record_fill(is_buy, fv_at_fill, self.clock.timestamp())
        self._request_requote()

    def on_stop(self) -> None:
        if _REQUOTE_ALERT in self.clock.timer_names():
            self.clock.cancel_timer(_REQUOTE_ALERT)
        self.requote_pending = False
        self.cancel_all_orders(self.cfg.instrument_id)
        if self.cfg.close_positions_on_stop:
            self.close_all_positions(self.cfg.instrument_id)

    def _request_requote(self) -> None:
        """Coalesce bursts: recompute once, debounce_ms after the first event."""
        if self.requote_pending:
            return
        self.requote_pending = True
        self.clock.set_time_alert(
            _REQUOTE_ALERT,
            self.clock.utc_now() + timedelta(milliseconds=self.cfg.debounce_ms),
            callback=self._on_requote_alert,
        )

    def _on_requote_alert(self, event) -> None:
        self.requote_pending = False
        self._recompute()

    def _levels(self, book, depth: int | None = None):
        bids = [(level.price.as_double(), level.size()) for level in book.bids(depth)]
        asks = [(level.price.as_double(), level.size()) for level in book.asks(depth)]
        return bids, asks

    def _risk(self, position: float, fv: float) -> tuple[bool, bool, float]:
        """Return (halt, reduce_only, size_scale)."""
        notional = position * fv
        if self.net_cash + notional <= -self.cfg.daily_loss_kill_usdc:
            return True, False, 0.0
        if notional >= self.cfg.max_market_notional_usdc:
            return False, True, 1.0
        return False, False, _headroom(notional, self.cfg.max_market_notional_usdc)

    def _hours_to_end(self, now: float) -> float | None:
        hours = (self.instrument.expiration_ns / 1e9 - now) / 3600.0
        return hours if hours > 0.0 else None

    def _recompute(self) -> None:
        cfg = self.cfg
        book = self.cache.order_book(cfg.instrument_id)
        if book is None:
            return
        bids, asks = self._levels(book, cfg.micro_levels)
        if not bids or not asks or bids[0][0] >= asks[0][0]:
            return  # one-sided or crossed: fair value is unreliable

        now = self.clock.timestamp()
        self.flow.decay_to(now)
        fv = compute_fair_value(microprice(bids, asks), self.flow.z, self.tick)
        prev_fv = self.last_fv
        self.vol.update(fv, now)
        self.markout.evaluate(fv, now)
        self.last_fv = fv

        position = max(0.0, float(self.portfolio.net_position(cfg.instrument_id)))
        halt, reduce_only, size_scale = self._risk(position, fv)
        sweep_flagged, self.sweep_flagged = self.sweep_flagged, False
        regime = self.regime_machine.decide(
            cfg,
            now=now,
            tick=self.tick,
            fv=fv,
            prev_fv=prev_fv,
            vol_ratio=self.vol.ratio,
            flow_z=self.flow.z,
            inventory_util=position * fv / cfg.q_max_usdc if cfg.q_max_usdc > 0 else 0.0,
            hours_to_end=self._hours_to_end(now),
            sweep_flagged=sweep_flagged,
            risk_halt=halt,
            risk_reduce_only=reduce_only,
        )
        self.regime_counts[regime.value] += 1

        targets = construct_yes_quotes(
            cfg,
            regime=regime,
            fv=fv,
            vol_short=self.vol.short,
            toxicity=self.markout.toxicity,
            best_bid=bids[0][0],
            best_ask=asks[0][0],
            position=position,
            tick=self.tick,
            decimals=self.decimals,
            risk_size_scale=size_scale,
        )
        self._apply_quotes(targets)

    def _apply_quotes(self, targets: list[Quote]) -> None:
        cfg = self.cfg
        open_orders = self.cache.orders_open(
            instrument_id=cfg.instrument_id,
            strategy_id=self.strategy_id,
        )
        # As in the live engine, never act on top of unconfirmed order state.
        if any(order.is_inflight or order.is_pending_cancel for order in open_orders):
            return

        live = [
            LiveOrder(
                order.client_order_id,
                order.side,
                order.price.as_double(),
                order.leaves_qty.as_double(),
            )
            for order in open_orders
        ]
        to_cancel, to_place = reconcile(
            targets,
            live,
            tick=self.tick,
            reprice_ticks=cfg.reprice_ticks,
            resize_frac=cfg.resize_frac,
        )
        if to_cancel:
            for client_order_id in to_cancel:
                self.cancel_order(client_order_id)
            self._request_requote()  # place replacements once cancels are confirmed
            return

        for quote in to_place:
            self.submit_order(
                self.order_factory.limit(
                    instrument_id=cfg.instrument_id,
                    order_side=quote.side,
                    quantity=self.instrument.make_qty(quote.size),
                    price=self.instrument.make_price(quote.price),
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                ),
            )

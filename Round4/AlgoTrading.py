from datamodel import Order, OrderDepth, TradingState, Symbol, ProsperityEncoder, Listing, Observation, Trade
from typing import Dict, List, Tuple, Optional
import json
import math


# ========================================================================
# CONFIG
# ========================================================================

LIMITS = {
    "HYDROGEL_PACK":       50,
    "VELVETFRUIT_EXTRACT": 400,
    "VEV_4000":            200,
    "VEV_4500":            200,
    "VEV_5000":            200,
    "VEV_5100":            200,
    "VEV_5200":            200,
    "VEV_5300":            200,
    "VEV_5400":            200,
    "VEV_5500":            200,
    "VEV_6000":            200,
    "VEV_6500":            200,
}

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

EXPIRY_DAY = 8
TIMESTAMPS_PER_DAY = 1_000_000

# ----- HYDROGEL_PACK -----
HYDROGEL_HALF        = 3
HYDROGEL_INV_SKEW    = 0.15     # v4 tried 0.20 → slightly worse. Back to v3 value.
HYDROGEL_MM_SIZE     = 12
HYDROGEL_HARD_CAP    = 15       # v5: properly enforced via bid/ask sizing
HYDROGEL_SNIPE_BUF   = 5
HYDROGEL_EOD_TS      = 95_000   # aggressive flatten after this timestamp
# v6: blended EMA fair to bias quotes contrarian to short-term moves.
# fair = (1 - W) * micro + W * ema, then clamp |fair - micro| ≤ MAX_SHIFT
HYDROGEL_EMA_ALPHA   = 0.005    # ~140-tick half-life
HYDROGEL_EMA_WEIGHT  = 0.20     # 80% micro / 20% ema
HYDROGEL_MAX_SHIFT   = 3        # safety: fair cannot deviate >3 ticks from micro

# ----- VOUCHER MM -----
# (sym, half_width, mm_size). Strikes not listed are not actively MM'd.
# v3 results in real Prosperity:
#   VEV_4500 (h=3): +86 ✓  | VEV_5100 (h=1): +46 ✓
#   VEV_5000 (h=2): -23 ✗  → DROPPED
#   VEV_5200 (h=1): 0 fills (lands at touch, end of queue) → DROPPED
#   VEV_5300 (h=1): 0 fills (quotes outside touch)         → DROPPED
# v4 adds:
#   VEV_4000 (21-tick spread, h=5) — same passive-MM pattern as VEV_4500
VOUCHER_MM_PARAMS: Dict[str, Tuple[int, int]] = {
    "VEV_4000": (5, 12),   # NEW: widest spread (~21)
    "VEV_4500": (3, 15),   # proven +86 in v3
    "VEV_5100": (1, 12),   # proven +46 in v3
}
VOUCHER_INV_SKEW = 0.02
VOUCHER_HARD_CAP = 100  # |pos|≥100 → only flatten side

# ----- VFE -----
# MM disabled. Hedge still happens if voucher trading turns on.

# ----- VOUCHER smile-trading -----
VOUCHER_TRADING_ENABLED = False
IV_EDGE_THRESHOLD       = 0.005
SMILE_MIN_POINTS        = 5

# ----- Hedging -----
DELTA_HEDGE_THRESHOLD = 5


# ========================================================================
# BLACK-SCHOLES UTILITIES
# ========================================================================

SQRT_2PI = math.sqrt(2 * math.pi)
def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
def _norm_pdf(x): return math.exp(-0.5 * x * x) / SQRT_2PI

def bs_call_price(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return S*_norm_cdf(d1) - K*_norm_cdf(d2)

def bs_call_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*sqrtT)
    return _norm_cdf(d1)

def bs_call_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*sqrtT)
    return S * _norm_pdf(d1) * sqrtT

def implied_vol(price, S, K, T, sigma_init=0.02, tol=1e-5, max_iter=30):
    intrinsic = max(0.0, S - K)
    if price <= intrinsic + 1e-6 or T <= 0: return None
    sigma = sigma_init
    for _ in range(max_iter):
        p = bs_call_price(S, K, T, sigma)
        v = bs_call_vega(S, K, T, sigma)
        if v < 1e-10: return None
        diff = p - price
        if abs(diff) < tol: return sigma
        sigma -= diff / v
        if sigma <= 1e-6: sigma = 1e-6
        if sigma > 5.0: sigma = 5.0
    return sigma if 1e-4 < sigma < 5.0 else None

def fit_parabolic_smile(points):
    n = len(points)
    if n < 3: return None
    s0=n; s1=sum(m for m,_ in points); s2=sum(m*m for m,_ in points)
    s3=sum(m**3 for m,_ in points); s4=sum(m**4 for m,_ in points)
    sy=sum(iv for _,iv in points); smy=sum(m*iv for m,iv in points)
    sm2y=sum(m*m*iv for m,iv in points)
    M=[[s4,s3,s2,sm2y],[s3,s2,s1,smy],[s2,s1,s0,sy]]
    for i in range(3):
        pivot = M[i][i]
        if abs(pivot) < 1e-12: return None
        for k in range(i+1,3):
            f = M[k][i]/pivot
            for j in range(i,4): M[k][j] -= f*M[i][j]
    coef=[0.0,0.0,0.0]
    for i in range(2,-1,-1):
        s = M[i][3]
        for j in range(i+1,3): s -= M[i][j]*coef[j]
        coef[i] = s/M[i][i]
    return tuple(coef)


# ========================================================================
# HELPERS
# ========================================================================

def best_bid_ask(od):
    bid = max(od.buy_orders.keys()) if od.buy_orders else None
    ask = min(od.sell_orders.keys()) if od.sell_orders else None
    return bid, ask

def microprice(od):
    bid, ask = best_bid_ask(od)
    if bid is None or ask is None: return None
    bv = od.buy_orders[bid]; av = -od.sell_orders[ask]
    tot = bv + av
    if tot <= 0: return 0.5*(bid+ask)
    return (bid*av + ask*bv) / tot


# ========================================================================
# Generic MM helper (used by HYDROGEL + vouchers)
# ========================================================================

def mm_quotes(sym: str, od: OrderDepth, fair: float, pos: int, limit: int,
              half: float, inv_skew: float, mm_size: int,
              hard_cap: Optional[int], force_flatten_only: bool = False) -> List[Order]:
    """Produce passive MM quotes with inventory skew, size cap, and optional
    hard one-sided cap. force_flatten_only suppresses the side that grows
    inventory (used for end-of-day cleanup)."""
    bid, ask = best_bid_ask(od)
    if bid is None or ask is None:
        return []
    sk = inv_skew * pos
    bid_target = fair - half - sk
    ask_target = fair + half - sk
    bid_px = int(math.floor(bid_target))
    ask_px = int(math.ceil(ask_target))
    # never quote outside the natural touch (would just sit useless)
    bid_px = min(bid_px, ask - 1)
    ask_px = max(ask_px, bid + 1)
    if bid_px >= ask_px:
        bid_px = ask_px - 1

    place_bid = True
    place_ask = True
    if hard_cap is not None:
        if pos >= hard_cap: place_bid = False
        if pos <= -hard_cap: place_ask = False
    if force_flatten_only:
        # only quote the side that REDUCES |pos|
        if pos > 0:
            place_bid = False
        elif pos < 0:
            place_ask = False
        else:
            place_bid = place_ask = False  # already flat

    orders: List[Order] = []
    # PROPERLY enforce hard_cap: the bid (which grows position) is sized so a
    # single fill cannot exceed +hard_cap. Same on ask side.
    if hard_cap is not None:
        bid_room = max(0, hard_cap - pos)   # how many more we can BUY before hitting +cap
        ask_room = max(0, hard_cap + pos)   # how many more we can SELL before hitting -cap
    else:
        bid_room = limit - pos
        ask_room = pos + limit
    bid_size = min(mm_size, bid_room) if place_bid else 0
    ask_size = min(mm_size, ask_room) if place_ask else 0
    if bid_size > 0:
        orders.append(Order(sym, bid_px, bid_size))
    if ask_size > 0:
        orders.append(Order(sym, ask_px, -ask_size))
    return orders


# ========================================================================
# TRADER
# ========================================================================

class Trader:
    def __init__(self):
        self.mark_log: list = []

    def get_tte(self, timestamp):
        day = timestamp // TIMESTAMPS_PER_DAY + 1
        intra = (timestamp % TIMESTAMPS_PER_DAY) / TIMESTAMPS_PER_DAY
        return max(EXPIRY_DAY - day - intra, 1e-6)

    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        self.mark_log = saved.get("mark_log", [])[-500:]
        self.hyd_ema = saved.get("hyd_ema", None)  # v6: persisted across ticks

        result: Dict[Symbol, List[Order]] = {}

        self._log_marks(state)

        # 1. HYDROGEL_PACK
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._mm_hydrogel(state)

        # 2. Vouchers (MM on wide-spread strikes)
        for sym in VOUCHER_MM_PARAMS:
            if sym in state.order_depths:
                ords = self._mm_voucher(state, sym)
                if ords:
                    result[sym] = ords

        # 3. Smile fit (diagnostic only; no orders + no hedge unless flag is on).
        # IMPORTANT: hedging VFE based on accumulated MM positions causes VFE to
        # blow out to its limit, since voucher MM accumulates inventory the
        # hedge tries to neutralize. Only run hedge when we're actually trading
        # vouchers off smile.
        if VOUCHER_TRADING_ENABLED and "VELVETFRUIT_EXTRACT" in state.order_depths:
            voucher_orders, vfe_hedge_qty = self._smile_trade_vouchers(state)
            for sym, ords in voucher_orders.items():
                result.setdefault(sym, []).extend(ords)
            if abs(vfe_hedge_qty) >= DELTA_HEDGE_THRESHOLD:
                hedge_orders = self._hedge_vfe(state, vfe_hedge_qty)
                if hedge_orders:
                    result["VELVETFRUIT_EXTRACT"] = hedge_orders



        out = {"mark_log": self.mark_log[-500:], "hyd_ema": self.hyd_ema}
        return result, 0, json.dumps(out)

    def _log_marks(self, state):
        for sym, trades in state.market_trades.items():
            for t in trades:
                if t.timestamp != state.timestamp - 100 and t.timestamp != state.timestamp:
                    continue
                buyer = t.buyer or ""; seller = t.seller or ""
                if buyer.startswith("Mark"):
                    self.mark_log.append((t.timestamp, buyer, "BUY", sym, t.price, t.quantity))
                if seller.startswith("Mark"):
                    self.mark_log.append((t.timestamp, seller, "SELL", sym, t.price, t.quantity))

    # -------- HYDROGEL ----------
    def _mm_hydrogel(self, state) -> List[Order]:
        sym = "HYDROGEL_PACK"
        od = state.order_depths[sym]
        bid, ask = best_bid_ask(od)
        if bid is None or ask is None:
            return []
        m = microprice(od)
        if m is None: return []

        # v6: EMA-blended fair, capped to prevent extreme deviation.
        # Updates self.hyd_ema in place; persisted to traderData by run().
        if self.hyd_ema is None:
            self.hyd_ema = m
        self.hyd_ema = HYDROGEL_EMA_ALPHA * m + (1 - HYDROGEL_EMA_ALPHA) * self.hyd_ema
        raw_blend = (1 - HYDROGEL_EMA_WEIGHT) * m + HYDROGEL_EMA_WEIGHT * self.hyd_ema
        deviation = raw_blend - m
        if deviation > HYDROGEL_MAX_SHIFT: deviation = HYDROGEL_MAX_SHIFT
        if deviation < -HYDROGEL_MAX_SHIFT: deviation = -HYDROGEL_MAX_SHIFT
        fair = m + deviation

        pos = state.position.get(sym, 0)
        limit = LIMITS[sym]
        orders: List[Order] = []
        new_pos = pos

        # Snipe extreme mispricings (respects hard_cap so a single fat ask can't blow past it)
        snipe_cap = HYDROGEL_HARD_CAP
        for px, vol in sorted(od.sell_orders.items()):
            if px < fair - HYDROGEL_SNIPE_BUF and new_pos < snipe_cap:
                qty = min(-vol, snipe_cap - new_pos)
                if qty > 0:
                    orders.append(Order(sym, px, qty)); new_pos += qty
        for px, vol in sorted(od.buy_orders.items(), reverse=True):
            if px > fair + HYDROGEL_SNIPE_BUF and new_pos > -snipe_cap:
                qty = min(vol, new_pos + snipe_cap)
                if qty > 0:
                    orders.append(Order(sym, px, -qty)); new_pos -= qty

        # End-of-day flush: only flatten side, very aggressive (quote at touch)
        eod = state.timestamp >= HYDROGEL_EOD_TS
        if eod and new_pos != 0:
            # Quote at touch on flatten side, full remaining size
            if new_pos > 0:
                orders.append(Order(sym, bid, -min(new_pos, HYDROGEL_MM_SIZE * 2)))
            else:
                orders.append(Order(sym, ask, min(-new_pos, HYDROGEL_MM_SIZE * 2)))
            return orders  # skip normal MM during EOD

        # Normal MM with inventory skew + hard cap
        passive = mm_quotes(
            sym, od, fair, new_pos, limit,
            half=HYDROGEL_HALF, inv_skew=HYDROGEL_INV_SKEW,
            mm_size=HYDROGEL_MM_SIZE, hard_cap=HYDROGEL_HARD_CAP,
        )
        orders.extend(passive)
        return orders

    # -------- Voucher MM ----------
    def _mm_voucher(self, state, sym: str) -> List[Order]:
        od = state.order_depths[sym]
        bid, ask = best_bid_ask(od)
        if bid is None or ask is None: return []
        fair = microprice(od)
        if fair is None: return []
        pos = state.position.get(sym, 0)
        limit = LIMITS[sym]
        half, sz = VOUCHER_MM_PARAMS[sym]
        return mm_quotes(
            sym, od, fair, pos, limit,
            half=half, inv_skew=VOUCHER_INV_SKEW,
            mm_size=sz, hard_cap=VOUCHER_HARD_CAP,
        )

    # -------- Smile fit (diagnostic; no trading unless flag) ----------
    def _smile_trade_vouchers(self, state) -> Tuple[Dict[Symbol, List[Order]], int]:
        vfe_od = state.order_depths["VELVETFRUIT_EXTRACT"]
        S = microprice(vfe_od)
        if S is None: return {}, 0
        T = self.get_tte(state.timestamp)

        ivs = []
        for sym, K in VEV_STRIKES.items():
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            mid = microprice(od)
            if mid is None: continue
            iv = implied_vol(mid, S, K, T)
            if iv is None: continue
            m = math.log(K/S) / math.sqrt(T)
            ivs.append((sym, K, m, iv, mid))

        smile = None
        if len(ivs) >= SMILE_MIN_POINTS:
            smile = fit_parabolic_smile([(m, iv) for _,_,m,iv,_ in ivs])

        orders: Dict[Symbol, List[Order]] = {}
        net_delta = 0.0
        for sym, K, m, iv_mkt, mid in ivs:
            if smile is None: continue
            a, b, c = smile
            iv_fair = a*m*m + b*m + c
            pos = state.position.get(sym, 0)
            net_delta += pos * bs_call_delta(S, K, T, max(iv_fair, 1e-4))

            if not VOUCHER_TRADING_ENABLED: continue
            od = state.order_depths[sym]
            bid_v, ask_v = best_bid_ask(od)
            edge = iv_mkt - iv_fair
            limit = LIMITS[sym]
            ord_list: List[Order] = []
            if edge > IV_EDGE_THRESHOLD and bid_v is not None and pos > -limit:
                qty = min(od.buy_orders[bid_v], pos + limit)
                if qty > 0: ord_list.append(Order(sym, bid_v, -qty))
            elif edge < -IV_EDGE_THRESHOLD and ask_v is not None and pos < limit:
                qty = min(-od.sell_orders[ask_v], limit - pos)
                if qty > 0: ord_list.append(Order(sym, ask_v, qty))
            if ord_list: orders[sym] = ord_list

        return orders, -int(round(net_delta))

    # -------- VFE delta hedge (only fires if voucher trading is on) ----------
    def _hedge_vfe(self, state, hedge_qty: int) -> List[Order]:
        sym = "VELVETFRUIT_EXTRACT"
        od = state.order_depths[sym]
        bid, ask = best_bid_ask(od)
        if bid is None or ask is None: return []
        pos = state.position.get(sym, 0)
        limit = LIMITS[sym]
        orders: List[Order] = []
        if hedge_qty > 0:
            qty = min(hedge_qty, limit - pos, -od.sell_orders.get(ask, 0))
            if qty > 0: orders.append(Order(sym, ask, qty))
        else:
            qty = min(-hedge_qty, pos + limit, od.buy_orders.get(bid, 0))
            if qty > 0: orders.append(Order(sym, bid, -qty))
        return orders

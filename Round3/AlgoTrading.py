from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json
POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":        50,
    "VELVETFRUIT_EXTRACT":  200,
    "VEV_4000": 200, "VEV_4500": 200, "VEV_5000": 200,
    "VEV_5100": 200, "VEV_5200": 200, "VEV_5300": 200,
    "VEV_5400": 200, "VEV_5500": 200,
    "VEV_6000": 200, "VEV_6500": 200,
}
SOFT_CAPS: Dict[str, int] = {
    "HYDROGEL_PACK":        60,      
    "VELVETFRUIT_EXTRACT":  80,       
    "VEV_4000": 80, "VEV_4500": 80, "VEV_5000": 80,
    "VEV_5100": 30,                                   
    "VEV_5200": 80, "VEV_5300": 80,
    "VEV_5400":  0,                                    
    "VEV_5500": 30,                                    
    "VEV_6000":  0, "VEV_6500":  0,
}
VEV_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
DEDICATED_MM_VOUCHERS: Dict[str, float] = {
    "VEV_4000": 5.0,    
    "VEV_4500": 4.0,    
}
VOUCHER_EDGE_DEFAULT = 2.5
VOUCHER_EDGE: Dict[str, float] = {
    "VEV_4000": 1.5, "VEV_4500": 1.5,                  
    "VEV_5000": 2.5, "VEV_5100": 2.5, "VEV_5200": 2.5,
    "VEV_5300": 2.5, "VEV_5400": 2.5, "VEV_5500": 1.5,  
    "VEV_6000": 2.5, "VEV_6500": 2.5,                   
}
IV_BIAS: Dict[str, float] = {
    "VEV_4000": -0.0073,   
    "VEV_4500":  0.0100,   
    "VEV_5000": -0.0040,    
    "VEV_5100":  0.0,       
    "VEV_5200":  0.0,       
    "VEV_5300":  0.0,       
    "VEV_5400":  0.0,     
    "VEV_5500":  0.0,       
    "VEV_6000":  0.0,       
    "VEV_6500":  0.0,      
}
HEDGE_THRESHOLD = 40
DAYS_TO_EXPIRY_AT_SIM_START = 4
ONE_DAY_TIMESTAMPS = 1_000_000
YEAR_DAYS = 365.0
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)
def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)
def implied_vol(price: float, S: float, K: float, T: float,
                lo: float = 1e-4, hi: float = 3.0,
                tol: float = 1e-5, max_iter: int = 60) -> Optional[float]:
    intr = max(S - K, 0.0)
    if price <= intr + 1e-6:
        return None
    if bs_call(S, K, T, hi) < price:
        return hi
    if bs_call(S, K, T, lo) > price:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
def best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba
def mid_price(od: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(od)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)
def microprice(od: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(od)
    if bb is None or ba is None:
        return None
    bvol = abs(od.buy_orders[bb])
    avol = abs(od.sell_orders[ba])
    tot = bvol + avol
    if tot == 0:
        return 0.5 * (bb + ba)
    return (bb * avol + ba * bvol) / tot
def fit_parabola(ms: List[float], ivs: List[float]) -> Optional[Tuple[float, float, float]]:
    n = len(ms)
    if n < 3:
        return None
    s0 = n
    s1 = sum(ms)
    s2 = sum(m * m for m in ms)
    s3 = sum(m ** 3 for m in ms)
    s4 = sum(m ** 4 for m in ms)
    t0 = sum(ivs)
    t1 = sum(iv * m for iv, m in zip(ivs, ms))
    t2 = sum(iv * m * m for iv, m in zip(ivs, ms))
    A = [[s0, s1, s2],
         [s1, s2, s3],
         [s2, s3, s4]]
    b = [t0, t1, t2]
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(A)
    if abs(D) < 1e-12:
        return None
    A1 = [[b[i] if j == 0 else A[i][j] for j in range(3)] for i in range(3)]
    A2 = [[b[i] if j == 1 else A[i][j] for j in range(3)] for i in range(3)]
    A3 = [[b[i] if j == 2 else A[i][j] for j in range(3)] for i in range(3)]
    return det3(A1) / D, det3(A2) / D, det3(A3) / D
class Trader:
    def _load_memory(self, state: TradingState) -> dict:
        if not state.traderData:
            return {"last_ts": -1, "day_count": 0,
                    "hydro_ema": None, "velv_ema": None,
                    "vev_emas": {},
                    "vb": {}} 
        try:
            mem = json.loads(state.traderData)
            mem.setdefault("vb", {})
            mem.setdefault("vev_emas", {})
            return mem
        except Exception:
            return {"last_ts": -1, "day_count": 0,
                    "hydro_ema": None, "velv_ema": None,
                    "vev_emas": {},
                    "vb": {}}
    def _tte_years(self, mem: dict, timestamp: int) -> float:
        frac = timestamp / ONE_DAY_TIMESTAMPS
        days_left = DAYS_TO_EXPIRY_AT_SIM_START - mem["day_count"] - frac
        days_left = max(days_left, 0.5 / ONE_DAY_TIMESTAMPS)
        return days_left / YEAR_DAYS
    def _tick_day_counter(self, mem: dict, timestamp: int) -> None:
        if mem["last_ts"] >= 0 and timestamp < mem["last_ts"]:
            mem["day_count"] += 1
        mem["last_ts"] = timestamp
    def _cap(self, product: str) -> int:
        return min(POSITION_LIMITS.get(product, 0), SOFT_CAPS.get(product, 0))
    def _apply_trade(self, entry: dict, signed_qty: int, price: float) -> None:
        old_q = entry["qty"]
        new_q = old_q + signed_qty
        if new_q == 0:
            entry["cost"] = 0.0
        elif old_q == 0 or (old_q > 0 and new_q > 0 and abs(new_q) > abs(old_q)) \
                or (old_q < 0 and new_q < 0 and abs(new_q) > abs(old_q)):
            entry["cost"] = (old_q * entry["cost"] + signed_qty * price) / new_q
        elif old_q * new_q < 0:
            entry["cost"] = price
        entry["qty"] = new_q
    def _update_cost_basis(self, mem: dict, state: TradingState) -> None:
        vb = mem["vb"]
        for symbol, trades in (state.own_trades or {}).items():
            if symbol not in VEV_STRIKES:
                continue
            if symbol not in vb:
                vb[symbol] = {"qty": 0, "cost": 0.0}
            for t in trades:
                if getattr(t, "buyer", "") == "SUBMISSION":
                    self._apply_trade(vb[symbol], t.quantity, float(t.price))
                elif getattr(t, "seller", "") == "SUBMISSION":
                    self._apply_trade(vb[symbol], -t.quantity, float(t.price))
    def _mm_passive(self, product: str, fair: float, od: OrderDepth,
                    position: int, spread: float = 1.0) -> List[Order]:
        orders: List[Order] = []
        cap = self._cap(product)
        if cap <= 0:
            return orders
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders
        skew = -(position / cap) * spread
        raw_bid = fair - spread + skew
        raw_ask = fair + spread + skew
        bid_px = int(math.floor(raw_bid))
        ask_px = int(math.ceil(raw_ask))
        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)
        if bid_px >= ask_px:
            return orders
        buy_qty = cap - position
        sell_qty = cap + position
        if position > cap * 0.8:
            buy_qty = 0
        if position < -cap * 0.8:
            sell_qty = 0
        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))
        return orders
    def _voucher_orders(
        self,
        state: TradingState,
        S: float,
        T: float,
        vb: dict,
    ) -> Tuple[Dict[str, List[Order]], float]:
        out: Dict[str, List[Order]] = {}
        net_delta = 0.0
        pts: List[Tuple[str, int, float, float]] = []  
        for name, K in VEV_STRIKES.items():
            if self._cap(name) <= 0:
                continue
            od = state.order_depths.get(name)
            if od is None:
                continue
            m = mid_price(od)
            if m is None or m <= 0.5 + 1e-6:
                continue
            log_mny = math.log(K / S) / math.sqrt(T)
            iv = implied_vol(m, S, K, T)
            if iv is None:
                continue
            pts.append((name, K, log_mny, iv))
        if len(pts) < 3:
            return out, 0.0
        fit = fit_parabola([p[2] for p in pts], [p[3] for p in pts])
        if fit is None:
            return out, 0.0
        a, b, c = fit
        for name, K in VEV_STRIKES.items():
            if name in DEDICATED_MM_VOUCHERS:
                continue
            cap = self._cap(name)
            if cap <= 0:
                continue
            od = state.order_depths.get(name)
            if od is None:
                continue
            bb, ba = best_bid_ask(od)
            if bb is None or ba is None:
                continue
            log_mny = math.log(K / S) / math.sqrt(T)
            iv_fit = max(a + b * log_mny + c * log_mny * log_mny + IV_BIAS.get(name, 0.0), 0.02)
            theo = bs_call(S, K, T, iv_fit)
            delta = bs_delta(S, K, T, iv_fit)
            edge = VOUCHER_EDGE.get(name, VOUCHER_EDGE_DEFAULT)
            pos = state.position.get(name, 0)
            orders: List[Order] = []
            per_entry = max(1, cap // 10)
            if ba <= theo - edge and pos < cap:
                vol_avail = -od.sell_orders[ba]
                qty = min(vol_avail, cap - pos, per_entry)
                if qty > 0:
                    orders.append(Order(name, ba, qty))
                    pos += qty
            if bb >= theo + edge and pos > -cap:
                vol_avail = od.buy_orders[bb]
                qty = min(vol_avail, pos + cap, per_entry)
                if qty > 0:
                    orders.append(Order(name, bb, -qty))
                    pos -= qty
            entry_info = vb.get(name)
            if entry_info is not None and entry_info["qty"] != 0:
                cost = entry_info["cost"]
                per_exit = max(1, cap // 5)
                if pos > 0:
                    target = int(math.ceil(cost))
                    exit_px = max(target, bb + 1) 
                    if exit_px >= cost and exit_px <= max(ba, target):
                        exit_qty = min(pos, per_exit)
                        if exit_qty > 0:
                            orders.append(Order(name, exit_px, -exit_qty))
                elif pos < 0:
                    target = int(math.floor(cost))
                    exit_px = min(target, ba - 1)
                    if exit_px <= cost and exit_px >= min(bb, target):
                        exit_qty = min(-pos, per_exit)
                        if exit_qty > 0:
                            orders.append(Order(name, exit_px, exit_qty))
            if orders:
                out[name] = orders
            net_delta += pos * delta
        return out, net_delta
    def _hedge_orders(self, state: TradingState, voucher_net_delta: float) -> List[Order]:
        prod = "VELVETFRUIT_EXTRACT"
        cap = self._cap(prod)
        if cap <= 0:
            return []
        cur = state.position.get(prod, 0)
        target = max(min(int(-voucher_net_delta), cap), -cap)
        residual = target - cur
        if abs(residual) < HEDGE_THRESHOLD:
            return []
        od = state.order_depths.get(prod)
        if od is None:
            return []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return []
        if residual > 0:
            return [Order(prod, bb, min(residual, cap - cur))]
        else:
            return [Order(prod, ba, max(residual, -(cap + cur)))]
    def run(self, state: TradingState):
        mem = self._load_memory(state)
        self._tick_day_counter(mem, state.timestamp)
        self._update_cost_basis(mem, state)
        T = self._tte_years(mem, state.timestamp)
        result: Dict[str, List[Order]] = {}
        prod = "HYDROGEL_PACK"
        if prod in state.order_depths:
            od = state.order_depths[prod]
            mp = microprice(od)
            if mp is not None:
                if mem["hydro_ema"] is None:
                    mem["hydro_ema"] = mp
                else:
                    mem["hydro_ema"] += 0.20 * (mp - mem["hydro_ema"])
                fair = mem["hydro_ema"]
                orders = self._mm_passive(prod, fair, od,
                                          state.position.get(prod, 0),
                                          spread=5.0)       
                if orders:
                    result[prod] = orders
        prod = "VELVETFRUIT_EXTRACT"
        S: Optional[float] = None
        velv_orders: List[Order] = []
        if prod in state.order_depths:
            od = state.order_depths[prod]
            mp = microprice(od)
            if mp is not None:
                if mem["velv_ema"] is None:
                    mem["velv_ema"] = mp
                else:
                    mem["velv_ema"] += 0.10 * (mp - mem["velv_ema"])
                fair = mem["velv_ema"]
                S = mp
                velv_orders = self._mm_passive(prod, fair, od,
                                               state.position.get(prod, 0),
                                               spread=1.5)  
        for vev_name, vev_spread in DEDICATED_MM_VOUCHERS.items():
            if vev_name not in state.order_depths:
                continue
            od = state.order_depths[vev_name]
            mp = microprice(od)
            if mp is None:
                continue
            cur_ema = mem["vev_emas"].get(vev_name)
            if cur_ema is None:
                cur_ema = mp
            else:
                cur_ema += 0.15 * (mp - cur_ema)
            mem["vev_emas"][vev_name] = cur_ema

            mm_orders = self._mm_passive(vev_name, cur_ema, od,
                                         state.position.get(vev_name, 0),
                                         spread=vev_spread)
            if mm_orders:
                result[vev_name] = mm_orders
        voucher_orders: Dict[str, List[Order]] = {}
        net_delta = 0.0
        if S is not None and T > 0:
            voucher_orders, net_delta = self._voucher_orders(state, S, T, mem["vb"])
        hedge = self._hedge_orders(state, net_delta)
        if velv_orders or hedge:
            result["VELVETFRUIT_EXTRACT"] = velv_orders + hedge
        for k, v in voucher_orders.items():
            result[k] = v
        return result, 0, json.dumps(mem)

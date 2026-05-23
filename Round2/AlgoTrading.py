import json
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
OSM_LIMIT = 80
PEP_LIMIT = 80  
LIMITS = {"ASH_COATED_OSMIUM": OSM_LIMIT, "INTARIAN_PEPPER_ROOT": PEP_LIMIT}
OSM_PRIOR_FAIR = 10000             
OSM_TAKE_EDGE = 2                  
OSM_TAKE_EDGE_LOADED = 4           
OSM_LOAD_THRESH = 30               
OSM_HIST_INTERVAL = 100            
OSM_HIST_MAX = 500                 
OSM_FIT_WINDOW = 200               
OSM_MIN_SAMPLES = 5                
OSM_FAIR_BLEND = 0.5               
OSM_SOFT_CLEAR = True              
OSM_SKEW_THRESH = 40               
PEP_DAY_LEN = 1_000_000
PEP_HIST_INTERVAL = 100
PEP_HIST_MAX = 200
PEP_FIT_WINDOW = 50
PEP_MIN_SAMPLES = 5
PEP_PRIOR_SLOPE = 0.001
PEP_SAFETY = 0.5
PEP_MAX_CHASE = 8
PEP_BEAR_SLOPE = -1e-5
PEP_NOISE_STD = 2.5
PEP_BASE_FRAC = 0.4
PEP_DIP_GAIN = 0.5
PEP_SELL_PEAK_Z = 3.5              
PEP_SELL_FRAC_PEAK = 0.25          
class Trader:
    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except Exception: td = {}
        result: Dict[str, List[Order]] = {}
        p = "ASH_COATED_OSMIUM"
        if p in state.order_depths:
            try:
                result[p] = self._trade_osmium(p, state.order_depths[p],
                                               state.position.get(p, 0), td)
            except Exception as e:
                print(f"[OSM ERROR] {type(e).__name__}: {e}")
        p = "INTARIAN_PEPPER_ROOT"
        if p in state.order_depths:
            try:
                result[p] = self._trade_pepper(p, state.order_depths[p],
                                               state.position.get(p, 0),
                                               state.timestamp, td)
            except Exception as e:
                print(f"[PEP ERROR] {type(e).__name__}: {e}")
        return result, 0, json.dumps(td)
    def _trade_osmium(self, product, od, pos, td):
        orders: List[Order] = []
        lim = LIMITS[product]
        if not od.buy_orders or not od.sell_orders:
            return orders
        best_bid = max(od.buy_orders); best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0
        hist = td.setdefault("osm_hist", [])
        ts = td.get("_ts", 0)
        td["_ts"] = ts + 1  
        if not hist or (len(hist) > 0 and (ts - hist[-1][0]) >= OSM_HIST_INTERVAL):
            hist.append([ts, mid])
            if len(hist) > OSM_HIST_MAX:
                del hist[: len(hist) - OSM_HIST_MAX]
        fair_est = self._estimate_osm_fair(hist)
        if fair_est is None:
            fair = OSM_PRIOR_FAIR
        else:
            fair = OSM_FAIR_BLEND * fair_est + (1 - OSM_FAIR_BLEND) * OSM_PRIOR_FAIR
        fair_round = round(fair)
        buy_cap = lim - pos; sell_cap = lim + pos
        buy_take_edge = OSM_TAKE_EDGE_LOADED if pos <= -OSM_LOAD_THRESH else OSM_TAKE_EDGE
        sell_take_edge = OSM_TAKE_EDGE_LOADED if pos >= OSM_LOAD_THRESH else OSM_TAKE_EDGE
        if pos <= -OSM_LOAD_THRESH:
            buy_take_edge = OSM_TAKE_EDGE        
            sell_take_edge = OSM_TAKE_EDGE_LOADED 
        elif pos >= OSM_LOAD_THRESH:
            buy_take_edge = OSM_TAKE_EDGE_LOADED  
            sell_take_edge = OSM_TAKE_EDGE        
        for px in sorted(od.sell_orders.keys()):
            if px <= fair - buy_take_edge and buy_cap > 0:
                v = min(buy_cap, -od.sell_orders[px])
                if v > 0:
                    orders.append(Order(product, px, v)); buy_cap -= v
            else:
                break
        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px >= fair + sell_take_edge and sell_cap > 0:
                v = min(sell_cap, od.buy_orders[px])
                if v > 0:
                    orders.append(Order(product, px, -v)); sell_cap -= v
            else:
                break
        if pos > 0 and sell_cap > 0 and fair_round in od.buy_orders:
            v = min(sell_cap, od.buy_orders[fair_round], pos)
            if v > 0:
                orders.append(Order(product, fair_round, -v)); sell_cap -= v
        elif pos < 0 and buy_cap > 0 and fair_round in od.sell_orders:
            v = min(buy_cap, -od.sell_orders[fair_round], -pos)
            if v > 0:
                orders.append(Order(product, fair_round, v)); buy_cap -= v
        if OSM_SOFT_CLEAR:
            if pos >= lim - 5 and sell_cap > 0:
                target_px = fair_round - 1
                if target_px in od.buy_orders:
                    v = min(sell_cap, od.buy_orders[target_px], pos)
                    if v > 0:
                        orders.append(Order(product, target_px, -v)); sell_cap -= v
            elif pos <= -(lim - 5) and buy_cap > 0:
                target_px = fair_round + 1
                if target_px in od.sell_orders:
                    v = min(buy_cap, -od.sell_orders[target_px], -pos)
                    if v > 0:
                        orders.append(Order(product, target_px, v)); buy_cap -= v
        bid_px = min(best_bid + 1, fair_round - 1)
        ask_px = max(best_ask - 1, fair_round + 1)
        if pos >= OSM_SKEW_THRESH:
            ask_px = max(fair_round + 1, best_ask - 2)
        elif pos <= -OSM_SKEW_THRESH:
            bid_px = min(fair_round - 1, best_bid + 2)
        STUCK_THRESH = 50
        STUCK_SIZE = 15
        if pos >= STUCK_THRESH and sell_cap > 0:
            sz = min(STUCK_SIZE, sell_cap)
            orders.append(Order(product, fair_round + 1, -sz))
            if sell_cap - sz > 0:
                orders.append(Order(product, ask_px, -(sell_cap - sz)))
            if buy_cap > 0:
                orders.append(Order(product, bid_px, buy_cap))
        elif pos <= -STUCK_THRESH and buy_cap > 0:
            sz = min(STUCK_SIZE, buy_cap)
            orders.append(Order(product, fair_round - 1, sz))
            if buy_cap - sz > 0:
                orders.append(Order(product, bid_px, buy_cap - sz))
            if sell_cap > 0:
                orders.append(Order(product, ask_px, -sell_cap))
        else:
            if buy_cap > 0:
                orders.append(Order(product, bid_px, buy_cap))
            if sell_cap > 0:
                orders.append(Order(product, ask_px, -sell_cap))
        return orders
    def _trade_pepper(self, product, od, pos, ts, td):
        orders: List[Order] = []
        lim = LIMITS[product]
        if not od.buy_orders or not od.sell_orders:
            return orders
        best_bid = max(od.buy_orders); best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0
        hist = td.setdefault("pep_hist", [])
        if not hist or (ts - hist[-1][0]) >= PEP_HIST_INTERVAL:
            hist.append([ts, mid])
            if len(hist) > PEP_HIST_MAX:
                del hist[: len(hist) - PEP_HIST_MAX]
        slope, trend_fair = self._estimate_pep_trend(hist, ts, mid)
        if slope is not None and slope <= PEP_BEAR_SLOPE:
            return orders
        eff_slope = slope if slope is not None else PEP_PRIOR_SLOPE
        remaining_drift = max(0, eff_slope) * max(0, PEP_DAY_LEN - ts)
        chase = min(PEP_MAX_CHASE, max(1, PEP_SAFETY * remaining_drift))
        dip = trend_fair - mid
        z = dip / PEP_NOISE_STD
        target_frac = max(PEP_BASE_FRAC, min(1.0, PEP_BASE_FRAC + PEP_DIP_GAIN * z))
        target_pos = int(lim * target_frac)
        buy_cap_agg = max(0, target_pos - pos)
        if buy_cap_agg > 0 and chase > 0:
            for px in sorted(od.sell_orders.keys()):
                if px <= mid + chase and buy_cap_agg > 0:
                    v = min(buy_cap_agg, -od.sell_orders[px])
                    if v > 0:
                        orders.append(Order(product, px, v))
                        buy_cap_agg -= v
                else:
                    break
        if pos > 0 and -dip / PEP_NOISE_STD >= PEP_SELL_PEAK_Z:
            sell_qty = int(pos * PEP_SELL_FRAC_PEAK)
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if sell_qty <= 0: break
                if px >= int(trend_fair):
                    v = min(sell_qty, od.buy_orders[px])
                    if v > 0:
                        orders.append(Order(product, px, -v))
                        sell_qty -= v
                else:
                    break
        pos_after = pos + sum(o.quantity for o in orders if o.quantity > 0)
        buy_cap_pass = lim - pos_after
        if buy_cap_pass > 0:
            bid_px = max(best_bid + 1, int(trend_fair) - 1)
            bid_px = min(bid_px, best_ask - 1)
            orders.append(Order(product, bid_px, buy_cap_pass))
        return orders
    def _estimate_osm_fair(self, hist):
        n = len(hist)
        if n < OSM_MIN_SAMPLES:
            return None
        recent = hist[-OSM_FIT_WINDOW:] if n > OSM_FIT_WINDOW else hist
        vals = sorted(h[1] for h in recent)
        m = vals[len(vals)//2]
        return m
    def _estimate_pep_trend(self, hist, ts, mid):
        n = len(hist)
        if n < PEP_MIN_SAMPLES:
            if hist:
                f0, t0 = hist[0][1], hist[0][0]
                return None, f0 + PEP_PRIOR_SLOPE * (ts - t0)
            return None, mid
        recent = hist[-PEP_FIT_WINDOW:] if n > PEP_FIT_WINDOW else hist
        n = len(recent)
        sx = sum(h[0] for h in recent); sy = sum(h[1] for h in recent)
        sxy = sum(h[0]*h[1] for h in recent); sxx = sum(h[0]*h[0] for h in recent)
        denom = n*sxx - sx*sx
        if denom <= 1e-9: return 0.0, mid
        slope = (n*sxy - sx*sy) / denom
        intercept = (sy - slope*sx) / n
        return slope, slope*ts + intercept

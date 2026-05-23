import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
class Trader:
    LIMITS = {"EMERALDS": 50, "TOMATOES": 50}
    EMERALD_FAIR = 10000
    ALPHA_MIN = 0.50
    ALPHA_MAX = 0.95
    ALPHA_SENSITIVITY = 1.5
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self._trade_emeralds(state, data)
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self._trade_tomatoes(state, data)
        return result, conversions, json.dumps(data)
    def _trade_emeralds(self, state: TradingState, data: dict) -> list[Order]:
        return self._trade_generic(state, "EMERALDS", self.EMERALD_FAIR)
    def _trade_tomatoes(self, state: TradingState, data: dict) -> list[Order]:
        od = state.order_depths["TOMATOES"]
        mid = self._vwap_mid(od)
        if mid is None:
            return []
        k = "t_ema"
        ema = data.get(k, mid)
        deviation = abs(mid - ema)
        alpha_range = self.ALPHA_MAX - self.ALPHA_MIN
        alpha = self.ALPHA_MIN + alpha_range * min(1.0, deviation / (self.ALPHA_SENSITIVITY * 3.0))
        ema = alpha * mid + (1 - alpha) * ema
        data[k] = ema
        fair = ema
        if hasattr(state, 'market_trades') and "TOMATOES" in state.market_trades:
            trades = state.market_trades["TOMATOES"]
            net_flow = 0
            for trade in trades:
                if trade.price >= ema:
                    net_flow += trade.quantity
                else:
                    net_flow -= trade.quantity
            flow_key = "t_flow"
            prev_flow = data.get(flow_key, 0.0)
            smoothed_flow = 0.4 * net_flow + 0.6 * prev_flow
            data[flow_key] = smoothed_flow
            flow_signal = max(-1.5, min(1.5, smoothed_flow * 0.05))
            fair += flow_signal
        return self._trade_generic(state, "TOMATOES", fair)
    def _trade_generic(self, state: TradingState, product: str,
                       fair: float) -> list[Order]:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []
        abs_pos = abs(pos)
        if pos > 0:
            if abs_pos > 35:
                buy_thresh = fair - 2
                sell_thresh = fair - 2
            else:
                buy_thresh = fair - 1
                sell_thresh = fair - 1
        elif pos < 0:
            if abs_pos > 35:
                buy_thresh = fair + 2
                sell_thresh = fair + 2
            else:
                buy_thresh = fair + 1
                sell_thresh = fair + 1
        else:
            buy_thresh = fair
            sell_thresh = fair

        if od.sell_orders:
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px <= buy_thresh:
                    vol = -od.sell_orders[ask_px]
                    qty = min(vol, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_px, qty))
                        pos += qty
                else:
                    break

        if od.buy_orders:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px >= sell_thresh:
                    vol = od.buy_orders[bid_px]
                    qty = min(vol, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_px, -qty))
                        pos -= qty
                else:
                    break
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        fair_floor = math.floor(fair)
        fair_ceil = math.ceil(fair)
        if fair_floor == fair_ceil:
            fair_ceil += 1

        pj_bid = (best_bid + 1) if best_bid is not None else (fair_floor - 1)
        pj_ask = (best_ask - 1) if best_ask is not None else (fair_ceil + 1)

        pj_bid = min(pj_bid, fair_floor)
        pj_ask = max(pj_ask, fair_ceil)
        abs_pos = abs(pos)
        if abs_pos <= 18:
            our_bid = pj_bid
            our_ask = pj_ask
        elif abs_pos <= 40:
            t = min((abs_pos - 18) / 22.0, 1.0)
            if pos > 0:
                target_ask = fair_ceil
                our_ask = round(pj_ask * (1 - t) + target_ask * t)
                our_ask = max(our_ask, fair_ceil)
                our_bid = pj_bid
            else:
                target_bid = fair_floor
                our_bid = round(pj_bid * (1 - t) + target_bid * t)
                our_bid = min(our_bid, fair_floor)
                our_ask = pj_ask
        else:
            if pos > 0:
                our_ask = fair_ceil
                our_bid = pj_bid
            else:
                our_bid = fair_floor
                our_ask = pj_ask

        if our_bid >= our_ask:
            our_bid = fair_floor - 1
            our_ask = fair_ceil + 1

        buy_qty = limit - pos
        sell_qty = limit + pos

        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))

        return orders

    @staticmethod
    def _vwap_mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bid_vol = od.buy_orders[best_bid]
            ask_vol = -od.sell_orders[best_ask]
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                return (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            return (best_bid + best_ask) / 2.0
        return None

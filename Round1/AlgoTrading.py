import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
class Trader:
    LIMITS = {"ASH_COATED_OSMIUM": 50, "INTARIAN_PEPPER_ROOT": 50}
    OSMIUM_FAIR = 10000
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._osmium(state)
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self._pepper(state, data)
        return result, conversions, json.dumps(data)
    def _osmium(self, state: TradingState) -> list[Order]:
        product = "ASH_COATED_OSMIUM"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        fair = self.OSMIUM_FAIR
        orders: list[Order] = []
        if od.sell_orders:
            for px in sorted(od.sell_orders):
                if px > fair - 1:
                    break
                vol = min(-od.sell_orders[px], limit - pos)
                if vol > 0:
                    orders.append(Order(product, px, vol))
                    pos += vol
        if od.buy_orders:
            for px in sorted(od.buy_orders, reverse=True):
                if px < fair + 1:
                    break
                vol = min(od.buy_orders[px], limit + pos)
                if vol > 0:
                    orders.append(Order(product, px, -vol))
                    pos -= vol
        r = fair
        math_r_floor = math.floor(r) - 1
        math_r_ceil = math.ceil(r) + 1
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            our_bid = min(best_bid + 1, math_r_floor)
        else:
            our_bid = min(fair - 12, math_r_floor)
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            our_ask = max(best_ask - 1, math_r_ceil)
        else:
            our_ask = max(fair + 12, math_r_ceil)
        if our_bid >= our_ask:
            our_bid = math.floor(r) - 2
            our_ask = math.ceil(r) + 2
        buy_qty = limit - pos
        sell_qty = limit + pos
        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))
        return orders
    def _pepper(self, state: TradingState, data: dict) -> list[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.LIMITS[product]
        orders: list[Order] = []
        mid = self._mid(od)
        if mid is None:
            return []
        ema = data.get("p_ema", mid)
        ema = 0.9 * ema + 0.1 * mid
        data["p_ema"] = ema
        if od.buy_orders and pos > 0:
            for px in sorted(od.buy_orders, reverse=True):
                if px > ema + 25:
                    vol_to_sell = min(od.buy_orders[px], pos)
                    if vol_to_sell > 0:
                        orders.append(Order(product, px, -vol_to_sell))
                        pos -= vol_to_sell
                else:
                    break
        buy_capacity = limit - pos
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if od.sell_orders and buy_capacity > 0 and best_ask is not None:
            take_left = min(buy_capacity, 16)
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px <= best_ask + 0:
                    if ask_px > ema + 10:
                        break
                    vol = min(-od.sell_orders[ask_px], take_left)
                    if vol > 0:
                        orders.append(Order(product, ask_px, vol))
                        take_left -= vol
                        buy_capacity -= vol
                else:
                    break
        if buy_capacity > 0:
            target_bid = best_bid + 1 if best_bid is not None else int(mid)
            if best_ask is not None:
                our_bid = min(best_ask - 1, target_bid)
            else:
                our_bid = target_bid
            orders.append(Order(product, our_bid, buy_capacity))

        return orders
    @staticmethod
    def _mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        elif od.buy_orders:
            return float(max(od.buy_orders))
        elif od.sell_orders:
            return float(min(od.sell_orders))
        return None

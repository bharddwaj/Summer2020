import sys
import time
import numpy as np
import shift

def algo_1(trader: shift.Trader):
    '''Moving average algorithm'''
    print(trader.get_last_price("MSFT"))
    best_price = trader.get_best_price("MSFT")
    print(f"ask price: {best_price.get_ask_price()}")
    print(f"local ask: {best_price.get_local_ask_price()}")
    print(f"global ask: {best_price.get_global_ask_price()}")
    print(f"bid price: {best_price.get_bid_price()}")
    print(f"local bid: {best_price.get_local_bid_price()}")
    print(f"global bid: {best_price.get_global_bid_price()}")
    print(trader.get_close_price("MSFT"))
    # portfolio_summary = trader.get_portfolio_summary()
    # print(trader.get_subscribed_order_book_list())
    # print(f" total bp: {portfolio_summary.get_total_bp()}")
    # print(f"total shares: {portfolio_summary.get_total_shares()}")
    # print(f"total realized pl: {portfolio_summary.get_total_realized_pl()}")
    # print(f"timestamp: {portfolio_summary.get_timestamp()}")
    # print("Symbol\t\tShares\t\tPrice\t\tP&L\t\tTimestamp")
    # for item in trader.get_portfolio_items().values():
    #     print(
    #         "%6s\t\t%6d\t%9.2f\t%7.2f\t\t%26s"
    #         % (
    #             item.get_symbol(),
    #             item.get_shares(),
    #             item.get_price(),
    #             item.get_realized_pl(),
    #             item.get_timestamp(),
    #         )
    #     )
    
    # prices = []
    # for i in range(10000):
    #     msft_last_price = trader.get_last_price("MSFT")
    #     msft_buy_market = shift.Order(shift.Order.Type.MARKET_BUY, "MSFT",1)
    #     msft_sell_market = shift.Order(shift.Order.Type.MARKET_SELL, "MSFT",1)
    #     if not prices:
    #         trader.submit_order(msft_buy_market)
    #         prices.append(msft_last_price)
    #     elif msft_last_price < np.mean(prices):
    #         trader.submit_order(msft_buy_market)
    #         prices.append(msft_last_price)
    #     elif msft_last_price > np.mean(prices):
    #         trader.submit_order(msft_sell_market)
    #         prices.append(msft_last_price)
    #     else:
    #         pass
    # print("----Finished------")
    # print(prices)
    # portfolio_summary = trader.get_portfolio_summary()
    # print(f" total bp: {portfolio_summary.get_total_bp()}")
    # print(f"total shares: {portfolio_summary.get_total_shares()}")
    # print(f"total realized pl: {portfolio_summary.get_total_realized_pl()}")
    # print(f"timestamp: {portfolio_summary.get_timestamp()}")
    
def main():
    # create trader object
    trader = shift.Trader("test001")

    # connect and subscribe to all available order books
    try:
        trader.connect("initiator.cfg", "password")
        trader.sub_all_order_book()
        time.sleep(1)
    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)
    algo_1(trader)
    trader.disconnect()
main()
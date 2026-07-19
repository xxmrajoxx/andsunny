"""Show the top 10 best performing stocks today using yfinance's screener."""

import yfinance as yf


def get_top_performers(n: int = 10):
    result = yf.screen("day_gainers", count=n)
    return result["quotes"][:n]


def main():
    quotes = get_top_performers(10)

    print(f"{'Symbol':<8}{'Name':<35}{'Price':>10}{'Change %':>12}")
    print("-" * 65)
    for q in quotes:
        symbol = q.get("symbol", "")
        name = (q.get("shortName") or "")[:34]
        price = q.get("regularMarketPrice", 0)
        change_pct = q.get("regularMarketChangePercent", 0)
        print(f"{symbol:<8}{name:<35}{price:>10.2f}{change_pct:>11.2f}%")


if __name__ == "__main__":
    main()
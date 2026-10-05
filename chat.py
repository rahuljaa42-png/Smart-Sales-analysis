"""
Talk to your sales data.

    python chat.py                      # interactive chat with the LLM
    python chat.py --ask "Why did the North region drop?"
    python chat.py --offline            # no LLM/API key: pick analysis tools from a menu

Set your LLM in config.py (LLM_MODEL) or with the LLM_MODEL environment variable.
"""
import argparse
import sys

EXAMPLES = [
    "Why did the North region drop?",
    "Which region is growing fastest, and what's the forecast for next month?",
    "Who are our champion customers, and who is at risk of churning?",
    "Were there any suspicious orders?",
]


def offline_menu():
    from src.analyst_tools import SalesAnalyst
    a = SalesAnalyst()
    menu = {
        "1": ("Overview", a.overview),
        "2": ("Explain a region's change", lambda: a.explain_region_change(input("Region: "))),
        "3": ("Compare regions", a.compare_regions),
        "4": ("Region monthly trend", lambda: a.region_trend(input("Region (or all): ") or "all")),
        "5": ("Sales forecast", a.sales_forecast),
        "6": ("Customer segments", a.customer_segments),
        "7": ("Churn-risk customers", a.churn_risk_customers),
        "8": ("Unusual orders", a.unusual_orders),
        "9": ("Top products", a.top_products),
    }
    while True:
        print("\n" + "\n".join(f"  {k}. {name}" for k, (name, _) in menu.items()) + "\n  q. Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "q":
            break
        if choice in menu:
            try:
                print("\n" + menu[choice][1]())
            except ValueError as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Smart Sales Analyst chatbot")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--offline", action="store_true", help="menu of tools, no LLM needed")
    parser.add_argument("--model", help='LangChain model string, e.g. "ollama:llama3.1"')
    args = parser.parse_args()

    if args.offline:
        return offline_menu()

    from src.chatbot import SalesChatbot
    try:
        bot = SalesChatbot(args.model)
    except Exception as e:
        print(f"Could not start the LLM: {e}\n"
              "Check your API key / model name in config.py, or run: python chat.py --offline")
        sys.exit(1)

    if args.ask:
        print(bot.ask(args.ask, show_tools=True))
        return

    print("Smart Sales Analyst - ask about your sales data (type 'exit' to quit).")
    print("Try:\n" + "\n".join(f"  - {q}" for q in EXAMPLES))
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            print(f"\nAnalyst: {bot.ask(q, show_tools=True)}")


if __name__ == "__main__":
    main()

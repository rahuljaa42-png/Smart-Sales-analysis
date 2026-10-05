# Smart Sales Analyst

An end-to-end ML project that takes a sales sheet in **Excel**, cleans it and finds customer
segments and suspicious orders with **scikit-learn**, forecasts next month's sales with
**PyTorch**, lets you ask questions in plain English through a **LangChain** chatbot, and
writes everything to an **Excel** report with charts.

```
sales_data.xlsx ──► clean (sklearn imputer) ──► KMeans segments + IsolationForest anomalies
                                           └──► PyTorch MLP forecast
                                                     │
                      saved results ◄────────────────┘
                        │          │
             LangChain chatbot   sales_report.xlsx (charts + formulas)
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py           # creates data/sales_data.xlsx (sample data)
python main.py                    # runs steps 1-5, writes outputs/sales_report.xlsx
python chat.py --offline          # try the analysis tools (no API key needed)
python chat.py                    # chat with the LLM
```

### Setting up the LLM (pick one, both free)

**Groq (cloud, free tier)** is the default:
1. Get a key at console.groq.com.
2. Set it:
   - Mac/Linux: `export GROQ_API_KEY="your-key"`
   - Windows: `set GROQ_API_KEY=your-key`

**Ollama (runs on your laptop, no key):**
1. Install Ollama, then run `ollama pull llama3.1`.
2. Run `pip install langchain-ollama`.
3. Run `python chat.py --model ollama:llama3.1`, or change `LLM_MODEL` in `config.py`.

Any LangChain `provider:model` string works, e.g. `openai:gpt-4o-mini` with `langchain-openai`.

## Project structure

| File | What it does | Tool |
|---|---|---|
| `generate_data.py` | Makes 2 years of messy sample sales data with a hidden story | pandas |
| `src/data_prep.py` | Removes duplicates, fixes "NORTH"/"north ", negatives, missing values | pandas, `SimpleImputer` |
| `src/segmentation.py` | RFM features → `KMeans` (best k by silhouette); `IsolationForest` anomalies | scikit-learn |
| `src/forecast.py` | Windowed MLP forecaster + back-test against a naive baseline | PyTorch |
| `src/analyst_tools.py` | Plain-Python analysis functions (the chatbot's "tools") | pandas |
| `src/chatbot.py` | Wraps the tools with `@tool`; `create_agent` picks which to call | LangChain |
| `src/report.py` | 6-sheet workbook, native Excel charts, live formulas | openpyxl |
| `main.py` | Runs the pipeline | |
| `chat.py` | Chat interface (`--ask`, `--offline`, `--model`) | |

## Using your own data

Your sheet needs these columns:
`OrderID, OrderDate, CustomerID, CustomerName, Region, Category, Product, Quantity, UnitPrice, Discount, Sales`.

It also needs at least 12 months of data. Then run:

```bash
python main.py --input path/to/your_sales.xlsx
```

## How each piece works

**Segmentation.** Each customer gets 3 numbers:
- **R**ecency: days since their last order
- **F**requency: number of orders
- **M**onetary: total spend

Monetary is log-scaled and all three are standardised. KMeans tries k = 3..6 and keeps the k with the best
silhouette score. Clusters are then named (Champions, Loyal Regulars, Lapsed...) from their
averages. A separate rule flags **churn risk**: top-25% spenders with no order in 60+ days.

**Anomalies.** Each order's quantity is compared with that customer's usual quantity *and*
that product's usual quantity. IsolationForest flags orders far above both. Comparing
per customer means regular bulk buyers don't get flagged. Flagged orders are excluded from
trends and the forecast.

**Forecast.** The model sees the last 6 months as `log(sales / latest month)` plus the
month-of-year as sin/cos. From that it predicts next month's relative change. All regions
are pooled into one training set.

The last 3 months are hidden and predicted to measure error (MAPE). This is compared with
a naive "next month = this month" guess. Be honest in your write-up: a neural net on 24
monthly points is small data. Where it doesn't beat naive, say so. No model can predict a
sudden structural break, like losing two key accounts, from history alone.

**Chatbot.** The LLM never does the maths itself. It chooses a tool, such as
`explain_region_change("North")`, then reads the facts the tool returns and explains them.
This keeps answers grounded in real numbers. Try asking:
- "Why did the North region drop?"
- "Which region is growing fastest, and what's next month's forecast?"
- "Who are our champion customers and who is at risk of churning?"
- "Were there any suspicious orders?"

## Ideas to extend it

- Swap the MLP for an `nn.LSTM`, and compare MAPE on the same back-test.
- Add weather, festival or holiday features to the forecast.
- Build a Streamlit UI: upload an Excel file, see the charts, and chat, all on one page.
- Add a `what_if_discount` tool that estimates sales at different discount levels.

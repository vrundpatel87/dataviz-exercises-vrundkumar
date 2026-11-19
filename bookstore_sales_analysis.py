"""
Bookstore Sales Analysis
------------------------
Run this script in the same folder as:
    orders.csv
    order_items.csv
    books.csv
    customers.csv
    returns.csv

It also accepts the uploaded filename variants such as orders(1).csv.

Requirements:
    pip install pandas matplotlib

Outputs:
    monthly_revenue_margin.csv
    q1_category_ranking.csv
    customer_segment_comparison.csv
    monthly_revenue_margin.png
    q1_category_ranking.png
    customer_segment_comparison.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR


def find_csv(base_name):
    """Find base_name.csv or a numbered variant such as base_name(1).csv."""
    candidates = [
        DATA_DIR / f"{base_name}.csv",
        *sorted(DATA_DIR.glob(f"{base_name}(*)*.csv")),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {base_name}.csv in {DATA_DIR}. "
        f"Place the dataset CSV files beside this Python script."
    )


def load_data():
    orders = pd.read_csv(find_csv("orders"))
    order_items = pd.read_csv(find_csv("order_items"))
    books = pd.read_csv(find_csv("books"))
    customers = pd.read_csv(find_csv("customers"))
    returns = pd.read_csv(find_csv("returns"))
    return orders, order_items, books, customers, returns


def prepare_sales_data(orders, order_items, books, returns):
    """Join source tables and calculate net quantity, revenue and gross margin."""
    required = {
        "orders": (orders, ["order_id", "order_date", "customer_id", "status", "discount_pct"]),
        "order_items": (order_items, ["order_item_id", "order_id", "book_id", "quantity", "unit_price"]),
        "books": (books, ["book_id", "category", "unit_cost"]),
        "returns": (returns, ["order_item_id", "quantity"]),
    }
    for table_name, (frame, columns) in required.items():
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise ValueError(f"{table_name}.csv is missing columns: {missing}")

    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    if orders["order_date"].isna().any():
        raise ValueError("Some order_date values could not be parsed as dates.")

    # The supplied dataset stores discount_pct as percentage points (e.g. 5 = 5%).
    returned_by_item = (
        returns.groupby("order_item_id", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "returned_quantity"})
    )

    sales = (
        order_items
        .merge(orders, on="order_id", how="inner")
        .merge(books[["book_id", "category", "unit_cost"]], on="book_id", how="left")
        .merge(returned_by_item, on="order_item_id", how="left")
    )

    sales["returned_quantity"] = sales["returned_quantity"].fillna(0)
    # A return cannot reduce sold quantity below zero.
    sales["returned_quantity"] = sales[
        ["returned_quantity", "quantity"]
    ].min(axis=1)

    sales["net_quantity"] = sales["quantity"] - sales["returned_quantity"]
    sales["discount_pct"] = pd.to_numeric(sales["discount_pct"], errors="coerce").fillna(0)
    sales["unit_price"] = pd.to_numeric(sales["unit_price"], errors="coerce")
    sales["unit_cost"] = pd.to_numeric(sales["unit_cost"], errors="coerce")

    sales["net_revenue"] = (
        sales["net_quantity"]
        * sales["unit_price"]
        * (1 - sales["discount_pct"] / 100.0)
    )
    sales["gross_margin"] = (
        sales["net_revenue"] - sales["net_quantity"] * sales["unit_cost"]
    )

    # Match the assignment analysis: completed orders only.
    sales = sales[sales["status"].astype(str).str.lower().eq("completed")].copy()
    sales["year"] = sales["order_date"].dt.year
    sales["month_number"] = sales["order_date"].dt.month
    sales["month"] = sales["order_date"].dt.strftime("%B")
    sales["quarter"] = sales["order_date"].dt.quarter
    return sales


def save_table_image(frame, filename, title, figsize=(10, 4)):
    """Save a readable screenshot-style image of a result table."""
    image_path = OUTPUT_DIR / filename
    display = frame.copy()

    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]) and column not in {
            "rank", "category_rank", "total_orders", "orders", "month_number"
        }:
            display[column] = display[column].map(
                lambda value: f"{value:,.2f}" if pd.notna(value) else ""
            )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, pad=14)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return image_path


def query_1_monthly_performance(sales):
    monthly = (
        sales[sales["year"] == 2025]
        .groupby(["month_number", "month"], as_index=False)
        .agg(
            net_revenue=("net_revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .sort_values("month_number")
    )
    monthly[["net_revenue", "gross_margin"]] = monthly[
        ["net_revenue", "gross_margin"]
    ].round(2)

    monthly.to_csv(OUTPUT_DIR / "monthly_revenue_margin.csv", index=False)
    save_table_image(
        monthly[["month", "net_revenue", "gross_margin"]],
        "monthly_revenue_margin.png",
        "Monthly Net Revenue and Gross Margin — 2025",
        figsize=(9, 4),
    )

    if not monthly.empty:
        strongest = monthly.loc[monthly["net_revenue"].idxmax()]
        print("\nQUESTION 1 — MONTHLY PERFORMANCE")
        print(monthly[["month", "net_revenue", "gross_margin"]].to_string(index=False))
        print(
            f"\nStrongest month by net revenue: {strongest['month']} "
            f"(net revenue {strongest['net_revenue']:,.2f}; "
            f"gross margin {strongest['gross_margin']:,.2f})"
        )
    else:
        print("\nNo 2025 completed-order data found for Question 1.")
    return monthly


def query_2_q1_category_ranking(sales):
    q1_sales = sales[
        (sales["year"] == 2025) & (sales["quarter"] == 1)
    ].copy()

    category_monthly = (
        q1_sales
        .groupby(["month_number", "month", "category"], as_index=False)
        .agg(category_revenue=("net_revenue", "sum"))
    )
    category_monthly["rank"] = (
        category_monthly.groupby("month_number")["category_revenue"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    category_monthly = category_monthly.sort_values(
        ["month_number", "rank", "category"]
    )
    category_monthly["category_revenue"] = category_monthly[
        "category_revenue"
    ].round(2)

    category_monthly.to_csv(OUTPUT_DIR / "q1_category_ranking.csv", index=False)
    save_table_image(
        category_monthly[["month", "category", "category_revenue", "rank"]],
        "q1_category_ranking.png",
        "Q1 2025 — Category Revenue Ranked Within Each Month",
        figsize=(10, 7),
    )

    print("\nQUESTION 2 — Q1 CATEGORY RANKING")
    if category_monthly.empty:
        print("No completed-order data found for Q1 2025.")
    else:
        print(
            category_monthly[
                ["month", "category", "category_revenue", "rank"]
            ].to_string(index=False)
        )
    return category_monthly


def query_3_customer_segments(sales, customers):
    if not {"customer_id", "segment"}.issubset(customers.columns):
        raise ValueError("customers.csv must contain customer_id and segment columns.")

    segment_sales = sales.merge(
        customers[["customer_id", "segment"]],
        on="customer_id",
        how="left",
    )

    result = (
        segment_sales
        .groupby("segment", dropna=False, as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            total_revenue=("net_revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
    )
    result["average_revenue_per_order"] = (
        result["total_revenue"] / result["total_orders"].replace(0, pd.NA)
    )
    result = result.sort_values("total_revenue", ascending=False)
    numeric_columns = [
        "total_revenue", "gross_margin", "average_revenue_per_order"
    ]
    result[numeric_columns] = result[numeric_columns].round(2)

    result.to_csv(OUTPUT_DIR / "customer_segment_comparison.csv", index=False)
    save_table_image(
        result,
        "customer_segment_comparison.png",
        "Customer Segment Comparison",
        figsize=(10, 4),
    )

    print("\nQUESTION 3 — CUSTOMER SEGMENTS")
    print(result.to_string(index=False))
    return result


def main():
    print("Loading bookstore dataset...")
    orders, order_items, books, customers, returns = load_data()
    sales = prepare_sales_data(orders, order_items, books, returns)

    print(f"Completed sales lines analysed: {len(sales):,}")
    query_1_monthly_performance(sales)
    query_2_q1_category_ranking(sales)
    query_3_customer_segments(sales, customers)

    print("\nAnalysis complete.")
    print(f"CSV result tables and PNG output images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_categorical_summary(df, max_unique=20, include_columns=None, show_categories_frequencies=False):
    """
    Summarizes categorical columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - max_unique: Max unique values to auto-include a column (default: 20)
    - include_columns: Columns to force include even if they exceed max_unique
    """

    column_summaries = []     # Holds summary info for each column
    value_counts_output = {}  # Holds value counts for later display

    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        is_included = (
            (include_columns and col in include_columns)
            or (nunique <= max_unique)
        )
        if is_included:
            counts = df[col].value_counts(dropna=False)
            total = len(df)
            nan_pct = df[col].isna().mean() * 100

            non_nan_counts = df[col].dropna().value_counts()
            # # Get major cat
            # if counts.idxmax() is pd.NA or pd.isna(counts.idxmax()):
            #     main_cat = non_nan_counts.idxmax()
            #     main_pct = non_nan_counts.max() / total * 100
            # else:
            #     main_cat = counts.idxmax()
            #     main_pct = counts.max() / total * 100
            # # Get minor cat
            # if counts.idxmin() is pd.NA or pd.isna(counts.idxmin()):
            #     minor_cat = non_nan_counts.idxmin()
            #     minor_pct = non_nan_counts.min() / total * 100
            # else:
            #     minor_cat = counts.idxmin()
            #     minor_pct = counts.min() / total * 100

            if not non_nan_counts.empty:
                main_cat = non_nan_counts.idxmax()
                main_pct = non_nan_counts.max() / total * 100
                minor_cat = non_nan_counts.idxmin()
                minor_pct = non_nan_counts.min() / total * 100
            else:
                main_cat = None
                main_pct = 0.0
                minor_cat = None
                minor_pct = 0.0


            column_summaries.append({
                "column": col,
                "n_unique": nunique,
                "nan_pct": round(nan_pct, 2),
                "main_cat": main_cat,
                "main_pct": round(main_pct, 2),
                "minor_cat": minor_cat,
                "minor_pct": round(minor_pct, 2),
            })

            value_counts_output[col] = counts

    # Create and display summary table
    summary_df = pd.DataFrame(column_summaries)
    print("=== CATEGORICAL COLUMN SUMMARY ===")
    display(summary_df)

    # Show value counts for each included column
    if show_categories_frequencies:
        for col, counts in value_counts_output.items():
            print(f"\n--- Column: {col} ---")
            print(counts)

def clean_dataframe(
    df: pd.DataFrame,
    string_cols: list,
    nan_threshold: float = 1.0,
    verbose: bool = True
):
    """
    Cleans a DataFrame using explicitly provided column types.
    
    Parameters:
    - df: DataFrame to clean
    - string_cols: list of string columns
    - numeric_cols: list of numeric columns
    - date_cols: list of datetime columns
    - nan_threshold: float [0, 1] threshold for dropping columns with too many NaNs
    - verbose: whether to print cleaning summary
    
    Returns:
    - Cleaned DataFrame
    """
    df = df.copy()
    summary = {
        "duplicated_rows_removed": 0,
        "columns_dropped": [],
        "column_changes": []
    }

    # Remove duplicated rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    summary["duplicated_rows_removed"] = dupes

    # Drop columns exceeding NaN threshold
    for col in df.columns:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > nan_threshold:
            df.drop(columns=[col], inplace=True)
            summary["columns_dropped"].append(col)

    # Normalize string columns
    for col in string_cols:
        if col in df.columns:
            before = df[col].copy()
            df[col] = df[col].astype(str).str.strip().str.lower().replace({'': np.nan, 'nan': np.nan})
            changes = (before != df[col]).sum(skipna=False)
            if changes > 0:
                summary["column_changes"].append({
                    "column": col,
                    "changed_values": int(changes),
                    "dtype_before": before.dtype,
                    "dtype_after": df[col].dtype
                })

    if verbose:
        print("=== DATA CLEANING SUMMARY ===")
        print(f"Duplicated rows removed: {summary['duplicated_rows_removed']}")
        if summary['columns_dropped']:
            print(f"Columns dropped due to NaN threshold: {summary['columns_dropped']}")
        if summary['column_changes']:
            print("Column transformations:")
            changes_df = pd.DataFrame(summary['column_changes'])
            display(changes_df)
        else:
            print("No column values changed.")

    return df

def map_region(region, hierarchy):
    if region in hierarchy:
        return hierarchy[region]
    # elif region not in hierarchy.values():
    #     return 'other region'
    return region

def map_region_list(region_str, hierarchy):
    if pd.isna(region_str) or not isinstance(region_str, str) or not region_str.strip():
        return region_str
    
    regions = [r.strip() for r in region_str.split(';')]
    mapped_regions = set()

    regions = sorted(regions)
    for region in regions:
        if region in hierarchy:
            mapped_regions.add(hierarchy[region])
        # elif region not in hierarchy.values():
        #     mapped_regions.append('other region')
        else:
            mapped_regions.add(region)

    # Remover duplicatas mantendo ordem
    return '; '.join(dict.fromkeys(mapped_regions))


def plot_stacked_revenue_bins(df_funded, df_not_funded, column='company_revenue_last_fy_usd', bins=6):
    """
    Plots a stacked bar chart of funded and not funded RFPs by equal-width revenue bins.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: column name with revenue values (default: 'company_revenue_last_fy_usd')
    - bins: number of equal-width bins (default: 6)
    """
    
    # Combine and clean revenue values
    funded_revenue = pd.to_numeric(df_funded[column], errors='coerce')
    not_funded_revenue = pd.to_numeric(df_not_funded[column], errors='coerce')
    
    all_revenue = pd.concat([funded_revenue, not_funded_revenue]).dropna()
    
    # Create equal-width bins
    bin_edges = np.histogram_bin_edges(all_revenue, bins=bins)
    funded_binned = pd.cut(funded_revenue, bins=bin_edges)
    not_funded_binned = pd.cut(not_funded_revenue, bins=bin_edges)
    
    # Count occurrences in each bin
    funded_counts = funded_binned.value_counts().sort_index()
    not_funded_counts = not_funded_binned.value_counts().sort_index()
    
    # Prepare bin labels
    bin_labels = [f"{int(interval.left / 1000):,}K - {int(interval.right / 1000):,}K" for interval in funded_counts.index]

    # Align counts
    funded_vals = funded_counts.values
    not_funded_vals = not_funded_counts.values
    total_vals = funded_vals + not_funded_vals

    # Plot
    x = np.arange(len(bin_labels))
    width = 0.6

    plt.figure(figsize=(10, 6))
    plt.bar(x, funded_vals, width, label='Funded', color='#4CAF50')         # Green
    plt.bar(x, not_funded_vals, width, bottom=funded_vals, label='Not Funded', color='#F44336')  # Red

    # Add percentages on top
    for i in range(len(bin_labels)):
        if total_vals[i] > 0:
            pct_funded = funded_vals[i] / total_vals[i] * 100
            pct_not_funded = not_funded_vals[i] / total_vals[i] * 100
            plt.text(x[i], total_vals[i] + 1, f"{pct_funded:.0f}%/{pct_not_funded:.0f}%", 
                     ha='center', va='bottom', fontsize=9)

    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel('Revenue Range (in Thousands USD)')
    plt.ylabel('Number of RFPs')
    plt.title('RFP Funding by Company Revenue Bins')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_deal_size_bins(df_funded, df_not_funded, column='deal_size_usd', bins=6):
    """
    Plots a stacked bar chart of funded and not funded RFPs by equal-width revenue bins.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: column name with deal_size_usd values
    - bins: number of equal-width bins (default: 6)
    """
    
    # Combine and clean deal_size values
    funded_deal_size = pd.to_numeric(df_funded[column], errors='coerce')
    not_funded_deal_size = pd.to_numeric(df_not_funded[column], errors='coerce')
    
    all_deal_size = pd.concat([funded_deal_size, not_funded_deal_size]).dropna()
    
    # Create equal-width bins
    bin_edges = np.histogram_bin_edges(all_deal_size, bins=bins)
    funded_binned = pd.cut(funded_deal_size, bins=bin_edges)
    not_funded_binned = pd.cut(not_funded_deal_size, bins=bin_edges)
    
    # Count occurrences in each bin
    funded_counts = funded_binned.value_counts().sort_index()
    not_funded_counts = not_funded_binned.value_counts().sort_index()
    
    # Prepare bin labels
    bin_labels = [f"{int(interval.left / 1000):,}K - {int(interval.right / 1000):,}K" for interval in funded_counts.index]

    # Align counts
    funded_vals = funded_counts.values
    not_funded_vals = not_funded_counts.values
    total_vals = funded_vals + not_funded_vals

    # Plot
    x = np.arange(len(bin_labels))
    width = 0.6

    plt.figure(figsize=(10, 6))
    plt.bar(x, funded_vals, width, label='Funded', color='#4CAF50')         # Green
    plt.bar(x, not_funded_vals, width, bottom=funded_vals, label='Not Funded', color='#F44336')  # Red

    # Add percentages on top
    for i in range(len(bin_labels)):
        if total_vals[i] > 0:
            pct_funded = funded_vals[i] / total_vals[i] * 100
            pct_not_funded = not_funded_vals[i] / total_vals[i] * 100
            plt.text(x[i], total_vals[i] + 1, f"{pct_funded:.0f}%/{pct_not_funded:.0f}%", 
                     ha='center', va='bottom', fontsize=9)

    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel('Deal Size Range (in Thousands USD)')
    plt.ylabel('Number of RFPs')
    plt.title('RFP Funding by deal size Bins')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_company_stage_impact(df_funded, df_not_funded, column='company_stage'):
    """
    Plots a stacked bar chart of funded and not funded RFPs by company stage.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: categorical column to compare (default: 'company_stage')
    """

    # Count occurrences
    funded_counts = df_funded[column].value_counts().sort_index()
    not_funded_counts = df_not_funded[column].value_counts().sort_index()

    # Align indices
    all_categories = sorted(set(funded_counts.index).union(set(not_funded_counts.index)))
    funded_counts = funded_counts.reindex(all_categories, fill_value=0)
    not_funded_counts = not_funded_counts.reindex(all_categories, fill_value=0)

    # Data for plotting
    funded_vals = funded_counts.values
    not_funded_vals = not_funded_counts.values
    total_vals = funded_vals + not_funded_vals

    x = np.arange(len(all_categories))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, funded_vals, label='Funded', color='#4CAF50')  # Green
    plt.bar(x, not_funded_vals, bottom=funded_vals, label='Not Funded', color='#F44336')  # Red

    # Add percentage labels above stacks (percentage of funded within total)
    for i in range(len(x)):
        total = total_vals[i]
        if total > 0:
            pct_funded = funded_vals[i] / total * 100
            plt.text(x[i], total + 1, f"{pct_funded:.0f}%", ha='center', fontsize=9)

    plt.xticks(x, all_categories, rotation=45)
    plt.xlabel('Company Stage')
    plt.ylabel('Number of RFPs')
    plt.title('RFP Funding by Company Stage')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_percentage_line_company_age_impact(df_funded, df_not_funded, use_submission_date=False):
    """
    Plots a line chart showing the percentage distribution of company ages at funding/request,
    with one line for funded and one for not funded RFPs.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - use_submission_date: If True, uses 'submission_date' instead of 'funding_date'
    """

    date_col = 'submission_date' if use_submission_date else 'funding_date'

    def prepare_company_age(df):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['company_founding_year'] = pd.to_numeric(df['company_founding_year'], errors='coerce')
        df = df.dropna(subset=[date_col, 'company_founding_year'])
        df = df[df['company_founding_year'] > 1900]
        df['company_age'] = df[date_col].dt.year - df['company_founding_year']
        df = df[(df['company_age'] >= 0) & (df['company_age'] <= 100)]
        return df

    # Prepare data
    df_funded = prepare_company_age(df_funded)
    df_not_funded = prepare_company_age(df_not_funded)

    # Count distributions
    funded_counts = df_funded['company_age'].value_counts(normalize=True).sort_index() * 100
    not_funded_counts = df_not_funded['company_age'].value_counts(normalize=True).sort_index() * 100

    # Align ages
    all_ages = sorted(set(funded_counts.index).union(set(not_funded_counts.index)))
    funded_counts = funded_counts.reindex(all_ages, fill_value=0)
    not_funded_counts = not_funded_counts.reindex(all_ages, fill_value=0)

    # Plot
    plt.figure(figsize=(11, 6))
    plt.plot(all_ages, funded_counts.values, label='Funded %', color='#4CAF50', marker='o')
    plt.plot(all_ages, not_funded_counts.values, label='Not Funded %', color='#F44336', marker='o')

    plt.xlabel('Company Age at Time of Request (Years)')
    plt.ylabel('Percentage of RFPs (%)')
    plt.title('Percentage Distribution of RFPs by Company Age at Time of Request')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_sorted_stacked_sector_impact(
    df_funded,
    df_not_funded,
    column='industry_sector',
    top_n=10,
    bottom_n=10,
    sort_by='funded'  # Options: 'funded', 'not_funded', 'funded_pct', 'not_funded_pct'
):
    """
    Plots a stacked bar chart of funded and not funded RFPs by a categorical column,
    sorted by volume or percentage, and showing top and bottom categories.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: Column name to analyze (e.g., 'industry_sector')
    - top_n: Number of top categories to display
    - bottom_n: Number of bottom categories to display
    - sort_by: Sorting method ('funded', 'not_funded', 'funded_pct', 'not_funded_pct')
    """

    # Raw counts
    funded_counts_raw = df_funded[column].value_counts()
    not_funded_counts_raw = df_not_funded[column].value_counts()

    # Combine all categories
    all_categories = set(funded_counts_raw.index).union(set(not_funded_counts_raw.index))
    all_categories = sorted(all_categories)

    # Reindex both count series with all categories
    funded_counts = funded_counts_raw.reindex(all_categories, fill_value=0)
    not_funded_counts = not_funded_counts_raw.reindex(all_categories, fill_value=0)
    total_counts = funded_counts + not_funded_counts

    # Calculate percentages
    funded_pct = funded_counts / total_counts * 100
    not_funded_pct = not_funded_counts / total_counts * 100

    # Define sort source
    if sort_by == 'funded':
        sort_source = funded_counts
    elif sort_by == 'not_funded':
        sort_source = not_funded_counts
    elif sort_by == 'funded_pct':
        sort_source = funded_pct
    elif sort_by == 'not_funded_pct':
        sort_source = not_funded_pct
    else:
        raise ValueError("Invalid 'sort_by'. Choose from 'funded', 'not_funded', 'funded_pct', 'not_funded_pct'.")

    # Select top and bottom categories
    top_categories = sort_source.sort_values(ascending=False).head(top_n).index
    bottom_categories = sort_source.sort_values(ascending=True).head(bottom_n).index
    selected_categories = list(top_categories) + list(bottom_categories)

    # Final filtered counts in desired order
    funded_plot = funded_counts.reindex(selected_categories, fill_value=0)
    not_funded_plot = not_funded_counts.reindex(selected_categories, fill_value=0)
    total_vals = funded_plot + not_funded_plot

    # Percentages for label (summing to 100%)
    funded_pct_display = (funded_plot / total_vals * 100).round(1).fillna(0)
    not_funded_pct_display = (not_funded_plot / total_vals * 100).round(1).fillna(0)

    # Plotting setup
    x = np.arange(len(selected_categories))
    funded_vals = funded_plot.values
    not_funded_vals = not_funded_plot.values

    plt.figure(figsize=(14, 6))
    plt.bar(x, funded_vals, label='Funded', color='#4CAF50')
    plt.bar(x, not_funded_vals, bottom=funded_vals, label='Not Funded', color='#F44336')

    # Labels above bars
    for i, label in enumerate(selected_categories):
        total = total_vals.iloc[i]
        if total > 0:
            lbl = f"{funded_pct_display.iloc[i]:.1f}% / {not_funded_pct_display.iloc[i]:.1f}%"
            plt.text(x[i], total + total * 0.01, lbl, ha='center', fontsize=9)

    # Dashed vertical line between top and bottom
    plt.axvline(top_n - 0.5, color='gray', linestyle='--', linewidth=1)

    # Aesthetics
    plt.xticks(x, selected_categories, rotation=45, ha='right')
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Number of RFPs')
    title_sort = sort_by.replace('_', ' ').title()
    plt.title(f"Top {top_n} & Bottom {bottom_n} by {title_sort}")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_region_impact(df_funded, df_not_funded, column='region'):
    """
    Plots a stacked bar chart of funded and not funded RFPs by region,
    sorted by total number of RFPs, with percentage labels per stack.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: Column to group by (default: 'region')
    """

    # Count occurrences
    funded_counts = df_funded[column].value_counts()
    not_funded_counts = df_not_funded[column].value_counts()

    # Union of all categories
    all_categories = sorted(set(funded_counts.index).union(set(not_funded_counts.index)))

    # Align and fill missing with 0
    funded_counts = funded_counts.reindex(all_categories, fill_value=0)
    not_funded_counts = not_funded_counts.reindex(all_categories, fill_value=0)

    # Calculate total and sort by total descending
    total_counts = funded_counts + not_funded_counts
    sorted_categories = total_counts.sort_values(ascending=False).index.tolist()

    # Reindex by sorted total
    funded_sorted = funded_counts.reindex(sorted_categories)
    not_funded_sorted = not_funded_counts.reindex(sorted_categories)
    total_vals = funded_sorted + not_funded_sorted

    # Percentages per stack
    funded_pct = (funded_sorted / total_vals * 100).round(1).fillna(0)
    not_funded_pct = (not_funded_sorted / total_vals * 100).round(1).fillna(0)

    # Plotting
    x = np.arange(len(sorted_categories))
    plt.figure(figsize=(14, 6))

    plt.bar(x, funded_sorted.values, label='Funded', color='#4CAF50')
    plt.bar(x, not_funded_sorted.values, bottom=funded_sorted.values, label='Not Funded', color='#F44336')

    # Add labels above each stack
    for i in range(len(x)):
        total = total_vals.iloc[i]
        if total > 0:
            label = f"{funded_pct.iloc[i]:.0f}% / {not_funded_pct.iloc[i]:.0f}%"
            plt.text(x[i], total + total * 0.01, label, ha='center', fontsize=9)

    # Aesthetics
    plt.xticks(x, sorted_categories, rotation=45, ha='right')
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Number of RFPs')
    plt.title(f'RFP Funding by {column.replace("_", " ").title()} (Sorted by Total)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_loan_type_impact(df_funded, df_not_funded, column='loan_type_requested'):
    """
    Plots a stacked bar chart of funded and not funded RFPs by loan type,
    sorted by total number of RFPs, with percentage labels per stack.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: Categorical column to analyze (default: 'loan_type_requested')
    """

    # Count occurrences
    funded_counts = df_funded[column].value_counts()
    not_funded_counts = df_not_funded[column].value_counts()

    # Union of all categories
    all_categories = sorted(set(funded_counts.index).union(set(not_funded_counts.index)))

    # Align and fill missing with 0
    funded_counts = funded_counts.reindex(all_categories, fill_value=0)
    not_funded_counts = not_funded_counts.reindex(all_categories, fill_value=0)

    # Sort by total RFPs
    total_counts = funded_counts + not_funded_counts
    sorted_categories = total_counts.sort_values(ascending=False).index.tolist()

    # Reindex sorted
    funded_sorted = funded_counts.reindex(sorted_categories)
    not_funded_sorted = not_funded_counts.reindex(sorted_categories)
    total_vals = funded_sorted + not_funded_sorted

    # Percentages per stack
    funded_pct = (funded_sorted / total_vals * 100).round(1).fillna(0)
    not_funded_pct = (not_funded_sorted / total_vals * 100).round(1).fillna(0)

    # Plotting
    x = np.arange(len(sorted_categories))
    plt.figure(figsize=(18, 6))
    plt.bar(x, funded_sorted.values, label='Funded', color='#4CAF50')
    plt.bar(x, not_funded_sorted.values, bottom=funded_sorted.values, label='Not Funded', color='#F44336')

    # Labels above bars
    for i in range(len(x)):
        total = total_vals.iloc[i]
        if total > 0:
            label = f"{funded_pct.iloc[i]:.0f}% / {not_funded_pct.iloc[i]:.0f}%"
            plt.text(x[i], total + total * 0.01, label, ha='center', fontsize=9)

    # Aesthetics
    plt.xticks(x, sorted_categories, rotation=45, ha='right')
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Number of RFPs')
    plt.title(f'RFP Funding by {column.replace("_", " ").title()} (Sorted by Total Volume)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_stacked_purpose_of_funds_impact(df_funded, df_not_funded, column='purpose_of_funds'):
    """
    Plots a stacked bar chart of funded and not funded RFPs by purpose of funds,
    sorted by total number of RFPs, with percentage labels per stack.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: categorical column to analyze (default: 'purpose_of_funds')
    """

    # Count occurrences
    funded_counts = df_funded[column].value_counts()
    not_funded_counts = df_not_funded[column].value_counts()

    # Union of all categories
    all_categories = sorted(set(funded_counts.index).union(set(not_funded_counts.index)))

    # Align and fill missing with 0
    funded_counts = funded_counts.reindex(all_categories, fill_value=0)
    not_funded_counts = not_funded_counts.reindex(all_categories, fill_value=0)

    # Sort by total RFPs
    total_counts = funded_counts + not_funded_counts
    sorted_categories = total_counts.sort_values(ascending=False).index.tolist()

    # Reindex sorted
    funded_sorted = funded_counts.reindex(sorted_categories)
    not_funded_sorted = not_funded_counts.reindex(sorted_categories)
    total_vals = funded_sorted + not_funded_sorted

    # Percentages per stack
    funded_pct = (funded_sorted / total_vals * 100).round(1).fillna(0)
    not_funded_pct = (not_funded_sorted / total_vals * 100).round(1).fillna(0)

    # Plotting
    x = np.arange(len(sorted_categories))
    plt.figure(figsize=(14, 6))
    plt.bar(x, funded_sorted.values, label='Funded', color='#4CAF50')
    plt.bar(x, not_funded_sorted.values, bottom=funded_sorted.values, label='Not Funded', color='#F44336')

    # Labels above bars
    for i in range(len(x)):
        total = total_vals.iloc[i]
        if total > 0:
            label = f"{funded_pct.iloc[i]:.0f}% / {not_funded_pct.iloc[i]:.0f}%"
            plt.text(x[i], total + total * 0.01, label, ha='center', fontsize=9)

    # Aesthetics
    plt.xticks(x, sorted_categories, rotation=45, ha='right')
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Number of RFPs')
    plt.title(f'RFP Funding by {column.replace("_", " ").title()} (Sorted by Total Volume)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_stacked_applications_bins(df_funded, df_not_funded, column='number_of_applications_received', bins=6):
    """
    Plots a stacked bar chart of funded and not funded RFPs by equal-width bins
    of number_of_applications_received.

    Parameters:
    - df_funded: DataFrame of funded RFPs
    - df_not_funded: DataFrame of not funded RFPs
    - column: Column name (default: 'number_of_applications_received')
    - bins: Number of equal-width bins to create (default: 6)
    """

    # Convert to numeric and clean
    funded_vals = pd.to_numeric(df_funded[column], errors='coerce').dropna()
    not_funded_vals = pd.to_numeric(df_not_funded[column], errors='coerce').dropna()

    all_vals = pd.concat([funded_vals, not_funded_vals])

    # Create equal-width bins
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    funded_binned = pd.cut(funded_vals, bins=bin_edges)
    not_funded_binned = pd.cut(not_funded_vals, bins=bin_edges)

    # Count in each bin
    funded_counts = funded_binned.value_counts().sort_index()
    not_funded_counts = not_funded_binned.value_counts().sort_index()

    # Prepare bin labels
    bin_labels = [f"{int(interval.left)} - {int(interval.right)} apps" for interval in funded_counts.index]

    # Stack values
    funded_vals_arr = funded_counts.values
    not_funded_vals_arr = not_funded_counts.values
    total_vals = funded_vals_arr + not_funded_vals_arr

    # Plot
    x = np.arange(len(bin_labels))
    width = 0.6

    plt.figure(figsize=(10, 6))
    plt.bar(x, funded_vals_arr, width, label='Funded', color='#4CAF50')
    plt.bar(x, not_funded_vals_arr, width, bottom=funded_vals_arr, label='Not Funded', color='#F44336')

    # Add percentage labels
    for i in range(len(bin_labels)):
        if total_vals[i] > 0:
            pct_funded = funded_vals_arr[i] / total_vals[i] * 100
            pct_not_funded = not_funded_vals_arr[i] / total_vals[i] * 100
            plt.text(x[i], total_vals[i] + max(total_vals) * 0.01,
                     f"{pct_funded:.0f}%/{pct_not_funded:.0f}%", ha='center', fontsize=9)

    # Aesthetics
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.xlabel('Number of Applications Received (Binned)')
    plt.ylabel('Number of RFPs')
    plt.title('RFP Funding by Applications Received Bins')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_lender_preference_matches(df):
    """
    Merges RFP and lender preference data, matches semicolon-separated preferences against actual RFP values,
    and plots match counts by funding status for industry, region, and loan type.

    Parameters:
    - df_rfps: DataFrame containing historical RFPs (should include 'awarded_lender_id', 'industry_sector', 'region', etc.)
    - df_lenders: DataFrame containing lender preferences (should include 'lender_id', 'preferred_industries', etc.)
    """

    # Match helper function
    def match_preference(preferred: str, value: str) -> bool:
        if pd.isna(preferred) or pd.isna(value):
            return False
        return value.lower() in [x.strip().lower() for x in preferred.split(';')]

    # Define fields to match
    match_fields = {
        'industry_match': ('preferred_industries', 'industry_sector', 'Industry'),
        'region_match': ('preferred_regions', 'region', 'Region'),
        'loan_type_match': ('preferred_loan_types', 'loan_type_requested', 'Loan Type')
    }

    # Apply matching logic
    for match_col, (pref_col, val_col, _) in match_fields.items():
        df[match_col] = df.apply(lambda row: match_preference(row[pref_col], row[val_col]), axis=1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#A8D5BA', '#F6C6C7']

    for ax, (match_col, (_, _, title_label)) in zip(axes, match_fields.items()):
        funded_df = df[df['deal_status'] == 'funded']

        match_counts = funded_df[match_col].value_counts()
        match_counts = match_counts.rename(index={True: 'was funded', False: 'was not funded'})

        total = match_counts.sum()
        percentages = (match_counts / total * 100).round(1)
        y_max = match_counts.max() * 1.15

        bars = ax.bar(match_counts.index, match_counts.values, color=colors)
        ax.set_title(f'Lender Preference Match: {title_label}')
        ax.set_xlabel('Funding Status')
        ax.set_ylabel('Matching RFPs')
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Add percentage labels
        for i, val in enumerate(match_counts.values):
            ax.text(i, val + total * 0.01, f"{percentages.iloc[i]}%", ha='center', va='bottom', fontsize=10)


    plt.tight_layout()
    plt.show()


def plot_combined_lender_preference_match(df):
    # Match helper function
    def match_preference(preferred: str, value: str) -> bool:
        if pd.isna(preferred) or pd.isna(value):
            return False
        return value.lower() in [x.strip().lower() for x in preferred.split(';')]

    # Define fields to match
    match_fields = {
        'industry_match': ('preferred_industries', 'industry_sector', 'Industry'),
        'region_match': ('preferred_regions', 'region', 'Region'),
        'loan_type_match': ('preferred_loan_types', 'loan_type_requested', 'Loan Type')
    }

    # Apply matching logic
    for match_col, (pref_col, val_col, _) in match_fields.items():
        df[match_col] = df.apply(lambda row: match_preference(row[pref_col], row[val_col]), axis=1)

    # Filter only funded RFPs
    funded_df = df[df['deal_status'] == 'funded'].copy()

    # Create any_match column
    funded_df['any_match'] = funded_df[['industry_match', 'region_match', 'loan_type_match']].any(axis=1)

    # Count matches
    match_counts = funded_df['any_match'].value_counts()
    match_counts = match_counts.rename(index={True: 'was funded', False: 'was not funded'})

    # Percentages for labels
    totals = match_counts.sum()
    percentages = (match_counts / totals * 100).round(1)

    # Plot
    ax = match_counts.plot(kind='bar', color=['#A8D5BA', '#F6C6C7'], figsize=(8, 6))

    # Add percentage labels
    for i, (index, val) in enumerate(match_counts.items()):
        pct = percentages.loc[index]
        ax.text(i, val + 1, f"{pct:.0f}%", ha='center', va='bottom', fontsize=10)

    ax.set_title("Any Lender Preference Match Among Funded RFPs")
    ax.set_xlabel("Match Status")
    ax.set_ylabel("Number of RFPs")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_strike_zone_analysis(df, 
                              deal_column='deal_size_usd',
                              funded_column='actual_funded_amount_usd',
                              min_col='preferred_deal_size_min_usd',
                              max_col='preferred_deal_size_max_usd'):
    """
    Plots a grouped bar chart comparing how many RFPs fall within the lender's preferred deal size range
    for both the requested and actual funded amounts.

    Parameters:
    - df: Merged DataFrame with both RFP and lender preference data
    - deal_column: column name for requested deal size
    - funded_column: column name for actual funded amount
    - min_col: column name for preferred minimum deal size
    - max_col: column name for preferred maximum deal size
    """

    # Strike zone helper
    def is_in_strike_zone(value, min_val, max_val):
        if pd.notnull(value) and pd.notnull(min_val) and pd.notnull(max_val):
            return min_val <= value <= max_val
        return False

    df = df.copy()

    # Apply strike zone logic
    df['deal_size_in_strike'] = df.apply(
        lambda row: is_in_strike_zone(row[deal_column], row[min_col], row[max_col]), axis=1
    )
    df['actual_funded_in_strike'] = df.apply(
        lambda row: is_in_strike_zone(row[funded_column], row[min_col], row[max_col]), axis=1
    )

    # Count matches
    all_categories = ['In Strike Zone', 'Out of Strike Zone']
    deal_counts = df['deal_size_in_strike'].value_counts().rename(
        {True: 'In Strike Zone', False: 'Out of Strike Zone'}
    ).reindex(all_categories, fill_value=0)
    funded_counts = df['actual_funded_in_strike'].value_counts().rename(
        {True: 'In Strike Zone', False: 'Out of Strike Zone'}
    ).reindex(all_categories, fill_value=0)

    deal_total = deal_counts.sum()
    funded_total = funded_counts.sum()

    # Plot
    x = range(len(all_categories))
    bar_width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x, deal_counts.values, width=bar_width, label='Requested Deal Size', color='#AED6F1')
    bars2 = plt.bar([i + bar_width for i in x], funded_counts.values, width=bar_width, label='Actual Funded Amount', color='#F9CB9C')

    plt.xticks([i + bar_width / 2 for i in x], all_categories)
    plt.ylabel('Number of RFPs')
    plt.title('Strike Zone Analysis: Requested vs. Actual Funded Amount')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add % labels above bars
    for i in range(len(all_categories)):
        val1 = deal_counts.values[i]
        val2 = funded_counts.values[i]
        pct1 = (val1 / deal_total * 100) if deal_total else 0
        pct2 = (val2 / funded_total * 100) if funded_total else 0
        plt.text(i, val1 + 1, f"{pct1:.1f}%", ha='center', fontsize=9)
        plt.text(i + bar_width, val2 + 1, f"{pct2:.1f}%", ha='center', fontsize=9)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bar_graph(series, title, column_name='', figsize=(10, 6), rotation=45):
    # Plot
    plt.figure(figsize=figsize)
    series.plot(kind='bar', color='#AED6F1')

    plt.title(title)
    plt.ylabel('Median Success Rate (%)')
    if column_name:
        plt.xlabel(column_name.replace('_', ' ').capitalize())
    plt.xticks(rotation=rotation)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_lender_deal_size_correlation_analysis(df):
    # Create deal_size_range column
    df['deal_size_range'] = df['preferred_deal_size_max_usd'] - df['preferred_deal_size_min_usd']

    # Select relevant columns
    cols = ['preferred_deal_size_min_usd', 'preferred_deal_size_max_usd', 'deal_size_range', 'historical_success_rate_pct']
    correlation = df[cols].corr()

    # Extract correlation with historical_success_rate_pct
    corr_with_success = correlation['historical_success_rate_pct'].drop('historical_success_rate_pct')

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr_with_success.index, y=corr_with_success.values, color="#A8D5BA")
    plt.ylabel("Correlation with Historical Success Rate (%)")
    plt.title("Correlation of Deal Size Preferences with Success Rate")
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()












from pathlib import Path

import openpyxl
import pandas as pd


RAW_DATA_FOLDER = Path(r"D:\MSEDCL")
OUTPUT_FILE = Path("data/energy_hourly.csv")

SOURCE_FILES = [
    "April 2023.xlsx",
    "May 2023.xlsx",
    "June 2023.xlsx",
    "July 2023.xlsx",
    "August 2023.xlsx",
]

TARGET_COLUMN_NAME = "NLDC_DEMAND|P"


def load_monthly_file(file_path):
    """Load timestamp and electricity demand from one monthly workbook."""

    print(f"Loading: {file_path.name}")

    workbook = openpyxl.load_workbook(
        file_path,
        read_only=True,
        data_only=True,
    )

    worksheet = workbook["Sheet1"]

    header_row = None
    target_column = None

    for row_number in range(1, 3):
        for column_number, cell in enumerate(
            worksheet[row_number],
            start=1,
        ):
            if cell.value == TARGET_COLUMN_NAME:
                header_row = row_number
                target_column = column_number
                break

        if target_column is not None:
            break

    if target_column is None:
        workbook.close()
        raise ValueError(
            f"{TARGET_COLUMN_NAME} was not found in {file_path.name}"
        )

    print(f"  Target column: {target_column}")
    print(f"  Header row: {header_row}")

    timestamps = []
    demand_values = []

    for row in worksheet.iter_rows(
        min_row=header_row + 1,
        values_only=True,
    ):
        if len(row) < target_column:
            continue

        timestamp = row[0]
        demand = row[target_column - 1]

        if timestamp is None or demand is None:
            continue

        timestamps.append(timestamp)
        demand_values.append(demand)

    workbook.close()

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "demand_mw": pd.to_numeric(
                demand_values,
                errors="coerce",
            ),
        }
    )

    dataframe = dataframe.dropna(
        subset=["timestamp", "demand_mw"]
    )

    print(f"  Valid rows: {len(dataframe):,}")

    return dataframe


def main():
    """Create a continuous hourly electricity-demand dataset."""

    monthly_data = []

    for filename in SOURCE_FILES:
        file_path = RAW_DATA_FOLDER / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {file_path}"
            )

        dataframe = load_monthly_file(file_path)
        monthly_data.append(dataframe)

    print("\nCombining monthly data...")

    data = pd.concat(
        monthly_data,
        ignore_index=True,
    )

    data = data.sort_values("timestamp")

    data = data.drop_duplicates(
        subset="timestamp",
    )

    data = data.set_index("timestamp")

    print(f"Raw observations: {len(data):,}")

    print("\nResampling to hourly demand...")

    hourly_data = (
        data["demand_mw"]
        .resample("1h")
        .mean()
        .dropna()
        .reset_index()
    )

    hourly_data["demand_mw"] = hourly_data[
        "demand_mw"
    ].round(3)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nHourly dataset created successfully.")
    print(
        f"Hourly observations: {len(hourly_data):,}"
    )
    print(
        f"First timestamp: "
        f"{hourly_data['timestamp'].min()}"
    )
    print(
        f"Last timestamp: "
        f"{hourly_data['timestamp'].max()}"
    )
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
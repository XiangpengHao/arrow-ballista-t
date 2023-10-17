import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TPCH_SCHEMAS = {
    "part.tbl": (
        [
            "p_partkey",
            "p_name",
            "p_mfgr",
            "p_brand",
            "p_type",
            "p_size",
            "p_container",
            "p_retailprice",
            "p_comment",
        ],
        [
            "int64",
            "string",
            "string",
            "string",
            "string",
            "int32",
            "string",
            "float64",
            "string",
        ],
    ),
    "supplier.tbl": (
        [
            "s_suppkey",
            "s_name",
            "s_address",
            "s_nationkey",
            "s_phone",
            "s_acctbal",
            "s_comment",
        ],
        ["int64", "string", "string", "int64", "string", "float64", "string"],
    ),
    "partsupp.tbl": (
        ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        ["int64", "int64", "int32", "float64", "string"],
    ),
    "customer.tbl": (
        [
            "c_custkey",
            "c_name",
            "c_address",
            "c_nationkey",
            "c_phone",
            "c_acctbal",
            "c_mktsegment",
            "c_comment",
        ],
        ["int64", "string", "string", "int64", "string", "float64", "string", "string"],
    ),
    "orders.tbl": (
        [
            "o_orderkey",
            "o_custkey",
            "o_orderstatus",
            "o_totalprice",
            "o_orderdate",
            "o_orderpriority",
            "o_clerk",
            "o_shippriority",
            "o_comment",
        ],
        [
            "int64",
            "int64",
            "string",
            "float64",
            "string",
            "string",
            "string",
            "int32",
            "string",
        ],
    ),
    "lineitem.tbl": (
        [
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        ],
        [
            "int64",
            "int64",
            "int64",
            "int32",
            "float64",
            "float64",
            "float64",
            "float64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
        ],
    ),
    "nation.tbl": (
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        ["int64", "string", "int64", "string"],
    ),
    "region.tbl": (
        ["r_regionkey", "r_name", "r_comment"],
        ["int64", "string", "string"],
    ),
}


def convert_data_types(df, data_types):
    for col, dtype in zip(df.columns, data_types):
        df[col] = df[col].astype(dtype)
    return df


def convert_tbl_to_parquet(input_dir, output_dir):
    # List all .tbl files in the input directory
    tbl_files = [f for f in os.listdir(input_dir) if f.endswith(".tbl")]

    for tbl_file in tbl_files:
        tbl_path = os.path.join(input_dir, tbl_file)

        print("Working on", tbl_file)

        # Get the schema for the table
        columns, data_types = TPCH_SCHEMAS.get(tbl_file)
        if not columns:
            print(f"Warning: No schema defined for {tbl_file}. Skipping.")
            continue


        df = pd.read_csv(tbl_path, sep="|", names=columns + ["_trailing"], engine="python")

        # Drop the last "_trailing" column
        df = df.drop(columns="_trailing")

        if df.isnull().any().any():
            print(f"Warning: Null values detected in {tbl_file}.")

        df = convert_data_types(df, data_types)

        # Convert the DataFrame to Parquet format
        table = pa.Table.from_pandas(df)

        # Construct the output path for the Parquet file
        parquet_path = os.path.join(output_dir, tbl_file.replace(".tbl", ".parquet"))
        pq.write_table(table, parquet_path)

        print(f"Converted {tbl_file} to Parquet format.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_to_parquet.py <input_directory> <output_directory>"
        )
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    convert_tbl_to_parquet(input_directory, output_directory)

import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TPCH_SCHEMAS = {
    "part.tbl": (["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"],
                ["int64", "object", "object", "object", "object", "int32", "object", "float64", "object"]),
    
    "supplier.tbl": (["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"],
                     ["int64", "object", "object", "int64", "object", "float64", "object"]),
    
    "partsupp.tbl": (["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
                     ["int64", "int64", "int32", "float64", "object"]),
    
    "customer.tbl": (["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
                     ["int64", "object", "object", "int64", "object", "float64", "object", "object"]),
    
    "orders.tbl": (["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
                   ["int64", "int64", "object", "float64", "object", "object", "object", "int32", "object"]),
    
    "lineitem.tbl": (["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                     ["int64", "int64", "int64", "int32", "float64", "float64", "float64", "float64", "object", "object", "object", "object", "object", "object", "object", "object"]),
    
    "nation.tbl": (["n_nationkey", "n_name", "n_regionkey", "n_comment"],
                   ["int64", "object", "int64", "object"]),
    
    "region.tbl": (["r_regionkey", "r_name", "r_comment"],
                   ["int64", "object", "object"])
}


def convert_data_types(df, data_types):
    for col, dtype in zip(df.columns, data_types):
        df[col] = df[col].astype(dtype)
    return df

def convert_tbl_to_parquet(input_dir, output_dir):
    # List all .tbl files in the input directory
    tbl_files = [f for f in os.listdir(input_dir) if f.endswith('.tbl')]

    for tbl_file in tbl_files:
        # Construct full path to the .tbl file
        tbl_path = os.path.join(input_dir, tbl_file)
        
        # Get the schema for the table
        columns, data_types = TPCH_SCHEMAS.get(tbl_file)
        if not columns:
            print(f"Warning: No schema defined for {tbl_file}. Skipping.")
            continue

        # Read the .tbl file into a pandas DataFrame
        df = pd.read_csv(tbl_path, sep='|', names=columns, engine='python')

        # Drop the last empty column that results from the trailing "|"
        df = df.dropna(axis=1, how='all')
        df = convert_data_types(df, data_types)

        # Convert the DataFrame to Parquet format
        table = pa.Table.from_pandas(df)
        
        # Construct the output path for the Parquet file
        parquet_path = os.path.join(output_dir, tbl_file.replace('.tbl', '.parquet'))
        pq.write_table(table, parquet_path)

        print(f"Converted {tbl_file} to Parquet format.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_parquet.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    convert_tbl_to_parquet(input_directory, output_directory)

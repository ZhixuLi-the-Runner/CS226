import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

npy_file_path = './auto_lang_ann.npy'
npy_file = np.load(npy_file_path, allow_pickle=True)

spark = SparkSession.builder.appName('FindMatchingTasks').getOrCreate()
target_task = 'turn_on_led'

df1 = spark.createDataFrame(enumerate(npy_file.item()['language']['task']), schema=["index", "task"])

schema = StructType([
    StructField("index", IntegerType(), True),
    StructField("range", StructType([
        StructField("first", IntegerType(), True),
        StructField("second", IntegerType(), True)
    ]))
])

lst2 = [(i, {"first": int(pair[0]), "second": int(pair[1])}) for i, pair in enumerate(npy_file.item()['info']['indx'])]
df2 = spark.createDataFrame(lst2, schema=schema)

matching_df = df1.filter(df1.task == target_task)
result_df = matching_df.join(df2, on="index", how="inner")
result_df.show(truncate=False)

import os
from pyspark.sql.functions import regexp_extract

npz_file_dir = './turn_on_led_npz/turn_on_led_npz/1/'
npz_files_names = os.listdir(npz_file_dir)

file_name_df = spark.createDataFrame(npz_files_names, "string").toDF("file_name")

df_with_numbers = file_name_df.withColumn("number", regexp_extract("file_name", r'_([0-9]+)', 1))

file_under_range_df = df_with_numbers.join(result_df, (df_with_numbers.number >= result_df.range.first) & (df_with_numbers.number <= result_df.range.second), how="inner")

file_under_range_df.show(truncate=False)

def load_and_combine_npz(file_paths):
    combined_array = None
    for file_path in file_paths:
        data = np.load(file_path)
        array = [data[key] for key in data][0]  # Assuming single 1D array per `.npz` file
        if combined_array is None:
            combined_array = array
        else:
            combined_array = np.vstack([combined_array, array])
    return combined_array

grouped_files = file_under_range_df.rdd.map(lambda row: (row["range"], os.path.join(npz_file_dir, row["file_name"]))).groupByKey().collect()

for range_name, file_paths in grouped_files:
    combined_array = load_and_combine_npz(file_paths)
    output_file = f"combined_data_{range_name}.npz"
    np.savez(output_file, combined_array=combined_array)
    print(f"Saved combined array for range {range_name} to {output_file}")
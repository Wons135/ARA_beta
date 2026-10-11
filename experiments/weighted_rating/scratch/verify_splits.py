import pandas as pd

base = "datasets/preprocessed"
count_test = pd.read_csv(f"{base}/count_stage1_test.csv", usecols=["review_id"])
bin_train = pd.read_csv(f"{base}/binary_label_train.csv", usecols=["review_id"])
bin_val = pd.read_csv(f"{base}/binary_label_val.csv", usecols=["review_id"])

count_test_ids = set(count_test["review_id"])
bin_train_ids = set(bin_train["review_id"])
bin_val_ids = set(bin_val["review_id"])

overlap_train = count_test_ids & bin_train_ids
overlap_val = count_test_ids & bin_val_ids

print(f"count_stage1_test rows: {len(count_test_ids)}")
print(f"binary_label_train rows: {len(bin_train_ids)}")
print(f"binary_label_val rows: {len(bin_val_ids)}")
print(f"Overlap count_test & binary_train: {len(overlap_train)}")
print(f"Overlap count_test & binary_val: {len(overlap_val)}")

# Also check binary test count
bin_test = pd.read_csv(f"{base}/binary_label_test.csv", usecols=["review_id"])
print(f"binary_label_test rows: {len(bin_test)}")
print(f"PASS" if len(overlap_train) == 0 and len(overlap_val) == 0 else "FAIL")

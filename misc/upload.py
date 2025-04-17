import pandas as pd
import requests
from datetime import datetime

# Load CSV (only from row 9999 onward)
df = pd.read_csv("plc_scan_dataset.csv")[9500:]

# API Endpoint
API_URL = "https://swift-cloud.ephlux.com/api-swift/ned/workflow/4I3NlgBIdN?SWIFT_API_KEY=j4VkUJ+Sw/BxkfhxGKRm96ioyU"

# Function to convert row to payload
def transform_row(row):
    # Handle "missed_scan"
    product_serial = "0" if row["ProductSerial"] == "missed_scan" else row["ProductSerial"]

    # Determine anomaly
    is_anomalous = len(product_serial) != 10

    return {
        "serial_no": product_serial,
        "batch_no": str(row["BatchNumber"]).replace("BATCH", ""),
        "is_anomalous": {
            "label": str(is_anomalous),
            "value": is_anomalous
        },
        "work_order":row["WorkOrder"],
        "selected_scanner": "1",
        "selected_plc_line": str(row["Plc_Line"]).replace("PLC", ""),
        "DateTimestamp": row["DateTimestamp"]
    }

# Send rows to API
for _, row in df.iterrows():
    payload = transform_row(row)
    response = requests.post(API_URL, json=payload)
    print(f"Sent serial {payload['serial_no']} | Status: {response.status_code} | Response: {response.text}")

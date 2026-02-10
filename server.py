from fastapi import FastAPI, Request
import uvicorn
import csv
import os
from datetime import datetime

app = FastAPI()

CSV_FILE = "received_data.csv"

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

@app.post("/data")
async def receive_data(request: Request):
    data = await request.json()
    # print(f"Received data: {data}") # Commented out to reduce noise
    
    # Extract data
    ax = data.get("ax")
    ay = data.get("ay")
    az = data.get("az")
    gx = data.get("gx")
    gy = data.get("gy")
    gz = data.get("gz")
    timestamp = datetime.now().isoformat()
    
    # Save to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, ax, ay, az, gx, gy, gz])
        
    return {"status": "success"}

@app.get("/")
def read_root():
    return {"message": "Fall Detection Data Server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

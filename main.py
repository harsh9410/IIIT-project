from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Body
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
import asyncio
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
import joblib
import io

# CNN Model Definition for NILM (optional)
if TORCH_AVAILABLE:
    class CNN_NILM_Model(nn.Module):
        def __init__(self, num_appliances=5):
            super(CNN_NILM_Model, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(128 * 25, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_appliances)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

app = FastAPI(title="NILM Electricity Predictor", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('nilm_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS power_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            voltage REAL,
            current REAL,
            power REAL,
            ac_power REAL,
            refrigerator_power REAL,
            washing_machine_power REAL,
            lights_power REAL,
            other_power REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appliance_signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            appliance_name TEXT,
            power_pattern TEXT,
            typical_usage REAL
        )
    ''')
    
    # Insert sample appliance signatures
    appliances = [
        ('AC', '[0.8, 1.2, 1.5, 1.2, 0.8]', 1.2),
        ('Refrigerator', '[0.2, 0.3, 0.1, 0.3, 0.2]', 0.3),
        ('Washing Machine', '[0.4, 0.9, 0.7, 0.5, 0.2]', 0.7),
        ('Lights', '[0.1, 0.15, 0.2, 0.15, 0.1]', 0.15),
        ('Other', '[0.05, 0.1, 0.15, 0.1, 0.05]', 0.1)
    ]
    
    cursor.executemany(
        'INSERT OR IGNORE INTO appliance_signatures (appliance_name, power_pattern, typical_usage) VALUES (?, ?, ?)',
        appliances
    )
    
    conn.commit()
    conn.close()

# Initialize model
def load_model():
    if TORCH_AVAILABLE:
        model = CNN_NILM_Model(num_appliances=5)
        # In production, you would load pre-trained weights here
        # model.load_state_dict(torch.load('nilm_cnn_model.pth'))
        return model
    # If torch not available, return None and keep using simulated pipeline
    return None

model = load_model()
init_db()

# Background simulator to keep the UI working without hardware
SIMULATE_REALTIME = True

async def simulator_loop():
    """Continuously generate simulated readings and broadcast them."""
    while SIMULATE_REALTIME:
        try:
            reading = get_simulated_reading()
            save_reading_to_db(reading)
            manager.last_reading = reading
            # Broadcast to connected WebSocket clients
            await manager.broadcast(reading)
        except Exception:
            # Keep simulator resilient even if an error occurs
            pass
        await asyncio.sleep(3)

@app.on_event("startup")
async def start_background_tasks():
    # Start realtime simulator so the website shows live data out-of-the-box
    asyncio.create_task(simulator_loop())

@app.get("/")
async def root():
    return {"message": "NILM Electricity Predictor API", "status": "active"}

@app.get("/api/current-reading")
async def get_current_reading():
    """Get current power reading with appliance breakdown"""
    conn = sqlite3.connect('nilm_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM power_readings 
        ORDER BY timestamp DESC LIMIT 1
    ''')
    reading = cursor.fetchone()
    conn.close()
    
    if reading:
        return {
            "timestamp": reading[1],
            "voltage": reading[2],
            "current": reading[3],
            "total_power": reading[4],
            "appliances": {
                "AC": reading[5],
                "Refrigerator": reading[6],
                "Washing Machine": reading[7],
                "Lights": reading[8],
                "Other": reading[9]
            }
        }
    else:
        raise HTTPException(status_code=404, detail="No real-time data yet. Send readings to /api/ingest.")

def save_reading_to_db(reading: dict):
    conn = sqlite3.connect('nilm_data.db')
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO power_readings (timestamp, voltage, current, power, ac_power, refrigerator_power, washing_machine_power, lights_power, other_power)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            reading.get('timestamp', datetime.now().isoformat()),
            reading['voltage'],
            reading['current'],
            reading['total_power'],
            reading['appliances'].get('AC'),
            reading['appliances'].get('Refrigerator'),
            reading['appliances'].get('Washing Machine'),
            reading['appliances'].get('Lights'),
            reading['appliances'].get('Other'),
        )
    )
    conn.commit()
    conn.close()

def get_simulated_reading():
    """Generate simulated power reading with CNN prediction"""
    voltage = 230 + np.random.normal(0, 2)
    current = 0.5 + np.random.normal(0, 0.1)
    total_power = voltage * current
    
    # Simulate CNN prediction for appliance breakdown
    appliance_powers = {
        "AC": total_power * 0.4 + np.random.normal(0, 0.1),
        "Refrigerator": total_power * 0.2 + np.random.normal(0, 0.05),
        "Washing Machine": total_power * 0.25 + np.random.normal(0, 0.08),
        "Lights": total_power * 0.1 + np.random.normal(0, 0.02),
        "Other": total_power * 0.05 + np.random.normal(0, 0.01)
    }
    
    # Normalize to match total power
    scale_factor = total_power / sum(appliance_powers.values())
    for appliance in appliance_powers:
        appliance_powers[appliance] *= scale_factor
    
    return {
        "timestamp": datetime.now().isoformat(),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "total_power": round(total_power, 2),
        "appliances": {k: round(v, 2) for k, v in appliance_powers.items()}
    }

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV data for NILM analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic CSV validation
        required_columns = ['timestamp', 'voltage', 'current', 'power']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required column: {col}"
                )
        
        # Process data with CNN model (simulated)
        results = process_with_cnn(df)
        
        return {
            "message": "File processed successfully",
            "rows_processed": len(df),
            "analysis": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def process_with_cnn(df):
    """Process data using CNN model for appliance disaggregation"""
    # Simulate CNN processing
    results = {
        "total_energy_consumed": round(df['power'].sum(), 2),
        "average_power": round(df['power'].mean(), 2),
        "peak_power": round(df['power'].max(), 2),
        "appliance_breakdown": {
            "AC": round(df['power'].mean() * 0.4, 2),
            "Refrigerator": round(df['power'].mean() * 0.2, 2),
            "Washing Machine": round(df['power'].mean() * 0.25, 2),
            "Lights": round(df['power'].mean() * 0.1, 2),
            "Other": round(df['power'].mean() * 0.05, 2)
        },
        "prediction_accuracy": round(0.85 + np.random.random() * 0.1, 2)  # Simulated accuracy
    }
    return results

@app.get("/api/history")
async def get_history(hours: int = 24):
    """Get historical power data"""
    conn = sqlite3.connect('nilm_data.db')
    
    query = f'''
        SELECT timestamp, voltage, current, power, ac_power, refrigerator_power,
               washing_machine_power, lights_power, other_power
        FROM power_readings 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        # Generate simulated historical data
        return generate_simulated_history(hours)
    
    return JSONResponse(content=df.to_dict(orient='records'))

def generate_simulated_history(hours):
    """Generate simulated historical data"""
    now = datetime.now()
    history = []
    
    for i in range(hours * 12):  # 5-minute intervals
        timestamp = now - timedelta(minutes=i*5)
        reading = get_simulated_reading()
        reading['timestamp'] = timestamp.isoformat()
        history.append(reading)
    
    return history

@app.get("/api/predict")
async def predict_usage(hours: int = 1):
    """Predict future electricity usage"""
    # Simulate CNN-based prediction
    base_power = 115  # Base power in watts
    
    predictions = []
    for i in range(hours):
        future_time = datetime.now() + timedelta(hours=i+1)
        
        # Simulate daily pattern (higher during day, lower at night)
        hour = future_time.hour
        if 6 <= hour <= 22:  # Day time
            multiplier = 0.8 + 0.4 * np.sin((hour - 6) * np.pi / 16)
        else:  # Night time
            multiplier = 0.3 + 0.2 * np.random.random()
        
        predicted_power = base_power * multiplier * (1 + np.random.normal(0, 0.1))
        
        predictions.append({
            "timestamp": future_time.isoformat(),
            "predicted_power": round(predicted_power, 2),
            "predicted_cost": round(predicted_power * 0.08 / 1000, 2),  # ₹0.08 per kWh
            "confidence": round(0.7 + np.random.random() * 0.25, 2)
        })
    
    return {
        "predictions": predictions,
        "model_used": "CNN-NILM",
        "prediction_horizon": f"{hours} hours"
    }

# ===============================
# \ud83d\udce1 WebSocket for Live Readings
# ===============================

class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.last_reading: dict | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)

    async def broadcast(self, data: dict):
        for conn in list(self.active_connections):
            try:
                await conn.send_json(data)
            except Exception:
                # Drop dead connections quietly
                self.disconnect(conn)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            if manager.last_reading:
                await manager.send_json(websocket, manager.last_reading)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/ingest")
async def ingest_reading(payload: dict = Body(...)):
    """Ingest sensor reading from ESP32.
    Expected payload: { timestamp?, voltage, current, total_power?, appliances? }
    If total_power missing, computed as voltage*current. If appliances missing, a simple split is applied.
    """
    try:
        voltage = float(payload.get('voltage'))
        current = float(payload.get('current'))
        total_power = float(payload.get('total_power')) if payload.get('total_power') is not None else voltage * current

        appliances = payload.get('appliances') or {}
        if not appliances:
            # Simple proportional split as a placeholder until NILM model
            appliances = {
                "AC": total_power * 0.4,
                "Refrigerator": total_power * 0.2,
                "Washing Machine": total_power * 0.25,
                "Lights": total_power * 0.1,
                "Other": total_power * 0.05,
            }
        # Normalize to match total
        scale = total_power / max(1e-6, sum(appliances.values()))
        for k in appliances:
            appliances[k] = round(appliances[k] * scale, 2)

        reading = {
            "timestamp": payload.get('timestamp') or datetime.now().isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "total_power": round(total_power, 2),
            "appliances": appliances,
        }

        save_reading_to_db(reading)
        manager.last_reading = reading
        # Fire-and-forget broadcast
        asyncio.create_task(manager.broadcast(reading))

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
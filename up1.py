import cv2
import numpy as np
import os
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
import time
import json
from datetime import datetime
from collections import deque
import threading
import logging

# ==================================
# LOGGING SETUP
# ==================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_log.txt'),
        logging.StreamHandler()
    ]
)

# ==================================
# VIDEO INPUT
video_source = input("Enter video file path or URL: ").strip()

# Check if the input is a URL
if video_source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
    cap = cv2.VideoCapture(video_source)
else:
    # Check if the file exists
    if not os.path.exists(video_source):
        print("Error: Video file not found!")
        exit()
    cap = cv2.VideoCapture(video_source)

# Verify that the video source opened successfully
if not cap.isOpened():
    print("Cannot open video source.")
    exit()

# ==================================
# EMA SMOOTHING
# ==================================

alpha = 0.4
smoothed_fill = 0

# ==================================
# FILL VALIDATION
# ==================================

fill_counter = 0
required_frames = 5


# ==================================
# PLC CONNECTION & MODBUS REGISTERS
# ==================================

PLC_IP = "192.168.1.10"      # Change to your PLC IP
PLC_PORT = 502               # Standard Modbus TCP port
PLC_RECONNECT_DELAY = 5      # seconds

# Modbus Registers Definition:
# Register 0: Fill Percentage (0-100)
# Register 1: Fill Status (1=FILL, 0=NO FILL)
# Register 2: Conveyor Control (1=RUN, 0=STOP)
# Register 3: Reject Cylinder Control (1=ACTIVATE, 0=DEACTIVATE)
# Register 4: Indicator Lamp Control (1=GREEN/OK, 0=RED/REJECT)
# Register 5: Good Parts Count
# Register 6: Bad Parts Count
# Register 7: Alarm Status (0=OK, 1=WARNING, 2=CRITICAL)

class PLCController:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.client = None
        self.connected = False
        self.last_values = {}
        self.connect()
    
    def connect(self):
        """Establish connection to PLC"""
        try:
            self.client = ModbusTcpClient(self.ip, port=self.port)
            if self.client.connect():
                self.connected = True
                logging.info(f"PLC Connected to {self.ip}:{self.port}")
                return True
            else:
                logging.error(f"Failed to connect to PLC at {self.ip}:{self.port}")
                self.connected = False
                return False
        except Exception as e:
            logging.error(f"PLC Connection Error: {str(e)}")
            self.connected = False
            return False
    
    def write_register(self, register, value):
        """Write single register with error handling"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            if register not in self.last_values or self.last_values[register] != value:
                result = self.client.write_register(register, int(value))
                if result.isError():
                    logging.warning(f"Register write error at {register}: {result}")
                    return False
                self.last_values[register] = value
                logging.debug(f"Register {register} set to {value}")
                return True
        except ModbusException as e:
            logging.error(f"Modbus exception on write: {str(e)}")
            self.connected = False
            return False
        except Exception as e:
            logging.error(f"Error writing register {register}: {str(e)}")
            return False
    
    def read_register(self, register):
        """Read single register with error handling"""
        if not self.connected:
            return None
        
        try:
            result = self.client.read_holding_registers(register, count=1)
            if result.isError():
                logging.warning(f"Register read error at {register}")
                return None
            return result.registers[0] if result.registers else None
        except Exception as e:
            logging.error(f"Error reading register {register}: {str(e)}")
            return None
    
    def disconnect(self):
        """Close PLC connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logging.info("PLC Disconnected")

# Initialize PLC Controller
plc = PLCController(PLC_IP, PLC_PORT)

# ==================================
# SCADA DATA LOGGING & TELEMETRY
# ==================================

class SCADALogger:
    def __init__(self, max_records=1000):
        self.records = deque(maxlen=max_records)
        self.alarms = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def log_production(self, fill_percentage, status, good_count, bad_count, alarm_status=0):
        """Log production data"""
        with self.lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'fill_percentage': fill_percentage,
                'status': status,
                'good_count': good_count,
                'bad_count': bad_count,
                'alarm_status': alarm_status
            }
            self.records.append(record)
            logging.info(f"Production: {status} | Fill: {fill_percentage:.1f}% | Good: {good_count} | Bad: {bad_count}")
    
    def log_alarm(self, alarm_type, message, severity='WARNING'):
        """Log alarm event"""
        with self.lock:
            alarm = {
                'timestamp': datetime.now().isoformat(),
                'type': alarm_type,
                'message': message,
                'severity': severity
            }
            self.alarms.append(alarm)
            logging.warning(f"ALARM [{severity}] {alarm_type}: {message}")
    
    def export_dashboard_data(self, filename='dashboard_data.json'):
        """Export data for live dashboard"""
        with self.lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'recent_records': list(self.records)[-50:],
                'alarms': list(self.alarms)[-20:],
                'summary': {
                    'total_good': sum(r['good_count'] for r in self.records) if self.records else 0,
                    'total_bad': sum(r['bad_count'] for r in self.records) if self.records else 0,
                }
            }
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logging.error(f"Error exporting dashboard data: {str(e)}")

scada = SCADALogger()

# ==================================
# PRODUCTION METRICS & CONTROL LOGIC
# ==================================

good_count = 0
bad_count = 0
previous_status = None
last_dashboard_export = time.time()
dashboard_export_interval = 5  # Export dashboard data every 5 seconds

def control_conveyor(should_run):
    """Control conveyor motor"""
    plc.write_register(2, 1 if should_run else 0)
    logging.info(f"Conveyor: {'RUN' if should_run else 'STOP'}")

def control_reject_cylinder(should_activate):
    """Control reject cylinder"""
    plc.write_register(3, 1 if should_activate else 0)
    logging.info(f"Reject Cylinder: {'ACTIVATED' if should_activate else 'DEACTIVATED'}")

def set_indicator_lamp(color='GREEN'):
    """Set indicator lamp (GREEN=OK, RED=REJECT)"""
    lamp_value = 1 if color == 'GREEN' else 0
    plc.write_register(4, lamp_value)
    logging.info(f"Indicator Lamp: {color}")

def update_plc_counters(good, bad):
    """Update production counters in PLC"""
    plc.write_register(5, good)
    plc.write_register(6, bad)

def trigger_alarm(alarm_type, message, severity='WARNING'):
    """Trigger alarm and log"""
    alarm_level = {'WARNING': 1, 'CRITICAL': 2, 'OK': 0}.get(severity, 1)
    plc.write_register(7, alarm_level)
    scada.log_alarm(alarm_type, message, severity)


# ==================================
# PROCESS VIDEO
# ==================================

while True:

    ret, img = cap.read()

    if not ret:
        print("Video ended or cannot read frame")
        break

    output = img.copy()

    # ==================================
    # MOLTEN METAL DETECTION
    # ==================================

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_channel = hsv[:, :, 2]

    v_channel = cv2.medianBlur(v_channel, 5)

    _, molten_mask = cv2.threshold(
        v_channel,
        220,
        255,
        cv2.THRESH_BINARY
    )

    # ==================================
    # MORPHOLOGY
    # ==================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    molten_mask = cv2.morphologyEx(
        molten_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    molten_mask = cv2.morphologyEx(
        molten_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ==================================
    # KEEP LARGEST MOLTEN REGION
    # ==================================

    contours, _ = cv2.findContours(
        molten_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            molten_mask
        )

        cv2.drawContours(
            clean_mask,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

        molten_mask = clean_mask

    # ==================================
    # CIRCLE DETECTION
    # ==================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=100,
        maxRadius=500
    )

    if circles is not None:

        circles = np.round(
            circles[0]
        ).astype(int)

        x, y, r = max(
            circles,
            key=lambda c: c[2]
        )

        # ==================================
        # CIRCLE MASK (INVISIBLE)
        # ==================================

        circle_mask = np.zeros_like(
            molten_mask
        )

        cv2.circle(
            circle_mask,
            (x, y),
            r,
            255,
            -1
        )

        # ==================================
        # MOLTEN INSIDE CIRCLE
        # ==================================

        molten_inside = cv2.bitwise_and(
            molten_mask,
            circle_mask
        )

        # ==================================
        # FILL PERCENTAGE
        # ==================================

        circle_pixels = cv2.countNonZero(
            circle_mask
        )

        filled_pixels = cv2.countNonZero(
            molten_inside
        )

        raw_fill = (
            filled_pixels / circle_pixels
        ) * 100 if circle_pixels > 0 else 0

        # ==================================
        # EMA SMOOTHING
        # ==================================

        smoothed_fill = (
            alpha * raw_fill +
            (1 - alpha) * smoothed_fill
        )

        fill_percentage = smoothed_fill
        
        # ==================================
        # WRITE FILL DATA TO PLC
        # ==================================
        plc.write_register(0, int(fill_percentage))

        # ==================================
        # CONSECUTIVE FRAME VALIDATION
        # ==================================

        if fill_percentage >= 60:
            fill_counter += 1
        else:
            fill_counter = 0

        if fill_counter >= required_frames:
            status = "FILL"
            color = (0, 255, 0)
            fill_status_reg = 1
        else:
            status = "NO FILL"
            color = (0, 0, 255)
            fill_status_reg = 0

        # ==================================
        # INDUSTRIAL CONTROL & DECISION LOGIC
        # ==================================
        
        # Update fill status register
        plc.write_register(1, fill_status_reg)
        
        # Handle FILL/NO FILL decision
        if status != previous_status:
            if status == "FILL":
                good_count += 1
                logging.info(f"✓ PART ACCEPTED - Good Count: {good_count}")
                
                # Control signals for FILL
                set_indicator_lamp('GREEN')  # Green indicator
                control_reject_cylinder(False)  # Deactivate reject
                control_conveyor(True)  # Keep conveyor running
                trigger_alarm('PART_ACCEPTED', f'Good part accepted. Count: {good_count}', 'OK')
                
            else:  # NO FILL
                bad_count += 1
                logging.info(f"✗ PART REJECTED - Bad Count: {bad_count}")
                
                # Control signals for NO FILL (REJECT)
                set_indicator_lamp('RED')  # Red indicator
                control_reject_cylinder(True)  # Activate reject cylinder
                control_conveyor(False)  # Stop conveyor momentarily
                trigger_alarm('PART_REJECTED', f'Bad part rejected. Count: {bad_count}', 'WARNING')
                
                # Resume conveyor after a brief delay
                time.sleep(0.5)
                control_conveyor(True)
            
            # Update PLC production counters
            update_plc_counters(good_count, bad_count)
            previous_status = status
        
        # Log production data to SCADA
        alarm_status = 0  # 0=OK, 1=WARNING, 2=CRITICAL
        scada.log_production(fill_percentage, status, good_count, bad_count, alarm_status)
        
        # Periodic export of dashboard data
        current_time = time.time()
        if current_time - last_dashboard_export > dashboard_export_interval:
            scada.export_dashboard_data('dashboard_data.json')
            last_dashboard_export = current_time


        # ==================================
        # RED MOLTEN OVERLAY
        # ==================================

        overlay = output.copy()

        overlay[molten_inside > 0] = (
            0,
            0,
            255
        )

        output = cv2.addWeighted(
            output,
            0.7,
            overlay,
            0.3,
            0
        )

        # ==================================
        # DISPLAY STATUS
        # ==================================

        cv2.putText(
            output,
            status,
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )
        cv2.putText(
            output,
            f"Good : {good_count}",
            (30,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            output,
            f"Reject : {bad_count}",
            (30,150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        cv2.putText(
            output,
            f"Fill : {fill_percentage:.1f}%",
            (30,190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

    else:

        fill_counter = 0


    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "Molten Metal Detection",
        output
    )

    cv2.imshow(
        "Molten Mask",
        molten_mask
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================================
# CLEANUP & SHUTDOWN
# ==================================

logging.info("=== PRODUCTION SESSION ENDED ===")
logging.info(f"Final Good Count: {good_count}")
logging.info(f"Final Bad Count: {bad_count}")
logging.info(f"Total Units: {good_count + bad_count}")
if good_count + bad_count > 0:
    acceptance_rate = (good_count / (good_count + bad_count)) * 100
    logging.info(f"Acceptance Rate: {acceptance_rate:.2f}%")

# Export final dashboard data
scada.export_dashboard_data('dashboard_data.json')
logging.info("Dashboard data exported to dashboard_data.json")

# Shutdown control signals
set_indicator_lamp('RED')
control_conveyor(False)
control_reject_cylinder(False)

# Disconnect PLC
plc.disconnect()

# Release video resources
cap.release()
cv2.destroyAllWindows()

logging.info("Program ended successfully")
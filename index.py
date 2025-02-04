import cv2
import os
import numpy as np
import pandas as pd
import face_recognition
import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
from PIL import Image, ImageTk

# Create required directories
if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists("attendance_records"):
    os.makedirs("attendance_records")

# Load existing encodings if available
ENCODINGS_FILE = "dataset/encodings.npy"
NAMES_FILE = "dataset/names.npy"

def load_encodings():
    if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
        encodings = np.load(ENCODINGS_FILE, allow_pickle=True)
        names = np.load(NAMES_FILE, allow_pickle=True)
        return encodings, names
    return [], []

encodings, names = load_encodings()

def save_encodings():
    np.save(ENCODINGS_FILE, encodings)
    np.save(NAMES_FILE, names)

def register_face():
    global encodings, names
    
    name = simpledialog.askstring("Input", "Enter Name:")
    if not name:
        return
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            encodings.append(face_encoding)
            names.append(name)
            save_encodings()
            messagebox.showinfo("Success", f"Face registered for {name}")
            break
        
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_path = f"attendance_records/{date_str}.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])
    
    if name not in df["Name"].values:
        df.loc[len(df)] = [name, datetime.now().strftime('%H:%M:%S')]
        df.to_csv(file_path, index=False)

def recognize_face():
    global encodings, names
    
    if not encodings:
        messagebox.showwarning("Warning", "No registered faces found!")
        return
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                matched_idx = np.where(matches)[0][0]
                name = names[matched_idx]
                mark_attendance(name)
                
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def create_gui():
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    
    tk.Label(root, text="Face Recognition Attendance System", font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="Register Face", command=register_face, font=("Arial", 14)).pack(pady=5)
    tk.Button(root, text="Recognize & Mark Attendance", command=recognize_face, font=("Arial", 14)).pack(pady=5)
    tk.Button(root, text="Exit", command=root.quit, font=("Arial", 14)).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()

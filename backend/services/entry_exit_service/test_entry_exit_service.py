import cv2
import json
import time
import os
import argparse
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path

# Import the core logic from your latest utils.py
from utils import ThreeLineCounter, DEFAULT_MODEL_PATH

def draw_dashboard(frame, stats):
    """Draws only the top statistics bar on the frame."""
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Counts
    cv2.putText(frame, f"ENTERED: {stats['entered']}", (30, 50), font, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"EXITED: {stats['exited']}", (350, 50), font, 1.2, (0, 0, 255), 3)
    
    # Demographics
    demographics = (
        f"M: {stats['gender_stats'].get('male', 0)}  "
        f"F: {stats['gender_stats'].get('female', 0)}  "
        f"Kids: {stats['age_stats'].get('child', 0)}"
    )
    text_size = cv2.getTextSize(demographics, font, 0.8, 2)[0]
    cv2.putText(frame, demographics, (w - text_size[0] - 30, 50), font, 0.8, (200, 200, 200), 2)

    return frame

def resolve_demographics(person):
    """Helper to safely get gender/age from the new list structure in PersonInfo."""
    g = person.final_gender
    a = person.final_age_group
    
    # Fallback to latest prediction if final is not set
    if not g and person.gender_preds:
        g = person.gender_preds[-1]
    if not a and person.age_preds:
        a = person.age_preds[-1]
        
    return g or "?", a or "?"

def main():
    parser = argparse.ArgumentParser(description="Test Entry/Exit Logic with Visualization")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--source", type=str, default=None, help="Override video source")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Determine Source
    source = args.source or config['input'].get('camera_url') or config['input'].get('video_path')
    if not source:
        print("Error: No video source found.")
        return

    print(f"Testing Entry/Exit Service on: {source}")

    # --- PARSE LINES ---
    lines = []
    for key in ['line_1', 'line_2', 'line_3']:
        if key in config['lines']:
            start = tuple(config['lines'][key]['start'])
            end = tuple(config['lines'][key]['end'])
            lines.append((start, end))

    # --- PARSE FILTER LINE (Explicit Coordinates) ---
    filter_line = None
    if 'filter_line' in config:
        fl_cfg = config['filter_line']
        if 'start' in fl_cfg and 'end' in fl_cfg:
            p1 = tuple(map(int, fl_cfg['start']))
            p2 = tuple(map(int, fl_cfg['end']))
            filter_line = (p1, p2)
            print(f"Loaded Filter Line: {filter_line}")

    # --- INITIALIZE COUNTER ---
    counter = ThreeLineCounter(
        lines=lines,
        model_path=config['model'].get('model_path', DEFAULT_MODEL_PATH),
        confidence=config['model'].get('confidence_threshold', 0.5),
        enable_gender_age=config['tracking'].get('enable_gender_age', True),
        filter_line=filter_line  # Passing the explicit tuple
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    processed_events = set()
    paused = False
    
    cv2.namedWindow("Test Entry/Exit Service", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break

            # Process
            processed_frame = counter.process_frame(frame)

            # Check Events
            for track_id, person in counter.tracked_people.items():
                
                # Check Entry
                if person.entered and person.entry_time:
                    event_key = (track_id, person.entry_time, "ENTRY")
                    if event_key not in processed_events:
                        processed_events.add(event_key)
                        
                        g, a = resolve_demographics(person)
                        t_str = datetime.now().strftime("%H:%M:%S")
                        
                        print(f"[{t_str}] ✅ ENTRY: ID {track_id} ({g}, {a}) [Group: {person.is_in_group}]")

                # Check Exit
                if person.exited and person.exit_time:
                    event_key = (track_id, person.exit_time, "EXIT")
                    if event_key not in processed_events:
                        processed_events.add(event_key)
                        
                        g, a = resolve_demographics(person)
                        t_str = datetime.now().strftime("%H:%M:%S")
                        
                        print(f"[{t_str}] ❌ EXIT: ID {track_id} ({g}, {a}) [Group: {person.is_in_group}]")

            # Draw
            stats = counter.get_statistics()
            final_frame = draw_dashboard(processed_frame, stats)
            cv2.imshow("Test Entry/Exit Service", final_frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('s'): cv2.imwrite(f"snapshot_{int(time.time())}.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
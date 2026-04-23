import redis
import json
import base64
import cv2
import numpy as np
import time
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "6")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:{CAMERA_ID}")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def decode_frame(frame_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def draw_table_boxes(img, table_bboxes, table_status, data={}):
    """
    Draw table bounding boxes with status-based colors.
    
    Colors (BGR):
    - EMPTY: Green (0, 255, 0)
    - OCCUPIED: Red (0, 0, 255) 
    - DIRTY: Yellow (0, 255, 255)
    """
    COLOR_EMPTY = (0, 255, 0)      # Green
    COLOR_OCCUPIED = (0, 0, 255)   # Red
    COLOR_DIRTY = (0, 255, 255)    # Yellow
    
    for table in table_bboxes:
        bbox = table["bbox"]
        table_id = table.get("id", "?")  # Use actual table ID from data
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Look up status using the actual table ID
        status = table_status.get(str(table_id), table_status.get(table_id, "EMPTY"))
        
        if status == "OCCUPIED":
            color = COLOR_OCCUPIED
        elif status == "DIRTY":
            color = COLOR_DIRTY
        else:
            color = COLOR_EMPTY
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        table_debug_info = data.get("table_debug_info", {})
        stats = table_debug_info.get(str(table_id), {})
        
        # Format label with stats if available
        # Example: T1: OCCUPIED | P:15/20 Emp:0/30
        label = f"T{table_id}: {status}"
        
        if stats:
            p_count = stats.get("person_count", 0)
            p_thresh = stats.get("occupied_threshold", 0)
            t_count = stats.get("tray_count", 0)
            t_thresh = stats.get("dirty_threshold", 0)
            
            c_empty = stats.get("consec_empty", 0)
            r_empty_thresh = stats.get("resolve_empty_threshold", 0)
            
            c_clean = stats.get("consec_clean", 0)
            r_clean_thresh = stats.get("resolve_clean_threshold", 0)
            
            # Sub-label for counts
            sub_label = ""
            if status == "OCCUPIED":
                sub_label = f"Unocc: {c_empty}/{r_empty_thresh}"
            elif status == "DIRTY":
                sub_label = f"Clean: {c_clean}/{r_clean_thresh}"
            else:
                # If Empty, show progress towards Occupied or Dirty
                sub_label = f"Occ: {p_count}/{p_thresh} | Dir: {t_count}/{t_thresh}"

            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            cv2.putText(
                img,
                sub_label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        else:
             cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )


def draw_persons(img, persons, debug_stats=None):
    """
    Draw person bounding boxes in Blue.
    """
    COLOR_PERSON = (255, 0, 0)  # Blue
    
    for idx, person in enumerate(persons):
        bbox = person["bbox"]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(
            img,
            f"Person {idx}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_PERSON,
            2,
        )

        # Draw debug stats if available
        if debug_stats:
            stats = debug_stats.get(str(idx))
            if stats:
                y_offset = y2 + 15
                for t_id, metrics in stats.items():
                    # STRICT FILTER: Show only if table_overlap > 0
                    if float(metrics.get('table_overlap', 0)) > 0:
                        
                        occ_mark = "[OCC]" if metrics.get("is_occupying") else ""
                        
                        # Helper to format "0.80 (True)" -> "0.8"
                        def fmt(val):
                            try:
                                parts = val.split(' ')
                                num = parts[0]
                                # We only want the number now as per request? 
                                # User said "up,down,left,right,priority and total_overlap"
                                # Assuming just values to keep it clean, or values+bool?
                                # Let's keep it compact: "0.8" or "0.8(T)"
                                # "values for tables with which the person has a non 0 overlap"
                                return num
                            except:
                                return val

                        # Format: T{id} P{prio} TO:{total} L:{l} R:{r} U:{u} D:{d} {OCC}
                        text = (f"T{t_id} P{metrics.get('priority')} TO:{metrics.get('table_overlap')} "
                                f"L:{fmt(metrics.get('left'))} R:{fmt(metrics.get('right'))} "
                                f"U:{fmt(metrics.get('up'))} D:{fmt(metrics.get('down'))} {occ_mark}")
                        
                        # Draw background for legibility
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(img, (x1, y_offset - h - 2), (x1 + w, y_offset + 2), (0,0,0), -1)
                        
                        cv2.putText(
                            img,
                            text,
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                        y_offset += 15


def draw_plates(img, plates):
    """
    Draw plate/tray bounding boxes in Cyan.
    """
    COLOR_PLATE = (255, 255, 0)  # Cyan
    
    for idx, plate in enumerate(plates):
        bbox = plate["bbox"]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        score = plate.get("score", 0)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PLATE, 2)
        cv2.putText(
            img,
            f"Plate ({score:.2f})" if score else f"Plate {idx}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_PLATE,
            2,
        )


def visualize():
    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    window_initialized = False
    while True:
        try:
            if window_initialized:
                 if cv2.waitKey(1) & 0xFF == ord("q"):
                     break
                 if cv2.getWindowProperty("Table Occupancy Visualization", cv2.WND_PROP_VISIBLE) < 1:
                     break

            # Consume ALL available frames to get to the latest one
            # blocked pop for the first one, then non-blocking for the rest
            msg = r.blpop(QUEUE_NAME, timeout=0.1)
            if msg is None:
                continue

            last_msg = msg
            # Drain the queue of any older buffered frames
            while True:
                next_msg = r.lpop(QUEUE_NAME)
                if next_msg is None:
                    break
                # Update last_msg to the newer one
                # Note: lpop returns value directly or None. blpop returns (queue, value).
                # standard redis lpop returns value. blpop returns tuple.
                # We need to format consistency.
                last_msg = (QUEUE_NAME, next_msg)
            
            _, payload = last_msg
            data = json.loads(payload)

            frame = decode_frame(data["frame"])

            print(data.get("table_status", {}))
            
            # Print debug stats to console as well
            debug_stats = data.get("debug_stats", {})
            if debug_stats:
                print("\n--- Frame Stats ---")
                for p_idx, p_data in debug_stats.items():
                    print(f"Person {p_idx}:")
                    for t_id, metrics in p_data.items():
                         if float(metrics['table_overlap']) > 0:
                            
                            occupy_mark = "[OCCUPYING]" if metrics.get("is_occupying") else ""
                            print(f"  -> Table {t_id} {occupy_mark}")
                            print(f"     Priority: {metrics.get('priority')}")
                            print(f"     IOU: {metrics.get('table_overlap')}")
                            print(f"     Left: {metrics.get('left')}")
                            print(f"     Right: {metrics.get('right')}")
                            print(f"     Up: {metrics.get('up')}")
                            print(f"     Down: {metrics.get('down')}")

            # Draw table bounding boxes with status colors
            draw_table_boxes(
                frame,
                data.get("table_bboxes", []),
                data.get("table_status", {}),
                data
            )

            # Draw person bounding boxes
            draw_persons(frame, data.get("persons", []), debug_stats)

            # Draw plate bounding boxes
            draw_plates(frame, data.get("plates", []))

            # # Draw configuration parameters
            # config = data.get("config", {})
            # if config:
            #     y_offset = 60
            #     for key, value in config.items():
            #         text = f"{key}: {value}"
            #         cv2.putText(
            #             frame,
            #             text,
            #             (10, y_offset),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6,
            #             (0, 255, 255),
            #             2,
            #         )
            #         y_offset += 25

            # Overlay metadata
            camera_id = data.get("camera_id", "unknown")
            timestamp = data.get("timestamp", "")
            
            debug_stats = data.get("debug_stats", {})
            if debug_stats:
                print("\n--- Frame Stats ---")
                for p_idx, p_data in debug_stats.items():
                    print(f"Person {p_idx}:")
                    for t_id, metrics in p_data.items():
                        # Only print relevant ones (non-zero or occupying) to avoid spam?
                        # Or print all if user requested.
                        # For now, let's print if ANY metric > 0 or occupying
                        if (float(metrics.get('table_overlap', 0)) > 0 or 
                            "True" in str(metrics.get('left', '')) or "True" in str(metrics.get('right', '')) or 
                            "True" in str(metrics.get('up', '')) or "True" in str(metrics.get('down', ''))):
                            
                            occupy_mark = "[OCCUPYING]" if metrics.get("is_occupying") else ""
                            print(f"  -> Table {t_id} {occupy_mark}")
                            print(f"     Priority: {metrics.get('priority')}")
                            print(f"     IOU: {metrics.get('table_overlap')}")
                            print(f"     Left: {metrics.get('left')}")
                            print(f"     Right: {metrics.get('right')}")
                            print(f"     Up: {metrics.get('up')}")
                            print(f"     Down: {metrics.get('down')}")

            cv2.putText(
                frame,
                f"Camera: {camera_id} | Time: {timestamp}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Table Occupancy Visualization", frame)
            window_initialized = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # Check if window was closed by user
            if cv2.getWindowProperty("Table Occupancy Visualization", cv2.WND_PROP_VISIBLE) < 1:
                break

        except Exception as e:
            print(f"[Visualizer Error] {e}")
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()

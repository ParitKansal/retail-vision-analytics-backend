Strict Status Logic (Hybrid Approach):
Opening (7-9 AM): The status is considered "Open" only when the lights are 100% ON (percentage_closed == 0.0). This ensures we don't alert "Early Opening" for partial lighting or setup work.
Closing (12-2 AM) & Other Times: The status is considered "Closed" only when the lights are 0% ON (Fully Dark). This ensures we warn about "Closing" only when the store is actually fully closed.
Transition Capture:
Added a frame_buffer (size 5) to look back at previous frames.
During Opening, the system captures and stores images for:
0%: The last frame before any light turned on.
Partial: The first frame detected with partial lighting (0% < lights < 100%).
100%: The first frame with full lighting.
These images are included in the "Store Open" alert payload.
Alerting Rules:
Opening:
Early: 7:00 - 7:50
Perfect: 7:50 - 8:15
Late: 8:15 - 9:00
Closing:
Early: 12:00 - 12:50
Perfect: 12:50 - 1:05
Late: 1:05 - 2:00
Stability:
The status must be stable for 80% of the buffer (approx. 48 out of 60 frames) before an alert is triggered.
The service is now updated with this logic. You can restart the service to apply these changes.
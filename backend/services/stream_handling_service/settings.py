CAMERAS = {

    "1": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.13:554/live",
        "sample_fps": 1,
        "consumer_groups": [
            "cg:cam:counter_staff_detection1",
            "cg:cam:video_classification",
            "cg:cam:customer_path",
            "cg:cam:dirty_floor_service",
#            "cg:cam:drawer_detection_service",
        ],
    },
    "2": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.21:554/live",
        "sample_fps": 1,
        "consumer_groups": [
            "cg:cam:counter_staff_detection2",
            "cg:cam:table_detection_and_cleaning3",
        ],
    },
    "3": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.14:554/live",
        "sample_fps": 0.5,
        "consumer_groups": [
            "cg:cam:table_detection_and_cleaning1",
            "cg:cam:video_classification",
        ],
    },
    "4": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.12:554/live",
        "sample_fps": 0.5,
        "consumer_groups": [
            "cg:cam:table_detection_and_cleaning2",
            "cg:cam:video_classification",
        ],
    },
    "5": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.15:554/live",
        "sample_fps": 1,
        "consumer_groups": [
#            "cg:cam:crew_room", 
            "cg:cam:video_classification"],
    },
    "6": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.64:554/live",
        "sample_fps": 1,
        "consumer_groups": [
            "cg:cam:kiosk_person_service",
            "cg:cam:table_detection_and_cleaning4",
            "cg:cam:video_classification",
            "cg:cam:store_opening_closing_service",
            "cg:cam:exit_emotion_detection_service",
        ],
    },
    "7": {
        "rtsp": "rtsp://CAMERA_USER:CAMERA_PASS@192.168.10.16:554/live",
        "sample_fps": 1,
        "consumer_groups": [
            "cg:cam:dirty_floor_service",
        ],
    },
}


# CAMERAS = {

#     "1": { #counter
#         "rtsp": "rtsp://192.168.2.106:8554/counter1",
#         "sample_fps": 1,
#         "consumer_groups": [
#             "cg:cam:counter_staff_detection1",
#             "cg:cam:video_classification",
#             "cg:cam:customer_path",
#             "cg:cam:dirty_floor_service",
# #            "cg:cam:drawer_detection_service",
#         ],
#     },
#     "2": { # cafe
#         "rtsp": "rtsp://192.168.2.106:8554/cafe",
#         "sample_fps": 1,
#         "consumer_groups": [
#             "cg:cam:counter_staff_detection2",
#             "cg:cam:table_detection_and_cleaning3",
#         ],
#     },
#     "3": { # back lobby
#         "rtsp": "rtsp://192.168.2.106:8554/back",
#         "sample_fps": 0.5,
#         "consumer_groups": [
#             "cg:cam:table_detection_and_cleaning1",
#             "cg:cam:video_classification",
#         ],
#     },
#     "4": { #lobby left
#         "rtsp": "rtsp://192.168.2.106:8554/left",
#         "sample_fps": 0.5,
#         "consumer_groups": [
#             "cg:cam:table_detection_and_cleaning2",
#             "cg:cam:video_classification",
#         ],
#     },
#     "5": { #crew room
#         "rtsp": "rtsp://192.168.2.106:8554/crew",
#         "sample_fps": 1,
#         "consumer_groups": [
# #            "cg:cam:crew_room", 
#             "cg:cam:video_classification"],
#     },
#     "6": { #kiosk
#         "rtsp": "rtsp://192.168.2.106:8554/kiosk",
#         "sample_fps": 1,
#         "consumer_groups": [
#             "cg:cam:kiosk_person_service",
#             "cg:cam:table_detection_and_cleaning4",
#             "cg:cam:video_classification",
#             "cg:cam:store_opening_closing_service",
#             "cg:cam:exit_emotion_detection_service",
#         ],
#     },
#     "7": { #kitchen
#         "rtsp": "rtsp://192.168.2.106:8554/kitchen",
#         "sample_fps": 1,
#         "consumer_groups": [
#             "cg:cam:dirty_floor_service",
#         ],
#     },
# }

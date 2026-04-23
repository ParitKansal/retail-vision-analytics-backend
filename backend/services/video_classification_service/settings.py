import json

with open("crew_room_mask.json", "r", encoding="utf-8") as f:
    crew_room_mask = json.load(f)

with open("counter_phone.json", "r", encoding="utf-8") as f:
    counter_1_mask = json.load(f)

with open("kiosk_mask.json", "r", encoding="utf-8") as f:
    kiosk_mask = json.load(f)

with open("counter_mask_floor_cleaning.json", "r", encoding="utf-8") as f:
    counter_mask_floor_cleaning = json.load(f)

with open("left_lobby_mask.json", "r", encoding="utf-8") as f:
    left_lobby_mask = json.load(f)

with open("back_lobby_mask.json", "r", encoding="utf-8") as f:
    back_lobby_mask = json.load(f)

SETTINGS = {
    "stream:cam:1": {  # counter 1
        "group_name": "cg:cam:video_classification",
        "fps": 1,
        "prompts": [
            {
                "prompt": [
                    "a person using mobile phone at workplace",
                    "a group of people standing idle",
                    "an empty workspace",
                    "a person working on a computer",
                ],
                "mask": counter_1_mask[0].get("points"),
            },
            {
                "prompt": [
                    "cleaning with mop",
                    "a person working on a computer",
                    "an empty workspace",
                    "a person talking over or using phone",
                ],
                "mask": counter_mask_floor_cleaning[0].get("points"),
            },
        ],
    },
    # "stream:cam:2": {  # cafe
    #     "group_name": "cg:cam:video_classification",
    #     "fps": 1,
    #     "prompts": [
    #         ["hygiene is maintained", "hygiene is not maintained"],
    #         ["person is using laptop", "person is not using laptop"],
    #     ],
    # },
    "stream:cam:3": {  # back lobby
        "group_name": "cg:cam:video_classification",
        "fps": 1,
        "prompts": [
            {
                "prompt": [
                    "cleaning with mop",
                    "a person working on a computer",
                    "an empty space",
                    "a person talking over or using phone",
                ],
                "mask": back_lobby_mask[0].get("points"),
            },
        ],
    },
    "stream:cam:4": {  # left lobby
        "group_name": "cg:cam:video_classification",
        "fps": 1,
        "prompts": [
            {
                "prompt": [
                    "cleaning with mop",
                    "a person working on a computer",
                    "an empty space",
                    "a person talking over or using phone",
                ],
                "mask": left_lobby_mask[0].get("points"),
            },
        ],
    },
    "stream:cam:5": {  # crew room
        "group_name": "cg:cam:video_classification",
        "fps": 1,
        "prompts": [
            {
                "prompt": [
                    "a person lying and sleeping on a bench",
                    "a person using a mobile phone",
                    "a group of people standing and talking",
                    "a person sitting and eating food",
                ],
                "mask": crew_room_mask[0].get("points"),
            }
        ],
    },
    "stream:cam:6": {  # kiosk
        "group_name": "cg:cam:video_classification",
        "fps": 1,
        "prompts": [
            {
                "prompt": [
                    "cleaning with mop",
                    "a person working on a computer",
                    "an empty floor",
                    "a person talking over or using phone",
                    "a person standing idle",
                ],
                "mask": kiosk_mask[0].get("points"),
            },
        ],
    },
}

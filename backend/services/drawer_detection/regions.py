
from typing import TypedDict, List, Optional
import copy

class Point(TypedDict):
    x: float
    y: float

class Region(TypedDict):
    points: List[Point]
    strokeColor: str
    id: str
    name: str

REGIONS_ORIGINAL: List[Region] = [
  {
    "points": [
      {"x": 694.4345227311311, "y": 298.9183914005311},
      {"x": 694.4345227311311, "y": 295.7718820173676},
      {"x": 655.1031577364871, "y": 283.1858444847137},
      {"x": 655.1031577364871, "y": 286.33235386787715}
    ],
    "strokeColor": "#00FFFF",
    "id": "left_region_1771070517576",
    "name": "Cyan"
  },
  {
    "points": [
      {"x": 732.1926331259893, "y": 206.0963645972083},
      {"x": 733.7658877257752, "y": 204.52310990562654},
      {"x": 692.8612681313454, "y": 195.08358175613608},
      {"x": 691.2880135315596, "y": 198.2300911392996}
    ],
    "strokeColor": "#FF00FF",
    "id": "left_region_1771070640297",
    "name": "Pink"
  },
  {
    "points": [
      {"x": 655.1031577364871, "y": 287.9056085594589},
      {"x": 692.8612681313454, "y": 300.49164609211283},
      {"x": 691.2880135315596, "y": 302.0649007836946},
      {"x": 655.1031577364871, "y": 291.0521179426224}
    ],
    "strokeColor": "#FFFF00",
    "id": "left_region_1771070723715",
    "name": "Yellow"
  },
  {
    "points": [
      {"x": 691.2880135315596, "y": 201.37660052246306},
      {"x": 730.6193785262036, "y": 207.66961928879005},
      {"x": 730.6193785262036, "y": 210.81612867195352},
      {"x": 691.2880135315596, "y": 204.52310990562654}
    ],
    "strokeColor": "#00FF00",
    "id": "left_region_1771070744842",
    "name": "Green"
  }
]

CHECKING_REGIONS: List[Region] = [
  {
    "points": [
      {"x": 653.5299031367013, "y": 277.52211799263483},
      {"x": 651.9566485369155, "y": 300.40117429161575},
      {"x": 692.8612681313454, "y": 312.98721182426965},
      {"x": 697.5810319307026, "y": 291.6814102168705}
    ],
    "strokeColor": "#FF0000",
    "id": "left_region_1771076530044",
    "name": "red"
  },
  {
    "points": [
      {"x": 692.8612681313454, "y": 187.53195443535142},
      {"x": 683.4217405326308, "y": 207.9842654259141},
      {"x": 732.1926331259893, "y": 218.9970482669863},
      {"x": 735.3391423255609, "y": 196.97148258484188}
    ],
    "strokeColor": "#0000FF",
    "id": "left_region_1771076557264",
    "name": "blue"
  }
]

# We are using the original regions as the source of truth for experiment 4.0
REGIONS = REGIONS_ORIGINAL

def get_region_by_name(regions_list: List[Region], name: str) -> Optional[Region]:
    for r in regions_list:
        if r['name'].lower() == name.lower():
            return r
    return None

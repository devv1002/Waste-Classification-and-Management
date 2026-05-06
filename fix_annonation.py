import json
import os

with open("annotations.json") as f:
    data = json.load(f)

valid_images = []
valid_ids = set()

for i, img in enumerate(data["images"]):
    new_name = f"{i}.jpg"
    
    if os.path.exists(f"images/{new_name}"):
        img["file_name"] = new_name
        valid_images.append(img)
        valid_ids.add(img["id"])

valid_annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] in valid_ids
]

data["images"] = valid_images
data["annotations"] = valid_annotations

with open("annotations_fixed.json", "w") as f:
    json.dump(data, f)

print("Done fixing!")
print("Images:", len(valid_images))
print("Annotations:", len(valid_annotations))
import json

with open('vaccine_camera.json', 'r') as f:
    cam = json.load(f)

first_cam = cam['data'][0]
cam['data'] = [first_cam] * 129

with open('vaccine_camera.json', 'w') as f:
    json.dump(cam, f)

print('完成，現在有', len(cam['data']), '幀')
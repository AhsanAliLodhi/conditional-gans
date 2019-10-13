cameras=[
    ("CameraRGB0",0.0),
    ("CameraRGB1",0.5),
    ("CameraRGB2",-0.3),
    ("CameraRGB3",1.0),
    ("CameraRGB4",-1.3),
    ("CameraRGB5",2.0),
]



def get_abs_pos(camera):
    if camera == "CameraRGB0":
        return 0.0
    if camera == "CameraRGB1":
        return 0.5
    if camera == "CameraRGB2":
        return -0.3
    if camera == "CameraRGB3":
        return 1.0
    if camera == "CameraRGB4":
        return -1.3
    if camera == "CameraRGB5":
        return 2.0
    return None

def get_cameras(reverse=False):
    return sorted(cameras, key=lambda x: x[1],reverse=reverse)

def get_rel_dists(pos,reverse=True,abs_pos = None,target_pos  = None):
    cameras = get_cameras(reverse=reverse)
    if abs_pos is None:
        if target_pos is None:
            cameras = [(camera[0],camera[1] - pos ) for camera in cameras]
        else:
            cameras = [(camera[0],camera[1] - pos ) for camera in cameras if target_pos == camera[1]]
    else:
        cameras = [(None,pos - camera) for camera in abs_pos]
    return cameras
import json
from unrealcv import client
from unrealcv.util import read_npy, read_png
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import glog as log

help_message = '''
A demo showing how to control a game using python
a, d: rotate camera to left and right.
q, e: move camera up and down.
left, right, up, down: move around
'''
plt.rcParams['keymap.save'] = ''

lock = 0

def move_3d(client, cam_id, delt_x, delt_y, delt_z):
    log.warn("Executing move_3d.")

    pose = self.get_pose(cam_id)
    location_now = self.cam[cam_id]['location']
    log.warn("Current location: {}".format(location_now))
    log.warn("Current rotation: {}".format(pose[-3:]))

    location_exp = [location_now[0] + delt_x, location_now[1]+delt_y, location_now[2]+delt_z]
    log.warn("Expecting to move to this location: {}".format(location_exp))

    while self.lock:
        log.warn("waiting for lock to open.")
        continue

    if not self.lock:
        log.warn("acquiring lock")
        lock = 1
        while self.lock:
            log.warn("locked.")
            self.moveto(cam_id, location_exp)
            self.lock = 0
    else:
        log.warn("process was locked, skipping this move")

    log.warn("Get Pose being called.")

    pose = self.get_pose(cam_id)
    location_now = self.cam[cam_id]['location']
    log.warn("moved to location: {}".format(location_now))
    log.warn("rotated to rotation: {}".format(pose[-3:]))

    if error < 36: # weird offset
        return False
    else:
        return True

def moveto(self, client, cam_id, loc):
    cmd = 'vset /camera/{cam_id}/moveto {x} {y} {z}'
    client.request(cmd.format(cam_id=cam_id, x=loc[0], y=loc[1], z=loc[2]))





def main():
    loc = None
    rot = None

    fig, ax = plt.subplots()
    img = np.zeros((480, 640, 4))
    ax.imshow(img)

    def onpress(event):
        rot_offset = 10 # Rotate 5 degree for each key press
        loc_offset = 10 # Move 5.0 when press a key

        # Up and Down in cam-plane
        if event.key == 'w': loc[0] += loc_offset
        if event.key == 's': loc[0] -= loc_offset
        # Left and Right movement in cam-plane
        if event.key == 'a': loc[1] -= loc_offset
        if event.key == 'd': loc[1] += loc_offset
        # In and Out movement into cam-plane
        if event.key == 'q': loc[2] += loc_offset
        if event.key == 'e': loc[2] -= loc_offset

        # cmd = 'vset /camera/0/rotation %s' % ' '.join([str(v) for v in rot])
        # client.request(cmd)
        cmd = 'vset /camera/0/moveto %s' % ' '.join([str(v) for v in loc])
        client.request(cmd)

        print(client.request('vget /camera/0/location'))
        print(client.request('vget /camera/0/rotation'))
        res = client.request('vget /camera/2/lit png')
        img = read_png(res)

        ax.imshow(img)
        fig.canvas.draw()

    client.connect()
    if not client.isconnected():
        print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
        return
    else:
        print(help_message)

    print("-------------------------------------------------------------------")
    print(client.request('vget /objects'))
    print("-------------------------------------------------------------------")
    print(client.request('vget /cameras'))
    print("-------------------------------------------------------------------")
    # print(client.request('vget /cameras'))
    # print("-------------------------------------------------------------------")

    init_loc = [float(v) for v in client.request('vget /camera/0/location').split(' ')]
    init_rot = [float(v) for v in client.request('vget /camera/0/rotation').split(' ')]

    loc = init_loc; rot = init_rot

    fig.canvas.mpl_connect('key_press_event', onpress)
    plt.title('Keep this window in focus, it will be used to receive key press event')
    plt.axis('off')
    plt.show() # Add event handler

if __name__ == '__main__':
    main()

import mujoco
import matplotlib.pylab as plt
import numpy as np


s_30 = np.sin(np.pi/6)
cos_30 = np.cos(np.pi/6)
def create_scene(x, y, car_ori, radar_ori, i):
    xml = f"""
<mujoco>
    <visual>
        <global offwidth="1280" offheight="1280"/>
    </visual>
    <worldbody>
        <body name="floor" pos="0 0 -0.1">
            <geom size="2.0 2.0 0.02" rgba="0.2 0.2 0.2 1" type="box" />
        </body>
        <body name="x_arrow" pos="0.5 0 0">
            <geom size="0.5 0.01 0.01" rgba="1 0 0 0.5" type="box" />
        </body>
        <body name="y_arrow" pos="0 0.5 0">
            <geom size="0.01 0.5 0.01" rgba="0 1 0 0.5" type="box" />
        </body>
        <body name="z_arrow" pos="0 0 0.5">
            <geom size="0.02 0.02 0.5" rgba="0 0 1 0.5" type="box" />
        </body>
        <body name="car" pos="{x} {y} 0.1" axisangle="0 0 1 {car_ori}">
            <body name="deck" pos="0 0 0">
                <geom size="0.2 0.1 0.02" rgba="1 1 1 0.9" type="box"/>
            </body>
            <body name="front" pos="-0.2 0 0">
                <geom size="0.01 0.01 0.01" rgba="1 0 0 0.5" type="box" />
            </body>
            <body name="wheel_1" pos="0.1 0.1 0" axisangle="1 0 0 90">
                <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
            </body>
            <body name="wheel_2" pos="-0.1 0.1 0" axisangle="1 0 0 90">
                <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
            </body>
            <body name="wheel_3" pos="0.1 -0.1 0" axisangle="1 0 0 90">
                <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
            </body>
            <body name="wheel_4" pos="-0.1 -0.1 0" axisangle="1 0 0 90">
                <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
            </body>
            <body name="radar" pos="0 0 0.1" axisangle="0 0 1 {radar_ori}">
                <body name="stick" pos="0 -0.1 0" axisangle="1 0 0 30">
                    <geom size="0.01 0.01 0.1" rgba="1 1 1 1" type="box"/>
                    <body name="head" pos="0 0 0.1">
                        <geom size="0.03 0.01" rgba="1 0 0 1" type="cylinder"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
    with open('test_xml.xml', 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1280, 1280)

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    plt.imsave(f"frame_{i:03d}.png", renderer.render())

n_frames = 50

angles = np.linspace(3/4 * 2 * np.pi, 2 * np.pi, n_frames)
xs = np.cos(angles)
ys = np.sin(angles)

xs -= 1

angles_in_degrees = np.linspace(270, 360, n_frames)
# create_scene(-1, -1, 180 +, 0, 0)
for i in range(len(angles)):
    create_scene(xs[i], ys[i], angles_in_degrees[i] - 90, 0, i)


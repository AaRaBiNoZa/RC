import mujoco
import matplotlib.pylab as plt


for i in range(10):
    xml = f"""
<mujoco>
  <visual>
     <global offwidth="1280" offheight="1280"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos="{.2 + i / 10} .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1280, 1280)

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    plt.imsave(f"frame_{i:03d}.png", renderer.render())

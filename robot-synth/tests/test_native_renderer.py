"""Optional real renderer test; skipped, never mocked, without graphics runtime."""

import importlib.util
import os

import numpy as np
import pytest


@pytest.mark.parametrize('with_floor', [False, True], ids=['box', 'box_with_infinite_floor'])
def test_native_offscreen_rgb_depth_segmentation(config, with_floor):
    if importlib.util.find_spec('mujoco') is None:
        pytest.skip('MuJoCo runtime is not installed')
    import ctypes.util
    if not ctypes.util.find_library('OSMesa'):
        pytest.skip('OSMesa is not installed')
    from importlib.metadata import version
    if version('mujoco') != config['simulation']['mujoco_version']:
        pytest.skip('Pinned MuJoCo runtime is required')
    from robot_synth.simulation.libero_replay import LiberoReplay
    from robot_synth.validation.geometry import validate_views
    xml = '''<mujoco><visual><global offwidth="64" offheight="64"/></visual><worldbody>
    <light pos="0 0 3"/><camera name="agentview" pos="0 0 2"/>
    <camera name="robot0_eye_in_hand" pos="0 0 2"/>
    <body name="box_1_main"><freejoint/><geom type="box" group="1" size=".2 .2 .2" rgba="1 .2 .1 1" mass="1"/></body>
    </worldbody></mujoco>'''
    if with_floor:
        xml = xml.replace('</worldbody>', '''<body name="floor" pos="0 0 -0.5">
            <geom name="infinite_floor" type="plane" group="1" size="3 3 .125" rgba=".2 .4 .6 1"/>
            </body></worldbody>''')
    replay = LiberoReplay(config, xml)
    try:
        state = np.zeros(14)
        state[4] = 1
        views = replay.render(state)
        assert validate_views(views, config)
        view = views['agentview']
        assert abs(float(view['depth'][16, 16]) - 1.8) < .002
        assert int(view['instance'][16, 16]) == 1
        assert [obj['body_name'] for obj in view['annotation']['objects']] == ['box_1_main']
        if with_floor:
            # An unannotated plane is still rendered and supplies metric depth.
            assert view['instance'][0, 0] == 0
            assert view['depth_valid'][0, 0]
            assert abs(float(view['depth'][0, 0]) - 2.5) < .002
            assert int(view['rgb'][0, 0].max()) > 0
            assert view['annotation']['objects'][0]['size_xyz'] == pytest.approx([.4, .4, .4])
    finally:
        replay.close()

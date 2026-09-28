from glob import glob
import os

from setuptools import setup

package_name = 'dua_camera_calibration'

setup(
    name=package_name,
    version='4.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        ('share/' + package_name, [package_name + '/' + package_name + '_params.yaml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dotX Automation s.r.l.',
    maintainer_email='info@dotxautomation.com',
    description='Interactive and offline calibration of monocular and stereo cameras.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'dua_camera_calibration_app = dua_camera_calibration.dua_camera_calibration_app:main',
            'dua_camera_calibration_cli = dua_camera_calibration.dua_camera_calibration_cli:main',
            'synthetic_camera = dua_camera_calibration.synthetic_camera:main',
        ],
    },
)

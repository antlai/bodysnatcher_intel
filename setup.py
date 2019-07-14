from setuptools import setup

setup(name='bodysnatcher_intel',
      version='0.1',
      description='Locate body parts with depth info with Intel D415',
      url='http://github.com/antlai/bodysnatcher_intel',
      author='Antonio Lain',
      author_email='antlai@cafjs.com',
      license='Apache 2.0',
      packages=['bodysnatcher_intel'],
      install_requires=[
          'cherrypy',
          'numpy',
#          'opencv-python',  manually installed
#          'pyrealsense2', manually installed
#          'tensorflow' has to be manually installed
      ],
      zip_safe=False)

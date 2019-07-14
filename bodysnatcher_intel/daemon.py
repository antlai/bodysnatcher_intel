import cherrypy
import time
import os
import sys
import json
from calibrate import mainCalibrate
from calibrate import mainSnapshot
from segmentProcess import mainSegment

class BodySnatcher(object):
    @cherrypy.expose
    def index(self):
        return "Hello from BodySnatcher"

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def calibrate(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        return mainCalibrate(options)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def snapshot(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        snap =  mainSnapshot(options)
        return snap


    @cherrypy.expose
    @cherrypy.tools.json_out()
    def points3D(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        print options
        options = json.loads(options) if options != None else {}
        options['onlyPoints3D'] = True
        return mainCalibrate(options)

    @cherrypy.expose
    def parts(self, options = None):
        print options
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        if options and options.get('display'):
            if (os.getenv('DISPLAY') is None) or \
               (os.getenv('XAUTHORITY') is None):
                print 'WARNING: disabling display, missing properties'
                options['display'] =  False
        return mainSegment(options)
    parts._cp_config = {'response.stream': True}

    @cherrypy.expose
    def reset(self):
        sys.exit(0)


def run():
    script_dir = os.path.dirname(__file__)
    cherrypy.config.update(os.path.join(script_dir, "server.config"))
    cherrypy.quickstart(BodySnatcher())


if __name__ == '__main__':
    run()

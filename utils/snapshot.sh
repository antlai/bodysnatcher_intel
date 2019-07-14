#!/bin/bash
curl http://192.168.1.21:7090/snapshot | jq '.data' | sed s/\"//g | base64 -d -

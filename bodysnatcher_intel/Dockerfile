# VERSION 0.2
# DOCKER-VERSION  1.7.0
# AUTHOR:         Antonio Lain <antlai@cafjs.com>
# DESCRIPTION:    Python image for Jetson TX2
# TO_BUILD:       docker build --rm -t gcr.io/cafjs-k8/root-bodysnatcherintel .
#
# TO_RUN PRIVILEGED: docker run -d --name=root-bodysnatcherintel --restart=always -p 7090:7090 -v /usr/local/cuda-9.0/lib64:/usr/local/cuda -v /usr/local/lib/realsense:/usr/local/lib/realsense -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/infer --privileged  gcr.io/cafjs-k8/root-bodysnatcherintel
# or with X11 see utils/x11setup.txt


FROM gcr.io/cafjs-k8/root-jetsonpython

ENTRYPOINT ["python"]

CMD [ "./rundaemon.py" ]

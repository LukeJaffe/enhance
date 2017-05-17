#!/bin/bash
gpudocker run -i lj:tf -c /bin/bash \
--docker_args="-it --net=host -v `pwd`/work:/project/work -w /project/work"


# 更改临时存储目录
```
 export SINGULARITY_TMPDIR=~/tmp
 singularity build  nginx.simg   docker://nginx:1.14
```
# 创建基础镜像，从cr文件文件基础镜像
cr文件内容
```
Bootstrap: docker
From: ubuntu:22.04
%post

    apt-get -y update

    apt-get -y install fortune cowsay lolcat
```

    
`singularity build  ubuntu-22.04.sif   cr`


 # 从基础镜像构建，无需docker权限,定义cr文件从基础镜像构建
 ```
 bootstrap: localimage
 From: ubuntu-22.04.sif
 ```
 
singularity build --fakeroot ubuntu-22.04_new.sif cr
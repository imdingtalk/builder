
# 更改临时存储目录
构建是默认使用/tmp目录作为临时目录，该目录在系统/分区下，系统/分区过小可能导致build失败，可以将临时存储目录设置为自己的目录，这样就不会影响系统/分区的大小，同时也可以将临时存储目录的内容持久化到用户目录下，这样下次启动容器时，临时存储目录也会是该目录
```
# 普通用户
# ~/tmp 设置为你自己的临时存储目录
 export SINGULARITY_TMPDIR=~/tmp
 sudo singularity build  nginx.sif   docker://nginx:1.14
# 此时临时存储目录则为上述目录，可以将配置持久化到 bashrc 中，这样每次启动容器时，临时存储目录也会是该目录
echo "export SINGULARITY_TMPDIR=~/tmp" >> ~/.bashrc


# 全局生效
# 注意/data/tmp要改为系统上一个较大的目录
sudo echo "export SINGULARITY_TMPDIR=/data/tmp" >> /etc/profile
# 重新加载环境变量
source /etc/profile
# 查看临时存储目录
echo $SINGULARITY_TMPDIR
# 查看临时存储目录的大小
du -sh /data/tmp
# 查看临时存储目录的内容
ls /data/tmp
# 查看临时存储目录的权限
```

# 创建基础镜像，从cr文件文件基础镜像，或者其他你想用的方式
场景： 如将转换好的大docker镜像，生成一个可供所有用户直接使用的singurity容器，而不需要授予特殊权限

cr文件内容,这里也涉及了后续容器后如何进行修改（如安装新软件和调整环境配置）,
具体参考： https://docs.sylabs.io/guides/2.6/user-guide/build_a_container.html#building-containers-from-singularity-recipe-files
cr文件内容如下，更多自定义信息参考官方文件
```
Bootstrap: docker
From: ubuntu:22.04
%post

    apt-get -y update

    apt-get -y install fortune cowsay lolcat
```

# 根据以上文件构建一个基础镜像    
`sudo singularity build  ubuntu-22.04.sif   cr`

生成一个如ubuntu-22.04.sif 基础镜像是存放在本地的，这个可以作为一个文件，给文件相应的权限（如744），或者更改文件的属组信息，分发给普通用户，普通用户后续使用该基础镜像构建新的容器，不需要docker权限

以下操作只需要普通用户执行即可，用户目录下放基础镜像文件，如前序步骤管理员构建出来的基础镜像文件为ubuntu-22.04.sif
后面普通用户基于这个基础镜像构建新的容器
 # 从基础镜像构建，无需docker权限,定义cr文件从基础镜像构建
 cr文件内容
 ```
 bootstrap: localimage
 From: ubuntu-22.04.sif
 ```
 普通用户执行构建新的镜像
`singularity build --fakeroot ubuntu-22.04_new.sif cr`




# 精简singularity镜像
在 Singularity 中，所有的内容最终都会被打包到一个单一的 .sif 文件中，并不会像 Docker 那样生成多层文件系统
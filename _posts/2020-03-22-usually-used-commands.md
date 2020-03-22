---
title: usually used commands
description: 
categories: scientists
---




```
  # https://github.com/docker/for-win/issues/2151

  # how to use Docker with WSL
  # https://nickjanetakis.com/blog/setting-up-docker-for-windows-and-wsl-to-work-flawlessly

  # The last thing we need to do is set things up so that volume mounts work. This tripped me up for a while because check this out…

  # When using WSL, Docker for Windows expects you to supply your volume paths in a format that matches this: /c/Users/nick/dev/myapp.

  # But, WSL doesn’t work like that. Instead, it uses the /mnt/c/Users/nick/dev/myapp format. Honestly I think Docker should change their path to use /mnt/c because it’s more clear on what’s going on, but that’s a discussion for another time.

  docker run -d -it --name devtest2 -v /d/Dropbox/Dropbox/:/app ubuntu
```


# windows

## vim mode ahk

shift + esp # enable vim mode
i # leave the vim mode

## adobe reader dc

  // https://helpx.adobe.com/cn/acrobat/using/keyboard-shortcuts.html
E # text selection 
shift + u: underline, delete line, highlight
h # hand
ctrl + 2: fit to window width

## change python error message lanugage

In windows set [environment variable](http://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows-7/) `LC_MESSAGES` to `English`. And then restart the server.

 







## shadowsocks on ubuntu

### 安装

```bash
pip3 install https://github.com/shadowsocks/shadowsocks/archive/master.zip
  # or download https://github.com/shadowsocks/shadowsocks/archive/master.zip, unzip and 
python setup.py install 
vim ~/shadowsocks.json
```

shadowsocks.json内容如下

```
{
  "server":"88.88.88.88",
  "local_address": "127.0.0.1",
  "local_port":1080,
  "server_port":8843,
  "password":"pAsswOrD",
  "timeout":300,
  "method":"aes-256-cfb"
}
```

local_address与local_port为本地设置，其余根据服务器参数进行填写

```
sslocal -c ~/shadowsocks.json  ## ssloacl -h 可显示sslocal使用方法
```

显示如下信息，说明成功
INFO: loading config from /home/user/shadowsocks.json
2018-07-17 17:19:12 INFO loading libcrypto from libcrypto.so.1.1
2018-07-17 17:19:12 INFO starting local at 127.0.0.1:1080

### 设置代理

使用系统网络代理设置，报错如下
2018-07-17 18:01:50 WARNING unsupported SOCKS protocol version 4

具体怎么更改系统网络代理为SOCKS5，博主也不清楚，所以我使用的是firefox浏览器代理，修改方式如下

首选项——常规——网络代理 设置：

- 选择手动代理配置
- SOCKS 主机填写为 127.0.0.1
- 端口为1080
- 勾选SOCKS_v5.
- 不使用代理填写localhost, 127.0.0.1
- 勾选使用 SOCKS v5 时代理 DNS 查询
- 点击确定



## total commander

Shift+F6 # Rename files in same directory
Shift+F5 # Copy files (with rename) in same directory
F6 # Rename or move files





## docker install
---

```bash
  # https://yeasy.gitbooks.io/docker_practice/install/ubuntu.html

  curl -fsSL get.docker.com -o get-docker.sh
  sudo sh get-docker.sh --mirror Aliyun

  # https://blog.csdn.net/xiangxianghehe/article/details/80897769
  sudo gpasswd -a ${USER} docker
  sudo usermod -aG docker $USER
  sudo su
  su xling
  sudo apt-get install docker-compose
  docker-compose up -d

  # 启动 Docker CE
  sudo systemctl enable docker
  sudo /etc/init.d/docker start

  # 测试 Docker 是否安装正确
  docker run hello-world
```

### 设置国内源

#### windows

Docker For Windows

在桌面右下角状态栏中右键 docker 图标，修改在 Docker Daemon 标签页中的 json ，把下面的地址:

```
http://f1361db2.m.daocloud.io
```

加到" `registry-mirrors`"的数组里。点击 Apply 。

![1570898415564](D:\Dropbox\Dropbox\base\shared-among-computers\typora\1570898415564.png)



#### [linux]( https://yeasy.gitbooks.io/docker_practice/install/mirror.html ) 

## install cuda @ windows

https://developer.download.nvidia.cn/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_426.00_win10.exe



Errors

![1572106951863](D:\Dropbox\Dropbox\base\shared-among-computers\typora\1572106951863.png)

## external tools

### virtualbox

#### reinstall ubuntu in virtualbox
[](~/base/shared-among-computers/documents/scripts/linux/reinstall_ubuntu.md)

##### v3.19.0-25 x64 of ubuntu has bugs
1. You can install x64 ubuntu on i386 PC. 
2. Seems that v3.19.0-25 x64 of ubuntu has bugs. I get error for lots of apt commands. [](https://bugs.launchpad.net/ubuntu/+source/linux/+bug/1479945)
    Error! Bad return status for module build on kernel: 3.19.0-25-generic (x86_64)

##### move virtual machine 
[](http://superuser.com/questions/633431/whats-the-recommended-way-to-move-a-virtualbox-vm-to-another-computer)
I have tried method 1 and method 2. Seems that once you create snapshots, the snapshots of the clone point to the old path and I dont know how to change.
Maybe reinstall the OS is faster.

###### method 1: use VB "export" feature (meet problems I cannot solve)
Generate .ovf file in VB manager.
Import the .ovf file in VB manager.

###### method 2: Manually copy the old vm folder to a new location and edit VirtualBox.xml (meet problems I cannot solve) [](https://forums.virtualbox.org/viewtopic.php?f=6&t=49624) 

#### reduce snapshot size
simply remove each snapshot (ctr+shift+d)

#### vboxmanage always fails for commands like controlvm
  Solution: remove and reinstall virtualbox

#### Make Windows vim use the vimrc of virtualbox

##### enable Windows to access VB''s folder using samba <#enable_windows_to_visit_vb>

0. Add an Host-Only adopter

You need two adapters:
* The first, NAT + forwarding, enables you to login the vm using putty.
  虚拟系统借助NAT(网络地址转换)功能，通过宿主机器所在的网络来访问公网。也就是说，使用NAT模式可以实现在虚拟系统里访问互联网。
  NAT模式下的虚拟系统的TCP/IP配置信息是由(NAT)虚拟网络的DHCP服务器提供的，无法进行手工修改，因此虚拟系统也就无法和本局域网中的其他真实主机进行通讯。
  采用NAT模式最大的优势是虚拟系统接入互联网非常简单，你不需要进行任何其他的配置，只需要宿主机器能访问互联网即可。
* The second, host-only, enable you to access vm's samba.
  Virtualbox在宿主机中模拟出一张专供虚拟机使用的网卡，所有虚拟机都是连接到该网卡上的，虚拟机可以通过该网卡IP访问宿主机，同时Virtualbox提供一个DHCP服务，虚拟机可以获得一个内部网IP，宿主机可以通过该IP访问虚拟机。如果单纯使用Host-only模式，则虚拟机不能连接外部公共网络。
  When you created this adapter, you get the follows when type ipconfig in windows cmd.

    Ethernet adapter VirtualBox Host-Only Network:
     Connection-specific DNS Suffix  . : 
     Link-local IPv6 Address . . . . . : fe80::6dd3:4cd2:60dd:d423%20
     IPv4 Address. . . . . . . . . . . : 192.168.56.1
     Subnet Mask . . . . . . . . . . . : 255.255.255.0
     Default Gateway . . . . . . . . . : 

  [](http://askubuntu.com/questions/281466/samba-how-can-i-access-a-share-on-a-virtualbox-guest-in-nat-mode)
  [http://blog.csdn.net/watkinsong/article/details/8878786]
  [http://blog.csdn.net/i_chips/article/details/19191957]
  [http://www.cnblogs.com/leezhxing/p/4482659.html]


1. Install samba 
    sudo apt-get -y install samba
    sudo apt-get -y install kdenetwork-filesharing
2. Configure /etc/samba/smb.conf
    [public]
    path = /
    browseable = yes
    writable = yes
    read only = no
    create mask = 0644
    directory mask = 0755
3. Create a new user and add the user to samba 
sudo smbpasswd -a xling

4. Restart samba
    sudo service smbd restart

5. Get the ip of this vm.
    xling@xling-VirtualBox:~$ ifconfig
    eth0      Link encap:Ethernet  HWaddr 08:00:27:26:b9:d2
            inet addr:10.0.2.15  Bcast:10.0.2.255  Mask:255.255.255.0
            inet6 addr: fe80::a00:27ff:fe26:b9d2/64 Scope:Link
            UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
            RX packets:9520 errors:0 dropped:0 overruns:0 frame:0
            TX packets:7935 errors:0 dropped:0 overruns:0 carrier:0
            collisions:0 txqueuelen:1000
            RX bytes:741066 (741.0 KB)  TX bytes:5696079 (5.6 MB)

  eth1      Link encap:Ethernet  HWaddr 08:00:27:f0:9a:99
            inet addr:192.168.56.100  Bcast:192.168.56.255  Mask:255.255.255.0
            inet6 addr: fe80::a00:27ff:fef0:9a99/64 Scope:Link
            UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
            RX packets:1555 errors:0 dropped:0 overruns:0 frame:0
            TX packets:1338 errors:0 dropped:0 overruns:0 carrier:0
            collisions:0 txqueuelen:1000
            RX bytes:255026 (255.0 KB)  TX bytes:693615 (693.6 KB)

  There are two eths that are corresponding to two adpaters (I am not sure). 
  192.168.56.100 is the vm's IP.

* cannot access samba?
background: 'sudo service smbd status' shows that samba is running. 
solution: sudo service smbd restart

6. create a driver mapping in windows
Z:\ -> \\192.168.56.100\public

##### Make gvim of windows load .vimrc of virtualbox
  [http://stackoverflow.com/questions/7109667/change-default-location-of-vimrc]
  [http://superuser.com/questions/361816/pass-command-line-arguments-to-windows-open-with].
    Edit regedit, change HKEY_CLASSES_ROOT\Applications\gvim.exe\shell\open\command from
      "D:\Program Files\Dropbox\base\usr\bin\Vim\vim74\gvim.exe" "%1"
    to  
      "D:\Program Files\Dropbox\base\usr\bin\Vim\vim74\gvim.exe" -u Z:\usr\share\vim\vimrc "%1"

##### login with putty

###### introduction
Install ssh server on the vm [](https://help.ubuntu.com/lts/serverguide/openssh-server.html)
  sudo apt-get install openssh-server

Setup the networking of the VM.
You can visit vm by setting vm's networking be NAT, host-only and bridge.
            Has unique ip?      Can access external network?      How to access putty
NAT         No. Need to use     Yes                               Port forwarding
Host only   Yes.                No                                ssh + VM's IP (192....)
Bridge      Yes.                YES                               ssh + VM's IP (192....)

[https://machinelearning1.wordpress.com/2013/02/02/connect-from-windows-to-ubuntu-on-virtualbox-using-putty-through-ssh/]
* Host-Only: 
  The VM will be assigned one IP, but it's only accessible by the box VM is running on. No other computers can access it.
  It will not have external access, but it will have access to the host any other virtual machines set to the same adapter mode
* Bridge: your VM has its own LAN IP. Your Windowns can directly communicate with this VM.
* NAT: the virtualbox has a LAN IP and assign IPs to VMs. Your Windows talks to virtualbox and then to VM. [www.virtualizationadmin.com/blogs/lowe/news/nat-vs-bridged-network-a-simple-diagram-178.html][http://superuser.com/questions/227505/what-is-the-difference-between-nat-bridged-host-only-networking]
  

###### Method 1 (NAT) [http://blog.csdn.net/ghyg525/article/details/18664725]

####### 2. Virtualbux -> setting -> network -> adaptor1 -> NAT
: ifconfig
    Link encap:Ethernet  HWaddr 08:00:27:c2:0d:9a
    inet addr:10.0.2.15  Bcast:10.0.2.255  Mask:255.255.255.0
    inet6 addr: fe80::a00:27ff:fec2:d9a/64 Scope:Link
    UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
    RX packets:6326 errors:0 dropped:0 overruns:0 frame:0
    TX packets:4468 errors:0 dropped:0 overruns:0 carrier:0
    collisions:0 txqueuelen:1000
    RX bytes:724249 (724.2 KB)  TX bytes:750425 (750.4 KB)

    Link encap:Local Loopback
    inet addr:127.0.0.1  Mask:255.0.0.0
    inet6 addr: ::1/128 Scope:Host
    UP LOOPBACK RUNNING  MTU:65536  Metric:1
    RX packets:2002 errors:0 dropped:0 overruns:0 frame:0
    TX packets:2002 errors:0 dropped:0 overruns:0 carrier:0
    collisions:0 txqueuelen:0
    RX bytes:9621544 (9.6 MB)  TX bytes:9621544 (9.6 MB)

  此时，相当于把虚拟机的22端口映射到本地的12100端口上，本地的Xshell就可以连接主机的12100端口来操作虚拟机了。
  I indeed do not understand the meaning of these settings well
  Maybe this is how things go
    # putty accesses 127.0.0.1:1111
    # VB intercepts this connection, forwards it to 10.0.2.15:22 ( 10.0.2.15 is the IP of the VM)

####### 3. Set forwarding rull (setting -> network -> adaptor1 -> NAT -> port forwarding)
      Host IP: 127.0.0.1 (can be empty)
      Host Port: 11111 
      Guest IP: 10.0.2.15 (can be empty)
      Guest Port: 22 

####### 4. Login with putty
      Host name: 127.0.0.1
      Host part: 11111

###### method 2 (host only)
Set the networking of the guest vm to be host-only.
The VM gets an IP 192.168.1.102. 
Use putty to connect to this ip.

[http://www.cnblogs.com/leezhxing/p/4482659.html]
nat模式下物理机是不能发现虚拟机的存在的，需要添加一块host only网卡来实现互访。

Con: cannot access external network [This will create a private virtual network on the host. It will not have external access, but it will have access to the host any other virtual machines set to the same adapter mode.](http://panosgeorgiadis.com/ive-installed-kali-linux-so-whats-next)

* If you try to access kali, extra configuration is needed [](http://www.drchaos.com/enable-ssh-on-kali-linux/)

###### Method 3 (Bridge)
```
  I need to give a static IP to VM, so that I do not need to update the IP
  address in putty each time virtual box is restarted. However, it seems
  that Bridge cannot give static IPs to VMs
  [http://askubuntu.com/questions/419327/how-can-i-make-virtualbox-guests-share-the-hosts-vpn-connection].
```

####### 2. Virtualbux -> setting -> network -> adaptor1 -> bridge ...
####### 3. Login with putty

###### putty Q: putty automatically disconnect with error "PuTTY Network Error: Software caused connection abort"

1. use kitty to automatically connect after disconnection
con: still need to reconnect

2. "enable tcp keepalives"
does not work.

3. disable server firewall
does not work

4. echo "ClientAliveInterval 60" | sudo tee -a /etc/ssh/sshd_config
does not work.

5. 
#### vim, python support is not enabled
Error message: UltiSnips requires py >= 2.6 or any py3

Cause: python is 64 bit. Use 32 bit python instead.

Install vim-nox instead of vim
  http://askubuntu.com/questions/764882/ubuntu-16-04-vim-without-python-support
    sudo apt install vim-nox-py2

##### Suddenly cannot access the network any more?
Disable NAT, host-only and reinstall again.

#### backup

##### Need to backup virtualbox vm on Windows.
  1. Do not need to shutdown the VM

    virtualbox snapshot
      VBoxManage snapshot "vm-name" take 

  2. need to shutdown VM

    Toucan [http://www.iplaysoft.com/toucan.html]
      Pro: diff backup, 
    Dropbox.
      Con: 
        Dropbox keeps writting hard disk and makes Windows slow. 
    rdiff-backup
      con:
        windows version is very old. 
        In windows, cannot backup files with long names.
        On windows, annot backup files whose names contain blank space.
    virtualbox clone
      VBoxManage controlvm "vm-name" poweroff
      VBoxManage clonevdi "D:\Program Files\Dropbox\base\virtualbox\ubuntu\ubuntu.vdi" "D:\Program Files\Dropbox\base\virtualbox_backup_to_windows\ubuntu14"

##### Create a backup script 'backup-virtualbox.py'
Create a windows task scheduler:
  schtasks /Create /it /tn "take_virtualbox_vm_snapshot" /sc DAILY /st 23:00 /tr "\"D:\Program Files\Python27\python.exe\" \"D:\Program Files\Dropbox\base\documents\scripts\backup\backup-virtualbox.py\" -b"
    # check [http://technet.microsoft.com/en-us/library/cc772785(v=ws.10).aspx] for command manual.

##### Dropbox cannot backup specific files. 
Need to give the snapshots files a separate folder. However, if a vm has created snapshots, vbox cannot change the snapshot folder any more.
  Clone ubuntu at vbox to ubuntu1404. These three files/folder, ubuntu1404.vbox, ubuntu1404.vdi and Logs, must be in the same directory. You cannot separate them.
  Give a separate folder snapshot_path for the snapshots of ubuntu1404. 
  Take snapshots for ubuntu1404. 
  Modify backup-virtualbox script for ubuntu1404.
  Enable dropbox to backup snapshot_path.   

  Remove ubuntu 

##### Remove virtualbox snapshots: start from the old snapshots

#### start virtualbox vm with windows:
  schtasks /Create /it /tn "start_virtualbox_vm_with_windows" /sc onstart /tr "\"D:\Program Files\Python27\python.exe\" \"D:\Program Files\Dropbox\base\documents\scripts\backup\backup-virtualbox.py\" -s"

#### cannot create symbolic links in a windows drive mounted in virtualbox.
Solution: http://michaelfranzl.com/2014/02/01/symlinks-within-shared-folders-virtualbox-operation-permitted-read-filesystem/

#### mount

##### umount
sudo umount /your/shared/folder

### programming

#### eclipse


##### c++
Problem: Symbol 'iostream' could not be resolved, Symbol 'cout' could not be resolved.
Solution: add mingw include directory to eclipse
  [http://stackoverflow.com/questions/10373788/c-unresolved-inclusion-iostream]
  [http://stackoverflow.com/questions/10803685/eclipse-cdt-symbol-cout-could-not-be-resolved]

However, still cannot build
  Description Resource  Path  Location  Type
  recipe for target 'src/test.o' failed subdir.mk /test2/Debug/src  line 18 C/C++ Problem

Cannot solve.


###### coding


####### exceptional c++ [http://m.blog.csdn.net/blog/u011444931/18816327]
######## 条款4：可扩充的模板：使用继承还是traits？ 难度7

如何检测某个模板参数是否具有某个函数或者继承自某个类？

几个方法的本质就是采用转换，比如说在析构函数中转换函数指针或类指针。（因为这一块我觉得用得不多，仅仅是让自己的代码检查更严格而已，故不用太仔细研究）。

知道模板的参数类型T派生于某个其它类型，这对模板来说有什么用处呢？知道这种派生关系能带来某种好处吗？而且，这种好处在没有继承关系的情况下就无法获得吗？

对于一个模板来说，就算知道它的一个模板参数从某个给定的基类继承，这也不能让它获得“使用traits无法获得”的额外好处。使用traits仅有的一个真正的缺点是，在一个庞大的继承体系中，为了处理大量的类，需要写大量的特殊化代码；不过，运用某些技术可以减轻或者消除这一缺点。

本条款的主要目的在于说明：与某些人的想法相反，“为了处理模板中的分类而使用继承”不足以成为使用继承的理由。traits提供了更通用的机制；当用一个新类型-例如来自第三方程序库中的某个类型-来实例化一个现有模板的时候，此类型可能很难从某个预先确定的基类派生，此时，traits体现了更强的可扩充性。

####### Effective c++

######## ITEM M4： 避免无用的缺省构造函数
If you do not have a default ctor, you cannot create an object array of this class. You can create a pointer array instead. 
(The name of this ITEM is misleading)

######## ITEM M5：谨慎定义类型转换函数
Simply do avoid using default conversion. 
Use functions with good name to convert.

######## ITEM M18：分期摊还期望的计算
ache previous results to save computing. 
http://www.weixueyuan.net/html/3160.html

######## ITEM M25：将构造函数和非成员函数虚拟化
Class A has a container that contains pointers of Class B.
A's copy ctor receives an object a.
The ctor needs to clone each element in a.container.
Elements in a.container may indeed be instances of derived classes of B. So, the ctor cannot simply new B(). 
You should give B virtual method clone().
[http://lixinzhang.github.io/book/c++/techniques,_idioms,_patterns.html]

######## ITEM M28：灵巧（SMART）指针
When pass auto_ptr as arguments, passing by value cause double-free. 
Should pass by reference instead of by value.

auto_ptr is replaced by unique_ptr.
Should use unque_ptr + passing by reference.

###### buiding

#### shell programming

##### powercmd:
Only CMD can find execute files under system folders (windows, system32). 
Other consoles (powercmd, console2) cannot. You need to move the exe file into other folders (d:\program ....)

##### alternative to windows cmd
  Problems of cmd: 
    Need mouse to copy/paste
    Cannot keep history.
  Alternative:
    PowerCmd
    git bash

#### visual studio 

##### free install 
visual studio 2013 community is free (need to sign in (free))

##### use gtest

Build gtest
  Download gtest
    http://code.google.com/p/googletest/downloads/list
  Extract. 
  Enter msvc.
  Double click gtest-md.sln. 
    (There is gtest.sln. This project generate libs in single thread mode which are not very useful.)

  build gtest.lib, gtest_main-md.lib, gtestd.lib, gtest_main-mdd.lib. 
    ('d' means debug. Later, your code use these libs. When debug (reliease) build, the compiler links gtestd.lib (gtest.lib). )

    Build gtest and gtest_main in debug mode (generate gtestd.lib and gtest_main-mdd.lib) 
      (clg) The building may failed.
      1>C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\Microsoft.CppBuild.targets(1361,5): warning MSB8012: TargetPath(D:\Program Files\gtest-1.7.0\msvc\gtest\Debug\gtest.lib) does not match the Library's OutputFile property value (D:\Program Files\gtest-1.7.0\msvc\gtest\Debug\gtestd.lib). This may cause your project to build incorrectly. To correct this, please make sure that $(OutDir), $(TargetName) and $(TargetExt) property values match the value specified in %(Lib.OutputFile).
      1>C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\Microsoft.CppBuild.targets(1363,5): warning MSB8012: TargetName(gtest) does not match the Library's OutputFile property value (gtestd). This may cause your project to build incorrectly. To correct this, please make sure that $(OutDir), $(TargetName) and $(TargetExt) property values match the value specified in %(Lib.OutputFile).
    
      (sln) 
        Right-click the gtest project -> Properties -> Configuration properties -> General.
        Change 
          $(SolutionName)/$(Configuration)\ 
        to
          $(SolutionName)\$(Configuration)\
    
        [](http://stackoverflow.com/questions/4650993/how-to-set-outdir-targetname-targetext-and-lib-outputfile-with-vi)
        Change 
          $(ProjectName)
        to
          $(ProjectName)d
    
    Build gtest and gtest_main in release mode (generate gtest.lib and gtest_main.lib). 

  Move gtest.lib, gtest_main-md.lib, gtestd.lib, gtest_main-mdd.lib to D:\Program Files\gtest-1.7.0\lib (create if not exists)

Use gtest
  Add include directory of gtest
    Right-click the gtest project -> Properties -> Configuration properties -> c/c++ -> General -> additional include directories.
      D:\Program Files\gtest-1.7.0\include;

  Add libs
    Right-click the gtest project -> Properties -> Configuration properties -> link -> General -> additional include directories.
      add "D:\Program Files\gtest-1.7.0\lib"

    Right-click the gtest project -> Properties -> Configuration properties -> link -> input -> additional include directories.
      add "gtestd.lib;gtest_main-mdd.lib;"

##### use boost

Install boost: http://blog.sina.com.cn/s/blog_6e0693f70100txlg.html
Set include path: 
  Project -> Boost_Test Properties -> Configuration Properties -> C/C++ -> General -> Additional Include Directories
  C:\Users\leonexu\Downloads\boost_1_58_0\boost_1_58_0\
Set lib path:
  Project -> Boost_Test Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories
  C:\Users\leonexu\Downloads\boost_1_58_0\boost_1_58_0\libs

#### coding
##### use vim to edit vba script and using snippets.
I hope to use code snippets for bas files in vim.
In short, it is impossible.

Macro scripts in MS Office are of .bas format - different from the vbs (VB script) format.
I made ultisnips to recognize support.
Two problems:
  vim does not understand bas and cannot auto indent. 
  vim cannot auto complete.

You can basically use Office's script editor to edit so that you get autocomplete. 
Use VbsEdit to indent. 
  This is not paid software. You can try it but I dont know when it expires.
Use vim to create/insert snips.
  TextEdit Anywhere can read buffer into a temperory file to edit. However, the file does not have an extension and hence ultisnips wont work.

However, office's vba editor read all content from the pptm file instead of an independent script file. 
Your update in vim cannot be reflected to the content in the vba editor.

#### debug python
  Need to debug a python script.
  Google says that I can use gdb [](https://wiki.python.org/moin/DebuggingWithGdb)
    xling@nodez:~/base/download/padc_repo\$ gdb --args python waf
  Basically, this method works for simple test code, but does not work for my code.

  The results:
    GNU gdb (Ubuntu 7.7.1-0ubuntu5~14.04.2) 7.7.1
    Copyright (C) 2014 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.
    For help, type "help".
    Type "apropos word" to search for commands related to "word"...

    warning: stl-views.gdb: No such file or directory
    Reading symbols from python...Reading symbols from /usr/lib/debug//usr/bin/python2.7...done.
    done.
    (gdb) catch throw
    Catchpoint 1 (throw)
    (gdb) r
    Starting program: /usr/bin/python waf
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
      File "waf", line 163, in <module>
        Scripting.waf_entry_point(cwd, VERSION, wafdir)
      File "/home4/xling/base/download/padc_repo/.waf-1.7.9-786a25f4411038005f1f2ec0d121c503/waflib/Scripting.py", line 90, in waf_entry_point
        set_main_module(Context.run_dir+os.sep+Context.WSCRIPT_FILE)
      File "/home4/xling/base/download/padc_repo/.waf-1.7.9-786a25f4411038005f1f2ec0d121c503/waflib/Scripting.py", line 115, in set_main_module
        Context.g_module=Context.load_module(file_path)
      File "/home4/xling/base/download/padc_repo/.waf-1.7.9-786a25f4411038005f1f2ec0d121c503/waflib/Context.py", line 279, in load_module
        raise Errors.WafError('Could not read the file %r'%path)
    
    $ $
    Could not read the file '/home4/xling/base/download/wscript'
    [Inferior 1 (process 25045) exited with code 01]
    (gdb) bt
    No stack.

  The program crash without leaving stack.
  I tried to add 'catch throw' but does not work.


#### test automation

##### test tool comparison

 |                    | support IE? | recorder in IE? | language                     | document                |
 |--------------------+-------------+-----------------+------------------------------+-------------------------|
 | selenium           | y           | N               | python                       | [5]                     |
 | SWAT [6]           | y           | Y. [2]          | self-defined script language | not complete            |
 | excel              | Y           | ?               | VBA                          |                         |
 | WatiN TestRecorder | y           | Y. [1]          | 'c#'                         | Need .net framework     |
 | imacros            | Y           | Y               | imacros script               | [3]                     |
 | others [4]         |             |                 |                              |                         |
 | winautomation      | Y           | y               |                              | [7]                     |
 | sikuli             |             |                 |                              | does not work with JDK7 |

 [1] Crash even I open a empty IE page. Maybe my installation has problem.
 [2] However, cannot deal with pop up window
 [3] Good. Handle popup window well. Can replay a arbitary piece of code.
 Not free for IE, but free for Firefox.
 [4] http://alternativeto.net/software/imacros-web-testing/
 [5] selenium: does not have a recoder on IE (focuse on FireFox)
 [6] Simple Web Automation Toolkit ie (SWAT)
 [7] Crash on popup windows.

selenium
  Has log info when error occurs.

##### GS script must be named as test.user.js instead of test.js

##### Greasemonkey

###### When to use Greasemonkey instead of selenium
Problem:
  Selenium scripts open page P1 and then P2, ... Pn one by one.
  Sometime, P1 is very complicated. 
  Hence,  I hope to manually open P1 and then let the script runs P2, ..., Pn. (G1) 
  However, when P1 contains some authentication process, the script cannot directly open P2 without passing P1. 

Solution:
  Greasemonkey can realize G1. 
  A Greasemonkey script is a Firefox plugin, not an auto test script. It is a part of the page and hence can easily read the authentication data.

###### GM script are writen in javascript

###### debug
1. alart()
2. console.log(). 

###### element selection

####### querySelector grammar (used in Greasemonkey)
[](http://www.w3.org/TR/css3-selectors/#content-selectors)
[](http://hakuhin.jp/js/selector.html)
(https://developer.mozilla.org/ja/docs/Web/API/Element/querySelectorAll)
(https://www.softel.co.jp/blogs/tech/archives/2085)
document.querySelector('#sample') // → id="sample"の要素を取得
document.querySelectorAll('input') // → input要素を全部取得
document.querySelectorAll("input[type='checkbox']") // → チェックボックスを全部取得
document.querySelectorAll('#xyz > .abc') // → id="xyz" 直下の class="abc" を取得

When name include '.', querySelector may fail. For example, the following does work (it works for someones but not for me)
  document.querySelector("#taishaJikokuKairiRiyuKbn\\.selected")
You need to replace it with full css pass. 
  "html body form div#container.ContentFrame div#content div#blankdeleteDiv div.FormCtrlArea.clearfix table#dataTable201.dataTable.ResultTable.ChgBorder tbody tr.DT td.DT_C select.TypeList"

It seems that firebug's css selectors cannot differentiate arraw items. Suppose that you want to select the 10-th element of an arraw A. Firebug's css always returns A[0]. The solution is to use :nth-child [](http://www.w3.org/TR/css3-selectors/#nth-child-pseudo)
  "html body form div#container.ContentFrame div#content div#blankdeleteDiv div.FormCtrlArea.clearfix table#dataTable201.dataTable.ResultTable.ChgBorder tbody tr.DT:nth-child(10) td.DT_C select.TypeList"
Firebug does not shorten css path. You can manually shorten it. 
  "#dataTable201.dataTable.ResultTable.ChgBorder tbody tr.DT:nth-child(15) td.DT_C select.TypeList"

####### css selector (used in selenium)
[](http://www.w3schools.com/cssref/css_selectors.asp), (http://qiita.com/okitan/items/cdf8809405821e057609)

other css selector:
  div.abc: equal to div[class='abc']
  div#abc: equal to div[id='abc']
  div a: matching <div><a></a></div> (In xpath: //div//a)
  div > a: matching <div>... <a></a> ... </div>  (In xpath: div//a)
  p: all <p> under the current element. E.g., ps = body.find_elements_by_css_selector('p')

####### xpath

jQueue, querySelector, xPath, .... [](http://stackoverflow.com/questions/12375008/how-can-my-userscript-get-a-link-based-on-the-links-text)

######## select an element expect certain child elements

######### exclude a element with certain attribute
<root xmlns:foo="http://www.foo.org/" xmlns:bar="http://www.bar.org">
	<actors>
		<actor id="1">Christian Bale</actor>
		<actor id="2">Liam Neeson</actor>
		<actor id="3">Michael Caine</actor>
	</actors>
	<foo:singers>
		<foo:singer id="4">Tom Waits</foo:singer>
		<foo:singer id="5">B.B. King</foo:singer>
		<foo:singer id="6">Ray Charles</foo:singer>
	</foo:singers>
</root>

Goal: exclude by tag
xpath: /root/*[not(actor)]
return: 
	<foo:singers>
		<foo:singer id="4">Tom Waits</foo:singer>
		<foo:singer id="5">B.B. King</foo:singer>
		<foo:singer id="6">Ray Charles</foo:singer>
	</foo:singers>

---------------------

<root xmlns:foo="http://www.foo.org/" xmlns:bar="http://www.bar.org">
	<actors>
		<actor class="abc">Christian Bale</actor>
		<actor id="2">Liam Neeson</actor>
		<actor id="3">Michael Caine</actor>
	</actors>
	<foo:singers>
		<foo:singer id="4">Tom Waits</foo:singer>
		<foo:singer id="5">B.B. King</foo:singer>
		<foo:singer id="6">Ray Charles</foo:singer>
	</foo:singers>
</root>

Get all elements that have "class=abc"
xpath: //*[@class="abc"]
return: <actor class="abc">Christian Bale</actor>

-----------------
<root xmlns:foo="http://www.foo.org/" xmlns:bar="http://www.bar.org">
	<actors>
		<actor class="abc">Christian Bale</actor>
		<actor id="2">Liam Neeson</actor>
		<actor id="3">Michael Caine</actor>
	</actors>
	<foo:singers>
		<foo:singer id="4">Tom Waits</foo:singer>
		<foo:singer id="5">B.B. King</foo:singer>
		<foo:singer id="6">Ray Charles</foo:singer>
	</foo:singers>
</root>

Goal: return all elements except <actor class="abc">Christian Bale</actor>

xpath: /root/*[@class!="abc"]
return: null
cause: [@class!="abc"] will select only node that have 'class' attribute and exclude these that have 'class=abc'. Only <actor class="abc">Christian Bale</actor> has 'class' attribute and is excluded. Hence, none is returned. [http://stackoverflow.com/questions/11113232/xpath-to-exclude-elements-that-have-a-class]
Solution: use [not(@class="abc")] to represent either 'does not have class attribute' or 'has class attribute but not equal to abc'

xpath: /root/*[not(@class="abc")] 
Return: 
Element='<actors>
  <actor class="abc">Christian Bale</actor>
  <actor>Liam Neeson</actor>
  <actor class="abc2">Michael Caine</actor>
</actors>'
Element='<foo:singers xmlns:foo="http://www.foo.org/">
  <foo:singer id="4">Tom Waits</foo:singer>
  <foo:singer id="5">B.B. King</foo:singer>
  <foo:singer id="6">Ray Charles</foo:singer>
</foo:singers>'

Cause: /root/* return direct child nodes under /root - namely, <actors> and <foo:singlers>. Both of them do not contain class attribute and hence both be returned unchanged.

path: /root/*[not(@class="abc")] 


######### exclude a certain tag
<root xmlns:foo="http://www.foo.org/" xmlns:bar="http://www.bar.org">
  <actors>
    <actor id="1">Christian Bale</actor>
    <actor id="2">Liam Neeson</actor>
    <actor id="3">Michael Caine</actor>
  </actors>
  <foo:singers>
    <foo:singer id="4">Tom Waits</foo:singer>
    <foo:singer id="5">B.B. King</foo:singer>
    <foo:singer id="6">Ray Charles</foo:singer>
  </foo:singers>
  <bob>
    <foo:singer id="4">Tom Waits</foo:singer>
    <foo:singer id="5">B.B. King</foo:singer>
    <foo:singer id="6">Ray Charles</foo:singer>
  </bob>
  <bob>
    <foo:singer id="4">Tom Waits</foo:singer>
    <foo:singer id="5">B.B. King</foo:singer>
    <foo:singer id="6">Ray Charles</foo:singer>
  </bob>
  <bob>
    <foo:singer id="4">Tom Waits</foo:singer>
    <foo:singer id="5">B.B. King</foo:singer>
    <foo:singer id="6">Ray Charles</foo:singer>
  </bob>
</root>

[](http://stackoverflow.com/questions/1068636/exclude-certain-elements-from-selection-in-xpath)
Goal: remove actors
xpath: /root/*[not(self::actors)]
Result: 
Element='<foo:singers xmlns:foo="http://www.foo.org/">
  <foo:singer id="4">Tom Waits</foo:singer>
  <foo:singer id="5">B.B. King</foo:singer>
  <foo:singer id="6">Ray Charles</foo:singer>
</foo:singers>'
Element='<bob>
  <foo:singer xmlns:foo="http://www.foo.org/" id="4">Tom Waits</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="5">B.B. King</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="6">Ray Charles</foo:singer>
</bob>'
Element='<bob>
  <foo:singer xmlns:foo="http://www.foo.org/" id="4">Tom Waits</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="5">B.B. King</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="6">Ray Charles</foo:singer>
</bob>'
Element='<bob>
  <foo:singer xmlns:foo="http://www.foo.org/" id="4">Tom Waits</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="5">B.B. King</foo:singer>
  <foo:singer xmlns:foo="http://www.foo.org/" id="6">Ray Charles</foo:singer>
</bob>'


However, it seems impossible to remove bob[1]
xpath: /root/*[not(self::bob[1])]
result: 
Element='<actors>
  <actor id="1">Christian Bale</actor>
  <actor id="2">Liam Neeson</actor>
  <actor id="3">Michael Caine</actor>
</actors>'
Element='<foo:singers xmlns:foo="http://www.foo.org/">
  <foo:singer id="4">Tom Waits</foo:singer>
  <foo:singer id="5">B.B. King</foo:singer>
  <foo:singer id="6">Ray Charles</foo:singer>
</foo:singers>'

##### imacros

Dont know why but fail to record some commands.

###### selenium

####### get xpath
In firefox, you can get xpath using firebug.
No such tools exist in ff.
You can use "developer tool" (F12), "select element by click" to manually find the element. 
You can also use bookmark.

####### Record in Firefox using selenium IDE
Install selenium IDE
  http://www.seleniumhq.org/download/#side_plugins
Dont know why but some time cannot find element.
  Possible reason 1:
    Some one say you should wait the element to show 
  Possible reason 2:
    seleminum IDE cannot correctly get the element id.
    You should better confirm the ID using firefox firebug.

###### Cannot get the dynamic id of pop up windows.
You may need to write script yourself instead of using the naive recorder.

##### Run IE pages (company page) in Firefox.
###### What you need to do is to override user-agent string.
To this end, you can use "UAControl" add-on.
Instruction

First, install UAControl (needs restart).
https://addons.mozilla.org/en-US/firefox/addon/uacontrol/

Uninstall firefox IE tab 
Then, open the preference dialog of UAControl (from Add-ons page, which can be opened by pressing Ctrl+Alt+a), and configure it as follows:
The user-agent value you have to input (Option -> Default for sites not listed):
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.04506.30; .NET CLR 3.0.04506.648)
Now you can access to companies sites with Firefox.
User scripts

###### Optionally, you may need to install use scripts developed by me.
Install scriptish,
https://addons.mozilla.org/ja/firefox/addon/scriptish/
Then go to my page http://fake.greenpf.cl.nec.co.jp/~masa/ and visit script files you want install.
For jinkin-sites, you should install http://fake.greenpf.cl.nec.co.jp/~masa/gm_scripts/kinrou_ff_workaround.user.js
This is how to intall http://wiki.greasespot.net/Greasemonkey_Manual:Installing_Scripts

### office


#### vim

##### Openning gvim takes too long time. 
  Open the file in console mode.
    Edit regdit, change
      /HKEY_CLASSES_ROOT/Applications/gvim.exe/shell/edit/command/default
      to 
      "D:\Program Files\ConEmuPack.141221\ConEmu.exe" "vim %1"
    Con: 
      Somehow, autocomplete does not work. 

##### gvim frequently crashes.

Solution: install python 32 instead of python 64. Then reinstall vim.

##### install ycm on windows

###### use neocomplcache on windows instead.

###### use install.py (does not work)
[](https://github.com/Valloric/YouCompleteMe#windows-installation)
install.py --clang-completer

####### Error:
  You have not specified which libclang to use.  You have several options:

     1. Set PATH_TO_LLVM_ROOT to a path to the root of a LLVM+Clang binary distribution. You can download such a binary distro from llvm.org. This is the recommended approach.
     2. Set USE_SYSTEM_LIBCLANG to ON; this makes YCM search for the system version of libclang.
     3. Set EXTERNAL_LIBCLANG_PATH to a path to whatever libclang.[so|dylib|dll] you wish to use.

Seems that cause is outdated tar command in windows

Try to compile without support for c-family languages
  python install.py

Error: 
   error: 'off64_t' does not name a type


###### manually compile
[](https://github.com/Valloric/YouCompleteMe/wiki/Windows-Installation-Guide)

Use vundle to download ycm.

If you installed VS2015:
  cmake -G "Visual Studio 14" -DPATH_TO_LLVM_ROOT="D:\Program Files (x86)\LLVM" . ..\third_party\ycmd\cpp
If you installed VS2013:
  cmake -G "Visual Studio 12" -DPATH_TO_LLVM_ROOT="D:\Program Files (x86)\LLVM" . ..\third_party\ycmd\cpp

Get error:
  Error	167	error C2027: use of undefined type 'boost::detail::shared_state_base'	D:\Program Files\Vim\vimfiles\bundle\YouCompleteMe\third_party\ycmd\cpp\BoostParts\libs\thread\src\win32\thread.cpp	64	1	BoostParts 

#### IE home page locked to hao.qq.com

Solution: Tencent "电脑管家实时防护" is the criminal. 


#### hope to give shortcut to my micros. Lots of articles on the network says there is such as setting but I cannot find in my office.
    Solution: seem that in office 2010, you cannot set shortcut.
    Instead, you can do things like alt + m + n.
    Most such shortcuts take three clicks.
    You can define 'quick access', which take two clicks. [](http://oshiete.goo.ne.jp/qa/6218097.html)

#### microsoft office

##### outlook 2010 

###### cannot type Japanese using Google JapanESe input.
  Solution: Use Text Editor Anywhere to input using vim.

###### outlook is full. Cannot send mails any more. 
I try to move mails to local directory. Outlook says that "You have no privilege to update the directory".
I created a new local directory ([Moving Messages to Local Storage]( http://agsci.psu.edu/it/how-to/move-seasonal-or-less-used-outlook-folder-to-local-storage)): control panel -> mail -> data files. Then, move mail from server to the newly created local directory.

###### 80070005. No permission to xxx
Cause: the pst file is not accessable. 
Solution: 
  Move the pst file from remote server to local server.
  (or) Restart the PC.

###### xxxx40825. Cannot access the server
Solution: restart outlook

###### Cannot read room booking status 

Solution: restart outlook

###### oultook cannot send mail. 
Error: 0x80070005
Solution: update address box [](https://rahuldpatel.wordpress.com/2014/06/03/this-message-could-not-be-sent-error-0x80070005-00000000-00000000-in-outlook-2010/)

###### Cannot view room booking condistion "空き時間情報を取得できませんでした"

####### cleanfreebusy => Does not work


##### excel

###### Open csv

Data -> text file 

###### You cannot organize tab in a tree manner.
You can only assign good names to tabs.

###### You can split the current sheet into multiple views.
But you cannot freely add views like vim. 
The number of view can be used is decided by excel.

###### highlight current row
[](http://www.extendoffice.com/product/kutools-for-excel.html)
This only work for one sheet. 
Need to update to apply to all sheets.

###### Use past link (リンク貼り付け) to avoid copy paste data.

###### Operate the same file in multiple windows.
View -> open in new window

###### (tbd) in vba, methods and property do not auto pop up.
Maybe IntelliSense is broken. 
Tried to reinstall/repair it but do not know how. 

###### label and reference a region
[](http://spreadsheets.about.com/od/lookupfunction1/ss/2012-06-01-excel-two-way-lookup-vlookup-part-1_3.htm)
Select the region. 
Type the name in the Named Box. 
  Do not use abc-123 (does not work). Use abc_123.
Press Enter key (Must!)

###### indent code
Some addon do this but may not secure. 
Use online solution: http://www.vbindent.com/?indent

###### excel: When click arrow keys, the sheet moves as a whole.
  Solution: Fn + NumLk

###### excel: Apply formula to a range of cells using Keyboard
  I have column A ranging from x to y. 
  I hope to create column B so that B[i]=2*A[i].
  Using mouse, you type =2*A1 in B1, double click the right bottom corner of B1. 

  I cannot find way to do the same using Keyboard only.
  The possible way:
    In B1, input =2*A1
    Press F5, input By.
    Press shift and then OK.
    ctr-D to apply

##### general 

###### clipart

When you search a clipart in office, office looks in bing but does not return full results.
You need to maintain you own clipart library
  Set the location of local clipart library: http://en.kioskea.net/faq/11400-word-2010-select-a-custom-location-for-your-clip-arts
  Find clipart 
    https://openclipart.org
    Bing: https://goo.gl/ylv1gJ (figures here are not well organized)

#### pdf

##### use mendelay to manage references
  When you add a paper into mendelay, mendelay can automatically copy the pdf paper into a folder
    Tools->Options->File Organizer

  You can also watch a certain folder
    Tools->Options->Watched Folders
  When a pdf is assed into the watched folder, mendelay automatically copy the pdf into the pdf database.

  Specifically, you can watch your download folder.


##### pdf to word
  https://www.pdftoword.com/ (Good conversion quality. Cannot access from company. May need to buy the software)


#### Google docs is not fully compatible with word. You may not be able to export Word files from Docs files.

### productivity

#### search

##### everything

###### Search results do not show any more
Solution: tool -> option -> indexes -> force rebuild

#### screenshot
screenshot captor
  Save shot to clipboard automatically 
    preference -> basic capting -> post capture options -> copy to clipboard -> image bitmap

#### screenshot does not work for Windows 7 enterprise version.
Restart OneNote [](http://superuser.com/questions/527367/why-is-the-windows-7-shortcut-for-screen-capture-not-working)


#### gesture control
Install strokeit.
Press moust right key, then make moust gestures.
When I use it for the first  time, did not work. After reinstallation, worked.

#### autohotkey

#### disable keys
disable the sleeping key [](http://superuser.com/questions/554431/how-do-i-disable-certain-keys-using-a-key-in-autohotkey)
  f7::return
##### mojibake/乱码 (AHK cannot short Japanese.)
Just save the script in ANSI instead of UTF8.

* People say that AutoHotkey_L can solve the problem (http://qiita.com/rohinomiya/items/b80707de5e8e0d0f840c). 
However, AutoHotkey_L has merged into AHK after v1.1 (http://rosettacode.org/wiki/AutoHotkey_1.1) and I am using v1.1.


#### windowtabs
Problem: 
  Too many windows opened (3 chromes, ie, firefox, putty, file explorer, ...). Too time consuming to swap windows.
Solution:
  windowtabs
  Once incstalled, you can trag multiple different windows and combine them into one in the same way chrome does. 
  Trag the title bar of window A to that of window B. When they are close, they are groupped togather.
Con:
  Too big. Risky to use in company
  Can combine at most three windows for free version.

#### virtual desktops
Many virtual desktop software exist. 
I need to be able to directly more an application between desktops.
Dexpot:
  right click icon. -> desktop windows -> trag applicant form list to the destination desktop.

virtual dimension
  Does not show at desktop sometime. Dont know why.

#### measure free disk size
windirstat

#### use linux tools
unxutils [](http://unxutils.sourceforge.net/)
coreutils [](http://gnuwin32.sourceforge.net/packages/coreutils.htm)
GetGnuWin32 [](http://getgnuwin32.sourceforge.net/)

#### cannot run convert.exe of ImageMagick and get error "access is denied"
Solution: change the convert.exe to any other name.
  There is a system file called convert.exe. Maybe Windows cannot differentiate the two files.

#### create ISO
LC ISOCreator (free and simple)

#### add tasks in task scheduler by commandline
Pro: quick (compared with using GUI)
Con: cannot add comment 

[](http://technet.microsoft.com/en-us/library/cc772785(v=ws.10).aspx)
/TN Task name. A unique name for the task
/ST Start time. Time when task should start.
/SC Schedule frequency. Valid types are: ONCE, MINUTE, HOURLY, DAYLY... 
  MINUTE, 1 - 1439, The task runs every n minutes.
  HOURLY, 1 - 23, The task runs every n hours.
  DAILY, 1 - 365, The task runs every n days.
  WEEKLY, 1 - 52, The task runs every n weeks.
  MONTHLY, 1 - 12, The task runs every n months.
/MO modifier for Schedule frequency.
/ET End time. Time when a task should stop execution. 
/TR - command to run

Run everyday
  schtasks /Create /it /tn "keep_mobile_disk_alive" /sc DAILY /st 23:00 /tr "\"D:\Program Files\Python27\python.exe\" \"D:\Program Files\Dropbox\base\documents\scripts\backup\backup-virtualbox.py\" -b"

Run every minutes
  schtasks /Create /it /tn "keep_mobile_disk_alive" /sc MINUTE /st 00:00 /mo 5 /tr "\"D:\Program Files\coreutils\touch.exe\" z:tmp/keep_disk_alive"

##### backup ie bookmarks
schtasks /Create /it /tn "backup_ie_bookmarks" /sc DAILY /st 13:00 /tr "\"D:\Program Files\rdiff-backup-1.2.8\rdiff-backup.exe\" \"%USERPROFILE%\Favorites\" \"Y:\base\know_how\company\office\ie_bookmarks\" "

### windows subsystem wsl

set default user

ubuntu1804 config --default-user root



#### 重装wsl

1. 打开 WSL 设定
   在开始菜单搜索 Ubuntu 右键点击应用设置。

![img](https://pic1.zhimg.com/80/v2-3555fbfda0cac762e7276902626f1c78_hd.jpg)

\2. 选择重置

![img](https://pic2.zhimg.com/80/v2-e527c18d1b01a232cabbbd8a8b0fb2c5_hd.jpg)

箭头指向处即是，这里说明重置不影响文件，但并未指明是哪个文件（夹），事实上重置操作模拟了系统重装效果，重置后需要重新设定用户和新密码。

\3. 重新设定用户及密码

![img](https://pic2.zhimg.com/80/v2-c806a2330670029bbe9a05776ebb3105_hd.jpg)



#### Install the Windows Subsystem for Linux

Before installing any Linux distros for WSL, you must ensure that the "Windows Subsystem for Linux" optional feature is enabled:

1. Open PowerShell as Administrator and run:

   PowerShellCopy

   ```powershell
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
   ```

2. Restart your computer when prompted.

##### Install your Linux Distribution of Choice

To download and install your preferred distro(s), you have three choices:

1. Download and install from the Microsoft Store (see below)
2. Download and install from the Command-Line/Script ([read the manual installation instructions](https://docs.microsoft.com/en-us/windows/wsl/install-manual))
3. Download and manually unpack and install (for Windows Server - [instructions here](https://docs.microsoft.com/en-us/windows/wsl/install-on-server))

##### Windows 10 Fall Creators Update and later: Install from the Microsoft Store

> This section is for Windows build 16215 or later. Follow these steps to [check your build](https://docs.microsoft.com/en-us/windows/wsl/troubleshooting#check-your-build-number).

1. Open the Microsoft Store and choose your favorite Linux distribution.

   ![View of Linux distros in the Microsoft Store](https://docs.microsoft.com/en-us/windows/wsl/media/store.png)

   The following links will open the Microsoft store page for each distribution:

   - [Ubuntu 16.04 LTS](https://www.microsoft.com/store/apps/9pjn388hp8c9)
   - [Ubuntu 18.04 LTS](https://www.microsoft.com/store/apps/9N9TNGVNDL3Q)
   - [OpenSUSE Leap 15](https://www.microsoft.com/store/apps/9n1tb6fpvj8c)
   - [OpenSUSE Leap 42](https://www.microsoft.com/store/apps/9njvjts82tjx)
   - [SUSE Linux Enterprise Server 12](https://www.microsoft.com/store/apps/9p32mwbh6cns)
   - [SUSE Linux Enterprise Server 15](https://www.microsoft.com/store/apps/9pmw35d7fnlx)
   - [Kali Linux](https://www.microsoft.com/store/apps/9PKR34TNCV07)
   - [Debian GNU/Linux](https://www.microsoft.com/store/apps/9MSVKQC78PK6)
   - [Fedora Remix for WSL](https://www.microsoft.com/store/apps/9n6gdm4k2hnc)
   - [Pengwin](https://www.microsoft.com/store/apps/9NV1GV1PXZ6P)
   - [Pengwin Enterprise](https://www.microsoft.com/store/apps/9N8LP0X93VCP)
   - [Alpine WSL](https://www.microsoft.com/store/apps/9p804crf0395)

2. From the distro's page, select "Get"

   ![View of Linux distros in the Microsoft store](https://docs.microsoft.com/en-us/windows/wsl/media/ubuntustore.png)

##### Complete initialization of your distro

Now that your Linux distro is installed, you must [initialize your new distro instance](https://docs.microsoft.com/en-us/windows/wsl/initialize-distro) once, before it can be used.



#### install docker on ubuntu

### 卸载旧版本

旧版本的 Docker 称为 `docker` 或者 `docker-engine`，使用以下命令卸载旧版本：

```bash
$ sudo apt-get remove docker \
               docker-engine \
               docker.io
```

```bash
https://yeasy.gitbooks.io/docker_practice/install/ubuntu.html

curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun

sudo systemctl enable docker
sudo /etc/init.d/docker start
```



使用国内源

## 安装bbr

一键安装最新内核并开启 BBR 加速脚本，https://umrhe.com/a-key-to-install-the-latest-kernel-and-open-the-bbr-acceleration-script.html

https://www.vultrgo.com/bbr/

wget http://hnd-jp-ping.vultr.com/vultr.com.100MB.bin



## Ubuntu 16.04+、Debian 8+、CentOS 7

对于使用 [systemd](https://www.freedesktop.org/wiki/Software/systemd/) 的系统，请在 `/etc/docker/daemon.json` 中写入如下内容（如果文件不存在请新建该文件）

```json
{
  "registry-mirrors": [
    "https://dockerhub.azk8s.cn",
    "https://reg-mirror.qiniu.com"
  ]
}
```

> 注意，一定要保证该文件符合 json 规范，否则 Docker 将不能启动。

之后重新启动服务。

```bash
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

> 



#### Install docker on wsl



##### 1. 安装Docker in Windows10

- 官网：https://docs.docker.com/docker-for-windows/install/

- 打开Docker Desktop设置:
  确保勾选：Expose daemon on localhost:2375 without TLS

  ![img](https://upload-images.jianshu.io/upload_images/3843091-f64c21546bb2aef4.png?imageMogr2/auto-orient/strip|imageView2/2/w/750/format/webp)

##### 3. Ubuntu18安装Docker CE

- 官网：https://docs.docker.com/install/linux/docker-ce/ubuntu/
- - apt-get install libltdl7
  - wget https://download.docker.com/linux/ubuntu/dists/bionic/pool/stable/amd64/docker-ce-cli_18.09.0~3-0~ubuntu-bionic_amd64.deb
- 授于当前用户以root权限运行Docker CLI

```bash
  # Allow your user to access the Docker CLI without needing root access.
  sudo usermod -aG docker $USER
```

- 安装Docker Compose

```bash
  # Install Python and PIP.
  sudo apt-get install -y python3 python3-pip

  # Install Docker Compose into your user's home directory.
  pip install --user docker-compose
```

#### 4. 连接Docker daemon

- 最关键的一行命令，打开Bash：

```ruby
echo "export DOCKER_HOST=tcp://localhost:2375" >> ~/.bashrc && source ~/.bashrc
```

不通过deamon连接的话，你在Ubuntu里运行docker，就会出现错误：

```cpp
docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

- 验证Docker启动成功：

```undefined
docker info
docker-compose --version
kevinqq@CN:/mnt/c/Users/xxx$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.
```

至此，已经在WSL Ubuntu里完美配置Docker成功了！

#### 5. mount host folder

```bash
ln -s /mnt/d/ /d  # This is must! 
### mysql 

Forget password: 忘记MySQL root密码解决方法，基于Ubuntu 12.04 LTS(http://blog.csdn.net/abbuggy/article/details/8245464)

## software install

### install java

sudo apt-get install default-jdk



### install windows on a new 1T SSH disk

Avoid install dropbox and virtualbox in a portable disk. The disk connection is unstalbe. dropbox and virtualbox crash frequently.

#### WinToUSB (does not work)
 http://lifehacker.com/how-to-run-a-portable-version-of-windows-from-a-usb-dri-1565509124
Download windows 7 iso 
Use WinToUSB => Get "disk read error" when boot from the portable disk.

#### use WinSetupFromUSB (fail)
Created a windows 7 boot in a portable disk. 
Use the 1T disk as the main disk. 
When rebook, get error 
  PXE-E61: media test failure. check cable
  PXE-M0F: existing intel boot agent

I guess that the PC is reading the SSD disk first instead of the portable disk. However, somehow I cannot enter BIOS any more - the bios selection selection screen does not show any more when system reboots.

#### method 3 (complicated. Need manual settings)
http://www.intowindows.com/how-to-install-windows-7-to-usb-external-hard-drive-must-read/

### batch uninstall
IObit: free, search autocomplete

### software batch uninstall
IObit Uninstaller

### list installed software

#### method 1: Get-WmiObject
[](http://cs.albis.jp/blogs/ms-18e/archive/2011/12/05/569518.aspx)
[](http://www.computerperformance.co.uk/powershell/powershell_wmi.htm)
[](http://www.howtogeek.com/165293/how-to-get-a-list-of-software-installed-on-your-pc-with-a-single-command/)
Open powershell (not CMD)
Run this line (replace "xling-lab" with your PC name)
  Get-WmiObject -Class Win32_product –ComputerName "xling-lab" | Select-Object Name, Vendor, Version | Export-Csv Y:\base\download\softwarelist.csv -encoding Default –NoTypeInformation

If get error: rpc server is unavailable. (Get-WmiObject : RPC サーバーを利用できません。 (HRESULT からの例外: 0x800706BA))
Solution [](http://bbs.winos.cn/thread-121544-1-1.html)
  出现此现象是防火墙设置的原因..
  解决方法：可以使用Gpedit.msc 进行以后，本地计算机策略--计算机配置--管理模板--网络--网络连接--WIndow防火墙--允许远程管理 (着信リモート管理の例外を許可する) 启用 即可

Problem: It seems that this method cannot list all software!
  [](http://stackoverflow.com/questions/673233/wmi-installed-query-different-from-add-remove-programs-list)
  One cause is that this class only displays products installed using Windows Installer (See Here). The uninstall registry key is your best bet. Here is some code to monitor the registry key.
  There is script to parse the register but I have not tried.

#### method 2: WMIC
http://helpdeskgeek.com/how-to/generate-a-list-of-installed-programs-in-windows/

However, the results contain many unrecognized characters. 

### Try to install both Ubuntu and Windows
  [http://chypot.blogspot.jp/2013/10/lets-note-sx2-ubuntu.html]
    リカバリディスク(DVD2枚必要)を作成する
      My PC: Lets note CF-SX2JEBDR
        # My PC should be able to write DVD.
        DVD: DVD±R/±RW/RAM/±RDL [http://kakaku.com/item/K0000385836/spec/#tab]

    Ubuntu 12.04のインストールディスクを用意する
    Lets NoteのBIOS設定を調整し、CD(DVD)からブートできるようにしてブート
    お試しUbuntuが起動するので、キーボードや無線LANなどが使えるか確認
    インストーラに従ってインストール


## hardware install

### ssd 

(clg) Add the SSD to a USB case, connect the case to the PC. PC cannot recognize the SSD. 
(hyp) Seems that the SSD is not formatted. 
(clg) Try to format it but the 'new sample volumn' button is grey [](https://www.youtube.com/watch?v=vCjpbgivnps).
(hyp) Use partassist [分区助手 – 轻松调整硬盘分区](http://www.disktool.cn/download.html) to format
Note to select '4k align' when format [](http://baike.baidu.com/view/7096264.htm)
Use HD革命/CopyDrive Ver.5 to migrate data.

## shell

### internet explorer 

#### ie crashes when accesses es-ras
Problem seems to occurs after juniper vpn update.
I installed the update plugins.
IE still crashes.
Clean caches. => does not work.
Reset IE => cannot access es-ras website anymore.

#### ie home page is moved to hao.qq.com

Reinstalling IE does not work.

Edit the start page entry in regedit. The entry is locked. Cannot unlock it after one-hour of try. Give up. 

### snapshot 
ScreenshotCaptor (free, but need to get free key each 6 months)

### remap keys
I used to accidently touch the sleep key on the keyboard. 
Need to disable it.

#### Method 1: I cannot find lets note official method

#### Method 2
[http://wenku.baidu.com/view/24b5b5cf0508763231121206.html]
Open regedit
Go to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout\
Create a binary key named 'Scancode Map'
Add value to the key something like this:
  00,00,00,00,00,00,00,00,02,00,00,00,01,00,02,00,00,00,00,00
In my lets note, I replace the sleep key F7 to ESC
  The key map.
  [http://zhidao.baidu.com/question/291163072.html]

### cannot paste text between local and remote machine when using windows remote desktop.
    Solution: use Task Manager to kill and restart the rdpclip.exe process on local
    and remote machines.
    [](http://superuser.com/questions/95609/cant-copy-and-paste-in-remote-desktop-connection-session)

  #include <iostream>
  #include <list>
  int main(int argc, char const *argv[])
  {
    std::list<int> l;
    // add watch point:
    //   watch l
    //   c
    std::cout << "start" << std::endl;
    l.push_back(1);
    l.push_back(2);
    std::cout << "done" << std::endl;
    return 0;
  }

  2. watch a member variable
 %%%%% snip [start] %%%%%

    #include <iostream>
    #include <vector>
    #include <list>
    class A{
      public:
        A(){ m_v.push_back(777); }
        void Insert(int v){
          l.push_back(v);
        }
      private:
        std::list<int> l;
        std::vector<int> m_v;
    };
    int main(int argc, char const *argv[])
    {
      A a1, a2;
      std::cout << "start" << std::endl;
      a1.Insert(10);
      a2.Insert(20);
      a2.Insert(30);
      a2.Insert(40);
      std::cout << "done" << std::endl;
      return 0;
    }
 %%%%% snip [end] %%%%%
    Suppose that you need to watch whether a2.l changed.
      watch a2.l
    Suppose that you have many objects of A. The l of one of the objects ax
    caused crash. You do not know where change of l occurs first. You do not
    know ax. Using 'where' command, you can get the address of ax and ax.l.

    ax may contain multiple elemetns and each element can change. You only need
    to detect the change lf ax.l. Hence, you should not use the address of ax
    to detect.
       
    You need to watch l by its address. First, you get l''s
    address. 
      p &a2.l
      (gdb) \$1 = (std::list<xxx>*) 0x123456
    You should not use
      watch 0x123456
    0x123456 is the base address of ax''s l. It does not change no matter whether l adds/removes elements.
       
    You should not use
      watch *0x123456
    *0x123456 is the value of the first byte at 0x123456 instead of a list object.
    
    Somehow, grammars like this are illegal.
      watch *((std::list *)(0x123456))
     
    You should use
      watch *$1

### the 'win' key does not work with left and right.
Problem:
  Suddenly, the win key does not work with left and right keys.
  But it works with down and up keys. 
Solution:
  I closed all windows. Suddenly, the problem disappeared. Mayb restart explorer also solve the problem.


### create symbolic links
[link shell extension](schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html)

### windows drive mapping. "The local device name is already in use."
Solution: restart PC. 

## os

### which process is locking a folder/file (prevent you from modifying a folder/file)?
  Solution: Process Explorer  [](http://stackoverflow.com/questions/1084482/how-to-find-out-what-processes-have-folder-or-file-locked)
     Ctrl-F will let you search for a file and list the process(es) that have that file open/locked. You can then close that handle using Process Explorer.

### windows: call program from cmd directly.
  Problem: you have a program d:\A\B\PowerCmd.exe. You hope to call PowerCmd under cmd directly. 
  Solution: 
    1. Add the path of PowerCmd into PATH. This is complicated.
    2. Add a symbolic of PowerCmd in c:\Windows\System32
      mkdir c:\Windows\System32\PowerCmd.exe d:\A\B\PowerCmd.exe
      
      However, PowerCmd cannot correctly run without running in its original directory. [http://stackoverflow.com/questions/14322980/windows-cmd-how-to-create-symbolic-link-to-executable-file]
    3. Add a bat script in c:\Windows\System32 # [http://stackoverflow.com/questions/5909012/windows-batch-script-launch-program-and-exit-console], [http://stackoverflow.com/questions/14322980/windows-cmd-how-to-create-symbolic-link-to-executable-file]
        # PowerCmd.bat 
        REM PowerCmd.cmd
        start "" "D:\Program Files (x86)\PowerCmd\PowerCmd.exe" %* 
      
      if use 
        start "" "D:\Program Files (x86)\PowerCmd\PowerCmd.exe" %*
      , the application shutdown its window immediately.
      If use 
        start "" "D:\Program Files (x86)\PowerCmd\PowerCmd.exe"
      , the application window does not disapear

### Cannot find 'systempropertiesprotection.exe'
    vim does not work correctly. Sometime shows error: cannot find python although you have added python into the system path.

  Cause: %PATH% is too long (longer than 2048). The following PATH may work.
    C:\Windows;C:\Windows\system32;C:\Windows\System32\Wbem;C:\Program Files (x86)\Symantec\VIP Access Client\;C:\ProgramData\Oracle\Java\javapath;C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\Intel\iCLS Client\;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Toshiba\Bluetooth Toshiba Stack\sys\;C:\Program Files (x86)\Toshiba\Bluetooth Toshiba Stack\sys\x64\;C:\Program Files (x86)\Intel\OpenCL SDK\2.0\bin\x86;C:\Program Files (x86)\Intel\OpenCL SDK\2.0\bin\x64;C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;c:\Program Files (x86)\Common Files\Roxio Shared\DLLShared\;c:\Program Files (x86)\Common Files\Roxio Shared\OEM\DLLShared\;c:\Program Files (x86)\Common Files\Roxio Shared\OEM\DLLShared\;c:\Program Files (x86)\Common Files\Roxio Shared\OEM\12.0\DLLShared\;c:\Program Files (x86)\Roxio 2010\OEM\AudioCore\;d:\Program Files (x86)\WinSCP\;d:\Program Files (x86)\MiKTeX 2.9\miktex\bin\;D:\Program Files\Python27;D:\Program Files\Python27\Scripts;C:\Program Files (x86)\Java\jre7\bin\;D:\Program Files\seltests\selenv;D:\Program Files (x86)\PuTTY;D:\Program Files\rdiff-backup-1.2.8;D:\Program Files\coreutils;D:\Program Files\Dropbox\base\usr\bin\Vim\vim73;d:\Program Files (x86)\LLVM\bin;d:\Program Files (x86)\Git\cmd;d:\Program Files (x86)\Git\bin;D:\Program Files (x86)\GnuWin32\bin;D:\Program Files (x86)\Java\jdk1.8.0_05\bin;

### Environment variable

#### Add directory to path
  Append "D:\lnk\mingw-bin" to %PATH%
    Open a cmd as administrator
    Run 
      setx PATH %PATH%;D:\lnk\mingw-bin /m

    However, seems that this method can only set very short EV path. 
    When you use My Computer -> Property -> Advanced ..., you can set much longer EN.

#### Externally set EV (not only in your current console)
  (need to run as administrator)
  setx NEWVAR SOMETHING

#### Environment variable too long
  Method 1: Use this bat script to shorten path using abbrievation. 
    [http://stackoverflow.com/questions/4405091/how-avoid-over-populate-path-environment-variable-in-windows]
    
    ////////////////////////
    @echo off
    
    SET MyPath=%PATH%
    echo %MyPath%
    echo --
    
    setlocal EnableDelayedExpansion
    
    SET TempPath="%MyPath:;=";"%"
    SET var=
    FOR %%a IN (%TempPath%) DO (
        IF exist %%~sa (
            SET "var=!var!;%%~sa
        ) ELSE (
            echo %%a does not exist
        )
    )
    
    echo --
    echo !var:~1!
    ////////////////////////
    However, my path is still much longer when abbrived using this script.

  Method 2: create symbolic shortcut for long paths 
    Con:
      Does not work. 
      Programs in a sybolic-link directory cannot work correctly!

    [http://stackoverflow.com/questions/4405091/how-avoid-over-populate-path-environment-variable-in-windows]
    if you are using windows vista or higher, you can make a symbolic link to the folder. for example:
    mklink /d C:\pf "C:\Program Files"
    would make a link so c:\pf would be your program files folder. I shaved off 300 characters from my path by using this trick.


  Method 3: Define multiple short alias for different directories and use alias to form the PATH. 
    Con:
      However, this indeed does not work. 
      When substituted into PATH, each alias will be extended to their original length.
    [http://stackoverflow.com/questions/21606419/set-windows-environment-variables-with-commandline-cmd-commandprompt-batch-file%]


### network

#### home network
problem: cannot access to internet 
reason: wrong so-net id. 
  ka45782@jc4.sonet.ne.jp => ka45782@jc4.so-net.ne.jp

### file system

#### who is locking a folder/file
    Process Explorer

#### share vimrc between windows vim and linux vim
I have driver Y mapped to a linux drive. 
I can simply use linux's vimrc and bundle. Avoid maintaining two vim settings.
  source Y:\.vimrc
  set rtp+=Y:\base\share\vim\vim73\bundle
My linux is using pathogen. Pathogen requires a pathogen.vim in vim/autoload to work. 
My windows vim's autoload does not have this file and hence some plugin cannot be loaded
by vim on windows. 
I create symbolic links of linux vim's autoload and bundle directories to windows vim's 
root direcotry (d:/program files/vim/)

#### symbolic
  Create a link in bat file like the below in windows/system32 [](http://stackoverflow.com/questions/14322980/windows-cmd-how-to-create-symbolic-link-to-executable-file)
    REM chrome.cmd
    start /b "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %*

    % the first "" means to hide the background cmd window.
    
    Link Shell Extension
      Install
      right click to source file/directory
      click 'pick source'
      go to the destination directory.
      right click mouse.
      click 'drop link'
    
    Con:
      Suppose that a.exe depends on a.dll. You need to create link for the a.dll in the destination directory as well.

#### Sort Windows folders according according to specific rules.
    Hope to order some folders.
    Method 1
      Add numbers to folder names.
      Con:
        These are folder for source codes. I will cite files according to there paths. Adding numbers in paths looks strange.
    Method 2:
      Add tags to folders.
      Con:
        Windows cannot assign tags to foldes.
    Method 3:
      Give number icons to folders.
      Con:
        You cannot do this for folders in a driver mapped from linux.
    Method 4:
      Make each folder contain less than 7 subfolders, so that you can easily understand the logic.

#### Hope to set shortcut for commands in Office 2010.
    You can use windows predefined shortcuts to call commands in Office.
    However, the predefined shortcuts are usually too long.
    You can add new commands to ribbon pannel.
    Seems that Chinese Office 2010 and Japanese Office 2007 can customize shortcuts. [](http://www.33lc.com/article/5386.html)
    However, looks like Japanese Office 2010 cannot only use predefined shortcuts but cannot customize shortcuts [](http://www.moug.net/tech/exopr/0020010.html).


#### dropbox

##### Move dropbox.

###### method 1: Using dropbox "move" feature
Con: symbolic links interrupt moving. 

###### method 2: when the dropbox folder is large, use fastcopy to copy the old dropbox folder to the new location.
Dropbox will gradually add files that are not corrected synchronized (if any) into the new folder.

##### what makes Windows slow
Cause: I use a script to create virtualbox snapshot everyday. Each time a new snapshot is created. Each snapshot is small and dropbox can quickly upload it to the cloud. 
Someday I moved the script and windows cannot automatically run it any more. Virtualbox keeps writing the same snapshot, the snapshot grows larger and larger. It take longer and longer time for dropbox to upload the snapshot. 

Solution: move the script back to its original location.

##### Sharing subfolders of shared folder
[](http://synappse.co/sharing-subfolders-of-shared-folder-dropbox/)

D:\Program Files\Dropbox\base\know_how\wealth\0.57_malaysia_ec>mklink /J 4_import-products 1_mei\5_smartphone_cases\4_import-products 

# linux

## install typora

​```bash
  # or use
## sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA300B7755AFCFAE
wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -

## add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update
sudo apt-get install typora

```

## 安装坚果云

```
wget https://www.jianguoyun.com/static/exe/installer/ubuntu/nautilus_nutstore_amd64.deb
sudo dpkg -i nautilus_nutstore_amd64.deb
sudo apt-get install -f
echo "fs.inotify.max_user_watches=32768" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p /etc/sysctl.conf

cat /proc/sys/fs/inotify/max_user_watches # default is 8192 
sudo sysctl fs.inotify.max_user_watches=1048576 # increase to 1048576
```



## ustc apt source 国内源 

```bash
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse 
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse 
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse 
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse 
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse 
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse 
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse 
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse 
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse 
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```

## install chrome on ubuntu

1. Add Key:

   ```sh
   wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
   ```

2. Set repository:

   ```sh
   echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list
   ```

3. Install package:

   ```sh
   sudo apt-get update 
   sudo apt-get install google-chrome-stable
   ```





## install shadowsocks on ubuntu

```bash
sudo apt-get install python3-pip
pip3 install http3://github.com/shadowsocks/shadowsocks/archive/master.zip
  # or download the zip, unzip it and then pip setup.py install
ssserver --version
  # should show 3.0.0. Otherwise, aes-256-gcm is not supported
```

vim ~/shadowsocks_client.json

```bash
{
    "server":"150.109.41.143",
    "server_port":11443,
    "local_address": "127.0.0.1",
    "local_port":1080,
    "password":"Bda@@x4h!!ZhaUr72VR",
    "timeout":300,
    "method":"aes-256-gcm"
}
```

```bash
sudo sslocal -c ~/shadowsocks_client.json
```

> ## 150.109.41.143
>
> 45.32.27.238
>
> 207.148.121.59



### 配置代理服务

在经过上面一番操作之后，我发现在linux下并不能直接通过上述设置直接翻墙，因为shawdowsocks是socks 5代理，需要客户端配合才能翻墙。

- 安装privoxy


```
sudo apt-get install privoxy
```

配置privoxy

```

sudo vi /etc/privoxy/config
```

我本地实在第1337行找到修改对象的，修改端口为上面shadowsocks配置的本地端口，如下：

```
forward-socks5t / 127.0.0.1:1080 .
```

privoxy监听接口默认开启的 `localhost：8118`，也是在这个文件中，这里我没有修改。

启动privoxy

- 开启privoxy 服务就行 


```
sudo service privoxy start
```

设置http 和 https 全局代理 



- ```
  export http_proxy=http://localhost:8118
  export https_proxy=https://localhost:8118
  ```

测试

```
wget www.google.com
```

如果把返回200 ，并且把google的首页下载下来了，那就是成功了



### 开机启动

```bash
vim /lib/systemd/system/shadowsocks.service
```

add 

```bash
[Unit]
Description=Shadowsocks client
After=network.target

[Service]
ExecStart=/home/xling/.local/bin/sslocal -c /home/xling/shadowsocks.json
Restart=on-abort

[Install]
WantedBy=multi-user.target
```



step 3：
运行shadowsocks.service
`systemctl start shadowsocks.service`
允许开机自动启动
`systemctl enable shadowsocks.service`
查看运行状态
`systemctl status shadowsocks.service`
![pic](http://wx2.sinaimg.cn/large/006ZPKkkly1frjcotzur6j30qv03c3yl.jpg)

 

google-chrome --proxy-server="socks5://127.0.0.1:1080"

## ubuntu 18.04 LTS 安装搜狗输入法



卸载ibus。

```shell
sudo apt-get remove ibus
```

清除ibus配置。

```shell
sudo apt-get purge ibus
```

卸载顶部面板任务栏上的键盘指示。

```shell
sudo  apt-get remove indicator-keyboard
```

安装fcitx输入法框架

```shell
sudo apt install fcitx-table-wbpy fcitx-config-gtk
```

切换为 Fcitx输入法

```shell
im-config -n fcitx
```

im-config 配置需要重启系统才能生效

```shell
sudo shutdown -r now
```

[下载搜狗输入法](https://links.jianshu.com/go?to=https%3A%2F%2Fpinyin.sogou.com%2Flinux%2F%3Fr%3Dpinyin)

```shell
wget http://cdn2.ime.sogou.com/dl/index/1524572264/sogoupinyin_2.2.0.0108_amd64.deb?st=ryCwKkvb-0zXvtBlhw5q4Q&e=1529739124&fn=sogoupinyin_2.2.0.0108_amd64.deb
```

安装搜狗输入法

```shell
sudo dpkg -i sogoupinyin_2.2.0.0108_amd64.deb
```

修复损坏缺少的包

```shell
 sudo apt-get install -f
```

打开 Fcitx 输入法配置



## install pinyin

https://blog.csdn.net/fx_yzjy101/article/details/80243710



## docker + cuda + gpu

###  Nvidia驱动安装配置

[http://hsiangyee.pixnet.net/blog/post/279753872-ubuntu-18.04-server-%E4%B8%8A%E5%BB%BA%E7%AB%8B-nvidia-docker2-%E4%B8%A6%E5%9C%A8%E5%AE%B9%E5%99%A8%E4%B8%AD](http://hsiangyee.pixnet.net/blog/post/279753872-ubuntu-18.04-server-上建立-nvidia-docker2-並在容器中)



可以`lspci | grep -i nvidia`查看自己机子的N卡，也可以`ubuntu-drivers devices`查看系统给出的配置及驱动推荐，这时，我们可以直接`ubuntu-drivers autoinstall`进行安装，完成并重启后，可通过`nvidia-smi`验证安装是否成功并查看N卡的状态



https://medium.com/@madmenhitbooker/install-tensorflow-docker-on-ubuntu-18-04-with-gpu-support-ed58046a2a56

```
sudo -i
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
```

测试：

```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

重要的是--runtime=nvidia，这样创建的容器就能识别cuda



It should return the same thing that was returned earlier if you remember:

![img](https://miro.medium.com/max/60/1*RdTeO1OC8SqFhIz77rEjNA.png?q=20)

![img](https://miro.medium.com/max/714/1*RdTeO1OC8SqFhIz77rEjNA.png)



创建能够运行GUI，GPU的容器

https://blog.csdn.net/ericcchen/article/details/79253416

```bash
sudo docker run \
-it \
--runtime=nvidia \
-p 8890:8890 \
-p 8024:22 \
-e DISPLAY=unix$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--privileged \
--mount src=/home/xling,target=/host_home,type=bind \
--name tf1.15_3 \
ubuntu18_gpu:v2 bash

  # GUI. Not work
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount src="$(pwd)",target=/host_home,type=bind --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --name tf1.15_1 tensorflow/tensorflow:1.15.0-gpu-jupyter bash

## tf 2.0
## tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter



Save image

​```bash
sudo docker commit tf1.15_2 ubuntu18_gpu:v3
```





### 在docker运行pycharm

  https://zhuanlan.zhihu.com/p/52827335

  （需要pycharm专业版）

  配置SSH服务

  接着我们在刚刚新建的容器里配置SSH服务，首先安装`*openssh-server*`:

  ```bash
  $ apt update
  $ apt install -y openssh-server
  ```

  然后建立一个配置文件夹并进行必要的配置：

  ```bash
  $ mkdir /var/run/sshd
  $ echo 'root:passwd' | chpasswd
  # 这里使用你自己想设置的用户名和密码，但是一定要记住！
  $ sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
  $ sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
  $ echo "export VISIBLE=now" >> /etc/profile
  ```

  重启SSH激活配置：

  ```bash
  $ service ssh restart
  ```

  在服务器（宿主机）上（不是服务器的docker里）测试刚刚新建docker容器中哪个端口转发到了服务器的22端口：

  ```bash
  $ sudo docker port tf1.15_2 22
  # 如果前面的配置生效了，你会看到如下输出
  # 0.0.0.0:8022
  ```

  最后测试能否用SSH连接到远程docker：

  ```bash
  $ ssh root@localhost -p 8022
  # 密码是你前面自己设置的
  ```

  到这里说明服务器的docker端已经完成配置。



## install anaconda

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-Linux-x86_64.sh

vim ~/.condarc

```bash
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```



```
conda create -n venv pip python=3.7
```



## install pycharm

wget https://download-cf.jetbrains.com/python/pycharm-community-2019.2.4.tar.gz



## install spacevim

```
wget https://spacevim.org/cn/install.sh
bash install.sh
```

## install zsh

```bash
sudo apt install zsh

## oh my zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

## autojump
git clone https://github.com/joelthelion/autojump.git
cd autojump
./install.py
```

Install auto sugession

```bash
## antigen
curl -L git.io/antigen > antigen.zsh
sudo apt-get install zsh-antigen

cp ~/jianguoyun/Dropbox/.antigen/ ~ -rp
```



1. Add the following to your `.zshrc`:

   ```
   antigen bundle zsh-users/zsh-autosuggestions
   ```

2. Start a new terminal session.



## 截图

**Shift + Ctrl + PrtSc** – *Copy the screenshot of a specific region to the clipboard.*

**Ctrl + PrtSc** – *Copy the screenshot of the entire screen to the clipboard.*
**Ctrl + Alt + PrtSc** – *Copy the screenshot of the current window to the clipboard.*

**PrtSc** – *Save a screenshot of the entire screen to the “Pictures” directory.*
**Shift + PrtSc** – *Save a screenshot of a specific region to Pictures.*
**Alt + PrtSc** – *Save a screenshot of the current window to Pictures*.

## 鼠标手势控制

```bash
sudo add-apt-repository ppa:easystroke/ppa
sudo apt-get update
```



## Install fzf 

```bash
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```



## 安装金山office

Search "wps" in ubuntu software center

Or



```bash
sudo apt-get install snapd
sudo snap install wps-office
```



Or

```bash
wget http://wdl1.pcfg.cache.wpscdn.com/wpsdl/wpsoffice/download/linux/8865/wps-office_11.1.0.8865_amd64.deb

sudo dpkg -i wps-office_11.1.0.8865_amd64.deb
```



## 安装tensorflow

pip install tensorflow-gpu==1.15



## ssh without password

```bash
ssh-keygen -t rsa
ssh-copy-id xling@150.109.41.143
```



## 安装chrome

```
sudo wget https://repo.fdzh.org/chrome/google-chrome.list -P /etc/apt/sources.list.d/
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
sudo apt-get update
sudo apt-get install google-chrome-stable
```



## zsh

```bash
sudo apt-get update
sudo apt-get install zsh

sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

mv ~/.zshrc{,.bak}
ln -s ~/base/shared-among-computers/documents/scripts/linux/.zshrc ~

mkdir ~/antigen
curl -L git.io/antigen > ~/antigen/antigen.zsh
zsh


git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions
source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh
```

## install java 

```
apt install default-jre
apt install openjdk-11-jre-headless
apt install openjdk-8-jre-headless
apt install openjdk-9-jre-headless
```



## install yarn

```console-bash
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add - 
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install yarn
```



## install nodejs

sudo apt-get install curl
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get install nodejs

## autojump

```
git clone git://github.com/wting/autojump.git
cd autojump
./install.py

  # add to .zshrc
. /usr/share/autojump/autojump.sh
```

## pycharm

```bash
curl -OL "https://download.jetbrains.com/python/pycharm-community-2018.3.tar.gz?_ga=2.28066617.44010933.1570078379-1029410450.1566607995"
mv pycharm-community-2018.3.tar.gz\?_ga=2.28066617.44010933.1570078379-1029410450.1566607995 pycharm-2018-3.tar.gz
tar zxvf pycharm-2018-3.tar.gz
ln -s ~/job/codes/install_src/pycharm-community-2018.3/bin/pycharm.sh /usr/bin/pycharm
```



## xclock

```
sudo apt-get install xorg openbox
```






## fzf install
```bash
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

## install Hexo+Github blog

超详细Hexo+Github博客搭建小白教程, https://zhuanlan.zhihu.com/p/35668237

​```bash
sudo npm i hexo-cli -g # 安装Hexo
mkdir blog
cd blog
hexo init
hexo g # 生成静态网页
hexo s # 打开本地服务器 
```


然后浏览器打开http://localhost:4000/

* add ssh key to github
https://help.github.com/en/enterprise/2.16/user/articles/adding-a-new-ssh-key-to-your-github-account

```bash
git config --global user.name "xling"
git config --global user.email "leonexu@qq.com"

ssh-keygen
cat ~/.ssh/id_rsa.pub
```



## ssh login without password

cat ~/.ssh/id_rsa.pub | ssh ubuntu@150.109.41.143 'cat >> .ssh/authorized_keys'

## neovim

```bash
sudo apt-get install software-properties-common
sudo apt-add-repository ppa:neovim-ppa/stable
sudo apt-get update
sudo apt-get install neovim
```



## pip upgrade

pip install --upgrade tensorflow

## ubuntu 18.04 apt aliyun源

  ```bash
  deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
  deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
  deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
  deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
  deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

  # deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
  # deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
  # deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
  # deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
  # deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
  ```



## pip 国内源
linux下，修改 ~/.pip/pip.conf (没有就创建一个)， 修改 index-url至tuna，内容如下：

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple



Windows下，C:\Users\86135\pip\pip.ini

> [global]
> index-url=https://pypi.tuna.tsinghua.edu.cn/simple 
> [install]  
> trusted-host=pypi.tuna.tsinghua.edu.cn
> disable-pip-version-check = true  
> timeout = 6000  

## install vscode

```console-bash
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code
```

## conda 

conda create --name venv36 python=3.6
conda env list



```bash
activate venv36  # for windows
```

国内源

查看当前使用源
conda config --show-sources

添加指定源
conda config --add channels 源名称或链接

删除指定源
conda config --remove channels 源名称或链接

恢复默认源
conda config --remove-key channels

清华源（TUNA）

> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/

> ##设置搜索时显示通道地址
> conda config --set show_channel_urls yes
> ##若需要PyTorch需添加
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

中科大源（USTC）

> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
> conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
> conda config --set show_channel_urls yes



## vim with python support 

```bash
sudo apt-get remove --purge vim 
sudo apt-get clean

sudo apt install libncurses5-dev \
libgtk2.0-dev libatk1.0-dev \
libcairo2-dev python-dev \
python3-dev git

git clone https://github.com/vim/vim.git && cd vim

./configure --with-features=huge \
--enable-multibyte \
--enable-pythoninterp=yes \
--with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu/ \ 
--enable-python3interp=yes \
--with-python3-config-dir=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/ \ 
--enable-gui=gtk2 \
--enable-cscope \ 
--prefix=/usr/local/

make
make install

sudo make VIMRUNTIMEDIR=/usr/local/share/vim/vim81 
Preparing deb package using checkinstall
And install that package

cd /usr/vim && sudo checkinstall
Or, if want to just create a package use --install=no option with checkinstall

Set Vim as a default editor
sudo update-alternatives --install /usr/bin/editor editor /usr/local/bin/vim 1
sudo update-alternatives --set editor /usr/local/bin/vim
sudo update-alternatives --install /usr/bin/vi vi /usr/local/bin/vim 1
sudo update-alternatives --set vi /usr/local/bin/vim   


```
```



mv ~/.SpaceVim.d/init.toml{,.bak}
ln -s ~/base/shared-among-computers/documents/scripts/crossplatform/vim/spacevim/init.toml ~/.SpaceVim.d/
mv ~/.SpaceVim/autoload/SpaceVim/default.vim{,.bak}
ln -s ~/base/shared-among-computers/documents/scripts/crossplatform/vim/spacevim/default.vim ~/.SpaceVim/autoload/SpaceVim/
```





## application

### package

#### dropbox

http://www.dropboxwiki.com/tips-and-tricks/using-the-official-dropbox-command-line-interface-cli

##### exclude

  dropbox.py exclude add base/ job/ ls/ wealth/

#### xming:

  Problem: installed xming, configured putty forwarding, but xclock does not start: "Error: Can't open display"   Solution: use XLaunch to start xserver instead of directly starting xming.

#### gnu_screen:

  If previous session exists, restore it; otherwise, create a new session      [http://stackoverflow.com/questions/10433922/gnu-screen-connect-if-exists-create-if-not]     [http://stackoverflow.com/questions/19327556/get-specific-line-from-text-file-using-just-shell-script]     cl; sessionId=`screen -ls | sed '2!d' | awk '{print $1}'`; screen -D -R -S $sessionId #restore_screen_session

#### editing

##### latex

###### proofreading

####### generate pdf from latex diff using [latexdiff](http://emeryblogger.com/2011/01/14/diff-your-latex-files-with-latexdiff/)

  mv draft.tex draft-latest.tex   git checkout id-of-the-last-version /path/to/the/draft.tex   latexdiff draft.tex draft-latest.tex > draft-diff.tex   platex draft-latex   clear; latexdiff draft.tex draft-latest.tex > draft-diff.tex && platexr_dvi_ps_pdf.sh draft-diff

####### latex diff

Some applications exist but none work perfectly.

\####### directly diff your old and new latex drafts. con: cannot detect the format errors.

####### others

\####### diffpdf Use diffpdf to proofread the difference between two versions of a pdf. diffpdf   sudo apt-get install diffpdf   con:      It takes pdf format into consideration.      If text location changes, it regards the text is different even if the text content is the same.

\####### DiffPdf (http://www.qtrac.eu/try) con:    not free   Not accurate as well. 

\####### pdfdiff.py http://www.cs.ox.ac.uk/people/cas.cremers/downloads/software/pdfdiff.py   Pro:      it turns the pdf into pure text, which removes the influence of format.

\####### diff-pdf (https://github.com/vslavik/diff-pdf) con:   The diff generated is blur. Does not work at all.

###### For latex files that require pLaTex2e, simply use platex.

platex manuscript.tex ...

Install pLaTex2e sudo apt-get install perl-tk wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz tar zxvf ... ./install-tl

###### install sty files

You could create a folder below your TeX home directory and put your .sty file therein.  Use this command at the command prompt to find out where:   kpsewhich -var-value=TEXMFHOME  In my machine, the path is ~/texmf 

###### Embed pdf font

When submit a paper, the conference may require the all fonts are embeded into the pdf. Method 1 latex valuetools2013-final2.tex dvips valuetools2013-final2.dvi ps2pdf13 valuetools2013-final2.ps valuetools2013-final2.pdf

Method 2

1. get pdf: latex->bibtex->latex->latex->dvipdfm
2. convert pdf to pdf with embeded fonts:
3. open pdf using GSViewer
4. file -> convert

##### vim

###### meet error when apt-get install vim

Errors were encountered while processing: /var/cache/apt/archives/vim-runtime_2%3a7.4.884-1~ppa1~t_all.deb

Solution: do not apt-get, install from source manually.

#### draw

##### Install plantuml, use script to draw charts, figures, diagrams

http://www.aise.ics.saitama-u.ac.jp/~gotoh/InstallPlantUMLonUbuntu.html Alternatively, you can use umlgraph.

##### Use gnuplot in cygwin

http://petpari.blogspot.jp/2013/04/octave-gnuplot-using-cygwins-x-server.html

#### backup

##### backup, rdiff-backup

I am backing up /home4/xling/ to /home4/share/linux [use cron to schedule the backup] crontab -e add this line: 15 1 * * * /usr/bin/rdiff-backup /home4/xling/ /home3/share/linux/

When rdiff-backup used up all spaces, you need to remove some backup files. You cannot directly remove files. You can only use --remove-older-than parameter: rdiff-backup --force --remove-older-than 1D /home3/share/linux/

##### I backed up linux using rdiff-backup 15:00 every day. The data is copied to /home4/xling/linux and you can directly access files there.

##### backup important files

I am mapping windows dropbox directory X to ubuntu. All files writen in X are automatically backuped.

#### window management

##### shell

  \subsubsection{switch from bash to zsh. Hard to import old bashrc. Given up.}   The benefit of zsh:     Expand [remote] paths ve/pl/re --> vendor/plugins/redmine     Expand variables: $PATH --> /your/full/path     Intelligent correction: /pbulic/html becomes /public/html     Spelling correction     Shared command history across running shells     Highly themable prompts     Most of your Bash configuration will work seamlessly     Can emulate Bash when run as /bin/sh     Supports vim mode commands     OhMyZsh support

  chsh -s /bin/zsh

  When you reboot console, you may get this error:     zsh compinit: insecure directories, run compaudit for list. Ignore insecure directories and continue [y] or abort compinit [n]?   The solution:     # Remove the 'write' priviliage of insecure files      compaudit | xargs chmod -R 755

  You need to load your bashrc.     Move contents of .bashrc to .profile     Add this line in ~/.zshrc      [[ -e ~/.profile ]] && emulate sh -c 'source ~/.profile'     # Directly doing the below does not work:     # [[ -e ~/.bashrc ]] && emulate sh -c 'source ~/.bashrc'

  However, some commmands in bashrc (e.g, 'shopt', 'fzf.bash') still cannot be loaded. Cannot find solution. Give up.

  \subsubsection{Do not use '-' as the initial character of file names}   fzf does not work when the current directory contain such a file.

  \subsubsection{Install fzf for fuzzy search.}   If does not work, ensure that

1. ~/.fzf/install is run
2. ~/.bashrc is reloaded

  \subsubsection{fzf, search file contents}     Go to the folder that may contains your target file.     run this:       grep --line-buffered --color=never -r "" * | fzf

##### transmission

  \subsubsection{make samba launch at startup (auto start after reboot)}   update-rc.d samba defaults 

\subsubsection{syncrhonize data in local network.}   GoodSync (good)   SyncBack:     Con:       Need long time to scan disk.

  ## screen -d -r session_name to reload ./screenrc

##### tmux

[~/.tmux.conf.md]  

###### tmux does not load .tmux.conf

Solution:

1. close zombie sessions. 
2. correct typos in .tmux.conf 

##### gnu screen

###### gnu screen is old. Use byobu

  byobu is an extension for screen and mux.   byobu make it easier to create windows.   You can still use the commands of screen.   ubuntu 14.04 has installed byobu by default.   Shortcuts:     F2 创建新的窗口     F3 回到先前窗口     F4:跳到下一个窗口     F5 重新加载文件     F6 释放该次对话     F7 进入 复制/回滚模式     F8 重新命名一个窗口     F9 配置菜单，也可以使用组合键Ctrl+a, Ctrl+@   If you login using putty, the shortcuts may not work.   The solution: got to putty setting -> session -> keyboard -> The function keys and keypad, change to Xterm R6.

###### run screen at startup

​       Simply do crontab -e in terminal     It will open in your selected editor, where you can type     @reboot <your command>     (make sure you give full path for all executables)     Save the file and exit

###### screen for multiple windows

  c-a A: rename window   c-a n: next window   c-a p: previous window   c-a '': show window list   c-a V: vertically split the current window into two regions. You need to create a new window or select an existing window in the new region before using it.   c-a S: horizentialy split the current window.   c-a tab: jump to the next region.   c-a [ ? xxx: search xxx on the current screen.   c-a [ u: cursor up   c-a [ d: cursor down   c-a [ b: screen uproll   c-a [ d: screen downroll   ctrl+a :number x # reorder a window [serverfault.com/questions/244294/how-to-re-order-windows-change-the-scroll-shortcut-and-modify-the-status-bar-c]   Ctrl-a : source ~/.screenrc # reload ~/.screenrc   exit # close current window.

  screen prevents some keys like F1, F2, ..., from being correctly recogniezd in vim. To patch: http://vim.wikia.com/wiki/GNU_Screen_integration

  Save session (windows)     # This method may not be able to survive reboot     screen -S abc # start a session "abc"     screen -R abc # resume the session.     screen -d -R abc # resume the session; shutdown session opened in other terminals.

```
  # This method may enable session to resume even after reboot.# however, this requires modified screen.# I have installed it but does know how to use yet.http://skoneka.github.io/screen-session/index.html
```

  Copy past problem.     You has a long string. The string is wrapped into multiple lines on the scree. You should not copy this string in a region. When you paste, the string will be regarded as multiple lines, connected by special charactors. You should copy the string in full window.

  Copy and paste     c-a [     press entry to start select, select a range, press entry to end select     c-a ]     ()

#### get/extract pdf meta data/auther

pdfinfo   Extract wrong data for many pdfs. 

cb2bib   Only GUI version. Hard to extract batch papers.

pdfssa4met    Need to get a Calais api, download pdf2xml. Complicated   Does not work     python headings.py z:\home\xling\base\download\afin2015\afin_2015\afin_2015_1_10_40012.pdf     Could not convert to XML. Are you sure you provided the name of a valid PDF?

#### ftp

Use openssh.  Openssh isself contains a FTP server. Tutorial: [http://stephen830.iteye.com/blog/2104480] Somehow, I have to change the folder owner to mysftp.  Otherwise, the FTP user can read files but cannot write.

### programming

[~/base/shared-among-computers/know_how/programming/programming.md]

#### Run ipython notebook in background

START /B ipython notebook

#### decompile: turn apk into source code in ubuntu

[https://ashwinishinde.wordpress.com/2014/03/21/how-to-extract-source-code-of-an-apk-using-ubuntu-system/]

```
Get source code  Use dex2jar on classes.dex to get "classes.dex.dex2jar.jar"  Use jd-gui on "classes.dex.dex2jar.jar" to get source codeGet manifest.xml  apktool if framework-res.apk  apktool d <your_app>.apk
```

#### Update from gcc4.6 to 4.8

[company.md]

#### bash:

  Goal: I try to avoid typing long commands like 'sudo apt-get install xxx'. Hope that bash popup a list for all history commands when I type apt-get .   Solutions:     fzf:        A grep-like search tool that can fuzzy search.        Install:         git clone https://github.com/junegunn/fzf.git ~/.fzf         ~/.fzf/install       Now, to popup history commands including apt-get,          ctr+r       autocomplate:         vim **<tab>     zsh

#### python

##### deal with special characters caused by encoding

I crawled a string text using scrapy.    bodies = response.xpath(targetSite.bodyXPath).extract() Here, bodies is an array of texts. The text seems contain charactores like \n, \u, \t, \xe2 and u201c. Need to remove them.

Method 1. simply    s = ''.join(bodies)   s contains no special characters.

  Seems that python purify the text for us by default.

Method 2.    def RemoveSpecialCharactors(self, text):     import re     assert text

```
  # remove \xe2 ... #text.replace('u\xa0', u' ') # work, but can only replace a certain charactor (e.g., \xa0) each time.#text = re.sub(u'\x..', '', text) # error: sre_constants.error: bogus escape: '\\x' [http://stackoverflow.com/questions/3328995/how-to-remove-xe2-from-a-list]text = re.sub(r'[\x90-\xff]', '', text) # [http://stackoverflow.com/questions/3328995/how-to-remove-xe2-from-a-list]# remove \n, \t, \utext = re.sub('\\n', '', text)text = re.sub('\\t', '', text)text = re.sub('\\u', '', text)# remove u201c, .. # [http://blog.jobbole.com/50345/]  text = text.encode('utf-8') return text
```

##### Draw figure in ipython notebook

import pandas as pd from pylab import * df = pd.DataFrame.from_csv('C : \results.csv', parse_dates=False) df.mtte_StCount.plot(color='g',lw=1.3) show()

##### Avoid the need of typing password during ssh by using public/private keys (http://cs.sru.edu/~zhou/linux/howto/sshpasswd.html)

ssh-keygen cd ~/.ssh cat id_rsa.pub | ssh xling@xx.xx.xx.xx 'cat - >> ~ / .ssh / authorized_keys'

##### the following pseudo-classes are implemented: nth-of-type

solution: replace nth-child with nth-of-type

#### test automation

##### Install sikuli

Make sure that Java 6 (instead of Java 7) is installed. If not, install oracal java 6 SE. Download and install http://www.sikuli.org/downloadrc3.html

##### selenium python install

(clg) "can not connect to the iedriver" (ctm) make all the four "protected mode" checkboxes in IE security settings panel checked.

#### bash

[/home4/xling/base/share/vim/vim73/bundle/neosnippet-snippets/neosnippets/vimshell.snip.md]

##### bash shortcuts (move cursor, delete) 

  Delete until previous punctuation mark in Bash     Esc + <back space>      # Some actions are not attached with a shortcut. You need to binding them.     # bind '"\C-p": shell-backward-kill-word'    ctrl+k: delete until the end of the current line

##### snippet autocomplate for shell

Install vim-conque  Run bash within vim using vim-conque Meanwhile, open a bash file. In it, get snippet using UltiSnips.  Copy the snippet to bash.

##### copy/paste in bash using keyboard without using mouse

  screen (enter)   c+a, ESC   (move the curse to the copy start point)   (enter)   (move the curse to the copy end point)   (enter)   (move the curse to the paste start point)   c+a, ]

Windows remote desktop, black screen   On laptop: ctrl + alt + Fn + End

##### Bash Keyboard Shortcuts

Moving the cursor:   Ctrl + a   Go to the beginning of the line (Home)   Ctrl + e   Go to the End of the line (End)   Ctrl + p   Previous command (Up arrow)   Ctrl + n   Next command (Down arrow)    Alt + b   Back (left) one word    Alt + f   Forward (right) one word   Ctrl + f   Forward one character   Ctrl + b   Backward one character   Ctrl + xx  Toggle between the start of line and current cursor position

##### Reduce keyboard typing

  vim utilization-500.txt weight-500.txt   ==>   nodes=500 && vim utilization-$nodes.txt weight-$nodes.txt

##### Move cursor under shell faster 

Bare Essentials   Use C-xC-e to open the current line in the editor specified by $FCEDIT or $EDITOR or emacs (tried in that order).   C-b Move back one character.   C-f Move forward one character.   [DEL] or [Backspace] Delete the character to the left of the cursor.   C-d Delete the character underneath the cursor.   C-_ or C-x C-u Undo the last editing command. You can undo all the way back to an empty line.

Movement   C-a Move to the start of the line.   C-e Move to the end of the line.   M-f Move forward a word, where a word is composed of letters and digits.   M-b Move backward a word.   C-l Clear the screen, reprinting the current line at the top.   Kill and yank   C-k Kill the text from the current cursor position to the end of the line.   M-d Kill from the cursor to the end of the current word, or, if between words, to the end of the next word. Word boundaries are the same as those used by M-f.   M-[DEL] Kill from the cursor the start of the current word, or, if between words, to the start of the previous word. Word boundaries are the same as those used by M-b.   C-w Kill from the cursor to the previous whitespace. This is different than M- because the word boundaries differ.   C-y Yank the most recently killed text back into the buffer at the cursor.   M-y Rotate the kill-ring, and yank the new top. You can only do this if the prior command is C-y or M-y.

///// comment_title [end] /////

#### run mutliple processes that may cause swap

I run muliple simulations. System memory can be used up and the machine hangs. Hope that OS suspend a part of processes when the memory is low, move the memory of  these processes into hard disk, finish left proccesses, load back suspended processes  and finish them. 

##### cryopid

CryoPID allows you to capture the state of a running process in Linux and save it to a file. This file can then be used to resume the process later on, either after a reboot or even on another machine. 

##### 

People are talking this problem but seems no tools can do this yet.

##### SLURM lists this as a furture work .

##### sun grid engine



#### padc-network does not show any message even the program crashes.

  When run in gdb, "segment fault" is shown.   When run in gdb but redirect result to file, no crash message is shown.   Other programs do show crash message ( "segment fault", e.g ) when crash.

## shell

### use virtual machine

#### knowledge

##### VM comparison

[東京エリアDebian勉強会 debootstrapを有効活用してみよう](http://pcdennokan.dip.jp/static/mypyapp2/files/debianmeetingresume201304-presentation-sugimoto.pdf)

xen and kvm have similar performance difference 

##### libvirt

libvirt is an open source API, daemon and management tool for managing platform virtualization. It can be used to manage KVM, Xen, VMware ESX, QEMU

##### QEMU

When working together, KVM arbitrates access to the CPU and memory, and QEMU emulates the hardware resources (hard disk, video, USB, etc.). When working alone, QEMU emulates both CPU and hardware.

##### kvm

KVM (for Kernel-based Virtual Machine) is a full virtualization solution for Linux on x86 hardware containing virtualization extensions (Intel VT or AMD-V). It consists of a loadable kernel module, kvm.ko, that provides the core virtualization infrastructure and a processor specific module, kvm-intel.ko or kvm-amd.ko.

#### kvm

##### Install kvm

(https://help.ubuntu.com/community/KVM/Installation)

Error: /dev/kvm does not exist Solution: sudo modprobe kvm_intel

Error:    Need to restart kvm    Run:     sudo modprobe -r kvm,    However, get error      rmmod: ERROR: Module kvm is in use by: kvm_intel Solution:         sudo modprobe -r kvm_intel # delete kvm_intel_«»     sudo modprobe -r kvm     sudo modprobe -a kvm # add kvm back     sudo modprobe -a kvm_intel # add kvm_intel back However, I get this problem again once I create /dev/kvm   sudo modprobe kvm_intel

##### run kvm to install vm

sudo apt-get install virt-manager

- Error:  Could not access KVM kernel module: Permission denied failed to initialize KVM: Permission denied

I tried to edit qemu.conf but does not work .

Solution: must restart kvm and kvm_intel (use modprobe -r and modprobe -a to kill and start kvm and kvm_intel)

##### 

sudo apt-get install qemu-system sudo apt-get install kvm qemu

##### 

##### (not work, cannot download os)

sudo apt install uvtool Download os image:   uvt-simplestreams-libvirt sync release=trusty arch=x86_64 uvt-simplestreams-libvirt query release=trusty arch=x86_64 label=release

##### setup network

 (does not work)

NAT   Host:      Get the address of virtual gateway       ifconfig -a         lo        Link encap:Local Loopback                   inet addr:127.0.0.1  Mask:255.0.0.0                   inet6 addr: ::1/128 Scope:Host                   UP LOOPBACK RUNNING  MTU:65536  Metric:1                   RX packets:1353157 errors:0 dropped:0 overruns:0 frame:0                   TX packets:1353157 errors:0 dropped:0 overruns:0 carrier:0                   collisions:0 txqueuelen:0                   RX bytes:1704100276 (1.7 GB)  TX bytes:1704100276 (1.7 GB)

```
    virbr0    Link encap:Ethernet  HWaddr 00:00:00:00:00:00              inet addr:192.168.122.1  Bcast:192.168.122.255  Mask:255.255.255.0              UP BROADCAST MULTICAST  MTU:1500  Metric:1              RX packets:0 errors:0 dropped:0 overruns:0 frame:0              TX packets:0 errors:0 dropped:0 overruns:0 carrier:0              collisions:0 txqueuelen:0              RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)    virbr1    Link encap:Ethernet  HWaddr 52:54:00:c1:2b:d6              inet addr:10.1.1.1  Bcast:10.1.1.255  Mask:255.255.255.0              UP BROADCAST MULTICAST  MTU:1500  Metric:1              RX packets:0 errors:0 dropped:0 overruns:0 frame:0              TX packets:0 errors:0 dropped:0 overruns:0 carrier:0              collisions:0 txqueuelen:0              RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)    Here, inet addr of virbr0 (192.168.122.1) is the gateway address. 
```

  Guest:     Set device model to 'NAT' in virt-manager

Bridge.   Host:      Create bridge       sudo vim /etc/network/interfaces       `auto br0         iface br0 inet dhcp         bridge_ports eth0         bridge_stp off         bridge_maxwait 1`       sudo /etc/init.d/networking restart   Guest:     Set device model to 'bridge' in virt-manager 

Common   Enable eth0 <#enable_eth0>     sudo vim /etc/network/interfaces     `auto eth0     iface eth0 inet static     address 192.168.122.8 # 192.168.122 should be equal the gateway     netmask 255.255.255.0     gateway 192.168.122.1`     sudo ifconfig eth0 up #     sudo /etc/init.d/networking restart   Get IP from DHCP server.   Set proxy:     export http_proxy="http://abc.cde.com"   Set DNS. Otherwise, abc.cde.com cannot be resolved.      Edit /etc/resolv.conf       `nameserver ...       nameserver ...       nameserver ...       search ...`

#### debootstrap

Con: only install a minimal version of os. Short of lots of software. Don't know how to install the software.

##### prepare

 // set environment variable sudo -E vim ~/.bashrc   cellularscheduler=/home4/xling/job/codes/cellularscheduler1

sudo apt-get update sudo apt-get install dchroot debootstrap

##### install os

mkdir $cellularscheduler sudo vim /etc/schroot/schroot.conf   [trusty]   description=Ubuntu trusty   location=/home4/xling/job/codes/cellularscheduler1   priority=3   users=demouser   groups=sbuild   root-groups=root

cd $cellularscheduler sudo debootstrap --variant=buildd trusty $cellularscheduler

##### mount

sudo -E vim /etc/fstab   // Do not use ev in fstab. The os may hang.   //proc $cellularscheduler/proc proc defaults 0 0   //sysfs $cellularscheduler/sys sysfs defaults 0 0   proc /home4/xling/job/codes/cellularscheduler1/proc proc defaults 0 0   sysfs /home4/xling/job/codes/cellularscheduler1/sys sysfs defaults 0 0

sudo mount proc $cellularscheduler/proc -t proc sudo mount sysfs $cellularscheduler/sys -t sysfs cp /etc/hosts $cellularscheduler/etc/hosts sudo chroot $cellularscheduler/ /bin/bash

##### install software

You need to install software again in the virtual environment (python, wget, ...)

- Install standard software  Go to https://repogen.simplylinux.ch/ to generate repository list of standard software vim /etc/apt/sources.list

[Install add-apt-repository](http://askubuntu.com/questions/38021/how-to-add-a-ppa-on-a-server)   PPAs are for non standard software/updates. They are generally used by people who want the latest and greatest. If you are going extra lengths to get this kind of software, then you are expected to know what you are doing. The selection in the Software Centre is ample for most human beings.

apt-get install software-properties-common

##### exist

exit sudo umount /test/proc sudo umount /test/sys

### autohotkey / ultisnip for bash

Tools to organize snippets. A list of plugins running shell in vim: 

#### from vim

##### vimshell

Error: vimproc_linux64 not found Solution: http://d.hatena.ne.jp/pospome/20140906

###### Error: vimshell + neocomplcache try to autocompete what is flushed on stdout.

Solution:  let g:vimshell_user_prompt = 'fnamemodify(getcwd(), ":~")'

###### autocomplete popup:

  Install neocomplcache 

###### autocomplete code snippets

  Install neosnippet .   create snippets in ~/.vim/bundle/neosnippet-snippets/neosnippets/vimshell.snip   In vimshell, use A-i to autocompelte (default is C-k, and I changed it)

###### History reverse search.

  Install vin unite.   C-l

###### con:

  still has problems when use pipe, e.g. the following line results in error.     find . -name 'out*' | xargs grep assert 

##### unite 

vimshell vs conque @ google Seems very powerful but I have not tried .

##### vim-conque

Try to run bash inside vim, so that I can get autocomplete bash command using UltiSnips. Use pathengon to install. Otherwise, the plugin cannot work.

Shortcut:   :ConqueTerm bash: open bash    <Esc>: switch between shell and vim

Con:   Slow. When you type fast, the shell may receive wrong commands.

#### boom (Boom)

sudo apt - get install rubygems sudo gem install boom boom master // create a list called ''master'' boom master meanu "cat RESULTS_1/Plot_PerTreeCreationEdge2Utilization.csv | awk ''{sum+=$6; ++n}END{print sum,n,sum/n}''" // create an entry called ''meanu'' under ''master'' boom meanu // copy meanu to copy board

You can edit the database at ~/.boom   {     "lists": [       {         "gifs": [           {             "shirt": "http://cl.ly/NwCS/shirt.gif"           }         ]       },       {         "test": [

```
    ]  }]
```

  }

##### pro

Can blur search: boom all | fzf

##### Con:

###### Cannot define long script name

###### Does not autocomplete script name. 

#### Other alternatives 

boom, Bang, sheet 

##### bang, 

Personal product. seems not stable. 

##### sheet, 

###### pro

Can search using fzf: sheet list | fzf

###### con

Save snippts in separated files. Cannot use Voom.

##### autohotkey, 

Cannot list all snippts and hence cannot use fzf No native linux implementation

##### c-R (reverse search)

Not permanent. History may get lost.

None is lots of users.

### useful commands

#### jump, shortcut, path:

  sudo apt-get install autojump # [https://github.com/joelthelion/autojump]



add into ~/.vimrc

  source path # path is the path to autojump.bash

#### path including blank space

  Work:    find . -name *.pdf | while read -r FILE; do pdfgrep network "$FILE"; done

  Does not work: ( space caused problems )     for path in `find . -name *.pdf`; do echo "aaa", $path; done     # output     aaa, ./research_paper_pdf/2.     aaa, cloud/1.     aaa, definition/others/08     aaa, Toward     aaa, a     aaa, Unified     aaa, Ontology     aaa, of     aaa, Cloud     aaa, Computing.pdf     aaa, ./network/icn/routing/CCN-based     aaa, Virtual     aaa, Private     aaa, Communityfor     aaa, Extended     aaa, Home     aaa, Media     aaa, Service.pdf

  Does not work (the results of `find` is shown in one line instead of mutiple lines)     %cl; OLDIFS="$IFS"; IFS=""; for path in `find . -name *.pdf`; do echo "aaa", $path; done; IFS=$OLDIFS

```
  #Results:aaa, ./research_paper_pdf/2. cloud/1. definition/others/08 Toward a Unified Ontology of Cloud Computing.pdf./network/icn/routing/CCN-based Virtual Private Communityfor Extended Home Media Service.pdf
```

#### linux, widecard

  List all matching files in the current folder     ls *.cc   List all matching files in the current folder, in the subdirectories     ls **/*.cc   List all matching files in the current folder, in the subdirectories, in the sub-sub-direcoties     ls **/**/*.cc

#### sort

##### sort sp2013.csv -r -n -o sp2013.csv

-r: reverse (high to low)  -n: sort by numbers instead of by strings  -o: destination

##### sort file names that contain numbers

  You have these files:     some.string_100_with_numbers.in-it.txt     some.string_101_with_numbers.in-it.txt     some.string_23_with_numbers.in-it.txt     some.string_24_with_numbers.in-it.txt

  run 'ls -lv'     some.string_23_with_numbers.in-it.txt     some.string_24_with_numbers.in-it.txt     some.string_100_with_numbers.in-it.txt     some.string_101_with_numbers.in-it.txt

#### cannot correctly print python results

  Prolem: I have a python script that generate four columns. However, it always only print out three columns.   Cause: I indeed used a awk to process the results generated by the python script.     ./plot_reserve.py --sfr_plain 'ConsumersPerNode FwRoutingTunneling'| awk -F $'\t' '{print $2,"\t", $1,"\t" $3}'

#### Use subproccess.Popen.stdout/stderr to redirect error into log file

  Problem:

#### Cannot access to samba after upgrading ubuntu

  Cause: ubuntu rewriten firewall policy.   Solution:     stop firewall (i.e., iptable)     restart samba.

#### Download redirection file

curl -Sso ./pathogen.vim https://raw.github.com/tpope/vim-pathogen/master/autoload/pathogen.vim

#### Print standard output and standard error into one file.

```
./abc > out 2>&1
```

  Print standard output and standard error into both screen and a file     ./abc 2>&1 | tee -a out

#### It is impossible to send mail from company remote servers (due to security restrictions? )

  Not possible to send mail using python + smtplib (security restriction?)   Maybe I need to create my own local sever client communication.

#### When the linux suddenly turns too slow.

  Use ''top'' command, these things may matters: VIRT, mem, RES, IO    When all the value looks normal, maybe you have used use all CPUs.

#### directory shortcut, linux

  autojump   Edit ~/.bashrc   source /usr/share/autojump/autojump.bash

#### process priority

Problem: my program may have bugy. it makes the system hang. Solution:    taskset 256 your-program-name   This forces your program run only on the first 8 cores.   It wont completely use up your resources.

  P.s.   You may want to simply adjust your program's priority.     nice -10 your-program-name   However, your code will still hang the system, even if your code only    use a part of cores, the system will still hang.

### Linux gnome shortcuts

  Install "Gnome Do" for searching application   Install "Synapse" for searching files   Use gnome-search-tool for searching files:   Enter compizConfig Settings Manager -> Commands   /usr/bin/gnome-search-tool: shift + alt + f   /usr/bin/gnome-open /home/ccnd/Documents/ndnSIM: ctrl + shift + alt + n

  Run application: alt + F2   Open console: ctrl + alt + t

  ctrl + up: maximize window   ctrl + down: minize window   ctrl + d: show desktop   alt + f2: open application   alt + f7: move current window using mouse

  cd -: return to the previous path (folder)

  Compiz Config Settings Manager   Chrome => /opt/google/chrome/google-chrome %U => shift + alt + b

  Move window     System setting -> keyboard -> windows

## batch rename

rename abc0001.png, abc0002.png, ... to 0001.png, 0002.png, ...

 rename -v 's/proposal-LowerHpcTccThreshold_do_UsePq-false_wholeTcc-0.8_//' *.png

_fefe

### periodically upgrade software

// write crontab using root account to avoid input password  // http://unix.stackexchange.com/questions/155044/choice-of-editor-when-running-under-sudo sudo env VISUAL=vim crontab -e  // add a line // -y: answer yes for apt-get directly  0 5 * * * (apt-get -y update && apt-get -y upgrade)

### replace incorrect global link mapping

// https://github.com/fex-team/fis/issues/83 sudo update-alternatives --install /usr/bin/node node /usr/bin/nodejs 10

## os

### install software

#### apt

##### apt-get install

  When apt-get install xxx, get error message like this:     libboost-all-dev : Depends: libboost-graph-parallel-dev but it is not going to be installed   Maybe you have some packege yyy not correctly installed.   Remove the package:     apt-get remove yyy

##### unmet dependencies

Problem:   "sudo apt-get build-dep mysql-workbench" fails:   The following packages have unmet dependencies:   libgl1-mesa-dev : Depends: libx11-xcb-dev but it is not going to be installed                    Depends: libxshmfence-dev but it is not going to be installed                    Depends: libxdamage-dev but it is not going to be installed                    Depends: libxfixes-dev but it is not going to be installed   libglib2.0-dev : Depends: libglib2.0-0 (= 2.40.2-0ubuntu1) but 2.42.0-2 is to be installed                   Depends: libglib2.0-bin (= 2.40.2-0ubuntu1) but 2.42.0-2 is to be installed   libgtkmm-2.4-dev : Depends: libgtkmm-2.4-1c2a (= 1:2.24.4-1ubuntu1) but 1:2.24.4-1.1 is to be installed                     Depends: libgtk2.0-dev (>= 2.24.0) but it is not going to be installed                     Depends: libglibmm-2.4-dev (>= 2.27.93) but it is not going to be installed                     Depends: libpangomm-1.4-dev (>= 2.27.1) but it is not going to be installed                     Depends: libatkmm-1.6-dev (>= 2.22.2) but it is not going to be installed   libpcre3-dev : Depends: libpcre3 (= 1:8.31-2ubuntu2.1) but 1:8.35-3ubuntu1 is to be installed   libpixman-1-dev : Depends: libpixman-1-0 (= 0.30.2-2ubuntu1) but 0.32.4-1ubuntu1 is to be installed   uuid-dev : Depends: libuuid1 (= 2.20.1-5.1ubuntu20.6) but 2.25.1-3ubuntu4 is to be installed   E: Build-dependencies for mysql-workbench could not be satisfied.

Cause:   Do you have a mixed /etc/apt/sources.list? It appears that you're trying to install one package from a newer repository but that it doesn't have access to a repository with the newer dependencies. 

Solution 

1. use 'sudo aptitude install package-name' install of 'sudo apt-get package-name'. aptitude is less likely to give up. 
2. Manually downgrade. (DON'T USE THIS APPROACH. IT IS RISKY!)
   
     [12:39:12 AM | Edited 12:39:20 AM] jamb0ss: have you installed all dependencies?
     [12:39:45 AM] Xu Ling: apt-get install libx11-xcb-dev
     [12:39:48 AM] Xu Ling: Like this?
     [12:40:39 AM] jamb0ss: all this libs, yes
     [12:40:43 AM] Xu Ling: As linux rejects to install them, I am afraid that manually installing them is risky.
     [12:40:45 AM] Xu Ling: No?
     [12:40:51 AM] jamb0ss: no
     [12:41:06 AM] jamb0ss: if some lib cannot be found by "sudo apt-get install NAME"
     [12:41:15 AM] jamb0ss: then you should use Synaptic manager
     [12:41:19 AM] jamb0ss: basically
     [12:41:27 AM] jamb0ss: you can use it to install all libs above
     [12:41:42 AM] jamb0ss: it has great search feature
     [12:41:58 AM] jamb0ss: when you don't know how exactly package is named
     [12:42:01 AM] jamb0ss: etc.
     [12:42:01 AM] jamb0ss: https://help.ubuntu.com/community/SynapticHowto
     [12:42:17 AM] jamb0ss: sudo apt-get install synaptic
     [12:42:24 AM] jamb0ss: System > Administration > "Synaptic Package Manager"
     [12:42:25 AM] jamb0ss: then start it in GUI
     [12:42:42 AM] jamb0ss: then paste there for example "libx11-xcb-dev" (in search)
     [12:43:16 AM] jamb0ss: then mark packages that match (not all though, you'll figure out)
     [12:43:21 AM] jamb0ss: and install them
     [12:43:38 AM] jamb0ss: sometimes it'll show similar packages (by common names) etc.
     [12:43:57 AM] jamb0ss: you'll find them unrealed and skip them, I'm sure
     [12:44:10 AM | Edited 12:44:21 AM] jamb0ss: but even if you install them too, there is no risk to break anything
     [12:44:14 AM] jamb0ss: don't worry about it
     [12:44:29 AM] jamb0ss: why Synaptic?
     [12:44:32 AM] jamb0ss: because
     [12:44:39 AM] jamb0ss: sudo apt-get install libx11-xcb-dev
     [12:44:51 AM] jamb0ss: can work and can not
     [12:45:13 AM] jamb0ss: I mean, not always "libx11-xcb-dev" will be the correct package name
     [12:45:28 AM] jamb0ss: and in this case Synaptic will help a lot
     [12:48:14 AM] Xu Ling: Ok.
     [12:48:32 AM] Xu Ling: Why linux dont install the packages?
     [12:49:21 AM] jamb0ss: linux is a core, it doesn't need this packages to work, it knows nothing about this packages
     [12:49:49 AM] jamb0ss: it's a question to ubuntu, not linux
     [12:50:09 AM] jamb0ss: but even ubuntu doesn't need this packages to work
     [12:50:21 AM] jamb0ss: so you should install them manually for your needs
     [12:50:43 AM] jamb0ss: why mysql-workbench has no "full installation"
     [12:50:49 AM] jamb0ss: is a good question
     [12:50:55 AM] jamb0ss: and I don't know why

##### E: Unable to correct problems, you have held broken packages.



#### install VNC

  xstartup (xfce4)   #!/bin/sh

  xrdb $HOME/.Xresources   xsetroot -solid grey   startxfce4 &

  xstartup (gnome)   #!/bin/sh



Uncomment the following two lines for normal desktop:

  unset SESSION_MANAGER



exec /etc/X11/xinit/xinitrc

  gnome-session --session=gnome-classic &

  [ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup   [ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources   xsetroot -solid grey   vncconfig -iconic &

  vncserver -geometry 1280x1024   vncserver -geometry 1280x800   vncserver -geometry 1024x768

#### after upgrading ubuntu, ndnsim cannot be built any more.

```
g++ says that "reference to 'uint64' is ambiguous". It shows that uint64 is defined mulitple times.
```

  Solution

#### reinstall ubuntu in virtualbox



### memory management

#### top command

Problem: after running a program, the 'used' fields show that lots of memory is used but indeed no process is running.  Cause: linux cached the results. The caches will be automatically released when you are short of memory. 

#### linux is slow

##### The system can be slow even if core dump is small.

###### whether swap in/out occurred?

[http://www.thegeekstuff.com/2011/07/iostat-vmstat-mpstat-examples/ Run    vmstat -w 1 If si (swap in) and so (swap out) is non zero, the system can be slow.

###### you can prohibit swap

  check os's aggressiveness in using swap     sysctl -a | grep swappiness

​     

  Temorarily stop swap     sudo bash -c 'echo 0 > /proc/sys/vm/swappiness'    Permanently stop swap     sudo bash -c 'echo 0 > /etc/sysctl.conf; sysctl -p'

\####$ core dump can be very large and crash your machine. Now I am using ulimit -c 0.  Only open core dump when need it.

##### you can limit the memory usage upper bound

  May try       ulimit -d 12345  // in kilobytes    Note that the os itself consumes memory (viewed by 'top' command).    Ensure that the sum of the system memory and memory set for ulimit -d is smaller than    the physical memory size.   Linux still swaps

  Seems that ulimit's setting is not permanent.   You need to set ulimit each time you run your code. 

  ulimit -c 0 -m 12000000; ./run_tcc.py -c -s   sudo bash -c 'echo 0 > /proc/sys/vm/swappiness'; ulimit -c 0 -m 12000000; ./run_tcc.py -c -s

However, I found that these cannot prevent swap from occurring.

##### Run a background process to suspend proccess when swap is likely to occur.

Con:   I though that when a memory is suspended, its memory is released. Indeed, it is not . Seems that you need to code your program to release memory when suspended.

Programming

Linux Access windows   In Windows,     Set the fold that you want share as "shared folder". Here, set the users that can access. Suppose that the user's ID is "NSL/0000011272737"   In Linux     Method 1: Access windows from linux by mount     //Create share from windows cmd:     net share data=d:\data /cache:no /remark="xxx"     //set the prevaligy via GUI     Right click d:\data -> "Properties" -> "Share" tab ->     "Advanced setting" -> "Grant access" -> "Add" ->     "NSL\0000011272737" -> "OK" -> add "full control" to the "0000011272737" user     //mount d:\data in linux     sudo mount -t cifs -o username=NSL/0000011272737,file_mode=0664,dir_mode=0775 //10.5 6.50.247/data /home/ccnd/Documents/win_storage

```
method 2: smbclient //10.56.50.247/data -U NSL/0000011272737* Bad. Although I can login windows, I cannot call windows applications. Do not know why
```

### network

#### basic network interface knowledge

 eth1: eth0，eth1，eth2……代表网卡一，网卡二，网卡三…… lo: 127.0.0.1，即localhost ppp0: 你進入internet的interface. 當你拔號後就會産生

# crossplatform

## machine learning

### save and recover model

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    saver = tf.train.Saver()
    for index in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if index % 100 == 0:
            print("index: %d" % index)
            path = saver.save(sess, "./mnist-model/model.ckpt", global_step=index) # , latest_filename="hello"

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # ckpt = tf.train.get_checkpoint_state("./mnist_model")
    ckpt = tf.train.get_checkpoint_state('./mnist-model')
    saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
    print(ckpt)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

```





```bash
>>> ckpt
model_checkpoint_path: "./mnist-model/model.ckpt-900"
all_model_checkpoint_paths: "./mnist-model/model.ckpt-500"
all_model_checkpoint_paths: "./mnist-model/model.ckpt-600"
all_model_checkpoint_paths: "./mnist-model/model.ckpt-700"
all_model_checkpoint_paths: "./mnist-model/model.ckpt-800"
all_model_checkpoint_paths: "./mnist-model/model.ckpt-900"
```

### tensorflow
```

```

#### debug
```bash
tensorboard --logdir ~/project_foo/model_bar_logdir --port 6006 --debugger_port 6064
```

```python
from tensorflow.python import debug as tf_debug
sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
sess.run(my_fetches)
```

#### tfdbg
```
from tensorflow.python import debug as tfdbg
mon_sess = tfdbg.LocalCLIDebugWrapperSession(mon_sess)
```

#### tensorboard
```python
import tensorflow as tf
a = tf.constant([1.0,2.0,3.0],name='input1')
b = tf.Variable(tf.random_uniform([3]),name='input2')
add = tf.add_n([a,b],name='addOP')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/",sess.graph)
    print(sess.run(add))
writer.close()
```

### graph neural network

### GATED GRAPH SEQUENCE NEURAL NETWORKS

背景：传统GNN的node编码循环很多次直到收敛，这使得node的编码与具体任务无关。

目标：把具体任务信息编码入node的embedding中

方案：在embedding node信息时，不是等到embedding收敛再截至，而是迭代固定次数。同时，根据任务给不同节点不同的标记（特征）。这样，embedding就会与具体任务相关。



### node2vec: Scalable Feature Learning for Networks

问题：在GNN领域，需要对节点的邻接信息编码。难以确定编码哪些邻接信息。比如，可以以BSF或DSF方式采样一个节点作为邻接节点。

提案：结合BSF和DSF



### RL

### install atari on windows 

Not work for me though.

 https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299 

```py
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

#### PG

##### Levin@UCB

![upload successful](/images/pasted-9.png)
![upload successful](/images/pasted-10.png)
![upload successful](/images/pasted-11.png)
![upload successful](/images/pasted-13.png)
![upload successful](/images/pasted-14.png)

![upload successful](/images/pasted-15.png)


##### David Silver
![upload successful](/images/pasted-2.png)

![upload successful](/images/pasted-3.png)

![upload successful](/images/pasted-4.png)

![upload successful](/images/pasted-5.png)

##### [Course](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/) 

![upload successful](/images/pasted-6.png)

![upload successful](/images/pasted-7.png)

##### 李宏毅

![upload successful](/images/pasted-8.png)

### check tensorflow version
python -c 'import tensorflow as tf; print(tf.__version__)'



### install tensorflow 

pip install tensorflow-gpu==1.14

### check tensorflow version

python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3

```bash
  # python 2 and 3, but seems to cause problems 
  # https://www.tigraine.at/2017/02/03/compiling-vim8-with-python-support-on-ubuntu
  # "Sorry, this command is disabled, the Python library could not be loaded"

  ./configure --with-features=huge --enable-python3interp --enable-pythoninterp --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu/ --with-python3-config-dir=/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/ --enable-multibyte --enable-cscope --prefix=/usr/local/vim/

  # only python 2
  ./configure --with-features=huge --enable-pythoninterp --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu/ --with-python3-config-dir=/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/ --enable-multibyte --enable-cscope --prefix=/usr/local/vim/

  # only python 3
  ./configure --with-features=huge --enable-python3interp --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu/ --with-python3-config-dir=/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/ --enable-multibyte --enable-cscope --prefix=/usr/local/vim/

  --with-features=huge：支持最大特性
  --enable-rubyinterp：打开对ruby编写的插件的支持
  --enable-pythoninterp：打开对python编写的插件的支持
  --enable-python3interp：打开对python3编写的插件的支持
  --enable-luainterp：打开对lua编写的插件的支持
  --enable-perlinterp：打开对perl编写的插件的支持
  --enable-multibyte：打开多字节支持，可以在Vim中输入中文
  --enable-cscope：打开对cscope的支持
  --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu/ 指定python 路径
  --with-python-config-dir=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/ 指定python3路径
```




## IT


### security

[](know_how/security.md)

### programming

[](programming/programming.md)()

#### use dropsync to backup folders outside of dropbox into dropbox. ###

#### print a code snip and keep the highlight ###
  Open the code using notepad++ (windows)

#### python ###

##### pip ####

###### get version of software installed by pip #####
pip freeze | grep keyword-of-the-software-to-query

###### update software #####
pip install -U Scrapy

### browser


#### Minic other browsers (firefox, chrome) as IE8 ###

* IE tab2 
cannot run script [](http://texags.com/forums/30/topics/1394996)

* user agent (Still cannot visit company system)
  * copy the kinro_system_button script to chrome / firefox. 
  * Confirm that your fake broswer and IE have the same user agent: http://tanimoto.to/PC_DIY/UserAgent.html

* Trixie 
Not opensourcel. Too risky to use.

#### chrome ###

##### shortcuts ####
g, then i: go to input box
o: collapse a conversation
p/n: previous / next message in the same conversation
enter: open the current message
j/k: previous / next conversion
?: shortcut list

##### manage tabs in tree structures ####
You can use chrome tab outliner to give name to a window. Each window manage a group of related tabs. By clicking the 'save & close all open windows' button, you can completely close all windows, including the outliner window, and reopen these windows the next time.

##### Use chrome tab outliner to group multiple tabs in to a separate window. ####
  Use WIN + x to switch chrome windows, where WIN is the windows key and x is the bar index of chrome 
  [](http://www.quora.com/Is-there-a-Chrome-and-Windows-7-shortcut-for-switching-between-Chrome-windows).

### develop web services


#### wordpress ###

##### created a new site B.com but always be redirected to A.com when access B.com. ####
Add in /var/www/newsite/wp_config.php 
define('WP_SITEURL','http://B.com');

#### host ###

##### hostgator ####

###### create multiple wordpress on hostgator using sub domain #####
(9:48 am) [Craig H.]() To have multiple Wordpress sites you just need to add an addon domain for each domain http://support.hostgator.com/articles/cpanel/how-do-i-create-and-remove-an-addon-domain
(9:48 am) [Craig H.]() Then go to QuickInstall and install Wordpress on each domain.

###### create multiple wordpress on hostgator using independent domain #####
http://support.hostgator.com/articles/hosting-guide/lets-get-started/dns-name-servers/how-to-change-godaddy-name-servers

#### apache server ###

##### windows ####

###### xampp #####

####### install ######

reset password: 
http://localhost/security/index.php

####### apache cannot be launched ######

######## vmware-hostd.exe occupied port 443 #######
1. change port in VMware Workstation edit > preferences shared VMs tab, disable, change port, enable

2. [](http://stackoverflow.com/questions/18300377/xampp-apache-error-apache-shutdown-unexpectedly)
Check who is using port 80
    netstat -abno | grep 80

Possiblely, 360, ....

3. DocumentRoot should point to exist directory [](http://stackoverflow.com/questions/18300377/xampp-apache-error-apache-shutdown-unexpectedly)

4. check Windows Event Viewer 
my computer -> right click -> manage -> manage computer -> system tool -> event viewer -> windows log -> system

5. Windows Event Viewer gives error: "Apache2.4 サービスは、サービス固有エラー ファンクションが間違っています。 で終了しました。"

Check httpd.conf's error: [](http://oshiete.goo.ne.jp/qa/2810109.html)
  httpd.exe -t

6. error: "RSA certificate configured for www.example.com:443 does NOT include an ID which matches the server name"
Most common reason: ports 80 and 443 are used
    in xampp/apache/conf/extra/httpd-ssl.conf, change listen 443 to aaaa
    in httpd.conf, change listen 80 to listen bbbb.

7. port should range in [1, 65535]

8. in httpd.conf, should be 
    Listen 192.168.0.1:9999
instead of 
    Listen 9999

9. reinstall xampp (sometime works)

###### install wordpress [](http://goo.gl/r9JAd8) #####

####### enable post via mails [](http://www.shoutmeloud.com/how-to-publish-wordpress-post-via-email.html) ######

######## use jetpack #######
http://www.wpexplorer.com/publish-wordpress-posts-email/
Note that jetpack does not work on local sites.

######## (does not work.) #######
Enter setting -> compose 
WP gives you some random strings like abc. 
Register a gmail abc@gmail.com. 
Write mails to abc@gmail.com and these mails will be submited to WP.
If you send from a mail that is not the WP's main mail, your post will be in the pending box: 
  http://localhost/wordpress/wp-admin/edit.php?post_status=pending&post_type=post 

However, seems that WP's post email address (abc@gmail.com) changes every time. So, this approach does not work.

#### office tool ###

##### vim ####

###### global #####
Problem: vim hangs. When open a file, the cursor flashes for one second and then vim does not response to any typing.
Observations:
  Hangs once open .sh files!

  When close pathogen in vimrc or remove ./vim/bundle, no hangs occurs.

  The problem occurs once I copy some string (bash script snaps, e.g). 
  The problem even occurs once vim open files contain these strings.

  The problem does not automatically disappear with time.
  Simply disabling and then enabling pathogen does not solve the problem.

Possible Solution:
  1. 

    move .vim/bundle to another place.
    remove the pathogen enabling code from ~/.vimrc
    open and then close vim once. 
    move bundle back to .vim
    recover pathogen.
  2. move some plugins out of bundle, use vim for a while, then move the plugins back to bundle.

    Even moving out YankRing.vim only solves the problem.

###### plugin #####

####### tab ######

######## CtrlSpace #######
CtrlSpace: open CtrlSpace control window
?: help
=: rename the current tab
tab: open the selected buffer and close the CtrlSpace window

w, (select workspace), ctrl-s: save to the selected WS
w, (select workspace), ctrl-l: load the selected WS
w, N, ctrl-s: close all opened buffers and save a new workspace.

l: show tab list
-: move tab up
+: move tab down

######### CtrlSpace conflict with vimgdb. ########
    Once CtrlSpace is installed, when call gdb, the cursor of the gdb console window is at the first line. However, the latest message is shown in the last line. When the console windown ask you things like "reload the program? Y/N", you cannot see it. You thought the gdb hungs.
    I tried to scroll down window when vim creates a new split but found no such command.
    The solution I use now is to temporarily disable CtrlSpace. Two ways to do it:
      1. move the CtrlSpace directroy out of bundle.
      2. set runtimepath-=~/.vim/bundle/vimacs
          [stackoverflow.com/questions/601412/how-to-turn-off-a-plugin-in-vim-temporarily]


######## flip / switch two splits (multiple rows/columns in the same tab) #######
    ctrl+w, ctrl+r [](http://stackoverflow.com/questions/6071266/how-to-flip-windows-in-vim)
    or
    [](http://stackoverflow.com/questions/2586984/how-can-i-swap-positions-of-two-open-files-in-splits-in-vim)

######## create table in vim #######
    Install vim-table-mode
    :TableModeEnable
    Enter the editing mode
    |: split
    ||: horizental line
    :TableModeDisable
    select lines and <leader>tt # create a table from the select lines

    Note:
      Seems cannot write long lines. You can give each lone line a ID attach the long line out of the table.
    
    |-------------+------+-------|
    | afefe       | fefe | affff |
    |-------------+------+-------|
    | feffefefefe | 12   | 32    |
    |-------------+------+-------|
    Note: each line starts with a '|'

######## Open multiple files into splits inside vim #######
  :args app/views/*.erb | all [](http://stackoverflow.com/questions/1272258/how-to-open-and-split-multiple-files)

######## group tabs #######
    Problem: I used to openning too many tabs. Need to group them.
    Approach:
      Open tab for a group of buffers.
      Use CtrlSpace plugin to name tabs.

####### Voom ######
  voom commands (for both single and multiple elements)
    aa: add a node
    dd: remove the current node

    <<: move left
    >>: move right
    ^^: move up
    __: move down
    
    o: unfold current node
    cc: fold current node 
    O: unfold current node recursively
    C: fold current node recursively


####### vim drawit ######
    start: \di
    stop: \ds

####### search ######

######## search content #######

######### search with ack ########
LAck makes results show in location list so that contents in quickfix window can be reserved.
LAck makes results show in location list so that contents in quickfix window can be reserved. 
After running LAck, vim pops up a split. It may not be the split of your search results. You need to close quickfix window and run lopen manually to see the search results.

######### Search and replace in all files in a tree [](http://vim.wikia.com/wiki/Search_and_replace_in_multiple_buffers) ########
    method 1: # cannot confirm for each replace
      find . -name "*.cc" -o -name "*.h" | xargs -0 sed -i 's/xxxxxxxx/yyyyyyyy/g' #tree_replace

    method 2: # very slow
      :args **/*.h # add all header files as arguments
      :argadd **/*.cc # add all source files as arguments
      :arg  Optional: Display the current arglist.
      :argdo %s/pattern/replace/ge | update Search and replace in all files in arglist.
       
    method 3: # very slow
      for file in \$(find . -name "*.cc" -o -name "*.h"); do vim -c '%s/xxxxxx/yyyyy/gc' -c 'wq' $file; done  #tree_replace


######## search file name #######

Search directory / files
  FuzzyFinder: popup a list of files in the current directory.
  FZF: fuzzy search. Find things in the current directory tree.
  CtrlP: havent use. Looks just like FuzzyFinder

Search inside the current file
  LogiPat, vim.sourceforge.net/scripts/script.php?script_id=1290
xxx_


####### programming ######

######## coding #######

######### YCM (ycm) ########

########## Install YCM on Windows #########
  Make sure that vim is 7.36+
  LLVM (indeed, only libclang is needed)
    Windows LLVM distribution [](http://llvm.org/builds/)
    Directly download libclang [](http://sourceforge.net/projects/clangonwin/ (not official))
  Download YCM
    https://github.com/xleng/YCM_WIN_X86

  Possible error: R6034 An application has made an attempt to load the C runtime library incorrectly. Please contact the application''s support team for more information
    Solution: http://stackoverflow.com/questions/9764341/runtime-error-with-vim-omnicompletion/10257098#10257098
      Note: close your vim.exe before step (3) (updating manifest)


########## Install YCM on ubuntu #########
http://www.html-js.com/article/1750

I keep receiving "user defined completion (^u^n^p) pattern not found " error.

* solution
It turns out that there are grammar error in my code.

* solution
Probably this line is problematic. 
  let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'
  * Try to remove this file.

* solution
When YCM stops working (showing messages "RuntimeError: Can''t jump to definition or decleration", "user defined completion (^U^N^P) Pattern not found")
  YCM does not show real time errors occurred in header files! YCM may recover by simply moving code from to headers to source files.
  Remove headers include ''ndnSIM'' (e.g., ns3/ndnSIM-module.h, ...) as much as possible.
  Add namespace to variable definitions.
  It may take two seconds for the completion list to pop up.
  In lines like this: 
    void Fun(A a, B b){...}
    AContainer ac;
    BContainer bc;
    Fun(ac.a, bc.b); //bc.b cannot be completed if ac.a contains errors.
  P.s The system contains multiple copies of ns3 headers. YouCompleteMe is using headers in 
    /usr/local/include/ns3-dev/ns3/. 
  Seems that this is set in 
    '~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'
  Add to the 'flags' array:
    '-isystem',
    '/home4/xling/job/codes/ns-3.24/build/ns3',


* solution
Disable all plugins except YCM to find the problem cause.
The possibility are neosnippet, neocompl*



########## Suddenly, YCM cannot recognize many notations. #########
  I found that it indeed manly cannot recognize std.
  I created a new .cc file, with simply code. It shows that YCM cannot find iostream.
  I recall that I updated gcc yesterday but failed, may be that is the problem.
  I reinstalled gcc (in utuntu, very simply http://ubuntuhandbook.org/index.php/2013/08/install-gcc-4-8-via-ppa-in-ubuntu-12-04-13-04/).
  YCM recovered.
########## YouCompleteMe does not complete template class well. This is a clang problem, which may take years to fix. Hence, avoid using template classes. #########

########## problem: dont know why but ycm cannot search header files in the current working directory any more. #########
YCM only search /usr/include.
solution: create a softlink of the working directory in /usr/include.

########## ValueError: Still no compileflags, no completions yet #########
Solution: copy a .ycm_ to the current directory or ~ and leave compilation_folder in .ycm_ empty. 

Reason:
Ycm needs .ycm_. It searches in the order of the current directory, ~, and .ycm_ specified in vimrc. 
If there is compilation_folder specified in .ycm_, ycm tries to use the compilation flags created in compilation_folder. If compilation_folder is empty, ycm uses clang to create the compilation database and flags. 
If your compilation_folder does not match your code, ycm cannot create the database / flags and fails.

########## RuntimeError: Can't jump to definition or declaration. #########
User defined completion (^U^N^P) Pattern not found

* Update the file 'ycm_extra_conf.py'
htt://stackoverflow.com/questions/30066100/youcompleteme-cant-autocomplete

* Enable ycm to correctly build the source code. 
  * how YCM works 
  ycm needs to read clang complation information (https://github.com/Valloric/YouCompleteMe/blob/master/doc/youcompleteme.txt)
  YCM creates a server that stores building context data. vim can ask the server information.
    * How does the the server get the building context data?
      ```
      YCM looks for a .ycm_extra_conf.py file in the directory of the opened file or in any directory above it in the hierarchy (recursively); when the file is found, it is loaded (only once!) as a Python module. YCM calls a FlagsForFile method in that module which should provide it with the information necessary to compile the current file. You can also provide a path to a global .ycm_extra_conf.py file, which will be used as a fallback.
      ```
      .ycm_extra_conf.py ==> 
      third_party/ycmd/ycmd/responses.py ==>
      ycmd/extra_conf_store.py ==> 
      ... => 
      third_party/ycmd/ycmd/completers/cpp/clang_completer.py

  When vim rums, YouCompleteMe/autoload runs the server by calling RestartServer from YouCompleteMe/python/ycm/youcompleteme.

  * Cause: 
  ns3 uses waf that does not generate this information.

  ~~
  * Solution:
[](https://waf.io/book/)
[](https://groups.google.com/forum/#!topic/waf-users/HRjeEmBJjHc)

  Download waf source code x.
  Enter x. 
  Run the following line to generate a waf binary that can generates compilation information.
    ./waf-light --make-waf --tools=clang_compilation_database
  Use x/waf to compile your ns3 code.
  * Evaluation: does not work. Still cannot autocomplete.
  ~~
  * Solution
  Looks like that ns3 has generated the compilition databse: 'build/compile_commands.json'
    [](http://network-simulator-ns-2.7690.n7.nabble.com/Bump-of-waf-version-td29937.html)

######### eclim ########
Problem: need to see the call graph of c++ code. 
Seems that eclim can import eclipse''s c++ plugin into vim.
download the eclim.jar file
java -jar eclim.jar
In bash, start the eclipse server and open  your workspace
  eclimd -Dosgi.instance.area.default=@/home4/xling/job/codes/eclipse/
In vim, 
  :ProjectCreate /path/to/my_cpp_project -n c++
  :ProjectList
  :CCallHierarchy (does not work)

And more commands: http://eclim.org/cheatsheet.html
Eclim does not support refactoring for all languages.
http://eclim.org/vim/refactoring.html#refactorundo

######### format code ########
  vim-clang-format: can wrap to 80 columns
    First, install clang-format 
      sudo apt-get install clang-format-3.5
    You system may have old clang-format that may not work. 
      Could not find checkout in any parent of the current path.
    You need to specify the location of the corrent clang-format
      /usr/bin/clang-format-3.5
  astyle: cannot wrap

  However, I feel that it has bug in processing macro. 
  Sometime it splits a macro without adding \ at the end of the previous line.
  Say, it split this line
    #define AAAA BBBB CCCC DDDD
  to 
    #define AAAA \
              BBBB CCCC 
              DDDD
  which cannot be built.
  Further more, it automatically removes things you add. So, you cannot manually add \ at the line end. 
  I have to stop using the vim-clang-format plugin.

########## automatically wrap lines longer than 80. #########
    # method 1. Lines are wrapped.
    :set textwidth=80
    :set formatoptions+=t
    #Problem: c++ strings are cut in the middle, which causes bugs.

    # Method2. Select some lines, press gq, and these lines will be wrapped.
    :set textwidth=80
    :set formatoptions+=w
    # Problem: cannot automatically wrap. Mutliples lines may be connected, which
    # may cause bugs in codes.

########## format c++ code #########
    You can use astyle command
      apt-get install astyle
      See ~/.vimrc for detail.
    Web says that you can use Google Cpplint style in astyle using '--style=google' [](http://stackoverflow.com/questions/2506776/is-it-possible-to-format-c-code-with-vim), but I cannot find such a switch in astyle.

  In files using ascii file encoding, astyle may write wrong charactors. 
  You need to convert file to utf-8:
    set bomb | set fileencoding=utf-8 | w

######## build #######
  vim-dispatch: run commands in asynchronously with your current program.

    // run in background
    :Make!      //build the code in background. 
    :Dispatch! cppcheck %  // run cppcheck in the background.
    :Copen:     //check the results of the previous command.
    
    // run in foreground
    :Make      //build the code in forground (only support tmux) 
    :Dispatch cppcheck %  // run cppcheck in the foreground.

######## debug #######

######## memory, double free #######
######### The frame reported by gdb may be incorrect. ########
  For example, this code crashed.
  a.x();
 // crashes
  a.y();

  gdb::bt shows that the crash occurs before a.y(). 
  However, the crash may indeed occur within a.y().

######### return vector<Ptr<T> > may cause double free ########
It seems that this code may cause double free:
class A{
public:
vector<Ptr<T> > fun(){
  return m_X;
}
private:
vector<Ptr<T> > m_X;
};

Indeed, Ptr<> is smart pointer. Shouldnt cause double free. 
Why...

######### How to detect double free ########
valgrind
  I dont know how to use. 
  valgrind crashes before it detects the bug. 

gdb
  GDB can reveal the name of the object that cause the crash.
  It also largely tells where the bug is.
  These info is largely sufficient to detect the bug.

####### Search command history ######
    backward: c-r
    forward: c-s

####### pathogen, ubuntu : pathogen does not get corrected loaded. ######
  Solution: use vundle instead of pathogen

  Other failed trials
    (hyp) pathogen.vim has these lines:
      if exists("g:loaded_pathogen") || &cp
        finish
      endif
      let g:loaded_pathogen = 1    

    cp is true means that vim is compatible with vi and more plugins should be disabled. 
    cp should be false but somehow it is true.
    Hence, pathogen is not loaded. 
    
    (eva) However, even if I ignore cp:
      if exists("g:loaded_pathogen")
        finish
      endif
      let g:loaded_pathogen = 1    
    it seems that pathogen still cannot be loaded.
    
    (hyp) Reinstall vim does not solve the problem. 
    (hyp) Not likely to be the problem of vim7.4.52 neither.
    
    (hyp) Removed and recreated bundle directory. 
    (eva) Does not work.
    
    (hyp)
      pathogen#infect() should be run after 'runtime! debian.vim'. 
      'runtime! debian.vim' sets variables like compatible
      'runtime! debian.vim' should have been called in /etc/vim/vimrc (before ~/.vimrc is loaded), but it is not for unknow reasons.
    (eva)
      I called call 'runtime! debian.vim' before "execute pathogen#infect()" in ~/.vimrc
      Now pathogen itself can be loaded, but pathogen failed to load other plugins

####### CtrlSpace ######
  :CtrlSpace : open control pannel
  =: rename tab 
  You need to save your working space. SaveSession of vim-session is not sufficient because it cannot save tab names you assigned. 
  w: open workspace list 

####### In NERDTree, open files in new tabs. ######
  Use NERDTreeTabs instead of NERDTree. 
    NERDTree opens one tree window for each tab. 
    NERDTreeTabs maintains a single tree window for all tabs. 
  When you need to open a file in a new tab, open a new tab, select the file from the tree window and press ENTER.

####### vundle ######
Problem: some plugins cannot be installed 
Solution: add this line
  call vundle#config#require(g:bundles) " otherwise some plugins cannot be loaded on windows [](https://github.com/gmarik/Vundle.vim/wiki#win2)

####### Installed YCM on windows, get R6034 error ######
  According to [1](), I need mt.exe to fix this problem. 
  But I do not have mt.exe
  Seems that I need to install windows sdk.
  I failed to install windows sdk 7.1. The reason may be that I installed Microsoft Visual C++ 2010, which is not compatible to sdk 7.1 [2,3]. 
  By installing sdk8.1, I get windows sdk and mt.exe installed.
  [1] http://stackoverflow.com/questions/9764341/runtime-error-with-vim-omnicompletion
  [2] http://support.microsoft.com/kb/2934068/zh-tw
  [3] http://www.cnblogs.com/sadgoblin/p/3327069.html
  []()

####### when use CtrlSpace, each tab have multiple windows. Directories of files opened in different tabs may be different. When you open direcotry browser like NERDTree, confussing occurs. ######
  Solution: 
    Set directory for each tab using plugin tcd.vim [https://github.com/vim-scripts/tcd.vim]
    :Tcd /abc/edf/

###### native-vim #####

####### single file ######


######## append selection into another (currently opened) file #######
[](http://stackoverflow.com/questions/9160570/append-or-prepend-selected-text-to-a-file-in-vim)
Select the content. 
  :silent '<,'>w! >> file-to-write.
  % without 'silent!', you cannot write opened file
In file-to-write, run 
  :edit
to reload. 

######## Open_file_under_cursor #######
[](http://vim.wikia.com/wiki/Open_file_under_cursor)
gf: open the path (e.g., "/abc/efg/hij") under the cursor. gf = "go file"
ctrl-w gf: open the path (e.g., "/abc/efg/hij") under the cursor in a new tab

######## insert the content behind a url #######
:r http://stackoverflow.com/questions/10505904/seeking-a-vim-function-to-insert-the-text-behind-a-hyperlink <CR>


######## lopen: open location list #######
cw: open quickfix

####### regular expression ######

######## extract paper titles from the reference section of papers. #######
Reference examples:
  * (E1) title has surrounding mark
      [2] J. Becker and H. Chen. "Measuring privacy risk in online social networks". In Proceedings of Web 2.0 Security and Privacy Workshop (W2SP’09), Oakland, CA, 2009.  
  * (E2) title has no surrounding mark
      [2] J. Becker and H. Chen. Measuring privacy risk in online social networks. In Proceedings of Web 2.0 Security and Privacy Workshop (W2SP’09), Oakland, CA, 2009.  
  * (E3) Index does not have '[]'
      102. Li, H.; Tan, J. Body Sensor Network Based Context Aware QRS Detection. In Proceedings of
      Pervasive Health Conference and Workshops, Innsbruck, Austria, 29 November–1 December
      2006; pp. 1-8.
      103. Ameen, M.A.; Nessa, A.; Kwak, K.S. QoS Issues with Focus on Wireless Body Area Networks.
      In Proceedings of 2008 Third International Conference on Convergence and Hybrid Information
      Technology, Busan, Korea, 11–13 November 2008; pp. 801-807. 

* split lines by '['
  :%s/\[/^M[/g
  []()

* For E1
  :%s/.\{-}“\(.*\)”.*/\1/g

* For E2
  :%s/.\{-}\(\([a-zA-Z-]\{1,}[ \.-\,]\)\{4,}\).*/\1/g

* remove lines containing no []
  g!/\[/d

* 
  * First, add a '!' in before each paper.
    %s/\(^[0-9]\{-}\.*\)/!^M\1/g
  * Merge all lines into one line: 'vGJ'
  * Split lines: 
    %s/!/
/g
  

Index list will be generated:
[1]
[2,3]
[11,12]

From the paper copy the indexed references, paste into vim, and run
  :1,$join | %s/\[/^M\[/g

Here, 
  :1,$join : join all lines
  :%s/\[/^M[/g : split lines by '['

######## lazy matching #######
[](http://stackoverflow.com/questions/1305853/how-can-i-make-my-match-non-greedy-in-vim)
      /^\[.\{-}\]

######## different applications have different regular expression #######
    In vim, lazy search is .\{-} [](http://stackoverflow.com/questions/1305853/how-can-i-make-my-match-non-greedy-in-vim)
      Instead of .* use .\{-}.
      %s/style=".\{-}"//g
    In other places, lazy search can be .+? [](http://stackoverflow.com/questions/2301285/what-do-lazy-and-greedy-mean-in-the-context-of-regular-expressions)

######### negative matching ########
   I want to find grammar errors in this example article:
    -------------------------
    (A) Figure \ref{tree1} and \ref{tree2} are trees.
    (B) Figure \ref{tree1} is tree. Second sentence: x and y are trees. 
    -------------------------

  Here, A is wrong and B is correct. 
  In A, 'Figure' should be 'Figures'
  I need to match A not B.

  The following search match A but also match B
    /Figure.\{-}and.*
  Maybe should use no greedy? 
  I try to use '.' to separate sentences. 
  But still match B:
    /Figure.\{-}and.\{-}\.
  The problem is that the second .\{-} matched '\.\
  This match A and does not match B. Since '.' is not included in [], RE searches within each sentence.
    /Figure[ {}\0-9a-zA-Z\-_]*

  However, writing [ {}\0-9a-zA-Z\-_] is error prone, I try to use negative matching. 
  First, some small tests.
    This works:
        /Figure\(.\)* 

    Work. Find lines not including .*tree
      /^\(.*tree\)\@!

  Then, try real work:
  In the following search, only the first charactor of the line is matched. 
    /^\(Figure.*\)\&\(.*tree\)\@!

  Need more investigation.

######## remove duplicated lines (match multiple lines) [](http://vim.wikia.com/wiki/Uniq_-_Removing_duplicate_lines) #######
  g/\%(^\1\n\)\@<=\(.*\)$/d
  g/                     /d  <-- Delete the lines matching the regexp
              \@<=           <-- If the bit following matches, make sure the bit preceding this symbol directly precedes the match
                  \(.*\)$    <-- Match the line into subst register 1
    \%(     \)               <-- Group without placing in a subst register.
       ^\1\n                 <-- Match subst register 1 followed the new line between the 2 lines

####### match English words ######
I need to extract paper titles from reference.
  [1] W. Grimson, D. Berry, J. Grimson, G. Stephens, E. Felton, P.  Given, R. O’Moore, Federated healthcare record server—the Synapses paradigm, Int. J. Med.  Inform. 52 (1998) 3–27.
  [2] J. Grimson, W. Grimson, D. Berry, G. Stephens, E. Felton, D.  Kalra, P. Toussaint, O.W. Weier, A CORBA-based integration of distributed electronic       healthcare records using the synapses approach, IEEE Trans. Inf. Technol. Biomed. 2 (1998) 124–138.
  [3] W. Grimson, B. Jung, E.M. van Mulligen, A.M. van Ginneken, S. Pardon, P.A. Sottile, Extensions to the HISA standard—the SynEx computing environment,     Methods Inf. Med. 41 (2002) 401–410.  [4] Harmonisation for the security of web technologies and applications (last accessed 2008/02/18); Available from: http://www.telecom.ntua.gr/(HARP/HARP/   INSIDE/Inside.htm. 

This RE expression works:
  %s/.\{-}\(\([a-zA-Z-—]\{1,}[, :]\)\{3,}\).*/\1/g

Explanation:
  I need to extract English words
  Seems there is not way to directly match whole words.
  So, I use [a-zA-Z-—]\{1,} to match a word. 
  I regard a title as more than 3 words connected by spaces or comma.
  So, (\([a-zA-Z-—]\{1,}[, :]\)\{3,}\)

####### copy a string in vim (via putty) and paste into bash ######
    It seems that windows has multiple copyboard. You can paste contents in copyboard A (B) using ctrl+v ( ctrl+<insert> ).
    In vim, select the string, type '"+y'.
    Now the string is in windows copyboard B. Paste the string somewhere in windows using ctrl + <insert>.
    Select the pasted content and copy again using ctrl+x.
    Paste the copyed string in bash using ctrl+<insert>.

####### vim scripting ######

######## optional arguments (arbitary number of arguments) #######
    [~/.vim/bundle/UltiSnips-2.2/UltiSnips/cpp.snippets]

######## string function: #######
  A=~=B: decide whether string A contains string B.
  Example:
    echo '/home4/xling/job/codes/padc2/' =~ 'padc2' # return 1
    echo '/home4/xling/job/codes/padc2/' =~ '1padc2' # return 0

######## Create function #######
You can use exe to call vim commands in your functions.
%  function! ChangeDirectoryToWorkingDirectory()
%    if getcwd()=~'job/codes/padc2'
%      exe 'cd' . $padcd
%    endif
%  endfunction

####### others ######

######## run multiple command in one line #######
  :cmd1 | cmd2
  When define shortcut, you need to use <bar> instead of |
  nmap <c-e> :clang-format % <bar> edit <cr>

######## Get error "ValueError: No sematic completer exists for filetype: ['conf']". #######
  Cause: I am editing a file wscript. This file has no .py extension and hence YCM cannot recoganize it.

######## cursor jumps to the top of the current file when I undo the previous action (Action1). #######
    Possible reason:
      I installed astyle. astyle rewrites the whole file (Action2) each time I save.
      When I undo, I first undo Action2 - hence the cursor moves.

######## folding does not work any more suddenly. #######
    reason: some unknown txt files are mistakenly put in .vim/bundle, making vim working incorrectly.

######## vim hangs from time to time. #######
    Not the problem of history size, folding, gnu_screen.
    May because I removed vim_nox yesterday?
    When I remove vimrc, vim does not hangs any more.
    Completely removed viminfo, problem still exists.
    remove vim-ruby
    Some random txt file is thrown in to ~/.vim/bundle. After removing the file, problem still exists.
    Pathogen conflicts with vundle? No.

######## vim history #######
    history is saved in ./viminfo.
    history only updates when the a file is closed.
    history is saved by catogory. Each catogary only store a fixed number of histories.


######## Tried to automatically include header (.h) files. #######
      vim-cpp-auto-include [](http://www.vim.org/scripts/script.php?script_id=4030)
        Require to run :ruby in vim. Does not know how to do.
        Very few users. May not stable.
    Tried to automatically add code guard/gate to source codes.
      vim-headerguard [](https://github.com/drmikehenry/vim-headerguard/tree/master/plugin)
        The generated guard does not obey the Google Cpplint style.
        I do not want to modify and maintain this plugin.



    ## Problem: cannot copy things into + register: "nothing in register +"
    Solution: simply restart vim.

######## Process big file ( xx Gb ) [](http://stackoverflow.com/questions/908575/how-to-edit-multi-gigabyte-text-files-vim-doesnt-work) #######
    '96%' # jump to 96% of the file.
    xling@nodez: split -l 10000 # split the big file into smaller ones

    ## Problem: When create a macro in vim, when you type some function name
    func_ABC, YCM will try to autocomplete the function name. Somehow YCM will
    search in a lots of folders, showing message like "scanning files ....",
    which makes vim hung up. 
       
    Solution:
      1. copy func_ABC into another register.  Instead of typing func_ABC, simply paste it.
      2. :set complete-=i [](http://stackoverflow.com/questions/915152/do-not-include-required-files-into-vim-omnicompletion )

######## Reload (update, syncrhonize) the current file #######
    :edit!

######## Using in putty, macro may not work. #######
    I defined  such a vim macro. It copy the word currently under the cursor, move to the line above the current line, and paste the word.
      wyO^[p   # ^[ means escape
    # This macro works on my local pc, but does not work on putty for remote login.
    As a solution, insted of using 'p', using '1p'. Namely,
      wyO^[1p

    This is not the problem of gnu screen. It occurs on vim independently.

######## special charactor list [](http://vimdoc.sourceforge.net/htmldoc/digraph.html#digraph-table) #######

######## Problem: vim complain "E492: Not an editor command" when I call :ack. #######
    Solution: it should be :Ack instead of :ack

######## write read-only file #######
    There seems to be some different approachs, depending on your current problem: [](http://superuser.com/questions/694450/using-vim-to-force-edit-a-file-when-you-opened-without-permissions)
      """
      Readonly by vi. If you file has :set readonly you can
      Use :w! to force write, or
      Issue :set noreadonly and then just use normal :w
      A permission problem (sudo): you can't write but you have sudo rights.
      Issue: :w !sudo tee %. This will write the buffer to tee, a command that receives pipe information and can write to files. And as tee is run with sudo powers, tee can modify the file.
      """


######## installation path #######
    Windows
      Exe: D:\Program Files\vim
      Bundle: ~\vimfiles\bundle
    Linux
      All: ~/base/share/vim

######## Normally, vim + putty does not accept alt key and your alt mapping does not work. Here is a solution [ Alt key shortcuts not working on gnome terminal with Vim @ stackoverflow ] #######

####### search ######
  vimgrep 
    con:
      slow [https://sites.google.com/site/fudist/Home/vim-nihongo-ban/vim-grep#TOC-grep-]
  ack
    pro:
      can exclude directories
    usage:
      :Ack string_to_search
    
  external grep 
    con:
      cannot exclude directories
  easiGrep:
    pro:
      can mutliple replace
    con:
      do not work correctly when the working directory is different from the directory of the current buffer.

##### latex ####
something

###### bibtex #####

####### generate rtf bibtex ######
Use this tool: [Bibtex Entries for IETF RFCs and Internet-Drafts](notesofaprogrammer.blogspot.jp/2014/11/bibtex-entries-for-ietf-rfcs-and.html)

###### create slide (beamer) #####

####### combine the advantage of beamer and powerpoint ######
Problem:
  To create slide, you can use latex beamer or ppt. 
  beamer:
    cannot freely add figures. 
  ppt:
    bad at reference management.
    cannot draw sequence charts using script.

Solution:
  Use ppt to create figures and save to a fixed location. 
  Beamer reads from that location.

####### add figures to the slide. ######

###### drawing, show figures in single-column mode and text in two-column mode. #####
Method: {figure*} + {minipage}
%  \documentclass{IEEEtran}
%  \usepackage{algpseudocode}% http://ctan.org/pkg/algorithmicx
%  \usepackage{lipsum}% http://ctan.org/pkg/lipsum
%  \usepackage{float}% http://ctan.org/pkg/float
%  \floatstyle{boxed} % Box...
%  \restylefloat{figure}% ...figure environment contents.
%  \begin{document}
%  ## A section
%  \lipsum[1]% dummy text
%  \begin{figure*}
%    \centering
%    \caption{Euclid’s algorithm}\label{euclid}
%    \begin{minipage}{0.5\columnwidth}
%    \begin{algorithmic}[1]
%      \Procedure{Euclid}{$a,b$}
%        \State $r\gets a\bmod b$
%        \While{$r\not=0$}
%          \State $a\gets b$
%          \State $b\gets r$
%          \State $r\gets a\bmod b$
%        \EndWhile\label{euclidendwhile}
%        \State \textbf{return} $b$
%      \EndProcedure
%    \end{algorithmic}
%    \end{minipage}
%  \end{figure*}
%  \lipsum[1]% dummy text
%  \end{document}


Method: [](http://tex.stackexchange.com/questions/34063/figure-span-to-one-column-on-double-column-page)
  \begin{figure*}
  ...
  \end{figure*}

  Con:
    Cannot work for {algorithmc} environment. The algorithm always sits in one column and cannot be central aligned.

Method:
  Using package multicols
  Example:
    \begin{multicols}[3]
      something_blabla
    \end{end}
  Con:
    If your page are currently using two columns. It can further split something_blabla into 3 columns. However, it cannot make your page one-single column.
    You can specify the column number of each part of the article. However, that is too much work.

Method:
  \onecolumn and \twocolumn commands [](http://tex.stackexchange.com/questions/88387/disable-two-column-mode-for-separate-part)
  Con: The two commands will start a new page.

Method:
  minipage
  con:
  The minipage overlap with one of the column. [](http://tex.stackexchange.com/questions/33307/how-to-prevent-two-column-text-to-overlay-with-minipage-of-entire-pagewidth)

  \subsubsection{Resize the figure to \textwidth}
    1. Bascially you do not need to adjust width. Just do not set horizental space yourself. For example, in this command,
      \newinst[distance]{variable}{InstanceName},
      do not set the distance argument.

    2. Resizebox {\columnwidth} {!} {} [](http://tex.stackexchange.com/questions/75449/specifying-the-width-and-height-of-a-tikzpicture)
      con: does not work
    
      \subsubsection{You can drow flowchar, sequency chart using plantuml or tikz}
  plantuml:
    con (examples: [/home4/xling/base/documents/scripts/programing/crossplatform/plantuml])
      Flowchart is not clever. Too small boxes/fonts. Too colorful.
      Sequence chart: comment cannot breakline.
      Cannot ref internal elements of figures.

  tikz:
    con:
      Grammar is too low-layer. Need to specific the position of each step block. Need to specify the line direction.
      It generates a pdf contains multiple figures. If you try to use a single figure, you need extra process to export figures.
    pro:
      can give label to each element of figures.

  MetaUML
    pro:
      export to latex.
      script
    con:
      does not support flowchart

  Dia edito / umlet
    pro:
      wysiwyg editor, easier than tikz. Good for compoistion diagrams
      export to latex
    con:
      Only support python 2.3 script.
      So, bascially cannot write script.
      Does not export tikz code.

  Use plantuml to generate pdf. Convert pdf to latex.
    Inkscape
      sudo apt-get install inkscape

  Graphviz
    pro:
      can programming
      can convert to latex using plugin dot2tex.
    con
      does not understand pseudo code. Still need you to assign box position. [](https://fsteeg.wordpress.com/2006/11/16/uml-activity-diagrams-with-graphviz/)

  inkscape (a file reader)
    Usage
      Import pdf
      To convert to tikz, you need to install an extension https://code.google.com/p/inkscape2tikz/
      Export -> export to tikz
    con:
      Does not work (?)
      No script (?)

  eps2pgf [](http://sourceforge.net/projects/eps2pgf/)
    con: convert eps to pgf, but not tikz.

  algoflow [](https://github.com/c-fos/algoflow), a opensource script to convert psedocode to tikz code.
    con: dont waste time on it. Not finished.

  Seems no good tool exist. Try to avoid draw diagram. Just write algorithm.

###### inverse search - jump to latex source code when double click pdf #####
http://leequangang.github.io/tech/2014/06/04/InverseSearch.html

###### latex pdf difference #####
//http://stackoverflow.com/questions/888414/git-checkout-older-revision-of-a-file-under-a-new-name
git show HEAD~1:manuscript.tex > manuscript-old.tex && latexdiff manuscript-old.tex manuscript.tex > diff.tex && platexr_dvi_ps_pdf.sh diff

##### markdown ####

###### use nodeppt to create slides. #####

// install nodejs
sudo apt-get install npm
// install nodeppt
npm install -g nodeppt
// fix some bug
// Error: EACCES, mkdir '/usr/local/lib/node_modules'
ln -s /usr/bin/nodejs /usr/bin/node
xx_


// enter path/for/ppts
nodeppt create ppt-name

// enable to view the slide in browsers
nodeppt start -p port -d path-to-the-slide-directory
  port: normally 8080

// access the slide
// in chrome, you may need to disable other markdown plugins
http://10.56.48.223:8080/


####### make headings left align ######

In nodeppt, headings like follows will be center aligned by default.
  # 
  ## 
The reason is that nodeppt has css in /css/nodeppt.css that centralizes 
headings.

I use headings a lot and need to align headings on left. 
To that end, I need to overwrite nodeppt's default css [](https://github.com/ksky521/nodePPT)
  感觉默认的模板不符合新意？可以支持自定义模板，查看theme.moon

  自定义后的模板路径在markdown的设置里填写：
  title: 这是演讲的题目
  speaker: 演讲者名字
  url: 可以设置链接
  transition: 转场效果，例如：zoomin/cards/slide
  files: /css/theme.moon.css

I need to add a css file, say xling.css, in /css. Use !important 
to overwrite.
  .centered {
      text-align: left !important;
  }

  .slides {
      text-align: left !important;
  }
  slides > slide .slide-wrapper {
      max-width: $slide-width;
      // height: $slide-height;
      text-align: left !important;
  }

In your slide markdown file, change 
  files: /css/theme.moon.css
to 
  files: /css/theme.moon.css, xling.css

#### host for web services ###

##### aws ####
* Pro:
you want to build distributed system like when your main instance is running backend and database is stored on aws rds cluster or so
* Con:
If you want to keep all things in one place when your entire app and backend and database are on the same server then I'd recommend not to use AWS

Also, if you app use a lot of CPU time you'll find that the perfomance will be down in some time on AWS EC2 instance that's how they work

##### cloud solution ####
* Pro: easy to use click-and-go
https://www.digitalocean.com/ or https://www.linode.com/.
SSD cloud hosting

##### fully dedicated server ####
* Pro: 
Get more powerful server with the lower price than on any cloud platform.
Also since you don't share its resources with anybody (it's a real server, real box, your box) you'll get a greater perfomance in most of cases
I use this hosting provider for dedicated servers: 
  https://www.soyoustart.com/
* Con: 
Administration tasks in case of dedicated server usually a little bit harder. 
I mean, in case of AWS or digitalocean you can bind you domain just in web adminl panel in two clicks.
In case of dedicated servers you may need to configure such things manually.
But in case of soyoustart.com, a task to add your domain also can be done by using their (poor) web admin panel though
Take a look at https://forum.ovh.us/showthread.php/1187-How-to-setup-DNS-for-soyoustart-dedicated-server

If you want to get a server in some particular region (country), then you should check offers and find out which provider has datacenters there.

#### amazon asw ###
jump to directory, directory shortcut
  autojump

Amazon AWS
  [moved to key database]()

Cannot use tab to auto compete in VNC xfce: remove the tab keyboard shortcut mapping (http://blog.163.com/thinki_cao/blog/static/83944875201303081111436/)

  //aws ec2 start-instances --instance-ids i-fe4b06ca

  [Login VNC]()
    Tutorial: https://www.youtube.com/watch?v=WeIw4CjwQ44
    Adjust resolution: vncserver -geometry 1920x1080

awk
  Get the second column separated by TAB: cat xxx.txt | awk -F $'\t' '{print $2}'

gnuplot
  plot './RESULTS_0/Plot_Dropped.txt' u 1:10 t "sts=1,rsv=1" w l, './RESULTS_1/Plot_Dropped.txt' u 1:10 t "sts=0,rsv=1" w l, './RESULTS_2/Plot_Dropped.txt' u 1:10 t "sts=1,rsv=0" w l, './RESULTS_3/Plot_Dropped.txt' u 1:10 t "sts=0,rsv=0" w l
gnuplot>

  To use gnuplot in cygwin
    []()http://petpari.blogspot.jp/2013/04/octave-gnuplot-using-cygwins-x-server.html



### web service


#### receive short message from China ###
Buy a Jego number (https://www.jego.me/app/internal/buyPack)
Con: out of service now. 

Buy a 沃信 number
Con: out of service now. 

#### Google search a specific date range [Unofficial Google Advanced Search [http://jwebnet.net/advancedgooglesearch.html#advDateRange]: ###
  "&as_qdr" # may be short for 'as query date range'. 
    '&as_qdr=y2': search past 2 years
    '&as_qdr=m2': search past 2 months
    '&as_qdr=M2': search past 2 Minutes

#### get bibtex of a paper ###
Use this API: https://github.com/ckreibich/scholar.py

./scholar.py -p 'PORs: Proofs of Retrievability for Large Files' -c 1 --citation=bt
  // bt: get citation in the bibtex format
  // -c 1: return 1 result
  // -p 'abc': search a paper contain 'abc'

#### share and edit files ###

##### git ####
Con: too complicated to use for oridnary people.

##### team cooperation ####

Dropbox is not suitable for team working. 
1. sync is slow
2. multiple peole can edit the same file at the same time. When they save, conflict occur.

I am looking for alternatives.
Looks google drive and MS office online are better [](http://www.makeuseof.com/tag/improve-collaborative-editing-office-files-dropbox-project-harmony/)

###### google drive + dropbox #####
Set google drive's root directory in dropbox. 
Use browse GD to edit files.

###### office online (onedrive.live.com/) #####
Con: cannot open files larger than 5mb

###### google doc #####

###### googledrive (different from google doc) #####

May not power enough as excel. For example, data validation. 

## vpn

sudo ssserver -p 11443 --user xling -k xxxx -m aes-256-cfb -d start
[create my own vpn using digitalocean and shadowsocks](https://github.com/iMeiji/shadowsocks_install/wiki/shadowsocks-libev-%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)

## bbr 

### install
https://github.com/google/bbr/blob/master/Documentation/bbr-quick-start.md

### test speed

https://www.howtoforge.com/tutorial/check-internet-speed-with-speedtest-cli-on-ubuntu/

sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install python-pip
sudo pip install speedtest-cli
sudo apt-get install unzip
cd /tmp
wget https://github.com/sivel/speedtest-cli/archive/master.zip
unzip master.zip
cd speedtest-cli-master/
chmod +x speedtest.py
./speedtest.py


* without bbr
Testing download speed................................................................................
Download: 742.96 Mbit/s
Testing upload speed................................................................................................
Upload: 211.68 Mbit/s

TCP cubic

* with bbr
Testing download speed................................................................................
Download: 640.99 Mbit/s
Testing upload speed................................................................................................
Upload: 469.74 Mbit/s

TCP bbr

## git


### resolve conflict


#### Always accept theirs: ###
  git merge --strategy-option theirs [](http://stackoverflow.com/questions/10697463/resolve-git-merge-conflicts-in-favor-of-their-changes-during-a-pull)

#### Script to accept the "theirs" version for files in a directory. ###
  [](http://stackoverflow.com/questions/105212/linux-recursively-list-all-files-in-a-directory-including-files-in-symlink-direc)
  for f in `find /dir/that/conflict -type f`; do git checkout --theirs $f; done

Or
  git checkout  --theirs  .

#### resolve conflict ###
  git merge test-development



Automatic merge failed, a bunch of conflicts!

  git checkout --theirs .
  git add .
  git commit

#### Unmerged paths ###
Unmerged 
  both modified:   documents/scripts/crossplatform/auto_test/selenium/get-citation/out.txt
  added by them:   documents/scripts/crossplatform/python/get_google_scholar_info 
  added by us:     documents/scripts/crossplatform/python/scholar

Solution:
  git mergetool [](http://stackoverflow.com/questions/17640019/git-same-file-added-by-both-branches-causes-weird-merge-conflict)


### basic


#### Create a local repo ###
  mkdir data.git
  cd data.git
  git init --bare

  cd ..
  mkdir data
  cd data
  git add .
  make some changes ....
  git commit -m "some changes are made"
  git remote add origin file:///d/data.git
  (git remote add origin file:///z///data.git)
  git push -u origin master

  git remote show origin

  sudo git clone git+ssh://xling@10.56.48.223/home3/share/data.git

#### fetch a branch to local ###
git checkout --track origin/daves_branch

#### Switch to a branch ###
  git clone https://leonexu@bitbucket.org/leonexu/padc2.git
  git fetch && git checkout refactoring_mtte

#### submodule vs clone ###
Using submodule, you can choose whether to commit the whole directory (ie.
module) or only a sub directory (i.e, submodule) 
[http://longair.net/blog/2010/06/02/git-submodules-explained/]
[the comment section, http://logicalfriday.com/2011/07/18/using-vim-with-pathogen/]

### find the commit id where a file is first added

  git log --diff-filter=A -- know_how/research/pdf-backup/mendeley/survey.pdf

### go to previous commit

HEAD~1: go to the previous commit
HEAD~2: go to the previous commit of the previous commit 

There is also HEAD^n. I don’t know the difference with HEAD~n
  1、“^”代表父提交,当一个提交有多个父提交时，可以通过在”^”后面跟上一个数字，表示第几个父提交，”^”相当于”^1”.
  2、~<n>相当于连续的<n>个”^”.

### Remove some data from multiple commits


Use git filter-branch [](http://code.csdn.net/help/CSDN_Code/progit/zh/09-git-internals/01-chapter9)
Specifically, remove a file:
  git filter-branch --index-filter 'git rm --cached --ignore-unmatch filename' HEAD [](http://loveky2012.blogspot.jp/2012/08/git-command-git-filter-branch.html)

  e.g.
    git filter-branch --index-filter 'git rm --cached --ignore-unmatch *venv*' HEAD

Then
  git push --force

Remove a file for certain commit range
  First, get the size of the current commit:
    对仓库进行 gc 操作，并查看占用了空间：

    $ git gc
    Counting objects: 21, done.
    Delta compression using 2 threads.
    Compressing objects: 100% (16/16), done.
    Writing objects: 100% (21/21), done.
    Total 21 (delta 3), reused 15 (delta 1)
    可以运行 count-objects 以查看使用了多少空间：
    
    $ git count-objects -v
    count: 4
    size: 16
    in-pack: 21
    packs: 1
    size-pack: 2016
    prune-packable: 0
    garbage: 0a

  Remove:
    git filter-branch --index-filter 'git rm --cached --ignore-unmatch git.tbz2' -- 6df7640^..

  Commit:
    git add -A; git commit -m 'xxx'; git push

### cannot checkout due to wrong path

Problem:
  Somehow I commited a file /x/y/z\a\b.txt into git repo. [](http://askubuntu.com/questions/144408/what-is-the-file-c-nppdf32log-debuglog-txt)
  On windows disk (or, a windows directory mapped into virtual box), git clone cannot correctly parse such path and will fail.
Solution:
  check out in a pure linux disk (e.g., /tmp).
    git rm /x/y/z\\a\\b.txt
    git commit 
    git push

### bitbucket


#### cannot git push. git hangs ###

Solution: use https instead git:
  Not this: git remote add origin git@bitbucket.org:leonexu/share.git
  Use this: git remote add origin https://leonexu@bitbucket.org/leonexu/share.git

#### push to bitbuket with default password ###
Idea: 
  Connect to BB via HTTPs.
  Let BB stores knows your public key.

##### create private key and upload public key to bitbucket ####
[](https://confluence.atlassian.com/display/BITBUCKET/Set+up+SSH+for+Git)

##### Errors: bad owner or permissions on ssh config ####
Solution: chmod 600 ~/.ssh/config [](http://www.haowt.info/archives/205.html)

##### Connect via the proxy. ####
HTTPs relays on SSH connection.
Seem that ssh needs special configuration to connect ssh via comany proxy. (http://www.zeitoun.net/articles/ssh-through-http-proxy/start) 
It likes risky to go throught the proxy.

### Error: git: pathspec.c:317: prefix_pathspec: Assertion `item->nowildcard_len <= item->len && item->prefix <= item->len' failed.


rename the folder, commit, rename back. [](http://stackoverflow.com/questions/23205961/why-is-git-erroring-with-assertion-failed-on-git-add)

## get paper citation graph
PaperCube: only work on a small database.

## docker

### create a docker vm in virtualbox with 80g disk (the default size is 10g, too small)
[https://stackoverflow.com/questions/32485723/docker-increase-disk-space]

docker-machine create --driver virtualbox --virtualbox-disk-size "80000" dockerId

### visit the vm in putty

Enter virtualbox, double click "default". 

Run ifconfig to know default's ip (i.g., 192.168.99.100)

Access 192.168.99.100 via putty.

### In China, change image repo

#### sudo vi /etc/docker/daemon.json
{
  "registry-mirrors": ["https://registry.docker-cn.com"]
}

#### sudo vi /etc/default/docker
DOCKER_OPTS="--registry-mirror=https://registry.docker-cn.com"


#### restart docker

##### the docker is in windows
Open cmd

> docker-machine restart dockerId

##### the docker is in linux
systemctl daemon-reload
systemctl restart docker && systemctl enable docker

### install first ubuntu

  // # install docker machines [https://docs.docker.com/machine/install-machine/#installing-machine-directly]
  // wget https://github.com/docker/machine/releases/download/v0.10.0/docker-machine-Linux-x86_64
  // chmod +x docker-machine-Linux-x86_64
  // sudo mv ./docker-machine-Linux-x86_64 /usr/bin/docker-machine

### create the first ubuntu container
[http://www.open-open.com/lib/view/open1410568733492.html]
docker pull kaggle/python
docker pull ubuntu

[mount](http://stackoverflow.com/questions/23439126/how-to-mount-host-directory-in-docker-container)
mkdir ~/win_dropbox
  ## Add a shared directory Dropbox in virtualbox 
```bash
sudo mount -t vboxsf Dropbox /home/docker/win_dropbox/

docker run -it -v /home/docker/win_dropbox/:/root/win_dropbox \
-p 15900:5900 -p 15901:5901 \
ubuntu bash
  # -p 5900:5900 -p 5901:5901: map 5900 port of host to the 5900 port of the container 
```

```
$ docker run -d \
  --name devtest \
  --mount source=/mnt/d/Dropbox/Dropbox,target=/app \
  ubuntu bash
```

[](reinstall_docker.md)

### restart a container 
(mkdir -p /home/docker/win_dropbox; sudo mount -t vboxsf Dropbox /home/docker/win_dropbox/; docker start fb9c2470da91; docker container attach fb9c2470da91)
docker container ps -a # get container id in the first column
sudo mount -t vboxsf Dropbox /home/docker/win_dropbox/
docker start 5dfb353dcc6f
docker container attach 5dfb353dcc6f 

### use Dockerfile

Run 'cmd' # not in putty
Create a new folder and enter it.
Create a file named 'Dockerfile'
docker build .
* In windows [](https://stackoverflow.com/questions/41286028/docker-build-error-checking-context-cant-stat-c-users-username-appdata)

### Run GUI application

https://github.com/fgrehm/docker-eclipse

docker pull fgrehm/eclipse:v4.4.1
L=$HOME/bin/eclipse && curl -sL https://github.com/fgrehm/docker-eclipse/raw/master/eclipse > $L && chmod +x $L
cd /path/to/java/project
eclipse

### install on the local ubuntu
fab -f ~/job/codes/devops.py SetupVnc -H localhost -i ~/.ssh/id_rsa

### increase disk size


## office_knowledge

Problem: in office, you cannot grouping tables with shapes.
Workround: cut and paste tables as figures. Then, grouping. [](http://www.indezine.com/products/powerpoint/learn/tables/2013/ungroup-a-table-in-ppt.html)

### Lab power

  The lab''s power is controlled by multiple switch boxes. The boxes are besides the outside of lab''s doors. Each machine, displayer is powered by a certain switch. Each switch is indexed. Devices beside each other may be have switches controlled by different switch boxes. So, be careful of the indexes.

  Switches start working once  powered. To stop the switch, you can directly shutdown the power.
  You need to power severs and routers respectively.

  Mutliple switches are connected. Each switch has muliple ports. The last port connect to the first port of the next switch.

### travel_abroad

  base
    插座转接口，接线板。
    携带式手机充电器。随身携带，飞机等上使用。
    公司pc
    感冒药
    company credit card
    拖鞋

  ls
    dress (衬衣，外套，裤子，鞋子) x 2。

  job
    Set an alarm to get conference receipt.
    Company credit card for hotel. Company travel agency does not pay when make reservation.

  job
    飞机上没有网。 手机中存一些paper。

  During travel
    Be careful of your hotel plan. It may not contain breakfast. Do not take hotel breakfast then. NEC cannot fully cover the fee.

### AFIN 2014

    Embedded Web Device Security
      Find network devices.
        www.shodanhq.com
    Keyword-Based Breadcrumbs: A Scalable Keyword-Based Search Feature in Breadcrumbs-Based Content-Oriented Network
      Scalability is a problem.
    Mobile edge computing
      Nokia is researching, implementing test bed. Currently mainly consider caching, also consider offloading ....
    
    Constraint-Based Distribution Method of In-Network Guidance Information in Content-Oriented Network
      Goal: reduce Breadcrumbs''s routing table size by remembering only 1% popular contents.
    Bluetooth low energy is only for applications that send packets at very low frequency. I can be used inside body for 10 years, sending packets each 4 seconds.


### AFIN

  masa?, previously ntt, nigata u
  prof TODE, okayama u, sensor, physical layer work, CCN
  About 100 people.

  Presentation Q&A
    Comment 1 (Prof Tode): why in high traffic delay(CCN) increase. Does not CCN always use shortest path routing?
    Comment 2 (Chair): any plan to put MTTE into market?
  Lot of sensor, material talks.
  Existing works similar to CloudLet
  * mobile agent.
  * jade agent development framework.
  * IBM aglets.
  * socket -> serialization -> dynamic method invocation
      The idea of CloudLet is not new. Maybe security + network is better research idea.



### enlarge images without loss quality

[](http://askubuntu.com/questions/301540/export-image-as-svg-in-gimp)
Use Inkscape.
Import the image.
Select the image.
Path -> trace bitmap -> uncheck "smooth".
Change canvas size: ctrl + shift + d.
Export as png.

## math


### 思维的乐趣


#### 密码学与协议 ###

##### Get salary sum. ####
Method 1: 
People give their salaries to a trustable third-party.

Method 2:
Everyone encrypts his salary f[i] = fun(s[i]). f(*) has the property that it allows calculation. 
effe*
An simply encryption can be sum().
Everyone adds a random number R[i] to his salary to get T[i]. 
Calculate sum(T).
The sum of salary is sum(T) - sum(R).

##### Use Chinese residue theorm to realize (n, k) encryption ####
Compute the residues of n on t prime numbers.
Let K be the key that encrypts the file.
Choose k-1 prime numbers so that their multiplications is less than K and adding any one prime number the multiplication is larger than K.

Goal: n people exist. The file can be and only can be decrypted when more than k people gather.

My idea:
  Everyone has a key. Generate a master key for each three personal keys using f(key1, ..., key_i) - C(n, k) master keys can be generated.
  Encrypt the file using C(n,k) master keys. 
  Everyone has all these C(n,k) ciphertexts. // May not need to have so many ciphertext.

Idea 2: polynomial
f() in this approach has an unique property - f(key1, ..., key_k) is equal for any combination of k keys. 

##### whether any insider paid the bill. ####

###### My idea: #####
Before eating, record the sum of all participants' money. After eating, if the sum changes, some participant paid the bill.
(clg) need trustful third party.

###### number of 'difference' is an even number. #####
Proof: 
Suppose that n =3, theorem hold by simply enumerating all cases

For n > 3, induction.
Suppose that the theorem holds for n nodes. 
In the case of n+1 nodes, we need to insert a new node between two existing nodes. 
case 1: the two nodes are different (suppose to be 0 and 1). No matter what the new node is, the number of "difference" after the inserting does not change. 
case 2: the two nodes are equal. After the inserting, the number of "difference" does not change or increases by two. 

###### Basic idea of insider detecting: #####
Design a protocol. We can calcuate the result when all users obey the protocol. If the real result does not match our expectation, an insider exists.

###### Basic idea of privacy-aware insider detecting: #####
Let each one picks a random number. Generate something R from the numbers according some rules. Ensure that R confirms to some theorem T if and only if the policy is obeyed. By checking whether T is confirmed, we know whether insider exists. 

###### what if two insiders exist? #####

##### privacy-aware broadcasting ####
My idea:
  Let 0 (1) be the event "insider exists" (insider does not exist). Running the
  insider-detection algorithm once generates one bit. By running the algorithm
  multiple times, we can generate a privacy-aware message.

##### undeniable contract ####
Approach 1: He encrypts the secret using my public key.
Approach 2: encrypt twice using my key and his key.

## laugh

[~/base/know_how/laugh.md]()


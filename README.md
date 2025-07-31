# Jazz
Code for paper "Edge-Served Congestion Control for Wireless Multipath Transmission with a Transformer Agent"

https://github.com/user-attachments/assets/bf57b1fd-a585-4b49-b2d2-87d0e6cf50e9

## Kernel and Environment Setup

This repository contains two versions of the Linux kernel (packaged as `.deb` installation files) in the `5.4.243-mininet` and `6.8.0` directories.

-   **Kernel `5.4.243-mininet`**: This version is recommended for testing in a Mininet environment.
-   **Kernel `6.8.0`**: We modified this kernel version because version `5.4.243` is incompatible with the NICs we used. However, we have not yet found a version of Mininet that functions correctly with this kernel.

### Kernel Installation Instructions

To install a kernel, copy all its corresponding `.deb` files into a dedicated directory and run the following commands:

Install the kernel packages:
```bash
sudo dpkg -i linux-*.deb
```

Next, edit the GRUB configuration to set the default kernel on boot:
```bash
sudo vim /etc/default/grub
```
Inside the file, modify the `GRUB_DEFAULT` value to the name of the kernel version you wish to use. After modification, run:
```bash
sudo update-grub
```
Then, reboot your system to apply the new kernel:
```bash
sudo reboot
```

### MPTCP Jazz Kernel Module

After kernel installation, you can check for the existence of the `mptcp_jazz.ko` kernel module in the following directory:

```bash
ls /lib/modules/$(uname -r)/kernel/net/ipv4/mptcp_jazz.ko
```

If the file exists, you can load the module using:

```bash
sudo modprobe mptcp_jazz.ko
```

Upon successful loading, verify the available congestion control algorithms:

```bash
sudo sysctl net.ipv4.tcp_available_congestion_control
```
You should see `jazz-1` or `jazz-2` among the listed algorithms.

## User-Space Deployment

To deploy the agent in user space on the client side (the TCP connection initiator), start the `user_space_server.py` script located in the `local_served` folder.
```bash
python user_space_server.py
```
Ensure that MPTCP connections use the `jazz` congestion control algorithm while `user_space_server.py` is running.

## Edge Deployment

In an edge deployment scenario:

1.  Deploy `backend_server (decision engine).py` and its dependencies on the decision engine server.
2.  Deploy `frontend_server (proxy).py` in the user space of the MPTCP connection initiator.
3.  Configure the IP address of the decision engine in the proxy script (`frontend_server (proxy).py`) to enable communication between the proxy and the decision engine.

## ⚠️ Important Notice

This demo is currently in the experimental stage. It is designed to work specifically with MPTCP connections that have **exactly two subflows**. Using a different number of subflows will likely result in a kernel panic.

Future development will focus on:
-   Supporting a variable number of subflows.
-   Implementing proper handling for fallback to standard TCP.

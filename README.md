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
sudo vim

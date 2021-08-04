This document helps you prepare environment for developing code for Sedna.
If you follow this guide and find some problem, please fill an issue to update this file.

## 1. Install Tools
### Install Git

Sedna is managed with [git], and to develop locally you
will need to install `git`.

You can check if `git` is already on your system and properly installed with 
the following command:

```
git --version
```

### Install Go(optional)

All Sedna's control components(i.e. [GM/LC][framework]) are written in the [Go][golang].
If you are planning to change them, you need to set up Go.

Sedna currently builds with Go 1.16, install or upgrade [Go using the instructions for your operating system][golang].

You can check if Go is in your system with the following command:

```
go version
```

## 2. Clone the code

Clone the `Sedna` repo:

```shell
git clone http://github.com/kubeedge/sedna.git
```

**Note**: If you want to add or change API in [pkg/apis](/pkg/apis), you need to checkout the code to `$GOPATH/src/github.com/kubeedge/sedna`.

## 3. Set up Kubernetes/KubeEdge(optional)
If you are planning to run or debug Sedna, you need to set up Kubernetes and KubeEdge.

Sedna requires Kubernetes version 1.16 or higher with CRD support.

Sedna requires KubeEdge version 1.5 or higher with edge support.

> **Note**: You need to check [the Kubernetes compatibility of KubeEdge][kubeedge-k8s-compatibility].

### Install Kubernetes

Follow [Kubernetes setup guides][k8s-setup] to set up and run Kubernetes, like:
> If you're learning Kubernetes, use the [tools][k8s-tools] to set up a Kubernetes cluster on a local machine, e.g.:
>	* [Installing Kubernetes with Kind][kind]
>	* [Installing Kubernetes with Minikube][minikube]


### Install KubeEdge

Please follow [the kubeedge instructions][kubeedge] to install KubeEdge.


## 4. What's Next?

Once you've set up the prerequisites, continue with:
- See [control plane development guide]
for more details about how to build & test Sedna.
- See [lib development guide TBD] for more details about how to develop AI algorithms and worker images based on [sedna lib code](/lib).

[git]: https://git-scm.com/
[framework]: /docs/proposals/architecture.md#architecture
[github]: https://github.com/
[golang]: https://golang.org/doc/install
[k8s-setup]: https://kubernetes.io/docs/setup/
[k8s-tools]: https://kubernetes.io/docs/tasks/tools
[minikube]: https://minikube.sigs.k8s.io/docs/start/
[kind]: https://kind.sigs.k8s.io
[kubeedge]: https://kubeedge.io/en/docs/
[kubeedge-k8s-compatibility]: https://github.com/kubeedge/kubeedge#kubernetes-compatibility

[control plane development guide]: ./control-plane/development.md

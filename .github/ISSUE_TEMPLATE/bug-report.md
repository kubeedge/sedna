---
name: Bug Report
about: Report a bug encountered while operating Sedna
labels: kind/bug

---

<!-- Please use this template while reporting a bug and provide as much info as possible. Thanks!-->
**What happened**:

**What you expected to happen**:

**How to reproduce it (as minimally and precisely as possible)**:

**Anything else we need to know?**:

**Environment**:
<details><summary>Sedna Version</summary>

```console
$ kubectl get -n sedna deploy gm -o jsonpath='{.spec.template.spec.containers[0].image}'
# paste output here

$ kubectl get -n sedna ds lc -o jsonpath='{.spec.template.spec.containers[0].image}'
# paste output here
```

</details>

<details><summary>Kubernets Version</summary>

```console
$ kubectl version
# paste output here
```

</details>

<details><summary>KubeEdge Version</summary>

```console
$ cloudcore --version
# paste output here

$ edgecore --version
# paste output here
```

</details>

**CloudSide Environment**:
<details><summary>Hardware configuration</summary>

```console
$ lscpu
# paste output here
```

</details>

<details><summary>OS</summary>

```console
$ cat /etc/os-release
# paste output here
```

</details>

<details><summary>Kernel</summary>

```console
$ uname -a
# paste output here
```

</details>

<details><summary>Others</summary>
</details>

**EdgeSide Environment**:
<details><summary>Hardware configuration</summary>

```console
$ lscpu
# paste output here
```

</details>

<details><summary>OS</summary>

```console
$ cat /etc/os-release
# paste output here
```

</details>

<details><summary>Kernel</summary>

```console
$ uname -a
# paste output here
```

</details>

<details><summary>Others</summary>
</details>

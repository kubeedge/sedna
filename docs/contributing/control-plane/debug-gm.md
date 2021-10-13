This document is intended to provide contributors with an introduction to develop [GM(GlobalManager)][framework] of the Sedna project. 

### Debug GM
1\. config GM:

The config file is the yaml format:

```yaml
kubeConfig: ""
namespace: ""
websocket:
  port: 9000
localController:
  server: http://localhost:9100
```

1. `kubeConfig`: kubernetes config file, default `""`
1. `namespace`: the namespace GM watches, `""` means that gm watches all namespaces, default `""`.
1. `websocket`: since the current limit of kubeedge(1.5), GM needs to build the websocket channel for communicating between GM and LCs.
1. `localController`:
   - `server`: to be injected into the worker to connect LC.

Generate a config yaml:

```shell
cat > gm.yaml <<EOF
kubeConfig: "${KUBECONFIG:-$HOME/.kube/config}"
namespace: ""
websocket:
  port: 9000
localController:
  server: http://localhost:9100
EOF
```

2\. compile and run GM directly:

If you have already run Sedna by following the [install doc],
you need to stop GM by `kubectl -n sedna scale --replicas=0 gm` before,
and reconfig `GM_ADDRESS` of LC by `kubectl -n sedna edit daemonset lc`.

```shell
make WHAT=gm
_output/bin/sedna-gm --config gm.yaml -v4
```

Alternatively you can debug GM with [golang delve]:

```shell
dlv debug cmd/sedna-gm/sedna-gm.go -- --config gm.yaml -v4
```



[install doc]: /docs/setup/install.md
[golang delve]: https://github.com/go-delve/delve
[framework]: /docs/proposals/architecture.md#architecture

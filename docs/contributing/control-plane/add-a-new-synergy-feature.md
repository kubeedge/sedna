### Discussion About New Feature
Discuss the new feature that you want to add with us on [our meeting](/README.md#meeting) in the following dimensions:
1. User Cases, Values
1. API
1. Architecture, A Main Process

#### Submitting Your Proposal
you can refer to [existing proposals] for proposal template.

### Developing

#### Developing on control plane
- [The development guide of control plane](./development.md)

Suppose you are going to add a synergy feature named `foobar` with versioned `v1alpha1`.
1. Add `foobar` APIs
	```shell
	cd pkg/apis/sedna/v1alpha1/
	# code the api
	touch foobar.go
	```

1. Code controller logic based on the [k8s operator pattern]
   
	```shell
	mkdir -p pkg/globalmanager/controllers/foobar/
	cd pkg/globalmanager/controllers/foobar/
	touch foobar.go
	```

1. Code upstream logic if any
	```shell
	# GM part
	cd pkg/globalmanager/controllers/foobar/
	touch upstream.go

	# LC part
	mkdir pkg/localcontroller/managers/foobar/
	cd pkg/localcontroller/managers/foobar/
	touch foobar.go
	```

1. Code downstream logic if any
	```shell
	# GM part
	cd pkg/globalmanager/controllers/foobar/
	touch downstream.go

	# LC part
	cd pkg/localcontroller/managers/foobar/
	touch foobar.go
	```

About debugging GM/LC:
  - [debug GM](debug-gm.md)
  - [debug LC](debug-lc.md)

Also see [coding conventions][k8s coding convention] for clean code.

#### Developing Workers TBD
<!--开发lib的流程 TBD -->

#### Submitting Your Code

When development has been done and ready to submit your work, see [pull request guide][kubernetes pull request guide] for more details if you don't know.

[existing proposals]: /docs/proposals

[k8s coding convention]: https://github.com/kubernetes/community/blob/master/contributors/guide/coding-conventions.md
[kubernetes pull request guide]: https://github.com/kubernetes/community/blob/master/contributors/guide/style-guide.md
[k8s operator pattern]: https://kubernetes.io/docs/concepts/extend-kubernetes/operator/

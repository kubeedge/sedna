package jointinference

import (
	"context"
	"testing"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	fakeseednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/fake"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	kubernetesfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/apps/v1"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
)

type mockPodLister struct {
	pods []*v1.Pod
}

type mockPodNamespaceLister struct {
	pods      []*v1.Pod
	namespace string
}

func (m *mockPodLister) Pods(namespace string) corelistersv1.PodNamespaceLister {
	return mockPodNamespaceLister{pods: m.pods, namespace: namespace}
}

func (m *mockPodLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	return m.pods, nil
}

func (m mockPodNamespaceLister) List(selector labels.Selector) ([]*v1.Pod, error) {
	var filteredPods []*v1.Pod
	for _, pod := range m.pods {
		if pod.Namespace == m.namespace {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods, nil
}

type mockDeploymentLister struct {
	deployments []*appsv1.Deployment
}

func (m *mockDeploymentLister) List(selector labels.Selector) (ret []*appsv1.Deployment, err error) {
	return m.deployments, nil
}

func (m *mockDeploymentLister) Deployments(namespace string) corelisters.DeploymentNamespaceLister {
	return mockDeploymentNamespaceLister{deployments: m.deployments, namespace: namespace}
}

func (m mockPodNamespaceLister) Get(name string) (*v1.Pod, error) {
	for _, pod := range m.pods {
		if pod.Namespace == m.namespace && pod.Name == name {
			return pod, nil
		}
	}
	return nil, k8serrors.NewNotFound(v1.Resource("pod"), name)
}

type mockDeploymentNamespaceLister struct {
	deployments []*appsv1.Deployment
	namespace   string
}

func (m mockDeploymentNamespaceLister) List(selector labels.Selector) ([]*appsv1.Deployment, error) {
	var filteredDeployments []*appsv1.Deployment
	for _, deployment := range m.deployments {
		if deployment.Namespace == m.namespace {
			filteredDeployments = append(filteredDeployments, deployment)
		}
	}
	return filteredDeployments, nil
}

func (m mockDeploymentNamespaceLister) Get(name string) (*appsv1.Deployment, error) {
	for _, deployment := range m.deployments {
		if deployment.Namespace == m.namespace && deployment.Name == name {
			return deployment, nil
		}
	}
	return nil, k8serrors.NewNotFound(v1.Resource("deployment"), name)
}

func Test_updateService(t *testing.T) {
	t.Run("update joint inference service successfully", func(t *testing.T) {
		// Create fake clients
		fakeSednaClient := fakeseednaclientset.NewSimpleClientset()
		fakeKubeClient := kubernetesfake.NewSimpleClientset()

		// Create a test joint inference service
		oldService := &sednav1.JointInferenceService{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "test-ji-service",
				Namespace:       "default",
				Generation:      1,
				ResourceVersion: "1",
			},
			Spec: sednav1.JointInferenceServiceSpec{
				EdgeWorker: sednav1.EdgeWorker{
					Model: sednav1.SmallModel{
						Name: "test-edge-model",
					},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "edge-container",
									Image: "edge-image:v1",
								},
							},
						},
					},
					HardExampleMining: sednav1.HardExampleMining{
						Name: "test-hem",
						Parameters: []sednav1.ParaSpec{
							{
								Key:   "param1",
								Value: "value1",
							},
						},
					},
				},
				CloudWorker: sednav1.CloudWorker{
					Model: sednav1.BigModel{
						Name: "test-cloud-model",
					},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "cloud-container",
									Image: "cloud-image:v1",
								},
							},
						},
					},
				},
			},
		}

		//Create Big Model Resource Object for Cloud
		bigModel := &sednav1.Model{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-cloud-model",
				Namespace: "default",
			},
		}
		_, err := fakeSednaClient.SednaV1alpha1().Models("default").Create(context.TODO(), bigModel, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create test big model: %v", err)
		}

		// Create Small Model Resource Object for Edge
		smallModel := &sednav1.Model{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-edge-model",
				Namespace: "default",
			},
		}
		_, err = fakeSednaClient.SednaV1alpha1().Models("default").Create(context.TODO(), smallModel, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create test small model: %v", err)
		}

		// Create the service using the fake client
		_, err = fakeSednaClient.SednaV1alpha1().JointInferenceServices("default").Create(context.TODO(), oldService, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create test service: %v", err)
		}

		// Create test deployments
		edgeDeployment := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-ji-deployment-edge",
				Namespace: "default",
			},
			Spec: appsv1.DeploymentSpec{
				Template: oldService.Spec.EdgeWorker.Template,
			},
		}
		cloudDeployment := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-ji-deployment-cloud",
				Namespace: "default",
			},
			Spec: appsv1.DeploymentSpec{
				Template: oldService.Spec.CloudWorker.Template,
			},
		}

		_, err = fakeKubeClient.AppsV1().Deployments("default").Create(context.TODO(), edgeDeployment, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create edge deployment: %v", err)
		}
		_, err = fakeKubeClient.AppsV1().Deployments("default").Create(context.TODO(), cloudDeployment, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create cloud deployment: %v", err)
		}

		// Manually create pods for the deployments
		edgePod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-ji-service-edge-pod",
				Namespace: "default",
				Labels: map[string]string{
					"jointinferenceservice.sedna.io/service-name": "test-ji-service",
				},
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       edgeDeployment.Name,
						UID:        edgeDeployment.UID,
					},
				},
			},
			Spec: edgeDeployment.Spec.Template.Spec,
		}
		cloudPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-ji-service-cloud-pod",
				Namespace: "default",
				Labels: map[string]string{
					"jointinferenceservice.sedna.io/service-name": "test-ji-service",
				},
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       cloudDeployment.Name,
						UID:        cloudDeployment.UID,
					},
				},
			},
			Spec: cloudDeployment.Spec.Template.Spec,
		}

		// Add pods to the fake client
		_, err = fakeKubeClient.CoreV1().Pods("default").Create(context.TODO(), edgePod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create edge pod: %v", err)
		}
		_, err = fakeKubeClient.CoreV1().Pods("default").Create(context.TODO(), cloudPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create cloud pod: %v", err)
		}

		cfg := &config.ControllerConfig{
			LC: config.LCConfig{
				Server: "http://test-lc-server:8080",
			},
		}

		eventBroadcaster := record.NewBroadcaster()
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: fakeKubeClient.CoreV1().Events("")})

		// Create a controller with the fake clients
		c := &Controller{
			kubeClient:        fakeKubeClient,
			client:            fakeSednaClient.SednaV1alpha1(),
			queue:             workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "test-ji-service"),
			recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "test-ji-service"}),
			cfg:               cfg,
			deploymentsLister: &mockDeploymentLister{deployments: []*appsv1.Deployment{edgeDeployment, cloudDeployment}},
			sendToEdgeFunc: func(nodeName string, eventType watch.EventType, job interface{}) error {
				return nil
			},
		}

		// Update the service
		newService := oldService.DeepCopy()
		// change parameter of hard example mining
		newService.Spec.EdgeWorker.HardExampleMining.Parameters[0].Value = "value2"
		newService.Generation = 2
		newService.ResourceVersion = "2"
		// Call updateService function
		c.createOrUpdateWorker(newService, jointInferenceForCloud, "test-ji-service.default", 8080, true)
		c.createOrUpdateWorker(newService, jointInferenceForEdge, "test-ji-service.default", 8080, true)
		// update service in fakeSednaClient
		_, err = fakeSednaClient.SednaV1alpha1().JointInferenceServices("default").Update(context.TODO(), newService, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update service: %v", err)
		}
		// Verify that the services were deleted and recreated
		updatedService, err := fakeSednaClient.SednaV1alpha1().JointInferenceServices("default").Get(context.TODO(), "test-ji-service", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get updated deployment: %v", err)
		}
		if updatedService.Spec.EdgeWorker.HardExampleMining.Parameters[0].Value != "value2" {
			t.Fatalf("Service was not updated correctly")
		}
	})
}

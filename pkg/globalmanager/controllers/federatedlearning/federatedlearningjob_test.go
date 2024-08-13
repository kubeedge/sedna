package federatedlearning

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	kubernetesfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	fakeseednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/fake"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

type mockPodLister struct {
	pods []*v1.Pod
}

func (m *mockPodLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	return m.pods, nil
}

func (m *mockPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return mockPodNamespaceLister{pods: m.pods, namespace: namespace}
}

type mockPodNamespaceLister struct {
	pods      []*v1.Pod
	namespace string
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

func (m mockPodNamespaceLister) Get(name string) (*v1.Pod, error) {
	for _, pod := range m.pods {
		if pod.Namespace == m.namespace && pod.Name == name {
			return pod, nil
		}
	}
	return nil, k8serrors.NewNotFound(corev1.Resource("pod"), name)
}

// unit test for deletePod function
func Test_deletePod(t *testing.T) {
	t.Run("delete existing pod successfully", func(t *testing.T) {
		// Create a fake client
		fakeClient := kubernetesfake.NewSimpleClientset()

		// Create a test pod
		testPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: "default",
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  "test-container",
						Image: "test-image",
					},
				},
			},
		}

		// Create the pod using the fake client
		_, err := fakeClient.CoreV1().Pods("default").Create(context.TODO(), testPod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create test pod: %v", err)
		}

		// Create a controller with the fake client
		controller := &Controller{
			kubeClient: fakeClient,
		}

		// Call deletePod function
		controller.deletePod(testPod)

		// Verify that the pod was recreated
		_, err = fakeClient.CoreV1().Pods("default").Get(context.TODO(), "test-pod", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Pod was not recreated")
		}
	})

	t.Run("delete non-existent pod", func(t *testing.T) {
		// Create a fake client
		fakeClient := kubernetesfake.NewSimpleClientset()

		// Create a controller with the fake client
		controller := &Controller{
			kubeClient: fakeClient,
		}

		// Call deletePod with a non-existent pod
		nonExistentPod := corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "non-existent-container",
					Image: "non-existent-image",
				},
			},
		}
		controller.deletePod(nonExistentPod)
		// No error should occur, and the function should complete
		// verify if the pod is deleted
		_, err := fakeClient.CoreV1().Pods("default").Get(context.TODO(), "test-pod", metav1.GetOptions{})
		if err == nil {
			t.Fatalf("Pod was not deleted")
		}
	})
}

// unit test for updateJob function
func Test_updateJob(t *testing.T) {
	t.Run("update correct job parameter successfully", func(t *testing.T) {
		// Create fake clients
		fakeSednaClient := fakeseednaclientset.NewSimpleClientset()

		// Create a test job
		oldJob := &sednav1.FederatedLearningJob{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-fl-job",
				Namespace: "default",
			},
			Spec: sednav1.FLJobSpec{
				AggregationWorker: sednav1.AggregationWorker{
					Model: sednav1.TrainModel{
						Name: "test-model",
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Name: "test-fl-job-aggregation-worker",
						},
						Spec: v1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "test-container",
									Image: "test-image",
								},
							},
						},
					},
				},
				TrainingWorkers: []sednav1.TrainingWorker{
					{
						Dataset: sednav1.TrainDataset{
							Name: "test-dataset1",
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name: "test-fl-job-training-worker-0",
							},
							Spec: v1.PodSpec{
								NodeName: "test-node1",
								Containers: []corev1.Container{
									{
										Name:            "test-container",
										Image:           "test-image",
										ImagePullPolicy: corev1.PullIfNotPresent,
										Env: []corev1.EnvVar{
											{
												Name:  "batch_size",
												Value: "32",
											},
											{
												Name:  "learning_rate",
												Value: "0.001",
											},
											{
												Name:  "epochs",
												Value: "2",
											},
										},
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceMemory: resource.MustParse("2Gi"),
											},
										},
									},
								},
							},
						},
					},
					{
						Dataset: sednav1.TrainDataset{
							Name: "test-dataset2",
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name: "test-fl-job-training-worker-1",
							},
							Spec: v1.PodSpec{
								NodeName: "test-node2",
								Containers: []corev1.Container{
									{
										Name:            "test-container",
										Image:           "test-image",
										ImagePullPolicy: corev1.PullIfNotPresent,
										Env: []corev1.EnvVar{
											{
												Name:  "batch_size",
												Value: "32",
											},
											{
												Name:  "learning_rate",
												Value: "0.001",
											},
											{
												Name:  "epochs",
												Value: "2",
											},
										},
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceMemory: resource.MustParse("2Gi"),
											},
										},
									},
								},
							},
						},
					},
				},
				PretrainedModel: sednav1.PretrainedModel{
					Name: "test-pretrained-model",
				},
			},
		}
		oldJob.ResourceVersion = "1"
		// Create the job using the fake client
		_, err := fakeSednaClient.SednaV1alpha1().FederatedLearningJobs("default").Create(context.TODO(), oldJob, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create test job: %v", err)
		}
		fakeKubeClient := kubernetesfake.NewSimpleClientset()

		// Create test pods
		testPods := []*v1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-fl-job-aggregation-worker",
					Namespace: "default",
				},
				Spec: oldJob.Spec.AggregationWorker.Template.Spec,
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-fl-job-training-worker-0",
					Namespace: "default",
				},
				Spec: oldJob.Spec.TrainingWorkers[0].Template.Spec,
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-fl-job-training-worker-1",
					Namespace: "default",
				},
				Spec: oldJob.Spec.TrainingWorkers[1].Template.Spec,
			},
		}

		// create pretrained model resource
		pretrainedModel := &sednav1.Model{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pretrained-model",
				Namespace: "default",
			},
		}
		_, err = fakeSednaClient.SednaV1alpha1().Models("default").Create(context.TODO(), pretrainedModel, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pretrained model: %v", err)
		}
		// create model resource
		model := &sednav1.Model{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-model",
				Namespace: "default",
			},
		}
		_, err = fakeSednaClient.SednaV1alpha1().Models("default").Create(context.TODO(), model, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create model: %v", err)
		}

		// create dataset1 resource
		dataset1 := &sednav1.Dataset{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dataset1",
				Namespace: "default",
			},
		}
		_, err = fakeSednaClient.SednaV1alpha1().Datasets("default").Create(context.TODO(), dataset1, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create dataset: %v", err)
		}

		// create dataset2 resource
		dataset2 := &sednav1.Dataset{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dataset2",
				Namespace: "default",
			},
		}
		_, err = fakeSednaClient.SednaV1alpha1().Datasets("default").Create(context.TODO(), dataset2, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create dataset: %v", err)
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
			kubeClient: fakeKubeClient,
			client:     fakeSednaClient.SednaV1alpha1(),
			podStore:   &mockPodLister{pods: testPods},
			flSelector: labels.SelectorFromSet(labels.Set{"federatedlearningjob.sedna.io/job-name": "test-fl-job"}),
			queue:      workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "test-fl-job"),
			recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "test-fl-job"}),
			cfg:        cfg,
			sendToEdgeFunc: func(nodeName string, eventType watch.EventType, job interface{}) error {
				return nil
			},
		}
		c.aggServiceHost = "test-fl-job-aggregation.default"

		// Update the job
		newJob := oldJob.DeepCopy()
		newJob.Spec.TrainingWorkers[0].Template.Spec.Containers[0].Env[0].Value = "16"
		newJob.Generation = 2
		newJob.ResourceVersion = "2"

		c.updateJob(oldJob, newJob)

		// Verify that the job was updated
		updatedJob, err := fakeSednaClient.SednaV1alpha1().FederatedLearningJobs("default").Get(context.TODO(), "test-fl-job", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get updated job: %v", err)
		}
		if updatedJob.Spec.TrainingWorkers[0].Template.Spec.Containers[0].Env[0].Value != "16" {
			t.Fatalf("Job was not updated correctly")
		}
	})
}

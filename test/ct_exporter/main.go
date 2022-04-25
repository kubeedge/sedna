package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"ct_exporter/api/types/v1alpha1"
	clientV1alpha1 "ct_exporter/clientset/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	//"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"log"
	//"math/rand"
	"net/http"
	//"path/filepath"
	//"strconv"
	//"strings"
)

var condMap = map[string]int{
	"Complete": 1,
	"Failed": 2,
	"Training": 3,
	"Pending": 4,
	"Running": 5,
	"Succeeded": 6,
}

var kubeconfig string

var (
	JobConditionType   		*prometheus.Desc
	NumberOfSamples    		*prometheus.Desc
	LastHeartBeatTime  		*prometheus.Desc
	LastProbeTime  			*prometheus.Desc
	StartTime       		*prometheus.Desc
	CompletionTime  		*prometheus.Desc
	ActivePodNumber  		*prometheus.Desc
	FailedPodNumber       	*prometheus.Desc
	SucceededPodNumber  	*prometheus.Desc
	Phase			  		*prometheus.Desc
)

type  Exporter struct {
	client    *clientV1alpha1.ExampleV1Alpha1Client
}

func NewExporter() (*Exporter, error) {
	var config *rest.Config
	var err error

	if kubeconfig == "" {
		log.Printf("using in-cluster configuration")
		config, err = rest.InClusterConfig()
	} else {
		log.Printf("using configuration from '%s'", kubeconfig)
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	}

	if err != nil {
		panic(err)
	}

	err = v1alpha1.AddToScheme(scheme.Scheme)
	if err != nil {
		panic(err)
	}
	clientSet, err := clientV1alpha1.NewForConfig(config)
	if err != nil {
		panic(err)
	}
	return &Exporter{
		client: clientSet,
	}, nil
}

func (e *Exporter) Describe(ch chan<- *prometheus.Desc) {
	ch <- JobConditionType
	ch <- NumberOfSamples
	ch <- LastHeartBeatTime
	ch <- LastProbeTime
	ch <- StartTime
	ch <- CompletionTime
	ch <- ActivePodNumber
	ch <- FailedPodNumber
	ch <- SucceededPodNumber
	ch <- Phase
}

func (e *Exporter) Collect(ch chan<- prometheus.Metric) {
	ctx := context.Background()
	federatedRes, err := e.client.FederatedClient("default").Get(ctx, "ct-yolo-v5", metav1.GetOptions{})
	if err != nil {
		panic(err)
	}

	conditions := federatedRes.Status.Conditions

	if conditions != nil {
		//condition type  complete/falied/training
		condType := conditions[len(conditions)-1].Type
		condTypeStr := string(condType)
		ch <- prometheus.MustNewConstMetric(
			JobConditionType,
			prometheus.GaugeValue,
			float64(condMap[condTypeStr]))

		lastHeartBeatTime := conditions[len(conditions)-1].LastHeartbeatTime
		heartBeatTimeUnix := lastHeartBeatTime.Unix()
		ch <- prometheus.MustNewConstMetric(
			LastHeartBeatTime,
			prometheus.GaugeValue,
			float64(heartBeatTimeUnix)*1000)

		lastProbeTime := conditions[len(conditions)-1].LastProbeTime
		probeTimeUnix := lastProbeTime.Unix()
		ch <- prometheus.MustNewConstMetric(
			LastProbeTime,
			prometheus.GaugeValue,
			float64(probeTimeUnix)*1000)
	}

	startTime := federatedRes.Status.StartTime
	startTimeUnix := startTime.Unix()
	ch <- prometheus.MustNewConstMetric(
		StartTime,
		prometheus.GaugeValue,
		float64(startTimeUnix)*1000)

	completionTime := federatedRes.Status.CompletionTime
	if completionTime != nil {
		completionTimeUnix := completionTime.Unix()
		ch <- prometheus.MustNewConstMetric(
			CompletionTime,
			prometheus.GaugeValue,
			float64(completionTimeUnix)*1000)
	}

	// 6.activePodNumber
	activePodNumber := federatedRes.Status.Active
	ch <- prometheus.MustNewConstMetric(
		ActivePodNumber,
		prometheus.GaugeValue,
		float64(activePodNumber))

	failedPodNumber := federatedRes.Status.Failed
	ch <- prometheus.MustNewConstMetric(
		FailedPodNumber,
		prometheus.GaugeValue,
		float64(failedPodNumber))

	succeededPodNumber := federatedRes.Status.Succeeded
	ch <- prometheus.MustNewConstMetric(
		SucceededPodNumber,
		prometheus.GaugeValue,
		float64(succeededPodNumber))

	//phase  Pending/Runing/Succeeded/Failed
	phase := federatedRes.Status.Phase
	phaseStr := string(phase)
	ch <- prometheus.MustNewConstMetric(
		Phase,
		prometheus.GaugeValue,
		float64(condMap[phaseStr]))

	datasetRes, err := e.client.DatasetClient("default").Get(ctx, "dataset-1", metav1.GetOptions{})
	if err != nil {
		panic(err)
	}
	number := datasetRes.Status.NumberOfSamples
	ch <- prometheus.MustNewConstMetric(
		NumberOfSamples,
		prometheus.GaugeValue,
		float64(number))
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "path to Kubernetes config file")
	flag.Parse()
}

func main() {
	JobConditionType = prometheus.NewDesc(
		"ct_conditiontype",
		"collabratedlearning_conditiontype",
		nil,
		nil)
	NumberOfSamples = prometheus.NewDesc(
		"number_of_samples",
		"number_of_samples",
		nil,
		nil)
	LastHeartBeatTime = prometheus.NewDesc(
		"LastHeartBeatTime",
		"LastHeartBeatTime",
		nil,
		nil)
	LastProbeTime = prometheus.NewDesc(
		"LastProbeTime",
		"LastProbeTime",
		nil,
		nil)

	StartTime = prometheus.NewDesc(
		"StartTime",
		"StartTime",
		nil,
		nil)

	CompletionTime = prometheus.NewDesc(
		"CompletionTime",
		"CompletionTime",
		nil,
		nil)

	ActivePodNumber = prometheus.NewDesc(
		"ActivePodNumber",
		"ActivePodNumber",
		nil,
		nil)

	FailedPodNumber = prometheus.NewDesc(
		"FailedPodNumber",
		"FailedPodNumber",
		nil,
		nil)

	SucceededPodNumber = prometheus.NewDesc(
		"SucceededPodNumber",
		"SucceededPodNumber",
		nil,
		nil)

	Phase = prometheus.NewDesc(
		"Phase",
		"Phase",
		nil,
		nil)


	exporter, err := NewExporter()
	if err != nil {
		panic(err)
	}
	reg := prometheus.NewPedanticRegistry()
	reg.MustRegister(exporter)

	gatherers := prometheus.Gatherers{
		prometheus.DefaultGatherer,
		reg,
	}
	h := promhttp.HandlerFor(gatherers,
		promhttp.HandlerOpts{
			ErrorHandling: promhttp.ContinueOnError,
		})
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
	})
	fmt.Printf("Start server at :9104")
	if err := http.ListenAndServe(":9104", nil); err != nil {
		fmt.Printf("Error occur when start server %v", err)
	}

}


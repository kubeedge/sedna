package main

import (
	"database/sql"
	"flag"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"log"
	"net/http"
	"time"
)

var addr = flag.String("listen-address", ":7070", "The address to listen on for HTTP requests.")

var metrics = []struct {
	name        string
	label       string
	note        string
	labelValues string
	value       float64
}{
	{"metric1", "method,handler", "This is metric1", "a, b", 0},
	{"metric2", "method,handler", "This is metric2", "a, b", 0},
	{"Recall", "model_name", "Model Recall", "a", 0},
	{"Precision", "model_name", "Model Precision", "b", 0},
}

func recordTaskNumM(db *sql.DB) {
	go func() {
		for {
			rows, err := db.Query("SELECT a.id, a.name, a.task_num, b.task_attr FROM ll_task_grp as a left join ll_tasks as b where a.id=b.id;")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var id int
				var name string
				var taskNum int
				var attr string
				err = rows.Scan(&id, &name, &taskNum, &attr)
				g, err := opsTaskTaskNum.GetMetricWithLabelValues(strconv.Itoa(id), name, attr)
				if err == nil {
					g.Set(float64(taskNum))
				}
			}
			time.Sleep(1 * time.Second)
		}
	}()
}

func recordTaskSampleMStatusM(db *sql.DB) {
	go func() {
		for {
			rows, err := db.Query("SELECT id, name, deploy, sample_num, task_num FROM ll_task_grp")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var id int
				var name string
				var deploy bool
				var sampleNum int
				var taskNum int
				err = rows.Scan(&id, &name, &deploy, &sampleNum, &taskNum)
				g, err := opsTaskSampleNum.GetMetricWithLabelValues(strconv.Itoa(id), name)
				if err == nil {
					g.Set(float64(sampleNum))
				}
				g, err = opsDeployStatus.GetMetricWithLabelValues(strconv.Itoa(id), name)
				if err == nil {
					if deploy {
						g.Set(1)
					} else {
						g.Set(0)
					}
				}
			}
			time.Sleep(1 * time.Second)
		}
	}()
}

func recordKnownTasks(db *sql.DB) {
	go func() {
		for {
			rows, err := db.Query("SELECT count(*) as c from ll_task_models where is_current = 1")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var c int
				err = rows.Scan(&c)
				g, err := opsKnowTaskNum.GetMetricWithLabelValues()
				if err == nil {
					g.Set(float64(c))
				}
			}
			time.Sleep(1 * time.Second)
		}
	}()
}

func recordTaskStatus(db *sql.DB) {
	go func() {
		for {
			rows, err := db.Query("SELECT task_id, model_url, is_current as c from ll_task_models")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var taskId int
				var modelUrl string
				var isCurrent bool
				err = rows.Scan(&taskId, &modelUrl, &isCurrent)
				g, err := opsTaskStatus.GetMetricWithLabelValues(strconv.Itoa(taskId), modelUrl)
				if err == nil {
					if isCurrent {
						g.Set(1)
					} else {
						g.Set(0)
					}
				}
			}
			time.Sleep(1 * time.Second)
		}
	}()
}

func recordTaskRelationShip(db *sql.DB) {
	go func() {
		for {
			rows, err := db.Query("select grp_id, task_id, transfer_radio from ll_task_relation")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var grpId int
				var taskId int
				var transferRatio float64
				err = rows.Scan(&grpId, &taskId, &transferRatio)
				g, err := opsTaskRelationShip.GetMetricWithLabelValues(strconv.Itoa(grpId), strconv.Itoa(taskId))
				if err == nil {
					g.Set(transferRatio)
				}
			}
			time.Sleep(1 * time.Second)
		}
	}()
}

func customMetrics(db *sql.DB, registry *prometheus.Registry) {
	// mock
	sql := "insert into metric (name, label, note, last_time) values (?, ?, ?, time('now'))"
	rows, err := db.Query("select count(*) from metric")
	if err != nil {
		fmt.Println(err)
	}
	mock := false
	for rows.Next() {
		var count int
		err = rows.Scan(&count)
		if err != nil {
			fmt.Println(err)
		}
		mock = count == 0
	}

	if mock {
		for i, metric := range metrics {
			_, err := db.Exec(sql, metric.name, metric.label, metric.note)
			if err != nil {
				fmt.Println(err)
			}
			_, err = db.Exec("insert into metric_value (metric_id, label_value, value, last_time) values (?, ?, ?, time('now'))",
				i, metric.labelValues, metric.value)
			if err != nil {
				fmt.Println(err)
			}
		}
		mockV := 1.0
		go func() {
			for {
				_, err := db.Exec("update metric_value set value=?", mockV)
				if err != nil {
					fmt.Println(err)
				}
				mockV = rand.Float64()
				time.Sleep(time.Second)
			}
		}()
	}

	// register metrics
	registeredMetrics := make([]*prometheus.GaugeVec, 0, 4)
	rows, err = db.Query("select name, label, note from metric order by id asc")
	if err != nil {
		fmt.Println(err)
	}
	for rows.Next() {
		var name string
		var label string
		var note string
		err = rows.Scan(&name, &label, &note)
		labels := strings.Split(label, ",")
		for i := range labels {
			labels[i] = strings.TrimSpace(labels[i])
		}
		met := promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: name,
			Help: note,
		}, labels)
		registry.MustRegister(met)
		registeredMetrics = append(registeredMetrics, met)
	}

	go func() {
		for {
			rows, err := db.Query("select metric_id, label_value, `value` from metric_value")
			if err != nil {
				fmt.Println(err)
			}
			for rows.Next() {
				var metricId int
				var labelValue string
				var value float64
				err = rows.Scan(&metricId, &labelValue, &value)
				labelValues := strings.Split(labelValue, ",")
				for i, _ := range labelValues {
					labelValues[i] = strings.TrimSpace(labelValues[i])
				}
				if len(registeredMetrics) <= metricId || registeredMetrics[metricId] == nil {
					continue
				}
				g, err := registeredMetrics[metricId].GetMetricWithLabelValues(labelValues...)
				if err == nil {
					g.Set(value)
				}
			}
			time.Sleep(time.Second)
		}
	}()
}

func fileScanner(p string, suffix string) {
	go func() {
		for {
			num := 0
			root := p
			err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
				if strings.HasSuffix(path, suffix) {
					num += 1
				}
				return nil
			})
			if err != nil {
				println(err)
			}
			g, err := opsFilesSuffixNum.GetMetricWithLabelValues()
			if err == nil {
				g.Set(float64(num))
			}
			num = 0
			time.Sleep(time.Second)
		}
	}()
}

var (
	// task metrics
	opsKnowTaskNum = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "know_task_num",
		Help: "Number of known tasks in the knowledge base",
	}, []string{})
	opsTaskSampleNum = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "task_sample_num",
		Help: "The total number of samples in task",
	}, []string{"id", "name"})
	opsTaskTaskNum = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "task_num",
		Help: "The total number of tasks",
	}, []string{"id", "name", "attr"})
	opsTaskRelationShip = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "task_relation_ship",
		Help: "Migration relationship between tasks",
	}, []string{"grp_id", "task_id"})
	opsTaskStatus = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "task_status",
		Help: "Whether the task can be deployed",
	}, []string{"task_id", "model_url"})
	opsDeployStatus = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "deploy_status",
		Help: "Enum(Waiting, OK, NotOK)",
	}, []string{"id", "name"})
	opsFilesSuffixNum = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "files_with_suffix_num",
		Help: "The number of files with suffix",
	}, []string{})
)

func main() {
	var scannerPath string
	var scannerSuffix string
	flag.StringVar(&scannerPath, "scanner-path", "-", "file scanner path")
	flag.StringVar(&scannerSuffix, "scanner-suffix", "", "file scanner suffix")
	flag.Parse()
	if scannerPath != "-" {
		fileScanner(scannerPath, scannerSuffix)
	}

	dbSrc := "kb.sqlite3"
	db, err := sql.Open("sqlite3", dbSrc)
	if err != nil {
		fmt.Printf("Can't open %s\n", dbSrc)
		return
	}

	dropTableMetric := "drop table metric"
	dropTableMetricValue := "drop table metric_value"
	_, err = db.Exec(dropTableMetric)
	if err != nil {
		println(err)
	}
	_, err = db.Exec(dropTableMetricValue)
	if err != nil {
		println(err)
	}
	createTableMetric := "create table if not exists metric(id int primary key, `name` text, label text, note text, last_time timestamp)"
	createTableMetricValue := "create table if not exists metric_value(id int primary key, metric_id int, label_value text, `value` float, last_time timestamp)"
	_, err = db.Exec(createTableMetric)
	if err != nil {
		println(err)
	}
	_, err = db.Exec(createTableMetricValue)
	if err != nil {
		println(err)
	}

	// Create a new registry.
	reg := prometheus.NewRegistry()
	recordTaskNumM(db)
	recordTaskSampleMStatusM(db)
	recordKnownTasks(db)
	recordTaskRelationShip(db)
	recordTaskStatus(db)
	customMetrics(db, reg)

	// Add Go module build info.
	reg.MustRegister(opsKnowTaskNum)
	reg.MustRegister(opsTaskSampleNum)
	reg.MustRegister(opsTaskTaskNum)
	reg.MustRegister(opsTaskRelationShip)
	reg.MustRegister(opsTaskStatus)
	reg.MustRegister(opsDeployStatus)
	reg.MustRegister(opsFilesSuffixNum)

	// Expose the registered metrics via HTTP.
	http.Handle("/metrics", promhttp.HandlerFor(
		reg,
		promhttp.HandlerOpts{
			// Opt into OpenMetrics to support exemplars.
			EnableOpenMetrics: true,
		},
	))
	log.Fatal(http.ListenAndServe(*addr, nil))
}

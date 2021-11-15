import threading, time
from kafka.metrics.metrics_reporter import AbstractMetricsReporter
from sedna.common.benchmark import FluentdHelper

class FluentdReporter(FluentdHelper, AbstractMetricsReporter):
    def init(self, metrics):
        """
        This is called when the reporter is first registered
        to initially register all existing metrics
        Arguments:
            metrics (list of KafkaMetric): All currently existing metrics
        """
        super().__init__()
        self.metrics = []
        self.interval = 5

        metrics_scheduler = threading.Thread(target=self.update)
        # Creating a daemon thread to not block shutdown
        metrics_scheduler.daemon = True
        metrics_scheduler.start()

    def metric_change(self, metric):
        """
        This is called whenever a metric is updated or added
        Arguments:
            metric (KafkaMetric)
        """
        if not metric in self.metrics:
            self.metrics.append(metric)
            # print(metric.metric_name.__str__() + " " + str(metric.value()))

    def metric_removal(self, metric):
        """
        This is called whenever a metric is removed
        Arguments:
            metric (KafkaMetric)
        """
        pass

    def configure(self, configs):
        """
        Configure this class with the given key-value pairs
        Arguments:
            configs (dict of {str, ?})
        """
        pass

    def close(self):
        """Called when the metrics repository is closed."""
        pass

    def update(self):
        while True:
            # for m in self.metrics:
                # self.send_json_msg(self.to_json(m))
            time.sleep(self.interval)

    def to_json(self, metric):
        return {
            'metric': metric.metric_name.name,
            'value': metric.value(),
            'description': metric.metric_name.description,
            'tags': metric.metric_name.tags        
        }
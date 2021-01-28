package trigger

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"
)

type Base interface {
	Trigger(stats map[string]interface{}) bool
}

type BinaryTrigger struct {
	Operator  string
	Metric    string
	Threshold float64
}

func convertFloat(v interface{}) (float64, error) {
	var realValue float64
	var err error
	switch v := v.(type) {
	case int:
		realValue = float64(v)
	case float32:
		realValue = float64(v)
	case float64:
		realValue = v
	case string:
		realValue, err = strconv.ParseFloat(v, 64)
	}
	return realValue, err
}

func (bt *BinaryTrigger) Trigger(stats map[string]interface{}) bool {
	var value interface{}
	var ok bool
	left := strings.Index(bt.Metric, "[")
	if left == -1 {
		value, ok = stats[bt.Metric]
		if !ok {
			return false
		}
	} else {
		// e.g. metric = precisions[3]
		right := strings.LastIndex(bt.Metric, "]")
		metric := strings.TrimSpace(bt.Metric[:left])
		topValue, ok := stats[metric]
		if !ok {
			return false
		}

		// only support the integer index
		subMetric := bt.Metric[left+1 : right]
		idx, err := strconv.Atoi(subMetric)
		if err != nil {
			return false
		}

		s := reflect.ValueOf(topValue)
		switch s.Kind() {
		case reflect.Slice, reflect.Array:
		default:
			return false
		}

		if !ok {
			return false
		}

		if idx >= s.Len() {
			return false
		}
		value = s.Index(idx).Interface()
	}
	realValue, err := convertFloat(value)
	if err != nil {
		return false
	}

	isEqual := math.Abs(realValue-bt.Threshold) < 1e-6

	switch bt.Operator {
	case "gt", ">":
		return !isEqual && realValue > bt.Threshold
	case "ge", ">=":
		return isEqual || realValue >= bt.Threshold
	case "eq", "=", "==":
		return isEqual
	case "ne", "!=":
		return !isEqual
	case "le", "<=":
		return isEqual || realValue <= bt.Threshold
	case "lt", "<":
		return !isEqual && realValue < bt.Threshold
	default:
		return false
	}
}

type TimerRangeTrigger struct {
	Start string
	End   string
	Type  string
}

func (tt *TimerRangeTrigger) Trigger(stats map[string]interface{}) bool {
	now := time.Now()
	start := tt.Start
	end := tt.End
	// now only support the 'daily' type
	var format string
	switch tt.Type {
	case "daily":
	default:
		format = "15:04"
	}

	v := now.Format(format)
	if start > end {
		// for daily type: [23:00, 01:00]
		return start <= v || v <= end
	}

	// for daily type: [01:00, 02:00]
	return start <= v && v <= end
}

type AndTrigger struct {
	Triggers []Base
}

func (at *AndTrigger) Trigger(stats map[string]interface{}) bool {
	for _, t := range at.Triggers {
		if !t.Trigger(stats) {
			return false
		}
	}
	return true
}

func newAndTrigger(triggers ...Base) *AndTrigger {
	var valid []Base
	for _, t := range triggers {
		if t != nil {
			valid = append(valid, t)
		}
	}
	return &AndTrigger{
		Triggers: valid,
	}
}

func NewTrigger(trigger map[string]interface{}) (Base, error) {
	var err error
	checkPeriodSeconds, ok := trigger["checkPeriodSeconds"]
	if !ok {
		checkPeriodSeconds = 60
	}
	_ = checkPeriodSeconds
	condVal, ok := trigger["condition"]
	var conditionTrigger Base
	if ok {
		cond, ok := condVal.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid condition value:%v",
				condVal)
		}

		threshold, err := convertFloat(cond["threshold"])

		if err != nil {
			return nil, fmt.Errorf("invalid threshold value:%v", cond["threshold"])
		}
		conditionTrigger = &BinaryTrigger{
			Operator:  cond["operator"].(string),
			Metric:    cond["metric"].(string),
			Threshold: threshold,
		}
	}
	var timerTrigger Base
	_, ok = trigger["timer"]
	if ok {
		timer := make(map[string]string)
		switch t := trigger["timer"].(type) {
		case map[string]interface{}:
			for k, v := range t {
				timer[k] = v.(string)
			}
		case map[string]string:
			for k, v := range t {
				timer[k] = v
			}
		default:
			err = fmt.Errorf("invalid timer %v", trigger["timer"])
		}
		timerTrigger = &TimerRangeTrigger{
			Start: timer["start"],
			End:   timer["end"],
			Type:  timer["type"],
		}
	}
	if err != nil {
		return nil, err
	}
	return newAndTrigger(timerTrigger, conditionTrigger), err
}

package trigger

import "testing"
import "time"

func TestTimerInRange(t *testing.T) {
	now := time.Now()
	hour, _ := time.ParseDuration("1h")

	var timerTest = []struct {
		start    time.Time
		end      time.Time
		expected bool
	}{
		{now.Add(-hour), now.Add(hour), true},
		{now.Add(-2 * hour), now.Add(-hour), false},
		{now.Add(hour), now.Add(2 * hour), false},
		{now.Add(hour), now.Add(-hour), false},
		{now.Add(2 * hour), now.Add(hour), true},
	}

	for _, tt := range timerTest {
		var timer = make(map[string]string)
		timer["start"] = tt.start.Format("15:04")
		timer["end"] = tt.end.Format("15:04")

		triggerExpr := map[string]interface{}{
			"timer": timer,
		}
		tg, _ := NewTrigger(triggerExpr)
		stats := make(map[string]interface{})
		if tg.Trigger(stats) != tt.expected {
			t.Errorf("failed to trigger timer %v, expected=%v", timer, tt.expected)
		}
	}
}

func TestCondition(t *testing.T) {
	metric := "numOfSamples"
	stats := map[string]interface{}{
		metric: 500,
	}
	var samplesTest = []struct {
		operator  string
		threshold interface{}
		expected  bool
	}{
		{"gt", 499, true},
		{">", 499, true},
		{"gt", 500, false},
		{">", 500, false},
		{"ge", 501, false},
		{">=", 501, false},
		{"ge", 500, true},
		{">=", 500, true},
		{"eq", 500, true},
		{"=", 500, true},
		{"=", 501, false},
		{"!=", 500, false},
		{"!=", 501, true},
		{"lt", 501, true},
		{"<", 501, true},
		{"lt", 500, false},
		{"<", 500, false},
		{"le", 501, true},
		{"<=", 501, true},
		{"le", 500, true},
		{"<=", 500, true},
		{"le", 499, false},
		{"<=", 499, false},
	}

	for _, st := range samplesTest {
		var condition = make(map[string]interface{})
		condition["operator"] = st.operator
		condition["metric"] = metric
		condition["threshold"] = st.threshold

		trigger := map[string]interface{}{
			"condition": condition,
		}
		tg, _ := NewTrigger(trigger)
		if st.expected != tg.Trigger(stats) {
			t.Errorf("failed to trigger condition:%+v, expected:%v", condition, st.expected)
		}
	}
}

func TestIndexCondition(t *testing.T) {
	metric := "pricision_delta"
	subMetric := metric + "[1]"

	stats := map[string]interface{}{
		metric: []float32{0.1, 0.2},
	}
	var samplesTest = []struct {
		operator  string
		threshold float32
		expected  bool
	}{
		{"gt", 0.3, false},
		{"gt", 0.2, false},
		{"gt", 0.1, true},
		{"ge", 0.2, true},
	}

	for _, st := range samplesTest {
		var condition = make(map[string]interface{})
		condition["operator"] = st.operator
		condition["metric"] = subMetric
		condition["threshold"] = st.threshold

		trigger := map[string]interface{}{
			"condition": condition,
		}
		tg, _ := NewTrigger(trigger)
		if st.expected != tg.Trigger(stats) {
			t.Errorf("failed to trigger condition:%+v, expected:%v", condition, st.expected)
		}
	}
}

func TestTimerAndCondition(t *testing.T) {
	now := time.Now()
	hour, _ := time.ParseDuration("1h")
	metric := "numOfSamples"
	stats := map[string]interface{}{
		metric: 500,
	}
	var samplesTest = []struct {
		start     time.Time
		end       time.Time
		operator  string
		threshold int
		expected  bool
	}{
		{now.Add(-hour), now.Add(hour), "ge", 500, true},
		{now.Add(-hour), now.Add(hour), "ge", 600, false},
		{now.Add(-2 * hour), now.Add(-hour), "ge", 500, false},
	}
	for _, st := range samplesTest {
		var timer = make(map[string]interface{})
		timer["start"] = st.start.Format("15:04")
		timer["end"] = st.end.Format("15:04")
		var condition = make(map[string]interface{})
		condition["operator"] = st.operator
		condition["metric"] = metric
		condition["threshold"] = st.threshold

		trigger := map[string]interface{}{
			"timer":     timer,
			"condition": condition,
		}
		tg, _ := NewTrigger(trigger)
		if tg.Trigger(stats) != st.expected {
			t.Errorf("failed to trigger %v, expected=%v", trigger, st.expected)
		}
	}
}
